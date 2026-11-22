package io.github.sceneview.demo.locusgs

import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.sqrt

internal data class ColmapImage(val path: String, val c2w: DoubleArray, val intrinsics: SceneIntrinsics)

internal fun colmapModels(root: File): List<File> = root.walkTopDown()
    .filter { it.isFile && it.name == "cameras.bin" && File(it.parentFile, "images.bin").isFile }
    .map { it.parentFile!! }
    .take(2).toList()

internal fun findColmapModel(root: File): File? {
    val models = colmapModels(root)
    require(models.size <= 1) { "Multiple COLMAP binary models in $root" }
    return models.singleOrNull()
}

/** Joins poses by full image-relative path, never by a basename or an array index. */
internal fun colmapPosesByFrame(frames: List<SceneFrame>, images: List<ColmapImage>): Map<String, DoubleArray> {
    val byPath = images.associateBy { it.path }
    require(byPath.size == images.size) { "Ambiguous COLMAP image paths" }
    val matchedPaths = HashSet<String>()
    return frames.associate { frame ->
        // COLMAP names are relative to images/. Some Nerfstudio exporters use
        // image paths relative to the dataset instead, so accept that exact key too.
        val exact = byPath[frame.id]
        val relative = frame.id.takeIf { it.startsWith("images/") }
            ?.removePrefix("images/")?.let(byPath::get)
        require(exact == null || relative == null) { "Ambiguous COLMAP image for ${frame.id}" }
        val image = exact ?: relative ?: throw IllegalArgumentException("No COLMAP image for ${frame.id}")
        require(matchedPaths.add(image.path)) { "Multiple frames match COLMAP image ${image.path}" }
        frame.id to image.c2w
    }
}

internal fun readColmap(dir: File, checkpoint: () -> Unit = {}): List<ColmapImage> {
    val cameras = DataInputStream(File(dir, "cameras.bin").inputStream().buffered()).use { input ->
        val count = input.recordCount("camera")
        val result = HashMap<Int, SceneIntrinsics>()
        repeat(count) {
            checkpoint()
            val id = input.intLE()
            val model = input.intLE()
            val width = input.dimension()
            val height = input.dimension()
            require(model == 0 || model == 1) {
                "Unsupported COLMAP camera model $model for camera $id (only SIMPLE_PINHOLE and PINHOLE are supported)"
            }
            val fx = input.doubleLE()
            val fy = if (model == 0) fx else input.doubleLE()
            val intrinsics = SceneIntrinsics(fx, fy, input.doubleLE(), input.doubleLE(), width, height)
            require(result.put(id, intrinsics) == null) { "Duplicate COLMAP camera ID: $id" }
        }
        require(input.read() == -1) { "Trailing data in cameras.bin" }
        result
    }
    return DataInputStream(File(dir, "images.bin").inputStream().buffered()).use { input ->
        val count = input.recordCount("image")
        val observationBuffer = ByteArray(64 * 1024)
        val paths = HashSet<String>()
        val ids = HashSet<Int>()
        val images = List(count) {
            checkpoint()
            val id = input.intLE()
            require(ids.add(id)) { "Duplicate COLMAP image ID: $id" }
            val qw = input.doubleLE()
            val qx = input.doubleLE()
            val qy = input.doubleLE()
            val qz = input.doubleLE()
            val tx = input.doubleLE()
            val ty = input.doubleLE()
            val tz = input.doubleLE()
            val camera = input.intLE()
            val lens = cameras[camera] ?: throw IllegalArgumentException("Unknown COLMAP camera ID: $camera")
            val path = input.imagePath()
            require(paths.add(path)) { "Ambiguous COLMAP image path: $path" }
            val pose = colmapPose(qw, qx, qy, qz, tx, ty, tz, path)
            val observations = input.longLE()
            // Each observation is (double x, double y, uint64 point3D_id).
            require(observations in 0..Int.MAX_VALUE.toLong() / 24) { "Invalid COLMAP observation count: $path" }
            var remaining = (observations * 24).toInt()
            while (remaining > 0) {
                checkpoint()
                val chunk = minOf(remaining, observationBuffer.size)
                input.readFully(observationBuffer, 0, chunk)
                remaining -= chunk
            }
            ColmapImage(path, pose, lens)
        }
        require(input.read() == -1) { "Trailing data in images.bin" }
        images
    }
}

private fun DataInputStream.recordCount(kind: String): Int = longLE().also {
    require(it in 1..MAX_SCENE_FRAMES.toLong()) { "Invalid COLMAP $kind count: $it" }
}.toInt()

private fun DataInputStream.dimension(): Int = longLE().also {
    require(it in 1..Int.MAX_VALUE.toLong()) { "Invalid COLMAP image dimension: $it" }
}.toInt()

private fun DataInputStream.intLE(): Int = Integer.reverseBytes(readInt())
private fun DataInputStream.longLE(): Long = java.lang.Long.reverseBytes(readLong())
private fun DataInputStream.doubleLE(): Double = Double.fromBits(longLE())

private fun DataInputStream.imagePath(): String {
    val bytes = ByteArrayOutputStream()
    while (true) {
        val value = readUnsignedByte()
        if (value == 0) break
        require(bytes.size() < 4096) { "COLMAP image path is too long" }
        bytes.write(value)
    }
    return scenePath(Charsets.UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes.toByteArray())).toString())
}

private fun colmapPose(
    qw: Double, qx: Double, qy: Double, qz: Double,
    tx: Double, ty: Double, tz: Double, path: String,
): DoubleArray {
    val norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    require(norm.isFinite() && norm > 1e-12) { "Invalid COLMAP rotation: $path" }
    val w = qw / norm; val x = qx / norm; val y = qy / norm; val z = qz / norm
    val r = doubleArrayOf(
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    )
    val pose = DoubleArray(16)
    for (row in 0..2) {
        val rx = r[row]; val ry = r[3 + row]; val rz = r[6 + row]
        // Invert world-to-camera then flip the camera Y and Z axes (CV to GL).
        pose[4 * row] = rx
        pose[4 * row + 1] = -ry
        pose[4 * row + 2] = -rz
        pose[4 * row + 3] = -(rx * tx + ry * ty + rz * tz)
    }
    pose[15] = 1.0
    require(pose.all { it.isFinite() }) { "Invalid COLMAP translation: $path" }
    return pose
}
