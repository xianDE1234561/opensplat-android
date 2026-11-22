package io.github.sceneview.demo.locusgs

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption.ATOMIC_MOVE
import java.nio.file.StandardOpenOption.CREATE_NEW
import java.nio.file.StandardOpenOption.WRITE
import java.util.UUID
import java.util.zip.CRC32
import java.util.zip.ZipFile

/** [root] is the private job directory; frame IDs are paths relative to it. */
data class ReconstructionScene(
    val id: String,
    val name: String,
    val root: File,
    val frames: List<SceneFrame>,
    val trainIds: List<String>,
    val validationIds: List<String>,
    val sceneScale: Double,
    /** Import time of the job directory; used to disambiguate same-named imports. */
    val importedAt: Long = root.lastModified(),
)

/** Intrinsics are in native image pixels; [c2w] is row-major OpenGL camera-to-world. */
data class SceneFrame(
    val id: String,
    val file: File,
    val c2w: DoubleArray,
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val width: Int,
    val height: Int,
)

internal data class SavedReconstruction(val id: String, val name: String, val file: File, val run: ReconstructionRun)

object SceneRepository {
    private const val RECORD = ".scene-repository.json"
    private const val MAX_ENTRIES = 32_768
    private const val MAX_FILE_BYTES = 512L * 1024 * 1024
    private const val MAX_ARCHIVE_BYTES = 4L * 1024 * 1024 * 1024

    /** Consumes and closes [input]. The final UUID directory appears only after validation. */
    suspend fun importZip(context: Context, name: String, input: InputStream): ReconstructionScene =
        input.use { source ->
            withContext(Dispatchers.IO) {
                val scenes = scenesDir(context)
                Files.createDirectories(scenes.toPath())
                val id = UUID.randomUUID().toString()
                val staging = Files.createTempDirectory(scenes.toPath(), ".import-").toFile()
                try {
                    val archive = File(staging, "archive.zip")
                    val buffer = ByteArray(64 * 1024)
                    val checkpoint = { ensureActive() }
                    // ZipFile requires the central directory, so a truncated stream cannot
                    // be accepted merely because its early local entries were complete.
                    archive.outputStream().use { out ->
                        var total = 0L
                        while (true) {
                            checkpoint()
                            val n = source.read(buffer)
                            if (n == -1) break
                            total += n
                            require(total <= MAX_ARCHIVE_BYTES) { "ZIP exceeds input size limit" }
                            out.write(buffer, 0, n)
                        }
                    }
                    val unpacked = File(staging, "data").apply { check(mkdir()) }
                    extract(archive, unpacked, buffer, checkpoint)
                    val dataset = findDataset(unpacked)
                    require(dataset.list().orEmpty().none { it.lowercase() in JOB_METADATA_PATHS }) {
                        "ZIP contains reserved reconstruction job metadata"
                    }
                    val scene = readScene(dataset, id, name, checkpoint)
                    val finalRoot = File(scenes, id)
                    val published = scene.copy(root = finalRoot, frames = scene.frames.map {
                        it.copy(file = File(finalRoot, it.id))
                    })
                    val record = JSONObject().put("name", name).toString().toByteArray(Charsets.UTF_8)
                    require(record.size <= MAX_RECORD_BYTES) { "Scene name is too long" }
                    Files.write(File(dataset, RECORD).toPath(), record, CREATE_NEW, WRITE)
                    checkpoint()
                    Files.move(dataset.toPath(), finalRoot.toPath(), ATOMIC_MOVE)
                    published
                } finally {
                    if (!staging.deleteRecursively()) Log.w("SceneRepository", "Could not clean $staging")
                }
            }
        }

    /** Rediscover committed jobs from private storage. Performs disk I/O; call off the UI thread. */
    fun list(context: Context): List<ReconstructionScene> = scenesDir(context).listFiles().orEmpty()
        .asSequence()
        .filter { it.isDirectory && isSceneId(it.name) }
        // Newest first: repeated imports of the same ZIP share a name, so time is the
        // only field that tells them apart in the picker.
        .sortedByDescending { it.lastModified() }
        .mapNotNull { root ->
            val record = File(root, RECORD)
            if (!record.isFile) return@mapNotNull null
            try {
                val name = readSceneJson(record, MAX_RECORD_BYTES).getString("name")
                readScene(root, root.name, name)
            } catch (e: Exception) {
                // One deleted or damaged job should not hide the others.
                Log.w("SceneRepository", "Could not reopen ${root.name}", e)
                null
            }
        }.toList()

    /** Browse published models without decoding every scene’s camera metadata. */
    internal fun results(context: Context): List<SavedReconstruction> = scenesDir(context).listFiles().orEmpty()
        .asSequence()
        .filter { it.isDirectory && isSceneId(it.name) }
        .sortedByDescending { it.lastModified() }
        .mapNotNull { root ->
            try {
                val run = ReconstructionRunStore.load(root)
                val name = run.finalPly ?: run.initialPly ?: return@mapNotNull null
                require(File(name).name == name)
                val file = File(root, name)
                if (!file.isFile || file.length() == 0L) return@mapNotNull null
                val title = readSceneJson(File(root, RECORD), MAX_RECORD_BYTES).getString("name")
                SavedReconstruction(root.name, title, file, run)
            } catch (e: Exception) {
                Log.w("SceneRepository", "Could not read result for ${root.name}", e)
                null
            }
        }.toList()

    private fun scenesDir(context: Context) = File(context.filesDir, "reconstruction/scenes")

    private fun isSceneId(value: String): Boolean = try {
        UUID.fromString(value).toString() == value
    } catch (_: IllegalArgumentException) {
        false
    }

    private fun extract(archive: File, root: File, buffer: ByteArray, checkpoint: () -> Unit) {
        ZipFile(archive).use { zip ->
            require(zip.size() in 1..MAX_ENTRIES) { "ZIP is empty or has too many entries" }
            val paths = HashSet<String>()
            val entries = zip.entries()
            var total = 0L
            while (entries.hasMoreElements()) {
                checkpoint()
                val entry = entries.nextElement()
                val path = scenePath(entry.name)
                require(paths.add(path)) { "Duplicate ZIP path: $path" }
                val target = File(root, path)
                require(entry.size in 0..MAX_FILE_BYTES && entry.size <= MAX_ARCHIVE_BYTES - total) {
                    "ZIP extraction size limit exceeded: $path"
                }
                if (entry.isDirectory) {
                    require(entry.size == 0L) { "ZIP directory has data: $path" }
                    Files.createDirectories(target.toPath())
                    continue
                }
                Files.createDirectories(target.parentFile!!.toPath())
                // ZIP links are never materialized as links. CREATE_NEW also catches file/dir
                // conflicts and aliases on a case-insensitive filesystem.
                val crc = CRC32()
                var size = 0L
                Files.newOutputStream(target.toPath(), CREATE_NEW, WRITE).use { out ->
                    zip.getInputStream(entry).use { data ->
                        while (true) {
                            checkpoint()
                            val n = data.read(buffer)
                            if (n == -1) break
                            size += n
                            total += n
                            require(size <= entry.size && total <= MAX_ARCHIVE_BYTES) {
                                "ZIP extraction size limit exceeded: $path"
                            }
                            out.write(buffer, 0, n)
                            crc.update(buffer, 0, n)
                        }
                    }
                }
                require(size == entry.size && crc.value == entry.crc) { "Corrupt ZIP entry: $path" }
            }
        }
    }

    private fun findDataset(root: File): File {
        val transforms = root.walkTopDown().filter { it.isFile && it.name == "transforms.json" }
            .take(2).toList()
        require(transforms.size <= 1) { "ZIP has multiple transforms.json datasets" }
        transforms.singleOrNull()?.let { return it.parentFile!! }

        val models = colmapModels(root)
        require(models.size == 1) { "ZIP must contain transforms.json or a single COLMAP model" }
        return generateSequence(models.single()) { if (it == root) null else it.parentFile }
            .firstOrNull { File(it, "images").isDirectory }
            ?: throw IllegalArgumentException("COLMAP dataset has no images/ directory")
    }

    private fun readScene(
        root: File, id: String, name: String, checkpoint: () -> Unit = {},
    ): ReconstructionScene {
        val transforms = File(root, "transforms.json")
        val doc = transforms.takeIf { it.isFile }?.let(::readSceneJson)
        val model = findColmapModel(root)
        val frames = if (doc != null) {
            readTransforms(root, doc, checkpoint).also { frames ->
                if (model != null) colmapPosesByFrame(frames, readColmap(model, checkpoint))
            }
        } else {
            requireNotNull(model) { "Missing transforms.json or COLMAP binary model" }
            readColmap(model, checkpoint).map { image ->
                checkpoint()
                sceneFrame(root, "images/${image.path}", image.c2w, image.intrinsics)
            }
        }
        val ids = frames.map { it.id }.sorted()
        require(ids.isNotEmpty() && ids.toSet().size == ids.size) { "Empty scene or duplicate frame paths" }
        // The holdout belongs to the path, not to the order used by an exporter.
        val (validation, train) = ids.withIndex().partition { it.index % 5 == 4 }

        // An optional scene_metadata.json works for either format. Its explicit override
        // takes precedence over scene_scale in transforms.json; poses are not rescaled here.
        val metadataFile = File(root, "scene_metadata.json")
        val metadata = metadataFile.takeIf { it.isFile }?.let { readSceneJson(it, MAX_RECORD_BYTES) }
        val scale = when {
            metadata?.has("scene_scale") == true -> metadata.getDouble("scene_scale")
            doc?.has("scene_scale") == true -> doc.getDouble("scene_scale")
            else -> 1.0
        }
        require(scale.isFinite() && scale > 0.0) { "scene_scale must be finite and positive" }
        return ReconstructionScene(id, name, root, frames, train.map { it.value }, validation.map { it.value }, scale)
    }

    private const val MAX_RECORD_BYTES = 16 * 1024
}

internal const val MAX_SCENE_FRAMES = 20_000
private const val MAX_TRANSFORMS_BYTES = 16 * 1024 * 1024

internal fun readSceneJson(file: File, maxBytes: Int = MAX_TRANSFORMS_BYTES): JSONObject {
    require(file.length() in 1..maxBytes.toLong()) { "Invalid metadata size: ${file.name}" }
    return JSONObject(file.readText(Charsets.UTF_8))
}

/** Normalize benign aliases but never interpret parent traversal or absolute paths. */
internal fun scenePath(raw: String): String {
    require(raw.length in 1..4096 && !raw.startsWith('/') && !raw.startsWith('\\') &&
        raw.none { it == ':' || it.code < 32 || it.code == 127 }) { "Unsafe scene path: $raw" }
    val parts = raw.replace('\\', '/').split('/').filter { it.isNotEmpty() && it != "." }
    require(parts.size in 1..32 && parts.none { it == ".." }) { "Unsafe scene path: $raw" }
    return parts.joinToString("/")
}

internal data class SceneIntrinsics(
    val fx: Double, val fy: Double, val cx: Double, val cy: Double,
    val width: Int?, val height: Int?,
)

private fun sceneFrame(root: File, id: String, pose: DoubleArray, lens: SceneIntrinsics): SceneFrame {
    require(id.substringBefore('/').lowercase() !in RESERVED_IMAGE_ROOTS) { "Reserved scene path: $id" }
    val file = File(root, id)
    require(file.isFile && file.canonicalFile.toPath().startsWith(root.canonicalFile.toPath())) {
        "Missing or unsafe scene image: $id"
    }
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.path, bounds)
    require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Unreadable scene image: $id" }
    require((lens.width == null || lens.width == bounds.outWidth) &&
        (lens.height == null || lens.height == bounds.outHeight)) {
        "Image dimensions do not match intrinsics: $id"
    }
    require(lens.fx.isFinite() && lens.fx > 0 && lens.fy.isFinite() && lens.fy > 0 &&
        lens.cx.isFinite() && lens.cy.isFinite()) { "Invalid intrinsics for $id" }
    return SceneFrame(id, file, pose, lens.fx, lens.fy, lens.cx, lens.cy, bounds.outWidth, bounds.outHeight)
}

private val JOB_METADATA_PATHS = setOf(
    ".scene-repository.json", "reconstruction.json", "reconstruction.json.new", "reconstruction.json.bak",
)

// Keep image paths distinct from metadata and entries generated in the Brush bundle.
private val RESERVED_IMAGE_ROOTS = JOB_METADATA_PATHS + setOf(
    "transforms.json", "transforms_val.json", "scene_metadata.json", "init.ply",
)

private fun readTransforms(root: File, doc: JSONObject, checkpoint: () -> Unit): List<SceneFrame> {
    val frames = doc.getJSONArray("frames")
    require(frames.length() in 1..MAX_SCENE_FRAMES) { "Invalid transforms frame count" }
    return List(frames.length()) { i ->
        checkpoint()
        val frame = frames.getJSONObject(i)
        fun source(key: String): JSONObject = if (frame.has(key)) frame else doc
        fun number(key: String): Double = source(key).getDouble(key)
        fun dimension(key: String): Int? = source(key).takeIf { it.has(key) }?.let {
            val value = it.getDouble(key)
            require(value in 1.0..Int.MAX_VALUE.toDouble() && value == value.toInt().toDouble()) {
                "Invalid image dimension $key"
            }
            value.toInt()
        }
        val path = scenePath(frame.getString("file_path"))
        val model = source("camera_model").optString("camera_model", "PERSPECTIVE")
        require(model in PINHOLE_MODELS) {
            "Unsupported camera model: $model ($path)"
        }
        for (key in DISTORTION_FIELDS) {
            val owner = source(key)
            require(!owner.has(key) || owner.getDouble(key) == 0.0) { "Unsupported distortion $key: $path" }
        }
        val matrix = frame.getJSONArray("transform_matrix")
        require(matrix.length() == 4) { "Expected a 4x4 transform: $path" }
        val pose = DoubleArray(16)
        for (row in 0..3) {
            val values = matrix.getJSONArray(row)
            require(values.length() == 4) { "Expected a 4x4 transform: $path" }
            for (col in 0..3) pose[row * 4 + col] = values.getDouble(col)
        }
        requireValidPose(pose, path)
        sceneFrame(root, path, pose, SceneIntrinsics(
            number("fl_x"), number("fl_y"), number("cx"), number("cy"), dimension("w"), dimension("h"),
        ))
    }
}

private val PINHOLE_MODELS = setOf("PERSPECTIVE", "OPENCV", "PINHOLE", "SIMPLE_PINHOLE")
private val DISTORTION_FIELDS = arrayOf("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")

private fun requireValidPose(m: DoubleArray, path: String) {
    // Permit rounding in JSON exports but reject non-rigid, reflected, or singular poses.
    val eps = 0.005
    require(m.all { it.isFinite() } && (12..14).all { kotlin.math.abs(m[it]) < eps } &&
        kotlin.math.abs(m[15] - 1) < eps) { "Invalid homogeneous transform: $path" }
    for (i in 0..2) for (j in i..2) {
        val dot = m[i] * m[j] + m[4 + i] * m[4 + j] + m[8 + i] * m[8 + j]
        require(kotlin.math.abs(dot - (if (i == j) 1.0 else 0.0)) < eps) { "Invalid rotation: $path" }
    }
    val det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
        m[2] * (m[4] * m[9] - m[5] * m[8])
    require(det > 0) { "Reflected rotation: $path" }
}
