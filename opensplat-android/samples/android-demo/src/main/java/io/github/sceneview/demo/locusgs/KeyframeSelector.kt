package io.github.sceneview.demo.locusgs

import android.graphics.Bitmap
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Chooses the 4 views LocusGS is fed, on the phone.
 *
 * Mirrors `android_infer/select_keyframes.py`: Laplacian-variance sharpness,
 * angular spacing around the camera-centre mean, and co-visibility from a
 * COLMAP point cloud. Weights are the Python defaults so a selection here can
 * be compared against one produced on the PC.
 *
 * The co-visibility term is what makes this worth doing on-device rather than
 * shipping a fixed list: on the truck scene it moves eval-view coverage from
 * 0.957 (the hand-picked set) to 0.992.
 */
object KeyframeSelector {

    data class Camera(val c2w: DoubleArray, val imageName: String)

    data class Selection(
        val indices: List<Int>,
        val sharpness: DoubleArray,
        val visiblePoints: List<Int>,
        val overlaps: List<Double>,
    )

    const val K = 4
    private const val MIN_SEP_DEG = 18.0
    private const val MIN_OVERLAP = 0.12
    private const val MAX_OVERLAP = 0.85
    private const val POINT_SUBSAMPLE = 2000

    // ---------------------------------------------------------------- input

    /** Reads a COLMAP points3D.bin; returns packed xyz triples. */
    fun readPoints(file: File): FloatArray {
        val bb = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val n = bb.long.toInt()
        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            bb.long                       // point id
            out[i * 3] = bb.double.toFloat()
            out[i * 3 + 1] = bb.double.toFloat()
            out[i * 3 + 2] = bb.double.toFloat()
            bb.position(bb.position() + 3 + 8)  // rgb + reprojection error
            val trackLen = bb.long.toInt()
            bb.position(bb.position() + trackLen * 8)
        }
        return out
    }

    /**
     * Reads COLMAP images.bin into camera-to-world 4x4 matrices.
     *
     * COLMAP stores world->camera (R, t); the centre is `-R^T t`. Getting that
     * backwards yields a mirrored scene where every view sees different
     * geometry, so the conversion is done once, here.
     *
     * The returned indices line up with [imageNames] for file-order alignment.
     */
    fun readCameras(file: File): Pair<List<Camera>, List<String>> {
        val bb = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val n = bb.long.toInt()
        val cams = ArrayList<Camera>(n)
        val names = ArrayList<String>(n)
        val buf = ByteArray(256)
        for (i in 0 until n) {
            bb.int                                  // camera id
            val qw = bb.double; val qx = bb.double; val qy = bb.double; val qz = bb.double
            val tx = bb.double; val ty = bb.double; val tz = bb.double
            bb.int                                  // image id
            var len = 0
            while (true) {
                val b = bb.get()
                if (b.toInt() == 0) break
                if (len < buf.size) buf[len++] = b
            }
            val name = String(buf, 0, len, Charsets.UTF_8)
            val npts = bb.long.toInt()
            bb.position(bb.position() + npts * 24)

            val qn = sqrt(qw * qw + qx * qx + qy * qy + qz * qz).coerceAtLeast(1e-12)
            val w = qw / qn; val x = qx / qn; val y = qy / qn; val z = qz / qn
            val r = doubleArrayOf(
                1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
            )
            // c2w = [R^T | -R^T t], stored row-major in a 16-element array.
            val m = DoubleArray(16)
            for (row in 0 until 3) for (col in 0 until 3) m[row * 4 + col] = r[col * 3 + row]
            for (row in 0 until 3) {
                m[row * 4 + 3] = -(m[row * 4] * tx + m[row * 4 + 1] * ty + m[row * 4 + 2] * tz)
            }
            m[15] = 1.0
            cams.add(Camera(m, name))
            names.add(name)
        }
        return cams to names
    }

    /** Reads cameras.bin; returns fx, fy, cx, cy, w, h. */
    fun readIntrinsics(file: File): DoubleArray {
        val raw = file.readBytes()
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        bb.int                    // camera id
        bb.int                    // model id
        // This writer emits u64 width/height at offset 8; stock COLMAP uses
        // i32. Probe by plausibility: a wrong read gives ~4.29e9.
        val probe = bb.long
        val w: Long; val h: Long; val paramsAt: Int
        if (probe in 1..100_000) {
            w = probe; h = bb.long; paramsAt = 24
        } else {
            w = bb.long; h = bb.long; paramsAt = 32
        }
        val p = ByteBuffer.wrap(raw, paramsAt, raw.size - paramsAt)
            .order(ByteOrder.LITTLE_ENDIAN)
        val p0 = p.double
        val p1 = p.double
        val p2 = p.double
        val p3 = p.double
        return if (kotlin.math.abs(p0 - p1) > 1e-9) {
            doubleArrayOf(p0, p1, p2, p3, w.toDouble(), h.toDouble())
        } else {
            doubleArrayOf(p0, p0, p1, p2, w.toDouble(), h.toDouble())
        }
    }

    // --------------------------------------------------------------- scoring

    /**
     * Variance of the Laplacian on a 128px thumbnail.
     *
     * Downscaling is not an approximation to avoid: the ranking is what
     * matters, and it is stable at this size while cutting cost ~20x.
     */
    fun sharpness(bitmap: Bitmap): Double {
        val scale = min(1.0, 128.0 / max(bitmap.width, bitmap.height))
        val w = max(3, (bitmap.width * scale).toInt())
        val h = max(3, (bitmap.height * scale).toInt())
        val small = if (w == bitmap.width && h == bitmap.height) bitmap
                    else Bitmap.createScaledBitmap(bitmap, w, h, true)
        val px = IntArray(w * h)
        small.getPixels(px, 0, w, 0, 0, w, h)
        if (small !== bitmap) small.recycle()
        return sharpnessFromArgb(px, w, h)
    }

    /**
     * The actual metric, over an ARGB pixel buffer.
     *
     * Split out so it is testable without an Android Bitmap -- the maths is
     * the part that can be wrong, and it should not need a device to check.
     */
    fun sharpnessFromArgb(px: IntArray, w: Int, h: Int): Double {
        if (w < 3 || h < 3 || px.size < w * h) return 0.0
        val gray = DoubleArray(w * h)
        for (i in 0 until w * h) {
            val c = px[i]
            gray[i] = ((c shr 16 and 0xFF) * 0.299 +
                       (c shr 8 and 0xFF) * 0.587 +
                       (c and 0xFF) * 0.114)
        }
        var sum = 0.0
        var sumSq = 0.0
        var count = 0
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val i = y * w + x
                val lap = -4.0 * gray[i] + gray[i - w] + gray[i + w] + gray[i - 1] + gray[i + 1]
                sum += lap; sumSq += lap * lap; count++
            }
        }
        if (count == 0) return 0.0
        val mean = sum / count
        return sumSq / count - mean * mean
    }

    // ----------------------------------------------------------- visibility

    private fun visiblePoints(
        poses: Array<Camera>, pts: FloatArray, K: DoubleArray,
    ): List<IntArray> {
        val n = pts.size / 3
        val step = max(1, n / POINT_SUBSAMPLE)
        val idx = (0 until n step step).toList()
        val fx = K[0]; val fy = K[1]; val cx = K[2]; val cy = K[3]
        val iw = K[4]; val ih = K[5]

        return poses.map { cam ->
            val m = cam.c2w
            val hit = ArrayList<Int>(idx.size)
            // M is row-major with M[0..2],[4..6],[8..10] = R^T, so R = (R^T)^T
            // and p_cam = R (p_world - centre). That is column-major indexing
            // into M: p_cam[row] = sum_col M[col*4 + row] * (p - c)[col].
            val tx = m[3]; val ty = m[7]; val tz = m[11]
            for (i in idx) {
                val x = pts[i * 3].toDouble() - tx
                val y = pts[i * 3 + 1].toDouble() - ty
                val z = pts[i * 3 + 2].toDouble() - tz
                val px = m[0] * x + m[4] * y + m[8] * z
                val py = m[1] * x + m[5] * y + m[9] * z
                val pz = m[2] * x + m[6] * y + m[10] * z
                if (pz <= 1e-6) continue
                val u = fx * px / pz + cx
                val v = fy * py / pz + cy
                if (u >= 0 && u < iw && v >= 0 && v < ih) hit.add(i)
            }
            hit.toIntArray()
        }
    }

    /** Visible-point count per camera. Exposed so tests can pin the pose math. */
    fun visiblePointCounts(
        cameras: List<Camera>,
        points: FloatArray,
        intrinsics: DoubleArray,
    ): IntArray = visiblePoints(cameras.toTypedArray(), points, intrinsics)
        .map { it.size }
        .toIntArray()

    private fun jaccard(a: IntArray, b: IntArray): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val sa = a.toHashSet()
        val sb = b.toHashSet()
        val inter = sa.count { it in sb }
        val union = sa.size + sb.size - inter
        return if (union == 0) 0.0 else inter.toDouble() / union
    }

    private fun coverageGain(chosen: List<Int>, cand: Int, vis: List<IntArray>): Double {
        val mine = vis[cand]
        if (mine.isEmpty()) return 0.0
        val already = HashSet<Int>()
        for (c in chosen) for (p in vis[c]) already.add(p)
        val new = mine.count { it !in already }
        return new.toDouble() / mine.size
    }

    private fun maxOverlap(cand: Int, chosen: List<Int>, vis: List<IntArray>): Double =
        chosen.maxOfOrNull { jaccard(vis[cand], vis[it]) } ?: 0.0

    private fun angularSep(a: Camera, b: Camera, centre: DoubleArray): Double {
        val va = doubleArrayOf(a.c2w[3] - centre[0], a.c2w[7] - centre[1], a.c2w[11] - centre[2])
        val vb = doubleArrayOf(b.c2w[3] - centre[0], b.c2w[7] - centre[1], b.c2w[11] - centre[2])
        val na = sqrt(va[0] * va[0] + va[1] * va[1] + va[2] * va[2])
        val nb = sqrt(vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2])
        if (na < 1e-9 || nb < 1e-9) return 0.0
        val dot = (va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]) / (na * nb)
        return acos(dot.coerceIn(-1.0, 1.0))
    }

    // -------------------------------------------------------------- selection

    fun select(
        cameras: List<Camera>,
        sharpness: DoubleArray,
        points: FloatArray,
        intrinsics: DoubleArray,
        k: Int = K,
    ): Selection {
        require(cameras.size == sharpness.size) {
            "cameras (${cameras.size}) and sharpness (${sharpness.size}) must match"
        }
        val poses = cameras.toTypedArray()
        val vis = visiblePoints(poses, points, intrinsics)

        val maxSharp = sharpness.maxOrNull()?.takeIf { it > 0 } ?: 1.0
        val sharpN = DoubleArray(sharpness.size) { sharpness[it] / maxSharp }
        val centre = doubleArrayOf(0.0, 0.0, 0.0)
        for (c in poses) { centre[0] += c.c2w[3]; centre[1] += c.c2w[7]; centre[2] += c.c2w[11] }
        centre[0] /= poses.size; centre[1] /= poses.size; centre[2] /= poses.size

        // Seed on the sharpest camera that actually faces the scene centre.
        var seed = 0
        var bestFace = -Double.MAX_VALUE
        for (i in poses.indices) {
            val m = poses[i].c2w
            val dx = centre[0] - m[3]; val dy = centre[1] - m[7]; val dz = centre[2] - m[11]
            val nd = sqrt(dx * dx + dy * dy + dz * dz).coerceAtLeast(1e-9)
            val face = max(0.0, (m[2] * dx + m[6] * dy + m[10] * dz) / nd)
            val score = sharpN[i] * (0.5 + 0.5 * face)
            if (score > bestFace) { bestFace = score; seed = i }
        }

        val chosen = arrayListOf(seed)
        var minSep = Math.toRadians(MIN_SEP_DEG)
        var lo = MIN_OVERLAP
        var hi = MAX_OVERLAP

        while (chosen.size < k) {
            var best = -1
            var bestScore = -Double.MAX_VALUE
            for (i in poses.indices) {
                if (i in chosen) continue
                val sep = chosen.minOf { angularSep(poses[i], poses[it], centre) }
                if (sep < minSep) continue

                var score = Math.pow(sep, 1.5) * (0.35 + 0.65 * sharpN[i])
                val ov = maxOverlap(i, chosen, vis)
                if (ov < lo || ov > hi) continue
                score *= 0.25 + 0.75 * coverageGain(chosen, i, vis)
                score *= 1.0 - 0.5 * max(0.0, ov - 0.5)
                if (score > bestScore) { bestScore = score; best = i }
            }
            if (best < 0) {
                minSep *= 0.5
                lo = max(0.02, lo * 0.6)
                hi = min(0.97, hi + 0.05)
                if (minSep < Math.toRadians(1.0)) {
                    val remaining = poses.indices.filter { it !in chosen }
                    if (remaining.isEmpty()) break
                    best = remaining.maxByOrNull { sharpN[it] }!!
                    chosen.add(best)
                }
                continue
            }
            chosen.add(best)
        }

        return Selection(
            indices = chosen,
            sharpness = sharpness,
            visiblePoints = chosen.map { vis[it].size },
            overlaps = chosen.mapIndexed { n, i ->
                if (n == 0) 0.0 else maxOverlap(i, chosen.subList(0, n), vis)
            },
        )
    }
}
