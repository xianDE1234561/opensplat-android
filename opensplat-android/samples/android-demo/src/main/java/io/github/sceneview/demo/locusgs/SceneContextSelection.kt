package io.github.sceneview.demo.locusgs

import android.graphics.BitmapFactory
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Select up to four training views. Start with the sharpest thumbnail, then
 * spread the views in position and viewing direction without dropping image
 * quality from the score. Path ordering breaks ties, even for stationary captures.
 *
 * When COLMAP is present, use its geometry keyed by image path, but always
 * return the scene's own frames (including its authoritative poses/intrinsics).
 * This does disk I/O and thumbnail decoding; call it on a worker thread.
 */
fun ReconstructionScene.selectContext(): List<SceneFrame> {
    val validation = validationIds.toHashSet()
    val train = trainIds.toHashSet()
    val candidates = frames.filter { it.id in train && it.id !in validation }.sortedBy { it.id }
    if (candidates.isEmpty()) return emptyList()
    val colmap = findColmapModel(root)?.let { colmapPosesByFrame(frames, readColmap(it)) }
    val poses = candidates.map { frame -> colmap?.getValue(frame.id) ?: frame.c2w }
    val sharpness = DoubleArray(candidates.size) { thumbnailSharpness(candidates[it]) }
    val peakSharpness = sharpness.maxOrNull()!!.takeIf { it > 0.0 } ?: 1.0

    // Put baseline lengths on a scene-relative scale. Divide before subtracting
    // so even very large finite camera centers cannot overflow the distance math.
    val centreMagnitude = poses.maxOf { maxOf(abs(it[3]), abs(it[7]), abs(it[11])) }
        .takeIf { it > 0.0 } ?: 1.0
    val centers = poses.map { doubleArrayOf(it[3] / centreMagnitude, it[7] / centreMagnitude, it[11] / centreMagnitude) }
    val span = DoubleArray(3) { j -> centers.maxOf { it[j] } - centers.minOf { it[j] } }
    val diagonal = sqrt(span.sumOf { it * it }).takeIf { it > 0.0 } ?: 1.0
    val minSeparation = DoubleArray(candidates.size) { Double.POSITIVE_INFINITY }
    val used = BooleanArray(candidates.size)
    val count = minOf(KeyframeSelector.K, candidates.size)
    return buildList(count) {
        repeat(count) { step ->
            var best = -1
            var bestScore = Double.NEGATIVE_INFINITY
            for (i in candidates.indices) {
                if (used[i]) continue
                val quality = sharpness[i] / peakSharpness
                val score = if (step == 0) quality else 0.75 * minSeparation[i] + 0.25 * quality
                if (score > bestScore) {
                    best = i
                    bestScore = score
                }
            }
            add(candidates[best])
            used[best] = true
            for (i in candidates.indices) {
                if (used[i]) continue
                val a = centers[i]
                val b = centers[best]
                val dx = a[0] - b[0]; val dy = a[1] - b[1]; val dz = a[2] - b[2]
                val baseline = sqrt(dx * dx + dy * dy + dz * dz) / diagonal
                // Both forward vectors are the negative Z column (OpenGL).
                val p = poses[i]; val q = poses[best]
                val dot = (p[2] * q[2] + p[6] * q[6] + p[10] * q[10]).coerceIn(-1.0, 1.0)
                val direction = (1.0 - dot) / 2.0
                minSeparation[i] = minOf(minSeparation[i], 0.7 * baseline + 0.3 * direction)
            }
        }
    }
}

private fun thumbnailSharpness(frame: SceneFrame): Double {
    var sample = 1
    while (maxOf(frame.width, frame.height) / sample > 256) sample *= 2
    val options = BitmapFactory.Options().apply { inSampleSize = sample }
    val bitmap = BitmapFactory.decodeFile(frame.file.path, options)
        ?: throw IllegalArgumentException("Cannot decode scene image: ${frame.id}")
    return try {
        KeyframeSelector.sharpness(bitmap).coerceAtLeast(0.0)
    } finally {
        bitmap.recycle()
    }
}
