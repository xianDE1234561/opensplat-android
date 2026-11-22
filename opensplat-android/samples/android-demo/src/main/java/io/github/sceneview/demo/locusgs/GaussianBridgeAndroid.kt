package io.github.sceneview.demo.locusgs

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * LocusGS 14-channel output -> Brush/3DGS PLY, on device.
 *
 * Mirrors `bridge/__init__.py` exactly; the Python version is the reference and
 * is covered by 16 unit tests. Three rules that must not drift:
 *
 *  * scale is stored as `ln(scale)` (LocusGS already applied the exponential),
 *  * opacity is stored as `logit(opacity)` and **not** shifted by the -2.0 head
 *    bias LocusGS used internally: Brush renders `sigmoid(raw)`, so adding the
 *    bias would make every splat more opaque than LocusGS rendered it,
 *  * colour is the SH DC term `(rgb - 0.5) / SH_C0`.
 */
object GaussianBridgeAndroid {
    const val SH_C0 = 0.28209479177387814f
    const val SCALE_CAP = 0.075f

    data class Stats(
        val count: Int,
        val opacityMedian: Float,
        val scaleMedian: Float,
        val scaleAtCapFraction: Float,
        val finite: Boolean,
        val saturated: Int,
    )

    /**
     * @param raw [N,14] float32: xyz(3), opacity(1), scale(3), quat wxyz(4), rgb(3)
     */
    fun writePly(raw: FloatArray, count: Int, outFile: File): Stats {
        require(raw.size == count * 14) { "expected ${count * 14} floats, got ${raw.size}" }

        val means = FloatArray(count * 3)
        val scales = FloatArray(count * 3)
        val rots = FloatArray(count * 4)
        val sh = FloatArray(count * 3)
        val opac = FloatArray(count)

        var finite = true
        var saturatedCount = 0
        for (i in 0 until count) {
            val o = i * 14
            for (k in 0 until 3) {
                means[i * 3 + k] = raw[o + k]
                val s = raw[o + 4 + k]
                if (s >= SCALE_CAP * 0.999f) saturatedCount++
                scales[i * 3 + k] = kotlin.math.ln(minOf(s, SCALE_CAP).coerceAtLeast(1e-12f))
            }
            val a = raw[o + 3].coerceIn(1e-12f, 1f - 1e-7f)
            opac[i] = kotlin.math.ln(a / (1f - a))
            for (k in 0 until 4) rots[i * 4 + k] = raw[o + 7 + k]
            for (k in 0 until 3) {
                sh[i * 3 + k] = (raw[o + 11 + k].coerceIn(0f, 1f) - 0.5f) / SH_C0
            }
            if (raw[o] .isNaN() || raw[o + 3].isNaN() || scales[i * 3].isNaN()) finite = false
        }

        val header = buildString {
            append("ply\nformat binary_little_endian 1.0\n")
            append("element vertex $count\n")
            for (f in listOf("x", "y", "z", "scale_0", "scale_1", "scale_2",
                             "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
                             "f_dc_0", "f_dc_1", "f_dc_2")) {
                append("property float $f\n")
            }
            append("end_header\n")
        }

        outFile.outputStream().buffered().use { os ->
            os.write(header.toByteArray(Charsets.US_ASCII))
            val buf = ByteBuffer.allocate(56).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until count) {
                buf.clear()
                for (k in 0 until 3) buf.putFloat(means[i * 3 + k])
                for (k in 0 until 3) buf.putFloat(scales[i * 3 + k])
                buf.putFloat(opac[i])
                for (k in 0 until 4) buf.putFloat(rots[i * 4 + k])
                for (k in 0 until 3) buf.putFloat(sh[i * 3 + k])
                os.write(buf.array(), 0, 56)
            }
        }

        return Stats(
            count = count,
            opacityMedian = median(opac),
            scaleMedian = median(FloatArray(count * 3) {
                kotlin.math.exp(scales[it])
            }),
            scaleAtCapFraction = saturatedCount.toFloat() / (count * 3),
            finite = finite,
            saturated = saturatedCount,
        )
    }

    private fun median(v: FloatArray): Float {
        if (v.isEmpty()) return 0f
        val c = v.clone(); c.sort()
        return c[c.size / 2]
    }
}
