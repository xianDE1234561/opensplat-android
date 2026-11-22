package io.github.sceneview.demo.locusgs

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Pins the on-device bridge to the same rules as `bridge/__init__.py`.
 *
 * Values below come from the Python implementation; if this test fails, the
 * device path and the verified PC path have diverged and the PLY fed to Brush
 * would be subtly wrong (wrong scale/opacity) rather than obviously broken.
 */
class GaussianBridgeTest {

    private fun writeAndRead(raw: FloatArray, count: Int): Map<String, FloatArray> {
        val f = File.createTempFile("gauss", ".ply")
        try {
            GaussianBridgeAndroid.writePly(raw, count, f)
            val bytes = f.readBytes()
            val header = String(bytes, Charsets.US_ASCII)
            val headerEnd = header.indexOf("end_header\n") + "end_header\n".length
            val n = (bytes.size - headerEnd) / 56
            val bb = java.nio.ByteBuffer.wrap(bytes, headerEnd, n * 56)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            // PLY stores vertices interleaved: x,y,z,scale_0,...,f_dc_2 per
            // vertex. Reading field-major would silently return vertex 1's
            // values for vertex 0's channel count > 1.
            val names = listOf("x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity",
                               "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
            val fields = names.associateWith { FloatArray(n) }
            for (i in 0 until n) {
                for (name in names) fields.getValue(name)[i] = bb.getFloat()
            }
            return fields
        } finally {
            f.delete()
        }
    }

    @Test
    fun scale_is_written_as_natural_log() {
        val raw = FloatArray(2 * 14)
        raw[4] = 0.0037f; raw[5] = 0.01f; raw[6] = 0.02f
        val out = writeAndRead(raw, 2)
        assertEquals(kotlin.math.ln(0.0037f), out["scale_0"]!![0], 1e-6f)
        assertEquals(kotlin.math.ln(0.01f), out["scale_1"]!![0], 1e-6f)
    }

    @Test
    fun opacity_is_logit_without_locusgs_head_bias() {
        // LocusGS stores the activated opacity (e.g. 0.0222); Brush expects
        // logit(0.0222) = -3.786. Adding the -2.0 head bias would give -5.786,
        // making every splat more transparent than LocusGS rendered it.
        val raw = FloatArray(14)
        raw[3] = 0.0222f
        val out = writeAndRead(raw, 1)
        assertEquals(kotlin.math.ln(0.0222f / (1f - 0.0222f)), out["opacity"]!![0], 1e-5f)
    }

    @Test
    fun colour_is_sh_dc() {
        val raw = FloatArray(14)
        raw[11] = 0f; raw[12] = 0.5f; raw[13] = 1f
        val out = writeAndRead(raw, 1)
        assertEquals(-0.5f / GaussianBridgeAndroid.SH_C0, out["f_dc_0"]!![0], 1e-6f)
        assertEquals(0f, out["f_dc_1"]!![0], 1e-6f)
        assertEquals(0.5f / GaussianBridgeAndroid.SH_C0, out["f_dc_2"]!![0], 1e-6f)
    }

    @Test
    fun quaternion_order_is_wxyz() {
        val raw = FloatArray(14)
        raw[7] = 1f
        val out = writeAndRead(raw, 1)
        assertEquals(1f, out["rot_0"]!![0], 0f)
        assertEquals(0f, out["rot_3"]!![0], 0f)
    }

    @Test
    fun saturated_scale_is_clamped_and_counted() {
        val raw = FloatArray(14)
        raw[4] = 0.5f
        val f = File.createTempFile("gauss", ".ply")
        try {
            val stats = GaussianBridgeAndroid.writePly(raw, 1, f)
            assertTrue(stats.saturated >= 1)
            assertTrue(stats.finite)
        } finally {
            f.delete()
        }
    }

    @Test
    fun rejects_wrong_buffer_size() {
        val f = File.createTempFile("gauss", ".ply")
        try {
            GaussianBridgeAndroid.writePly(FloatArray(10), 2, f)
            throw AssertionError("expected IllegalArgumentException")
        } catch (_: IllegalArgumentException) {
        } finally {
            f.delete()
        }
    }
}
