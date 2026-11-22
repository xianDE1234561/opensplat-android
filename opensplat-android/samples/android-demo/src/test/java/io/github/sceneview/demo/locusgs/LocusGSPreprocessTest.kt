package io.github.sceneview.demo.locusgs

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the Kotlin preprocessing port to the PyTorch reference.
 *
 * A wrong pose normalisation or a transposed rotation does not throw -- it
 * produces Gaussians in the wrong coordinate frame, which renders as a
 * plausible but shifted scene. These tests catch that before it reaches a
 * device, using numbers produced by the real `Provider._preprocess` and
 * `ray_condition` (see `scripts/dump_preprocess_reference.py`).
 */
class LocusGSPreprocessTest {

    /** The pose pair the reference script used. */
    private fun poses(): Array<DoubleArray> {
        val a = doubleArrayOf(
            1.0, 0.0, 0.0, 0.5,
            0.0, -1.0, 0.0, -0.3,
            0.0, 0.0, -1.0, 2.0,
            0.0, 0.0, 0.0, 1.0,
        )
        val ang = 0.35
        val c = kotlin.math.cos(ang)
        val s = kotlin.math.sin(ang)
        val b = doubleArrayOf(
            c, 0.0, s, 1.1,
            0.0, 1.0, 0.0, 0.2,
            -s, 0.0, c, 2.4,
            0.0, 0.0, 0.0, 1.0,
        )
        return arrayOf(a, b)
    }

    @Test
    fun invert4x4_round_trips() {
        val m = poses()[1]
        val inv = LocusGSPreprocess.invert4x4(m)
        val id = LocusGSPreprocess.multiply4x4(m, inv)
        val expected = doubleArrayOf(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        for (i in expected.indices) assertEquals(expected[i], id[i], 1e-12)
    }

    @Test
    fun invert4x4_handles_a_pivot_in_the_last_row() {
        // A pose whose rotation puts the largest magnitude in the bottom row:
        // naive elimination without pivoting would divide by ~1e-17 here.
        val m = doubleArrayOf(
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        val inv = LocusGSPreprocess.invert4x4(m)
        for (i in m.indices) assertEquals(m[i], inv[i], 1e-12)  // its own inverse
    }

    @Test
    fun first_camera_normalises_to_identity() {
        // inv(c2w[0]) @ c2w[0] is the identity, whatever c2w[0] was.
        val n = LocusGSPreprocess.normalizeCameras(poses())
        val expected = doubleArrayOf(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        for (i in expected.indices) assertEquals("element $i", expected[i], n[0][i], 1e-9)
    }

    @Test
    fun normalised_poses_match_pytorch() {
        // Values from android_infer/preprocess_ref.json, produced by
        // Provider._preprocess with camera_normalization_method='first_cam'
        // and scene_scale=0.15.
        val n = LocusGSPreprocess.normalizeCameras(poses())
        val expected = doubleArrayOf(
            // cam 0: identity (translation is zero, so nothing to scale)
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        for (i in expected.indices) assertEquals("cam0 element $i", expected[i], n[0][i], 1e-9)

        // Camera 1 is expressed relative to camera 0 and its translation is
        // scaled by scene_scale. These are the reference values verbatim.
        val ref1 = REF_CAM1
        for (i in ref1.indices) {
            assertEquals("cam1 element $i", ref1[i], n[1][i], 1e-9)
        }
    }

    @Test
    fun plucker_matches_pytorch_probe_pixels() {
        val n = LocusGSPreprocess.normalizeCameras(poses())
        val h = 8
        val w = 8
        val pk = LocusGSPreprocess.rayCondition(40.0, 41.0, 4.0, 3.5, n[0], h, w)

        // Probe pixels from the reference file.
        fun at(x: Int, y: Int, c: Int) = pk[c * h * w + y * w + x]

        val ref00 = doubleArrayOf(0.0, -0.0, 0.0, -0.086936, -0.072699, 0.993558)
        val refMid = doubleArrayOf(0.0, 0.0, 0.0, 0.012495, 0.024381, 0.999625)
        for (c in 0 until 6) {
            assertEquals("v0[0,0] channel $c", ref00[c], at(0, 0, c).toDouble(), 1e-5)
            assertEquals("v0[mid] channel $c", refMid[c], at(w / 2, h / 2, c).toDouble(), 1e-5)
        }
    }

    @Test
    fun plucker_directions_are_unit_length() {
        val n = LocusGSPreprocess.normalizeCameras(poses())
        val h = 8
        val w = 8
        val pk = LocusGSPreprocess.rayCondition(40.0, 41.0, 4.0, 3.5, n[0], h, w)
        val plane = h * w
        for (p in 0 until plane) {
            val dx = pk[3 * plane + p]
            val dy = pk[4 * plane + p]
            val dz = pk[5 * plane + p]
            val norm = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
            // get_rays_from_uvs normalises the direction before rotating it,
            // and the rotation is orthonormal, so this must hold everywhere.
            assertEquals("pixel $p", 1.0, norm.toDouble(), 1e-5)
        }
    }

    @Test
    fun plucker_sum_matches_pytorch() {
        // The strongest single check: a wrong axis order, a transposed
        // rotation, or an off-by-one in the pixel grid all move this.
        val n = LocusGSPreprocess.normalizeCameras(poses())
        val h = 8
        val w = 8
        var sum = 0.0
        for (v in 0 until 2) {
            val pk = LocusGSPreprocess.rayCondition(40.0, 41.0, 4.0, 3.5, n[v], h, w)
            for (x in pk) sum += x.toDouble()
        }
        assertEquals(REF_PLUCKER_SUM, sum, 1e-3)
    }

    @Test
    fun crop_geometry_matches_the_python_order() {
        // 640x480 source into 448x256: the target aspect (1.75) is wider than
        // the source (1.333), so the fitted crop is limited by height and the
        // shift is a negative (crop) on x and zero on y.
        val g = LocusGSPreprocess.cropGeometry(640, 480, 448, 256)
        val ratio = minOf(480.0 / 256.0, 640.0 / 448.0)
        assertEquals((448 * ratio).toInt().toDouble(), g[4], 0.0)  // cropW
        assertEquals((256 * ratio).toInt().toDouble(), g[5], 0.0)  // cropH
        assertEquals((g[4] - 640) / 2.0, g[0], 1e-9)               // shiftX
        assertEquals((g[5] - 480) / 2.0, g[1], 1e-9)               // shiftY
        assertEquals(448.0 / g[4], g[2], 1e-12)                    // scaleX
        assertEquals(256.0 / g[5], g[3], 1e-12)                    // scaleY
    }

    @Test
    fun intrinsics_are_shifted_then_scaled() {
        // Matches Provider._preprocess: [(fx*scaleX), (fy*scaleY),
        // ((cx+shiftX)*scaleX), ((cy+shiftY)*scaleY)].
        val k = LocusGSPreprocess.adjustIntrinsics(
            fx = 100.0, fy = 110.0, cx = 50.0, cy = 40.0,
            shiftX = -8.0, shiftY = -4.0, scaleX = 2.0, scaleY = 2.0,
        )
        assertEquals(200.0, k[0], 1e-12)
        assertEquals(220.0, k[1], 1e-12)
        assertEquals(84.0, k[2], 1e-12)   // (50 - 8) * 2
        assertEquals(72.0, k[3], 1e-12)   // (40 - 4) * 2
    }

    companion object {
        /** From preprocess_ref.json, `c2w_normalised[1]`, row-major. */
        val REF_CAM1 = doubleArrayOf(
            0.9393727128473789, 0.0, -0.34289780745545134, 0.6000000000000001,
            0.0, -1.0, 0.0, 0.5,
            -0.34289780745545134, 0.0, -0.9393727128473789, 0.3999999999999999,
            0.0, 0.0, 0.0, 1.0,
        )

        /** Plucker sum over both views at 8x8, from preprocess_ref.json. */
        const val REF_PLUCKER_SUM = -9.981807388365269
    }
}
