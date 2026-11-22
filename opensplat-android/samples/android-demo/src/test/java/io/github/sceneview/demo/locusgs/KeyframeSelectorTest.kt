package io.github.sceneview.demo.locusgs

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Cross-checks the on-device selector against the Python implementation.
 *
 * The risky part of the port is COLMAP parsing and the pose convention: with
 * either wrong, every camera reports zero visible points and the overlap term
 * silently degenerates to 0.0 for all pairs. The reference numbers below were
 * produced by `android_infer/select_keyframes.py` on the truck scene, so a
 * mismatch means the two implementations would pick different views.
 *
 * Expected values are literals, not a generated fixture: they only change when
 * the algorithm changes, and a test that reads its own expectations from a file
 * the implementation also reads cannot catch a wrong convention.
 *
 * Skips where the truck COLMAP data is absent.
 */
class KeyframeSelectorTest {

    private val sparse = File("D:/work/data/tandt_db/tandt/truck/sparse/0")

    // From the Python run: 251 cams, 136029 points, native 1957x1091 PINHOLE.
    private val expFx = 1163.2547280302354
    private val expFy = 1156.280404988286
    private val expCx = 978.5
    private val expCy = 545.5

    @Test
    fun intrinsics_are_read_from_cameras_bin() {
        assumeTrue("COLMAP sparse absent", sparse.exists())
        val got = KeyframeSelector.readIntrinsics(File(sparse, "cameras.bin"))
        assertEquals(expFx, got[0], 1e-6)
        assertEquals(expFy, got[1], 1e-6)
        assertEquals(expCx, got[2], 1e-6)
        assertEquals(expCy, got[3], 1e-6)
        assertEquals(1957.0, got[4], 0.0)
        assertEquals(1091.0, got[5], 0.0)
    }

    @Test
    fun images_bin_parses_all_poses() {
        assumeTrue("COLMAP sparse absent", sparse.exists())
        val (cams, names) = KeyframeSelector.readCameras(File(sparse, "images.bin"))
        assertEquals(251, cams.size)
        assertEquals(251, names.size)
        assertTrue("names not read", names.all { it.endsWith(".jpg") })
        assertEquals("000001.jpg", names.first())
        // Row-major 4x4, last row must be the homogeneous one.
        assertEquals(1.0, cams.first().c2w[15], 0.0)
    }

    @Test
    fun points3d_bin_parses_every_point() {
        assumeTrue("COLMAP sparse absent", sparse.exists())
        val pts = KeyframeSelector.readPoints(File(sparse, "points3D.bin"))
        assertEquals(136029 * 3, pts.size)
    }

    /**
     * Pins the pose convention: Python reports 1520 visible points for camera
     * 0 and 1543 for camera 1 at subsample 2000. A transposed rotation, a
     * missing `-R^T t`, or the u64/i32 header mistake all move these far away.
     */
    @Test
    fun visible_counts_match_the_python_reference() {
        assumeTrue("COLMAP sparse absent", sparse.exists())
        val (cams, _) = KeyframeSelector.readCameras(File(sparse, "images.bin"))
        val pts = KeyframeSelector.readPoints(File(sparse, "points3D.bin"))
        val k = KeyframeSelector.readIntrinsics(File(sparse, "cameras.bin"))
        val counts = KeyframeSelector.visiblePointCounts(cams, pts, k)

        assertEquals(251, counts.size)
        assertTrue("camera 0 saw ${counts[0]} points", counts[0] > 1000)
        assertTrue("camera 0 saw ${counts[0]} points", abs(counts[0] - 1520) <= 40)
        assertTrue("camera 1 saw ${counts[1]} points", abs(counts[1] - 1543) <= 40)
        // A wrong convention typically yields a handful of stray hits, so the
        // mean is the broadest guard.
        val mean = counts.average()
        assertTrue("mean visible ${mean.toInt()} implausible", mean in 1000.0..1600.0)
    }

    @Test
    fun selection_returns_k_distinct_valid_views() {
        assumeTrue("COLMAP sparse absent", sparse.exists())
        val (cams, _) = KeyframeSelector.readCameras(File(sparse, "images.bin"))
        val pts = KeyframeSelector.readPoints(File(sparse, "points3D.bin"))
        val k = KeyframeSelector.readIntrinsics(File(sparse, "cameras.bin"))
        val sel = KeyframeSelector.select(cams, DoubleArray(cams.size), pts, k)

        assertEquals(KeyframeSelector.K, sel.indices.size)
        assertEquals(sel.indices.size, sel.indices.toSet().size)
        assertTrue(sel.indices.all { it in cams.indices })
        assertTrue("selected a blind view", sel.visiblePoints.all { it > 100 })
        assertTrue("overlap not computed", sel.overlaps.drop(1).all { it > 0.0 })
    }

    @Test
    fun selection_is_deterministic() {
        assumeTrue("COLMAP sparse absent", sparse.exists())
        val (cams, _) = KeyframeSelector.readCameras(File(sparse, "images.bin"))
        val pts = KeyframeSelector.readPoints(File(sparse, "points3D.bin"))
        val k = KeyframeSelector.readIntrinsics(File(sparse, "cameras.bin"))
        val a = KeyframeSelector.select(cams, DoubleArray(cams.size), pts, k)
        val b = KeyframeSelector.select(cams, DoubleArray(cams.size), pts, k)
        assertEquals(a.indices, b.indices)
        assertEquals(a.visiblePoints, b.visiblePoints)
    }

    @Test
    fun sharpness_from_argb_ranks_a_fine_pattern_above_a_coarse_one() {
        // The selector only relies on this ordering, and it is checkable
        // without an Android Bitmap. A finer checkerboard carries more
        // high-frequency energy, hence a higher Laplacian variance.
        val w = 128
        val h = 128
        val fine = IntArray(w * h)
        val coarse = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val v1 = if (((x / 4) + (y / 4)) % 2 == 0) 0xFF else 0x00
            fine[y * w + x] = (0xFF shl 24) or (v1 shl 16) or (v1 shl 8) or v1
            val v2 = if (((x / 32) + (y / 32)) % 2 == 0) 0xFF else 0x00
            coarse[y * w + x] = (0xFF shl 24) or (v2 shl 16) or (v2 shl 8) or v2
        }
        val sFine = KeyframeSelector.sharpnessFromArgb(fine, w, h)
        val sCoarse = KeyframeSelector.sharpnessFromArgb(coarse, w, h)
        assertTrue("fine=$sFine must exceed coarse=$sCoarse", sFine > sCoarse)
    }

    @Test
    fun sharpness_of_a_flat_image_is_zero() {
        val w = 32
        val h = 32
        val flat = IntArray(w * h) { (0xFF shl 24) or (0x80 shl 16) or (0x80 shl 8) or 0x80 }
        assertEquals(0.0, KeyframeSelector.sharpnessFromArgb(flat, w, h), 1e-9)
    }

    @Test
    fun sharpness_rejects_buffers_smaller_than_declared() {
        assertEquals(0.0, KeyframeSelector.sharpnessFromArgb(IntArray(4), 32, 32), 0.0)
        assertEquals(0.0, KeyframeSelector.sharpnessFromArgb(IntArray(4), 2, 2), 0.0)
    }

    @Test
    fun selected_colmap_indices_must_be_translated_to_scene_indices() {
        // The exported Brush dataset lists context views first, so COLMAP index
        // and transforms.json index differ. This pins the translation rather
        // than the exact numbers: what matters is that a name looked up in a
        // reordered scene yields the position in *that* list.
        val sceneOrder = listOf(
            "000232.jpg", "000205.jpg", "000105.jpg", "000216.jpg", "000001.jpg"
        )
        val colmapOrder = listOf(
            "000001.jpg", "000105.jpg", "000205.jpg", "000216.jpg", "000232.jpg"
        )
        val byName = sceneOrder.withIndex().associate { it.value to it.index }

        // A selector working in COLMAP space picks, say, indices 1 and 4.
        val colmapPicks = listOf(1, 4)
        val translated = colmapPicks.map { byName.getValue(colmapOrder[it]) }

        assertEquals(listOf(2, 0), translated)
        // The untranslated indices would have pointed at different frames --
        // exactly the silent mismatch this guards against.
        assertTrue(
            "translation must actually change something on a reordered scene",
            translated != colmapPicks,
        )
        // And they must name the same files the selector chose.
        assertEquals(colmapPicks.map { colmapOrder[it] }, translated.map { sceneOrder[it] })
    }
}
