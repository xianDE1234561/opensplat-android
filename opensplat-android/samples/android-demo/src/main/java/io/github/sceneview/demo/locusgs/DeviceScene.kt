package io.github.sceneview.demo.locusgs

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONObject
import java.io.File

/**
 * A scene on the phone: frames, poses and intrinsics, ready for LocusGS.
 *
 * This is what replaces the PC side of the pipeline. Previously the device was
 * fed `images.f32` and `plucker.f32` that `scripts/export_device_inputs.py`
 * produced, which meant "on-device inference" still needed a PC for every
 * preprocessing step. Now the phone opens the same nerfstudio-format dataset
 * Brush trains on, picks the views itself, and does the crop/resize/normalise
 * and ray construction locally.
 *
 * Format is nerfstudio `transforms.json`:
 *   { fl_x, fl_y, cx, cy, w, h, frames: [{ file_path, transform_matrix, ... }] }
 * `transform_matrix` is a 4x4 camera-to-world in the OpenGL convention, which
 * is what LocusGS's Provider consumes directly.
 */
class DeviceScene(
    val root: File,
    val frames: List<Frame>,
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val width: Int,
    val height: Int,
) {
    data class Frame(
        val file: File,
        val c2w: DoubleArray,
        val fx: Double,
        val fy: Double,
        val cx: Double,
        val cy: Double,
    )

    /**
     * Loads and decodes [indices].
     *
     * Bitmaps are decoded at their native size and cropped/resized inside
     * [LocusGSPreprocess] so the same crop geometry drives both the pixels and
     * the intrinsics. Decoding straight to the target size would make the
     * effective crop ambiguous and shift the image relative to the rays.
     */
    fun loadViews(indices: List<Int>): List<DecodedView> = indices.map { i ->
        val f = frames[i]
        val bmp = BitmapFactory.decodeFile(f.file.absolutePath)
            ?: throw IllegalStateException("cannot decode ${f.file.absolutePath}")
        DecodedView(i, bmp, f)
    }

    data class DecodedView(val index: Int, val bitmap: Bitmap, val frame: Frame)

    companion object {
        fun load(root: File): DeviceScene {
            val tf = File(root, "transforms.json")
            require(tf.isFile) { "missing ${tf.absolutePath}" }
            val doc = JSONObject(tf.readText())
            val framesJson = doc.getJSONArray("frames")
            val frames = (0 until framesJson.length()).map { i ->
                val o = framesJson.getJSONObject(i)
                val path = o.getString("file_path").removePrefix("./")
                val m = o.getJSONArray("transform_matrix")
                val c2w = DoubleArray(16) { k ->
                    m.getJSONArray(k / 4).getDouble(k % 4)
                }
                Frame(
                    file = File(root, path),
                    c2w = c2w,
                    fx = o.optDouble("fl_x", doc.getDouble("fl_x")),
                    fy = o.optDouble("fl_y", doc.getDouble("fl_y")),
                    cx = o.optDouble("cx", doc.getDouble("cx")),
                    cy = o.optDouble("cy", doc.getDouble("cy")),
                )
            }
            require(frames.isNotEmpty()) { "${tf.absolutePath} has no frames" }
            val missing = frames.take(64).map { it.file }.filter { !it.isFile }
            require(missing.isEmpty()) {
                "${missing.size} image files missing under $root, e.g. ${missing.first().name}"
            }
            return DeviceScene(
                root = root,
                frames = frames,
                fx = doc.getDouble("fl_x"),
                fy = doc.getDouble("fl_y"),
                cx = doc.getDouble("cx"),
                cy = doc.getDouble("cy"),
                width = doc.getInt("w"),
                height = doc.getInt("h"),
            )
        }
    }
}
