package io.github.sceneview.demo.locusgs

import android.graphics.Bitmap
import java.io.File
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Everything LocusGS's encoder expects, computed on the phone.
 *
 * The exported ONNX graphs start at the encoder, so whatever `Provider._preprocess`
 * does before that has to happen here or the pipeline is not really on-device:
 *
 *   center-crop -> resize -> ImageNet normalize   (images)
 *   first_cam pose normalization -> scene scale   (cameras)
 *   intrinsics rescaled for the crop/resize       (cameras)
 *   ray_condition                                 (Plucker)
 *
 * Each step mirrors `locusgs/data/provider.py` and `locusgs/utils/data.py`.
 * Getting any of them wrong does not crash -- it produces plausible Gaussians in
 * the wrong place -- so the port is pinned by tests against Python output.
 */
object LocusGSPreprocess {

    /** ImageNet statistics, the values `Provider.input_normalizer` uses. */
    val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /**
     * Camera-to-world poses in, LocusGS-normalised poses out.
     *
     * `Provider._normalize_camera_mean_cam` is the other option, but the
     * exported graphs were built from a run using `first_cam` (the checkpoint's
     * default), so that is what this reproduces: every pose is pre-multiplied
     * by the inverse of the first one, which puts camera 0 at the origin
     * looking down +Z. Translations are then scaled by `sceneScale`.
     *
     * Both operations are isometries followed by a uniform scale, so the
     * geometry is preserved; only the coordinate frame changes.
     */
    /** Flips a nerfstudio/OpenGL camera-to-world into COLMAP's OpenCV frame. */
    private val GL_TO_CV = doubleArrayOf(
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )

    /**
     * Turns nerfstudio `transform_matrix` values into the poses LocusGS was
     * trained with.
     *
     * Two conversions are required, and both were missing until a full
     * inference comparison caught it (see RESULTS.md "Pose frame mismatch"):
     *
     *  1. nerfstudio stores OpenGL c2w (y and z flipped relative to OpenCV),
     *     so each pose is post-multiplied by diag(1,-1,-1,1). Without this the
     *     rotation comes out with mirrored signs and every Gaussian lands in
     *     the wrong orientation -- no error, just a wrong scene.
     *  2. `Provider._preprocess` applies `first_cam` normalisation and then
     *     scales only the *input* poses. The `c2w_all` that `ray_condition`
     *     consumes is the normalised, **unscaled** one, so `sceneScale` only
     *     affects the rendered view matrices, not the Plücker rays. Applying it
     *     here moved every point along its ray.
     *
     * Verified against `provider._preprocess`: poses match to 1e-6.
     */
    fun normalizeCameras(c2w: Array<DoubleArray>, sceneScale: Double = 1.0): Array<DoubleArray> {
        require(c2w.isNotEmpty()) { "need at least one camera" }
        require(c2w.all { it.size == 16 }) { "each c2w must be a flat 4x4 (16 doubles)" }

        val cv = Array(c2w.size) { i -> multiply4x4(c2w[i], GL_TO_CV) }
        val firstInv = invert4x4(cv[0])
        return Array(cv.size) { i ->
            val m = multiply4x4(firstInv, cv[i])
            // `sceneScale` is kept as an argument for callers that need the
            // scaled variation, but the default of 1.0 is what matches the
            // reference; scaling here would break the Plücker rays.
            if (sceneScale != 1.0) {
                for (r in 0 until 3) m[r * 4 + 3] *= sceneScale
            }
            m
        }
    }

    /**
     * Rescales intrinsics for a center crop followed by a resize.
     *
     * Mirrors `Provider._preprocess`: the crop shifts the principal point by
     * half the size delta (negative for a crop), then the resize scales both
     * focal lengths and the shifted principal point.
     */
    fun adjustIntrinsics(
        fx: Double, fy: Double, cx: Double, cy: Double,
        shiftX: Double, shiftY: Double,
        scaleX: Double, scaleY: Double,
    ): DoubleArray = doubleArrayOf(
        fx * scaleX,
        fy * scaleY,
        (cx + shiftX) * scaleX,
        (cy + shiftY) * scaleY,
    )

    /**
     * Center-crop geometry for one source size, matching `ImageTransform`.
     *
     * Returns (shiftX, shiftY, scaleX, scaleY, cropW, cropH). With `maxCrop`
     * the crop is the largest rectangle of the target aspect that fits inside
     * the source, which is what `crop_ratio` does; otherwise the crop is the
     * fixed target size clamped to the source.
     */
    fun cropGeometry(
        srcW: Int, srcH: Int,
        targetW: Int, targetH: Int,
        maxCrop: Boolean = true,
    ): DoubleArray {
        val cropW: Int
        val cropH: Int
        if (maxCrop) {
            val ratio = min(srcH.toDouble() / targetH, srcW.toDouble() / targetW)
            cropW = (targetW * ratio).toInt().coerceAtMost(srcW)
            cropH = (targetH * ratio).toInt().coerceAtMost(srcH)
        } else {
            cropW = targetW.coerceAtMost(srcW)
            cropH = targetH.coerceAtMost(srcH)
        }
        val shiftX = (cropW - srcW) / 2.0
        val shiftY = (cropH - srcH) / 2.0
        val scaleX = targetW.toDouble() / cropW
        val scaleY = targetH.toDouble() / cropH
        return doubleArrayOf(shiftX, shiftY, scaleX, scaleY, cropW.toDouble(), cropH.toDouble())
    }

    /**
     * Center-crops to [cropW]x[cropH] then resizes to [targetW]x[targetH],
     * normalising with ImageNet statistics. Output is planar CHW.
     *
     * The crop is applied first and the resize second, matching the order in
     * `preprocess_images`; doing it the other way changes which pixels survive
     * and would silently shift the image relative to the Plucker rays.
     */
    fun prepareImage(
        bitmap: Bitmap,
        targetW: Int,
        targetH: Int,
        maxCrop: Boolean = true,
    ): FloatArray = FloatArray(3 * targetW * targetH).also {
        prepareImageInto(bitmap, targetW, targetH, maxCrop, it, 0)
    }

    /**
     * Same as [prepareImage], writing into [out] at [offset] instead of allocating.
     *
     * The batch path used to build one array per view and `System.arraycopy` it
     * into the encoder input: an extra copy of every pixel and, for four 256×256
     * views, ~3 MB of avoidable peak. Writing straight into the destination
     * removes both.
     */
    fun prepareImageInto(
        bitmap: Bitmap,
        targetW: Int,
        targetH: Int,
        maxCrop: Boolean,
        out: FloatArray,
        offset: Int,
    ) {
        val targetPlane = targetW * targetH
        require(offset >= 0 && offset + 3 * targetPlane <= out.size) {
            "destination too small for ${targetW}x$targetH at $offset"
        }
        val srcW = bitmap.width
        val srcH = bitmap.height
        val g = cropGeometry(srcW, srcH, targetW, targetH, maxCrop)
        val cropW = g[4].toInt()
        val cropH = g[5].toInt()
        val left = (srcW - cropW) / 2
        val top = (srcH - cropH) / 2

        // Reused per call: one view's worth of ARGB, not the whole batch's.
        val px = IntArray(targetPlane)
        val cropped = Bitmap.createBitmap(bitmap, left, top, cropW, cropH)
        try {
            val resized = if (cropW == targetW && cropH == targetH) cropped
                          else Bitmap.createScaledBitmap(cropped, targetW, targetH, true)
            try {
                resized.getPixels(px, 0, targetW, 0, 0, targetW, targetH)
            } finally {
                if (resized !== bitmap && resized !== cropped) resized.recycle()
            }
        } finally {
            // createBitmap can return the source for a full-image crop.
            if (cropped !== bitmap) cropped.recycle()
        }

        for (p in 0 until targetPlane) {
            val c = px[p]
            val r = ((c shr 16) and 0xFF) / 255f
            val gg = ((c shr 8) and 0xFF) / 255f
            val b = (c and 0xFF) / 255f
            out[offset + p] = (r - MEAN[0]) / STD[0]
            out[offset + targetPlane + p] = (gg - MEAN[1]) / STD[1]
            out[offset + 2 * targetPlane + p] = (b - MEAN[2]) / STD[2]
        }
    }

    /**
     * Plucker coordinates for one camera, [6,H,W] planar.
     *
     * `ray_condition` builds `cross(o, d)` and appends `d`. The ray origin is
     * constant per view, so the cross product is computed once per pixel from
     * the constant origin -- the only per-pixel work is the direction.
     */
    fun rayCondition(
        fx: Double, fy: Double, cx: Double, cy: Double,
        c2w: DoubleArray,
        h: Int, w: Int,
    ): FloatArray = FloatArray(6 * h * w).also {
        rayConditionInto(fx, fy, cx, cy, c2w, h, w, it, 0)
    }

    /** [rayCondition] writing into [out] at [offset], for the batched path. */
    fun rayConditionInto(
        fx: Double, fy: Double, cx: Double, cy: Double,
        c2w: DoubleArray,
        h: Int,
        w: Int,
        out: FloatArray,
        offset: Int,
    ) {
        // c2w is row-major: R in [0,1,2],[4,5,6],[8,9,10], t in [3,7,11].
        val r00 = c2w[0]; val r01 = c2w[1]; val r02 = c2w[2]
        val r10 = c2w[4]; val r11 = c2w[5]; val r12 = c2w[6]
        val r20 = c2w[8]; val r21 = c2w[9]; val r22 = c2w[10]
        val ox = c2w[3]; val oy = c2w[7]; val oz = c2w[11]

        val plane = h * w
        require(offset >= 0 && offset + 6 * plane <= out.size) {
            "destination too small for ${w}x$h rays at $offset"
        }
        val eps = 1e-6
        val fxe = if (kotlin.math.abs(fx) < eps) eps else fx
        val fye = if (kotlin.math.abs(fy) < eps) eps else fy
        // The x term depends only on the column and the focal length, so it is
        // computed once for the whole view instead of h times.
        val xsByColumn = DoubleArray(w) { x -> ((x + 0.5) - cx) / fxe }

        // Axis order matches `custom_meshgrid(linspace(0,H-1,H), linspace(0,W-1,W))`
        // then `+0.5`, with a stride of 1 because nh=H and nw=W.
        var y = 0
        while (y < h) {
            val j = y + 0.5
            val ys = (j - cy) / fye
            val rowBase = offset + y * w
            var x = 0
            while (x < w) {
                val xs = xsByColumn[x]
                val n = sqrt(xs * xs + ys * ys + 1.0).coerceAtLeast(eps)
                val dxn = xs / n
                val dyn = ys / n
                val dzn = 1.0 / n

                // rays_d = directions @ R^T  ->  R * direction
                var dx = r00 * dxn + r01 * dyn + r02 * dzn
                var dy = r10 * dxn + r11 * dyn + r12 * dzn
                var dz = r20 * dxn + r21 * dyn + r22 * dzn
                if (!dx.isFinite()) dx = 0.0
                if (!dy.isFinite()) dy = 0.0
                if (!dz.isFinite()) dz = 0.0

                // cross(rays_o, rays_d)
                val px = oy * dz - oz * dy
                val py = oz * dx - ox * dz
                val pz = ox * dy - oy * dx

                val k = rowBase + x
                out[k] = sanitize(px)
                out[k + plane] = sanitize(py)
                out[k + 2 * plane] = sanitize(pz)
                out[k + 3 * plane] = dx.toFloat()
                out[k + 4 * plane] = dy.toFloat()
                out[k + 5 * plane] = dz.toFloat()
                x++
            }
            y++
        }
    }

    private fun sanitize(v: Double): Float =
        if (v.isFinite()) v.toFloat() else 0f

    // ------------------------------------------------------------ 4x4 helpers

    /** Inverts a row-major 4x4 via Gauss-Jordan; LocusGS poses are rigid so this is exact. */
    fun invert4x4(m: DoubleArray): DoubleArray {
        require(m.size == 16) { "expected 16 elements" }
        val a = DoubleArray(32)
        for (r in 0 until 4) {
            for (c in 0 until 4) a[r * 8 + c] = m[r * 4 + c]
            a[r * 8 + 4 + r] = 1.0
        }
        for (col in 0 until 4) {
            // Partial pivot: the translation column can be the largest entry,
            // so a fixed-order elimination is not safe in general.
            var pivot = col
            for (r in col + 1 until 4) {
                if (kotlin.math.abs(a[r * 8 + col]) > kotlin.math.abs(a[pivot * 8 + col])) pivot = r
            }
            if (pivot != col) {
                for (k in 0 until 8) {
                    val t = a[col * 8 + k]; a[col * 8 + k] = a[pivot * 8 + k]; a[pivot * 8 + k] = t
                }
            }
            val d = a[col * 8 + col]
            require(kotlin.math.abs(d) > 1e-12) { "matrix is singular at column $col" }
            for (k in 0 until 8) a[col * 8 + k] /= d
            for (r in 0 until 4) {
                if (r == col) continue
                val f = a[r * 8 + col]
                if (f == 0.0) continue
                for (k in 0 until 8) a[r * 8 + k] -= f * a[col * 8 + k]
            }
        }
        val out = DoubleArray(16)
        for (r in 0 until 4) for (c in 0 until 4) out[r * 4 + c] = a[r * 8 + 4 + c]
        return out
    }

    /** Row-major 4x4 multiply. */
    fun multiply4x4(x: DoubleArray, y: DoubleArray): DoubleArray {
        require(x.size == 16 && y.size == 16) { "expected 16 elements each" }
        val out = DoubleArray(16)
        for (r in 0 until 4) {
            for (c in 0 until 4) {
                var s = 0.0
                for (k in 0 until 4) s += x[r * 4 + k] * y[k * 4 + c]
                out[r * 4 + c] = s
            }
        }
        return out
    }

    /**
     * Full preprocess for a batch of views.
     *
     * Returns the encoder's two inputs, exactly the shapes the exported graphs
     * declare: images `[V,3,H,W]` and Plucker `[V,6,H,W]`, both planar.
     */
    fun prepare(
        bitmaps: List<Bitmap>,
        c2w: Array<DoubleArray>,
        fx: Double, fy: Double, cx: Double, cy: Double,
        targetW: Int, targetH: Int,
        // 1.0 is correct: the reference Plücker rays use the unscaled poses.
        sceneScale: Double = 1.0,
        maxCrop: Boolean = true,
    ): Prepared {
        checkBatch(bitmaps, c2w, targetW, targetH)

        // Intrinsics are adjusted from the first image's geometry; every frame
        // in a capture shares the sensor, which LocusGS assumes too.
        val g = cropGeometry(bitmaps[0].width, bitmaps[0].height, targetW, targetH, maxCrop)
        val k = adjustIntrinsics(fx, fy, cx, cy, g[0], g[1], g[2], g[3])
        return prepareBatch(bitmaps, c2w, List(bitmaps.size) { k }, targetW, targetH, sceneScale, maxCrop)
    }

    /**
     * Like [prepare], but each (fx, fy, cx, cy) is in its own bitmap's pixel coordinates.
     * Crop and resize the calibration independently for each view, just like its pixels.
     */
    fun prepareViews(
        bitmaps: List<Bitmap>,
        c2w: Array<DoubleArray>,
        intrinsics: List<DoubleArray>,
        targetW: Int,
        targetH: Int,
        sceneScale: Double,
    ): Prepared {
        checkBatch(bitmaps, c2w, targetW, targetH)
        require(intrinsics.size == bitmaps.size) {
            "got ${bitmaps.size} images but ${intrinsics.size} calibrations"
        }
        val adjusted = List(bitmaps.size) { i ->
            val k = intrinsics[i]
            require(k.size == 4 && k.all { it.isFinite() } && k[0] > 0 && k[1] > 0) {
                "view $i calibration must have finite (fx, fy, cx, cy) with positive focal lengths"
            }
            val g = cropGeometry(bitmaps[i].width, bitmaps[i].height, targetW, targetH)
            require(g[4] >= 1 && g[5] >= 1) { "view $i crop is empty" }
            adjustIntrinsics(k[0], k[1], k[2], k[3], g[0], g[1], g[2], g[3])
        }
        return prepareBatch(bitmaps, c2w, adjusted, targetW, targetH, sceneScale, maxCrop = true)
    }

    private fun checkBatch(bitmaps: List<Bitmap>, c2w: Array<DoubleArray>, targetW: Int, targetH: Int) {
        require(bitmaps.isNotEmpty()) { "need at least one view" }
        require(bitmaps.size == c2w.size) { "got ${bitmaps.size} images but ${c2w.size} poses" }
        require(targetW > 0 && targetH > 0) { "target dimensions must be positive" }
    }

    /**
     * Decodes each view, converts it into the shared encoder buffer, and recycles
     * the bitmap immediately.
     *
     * The caller previously decoded every view first and then preprocessed them,
     * so four 1280×720 ARGB bitmaps (~15 MB) were resident at once on top of the
     * float buffers. One at a time halves that peak; the crop still comes from
     * the decoded image's real size, so the geometry is unchanged.
     */
    fun prepareDecoding(
        files: List<File>,
        c2w: Array<DoubleArray>,
        intrinsics: List<DoubleArray>,
        targetW: Int,
        targetH: Int,
        sceneScale: Double,
    ): Prepared {
        require(files.size == c2w.size) { "got ${files.size} images but ${c2w.size} poses" }
        require(intrinsics.size == files.size) { "got ${files.size} images but ${intrinsics.size} calibrations" }
        val v = files.size
        require(v > 0) { "need at least one view" }
        require(targetW > 0 && targetH > 0) { "target dimensions must be positive" }

        val adjusted = List(v) { i ->
            val k = intrinsics[i]
            require(k.size == 4 && k.all { it.isFinite() } && k[0] > 0 && k[1] > 0) {
                "view $i calibration must have finite (fx, fy, cx, cy) with positive focal lengths"
            }
            val bounds = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
            android.graphics.BitmapFactory.decodeFile(files[i].path, bounds)
            require(bounds.outWidth > 0 && bounds.outHeight > 0) { "cannot decode ${files[i].path}" }
            val g = cropGeometry(bounds.outWidth, bounds.outHeight, targetW, targetH)
            require(g[4] >= 1 && g[5] >= 1) { "view $i crop is empty" }
            adjustIntrinsics(k[0], k[1], k[2], k[3], g[0], g[1], g[2], g[3])
        }

        val images = FloatArray(v * 3 * targetH * targetW)
        val perView = 3 * targetH * targetW
        for (i in 0 until v) {
            val bmp = requireNotNull(android.graphics.BitmapFactory.decodeFile(files[i].path)) {
                "cannot decode ${files[i].path}"
            }
            try {
                prepareImageInto(bmp, targetW, targetH, maxCrop = true, images, i * perView)
            } finally {
                bmp.recycle()
            }
        }

        val poses = normalizeCameras(c2w, sceneScale)
        val plucker = FloatArray(v * 6 * targetH * targetW)
        val plane = targetH * targetW
        for (i in 0 until v) {
            val k = adjusted[i]
            rayConditionInto(k[0], k[1], k[2], k[3], poses[i], targetH, targetW, plucker, i * 6 * plane)
        }

        return Prepared(images, plucker, poses, adjusted.first(), v, targetH, targetW, adjusted)
    }

    private fun prepareBatch(
        bitmaps: List<Bitmap>,
        c2w: Array<DoubleArray>,
        adjusted: List<DoubleArray>,
        targetW: Int,
        targetH: Int,
        sceneScale: Double,
        maxCrop: Boolean,
    ): Prepared {
        val v = bitmaps.size

        val images = FloatArray(v * 3 * targetH * targetW)
        val perView = 3 * targetH * targetW
        for (i in 0 until v) {
            prepareImageInto(bitmaps[i], targetW, targetH, maxCrop, images, i * perView)
        }

        val poses = normalizeCameras(c2w, sceneScale)
        val plucker = FloatArray(v * 6 * targetH * targetW)
        val plane = targetH * targetW
        for (i in 0 until v) {
            val k = adjusted[i]
            rayConditionInto(k[0], k[1], k[2], k[3], poses[i], targetH, targetW, plucker, i * 6 * plane)
        }

        return Prepared(images, plucker, poses, adjusted.first(), v, targetH, targetW, adjusted)
    }

    data class Prepared(
        val images: FloatArray,
        val plucker: FloatArray,
        val c2wNormalised: Array<DoubleArray>,
        /** First view for compatibility with the shared-K debug API. */
        val intrinsics: DoubleArray,
        val views: Int,
        val height: Int,
        val width: Int,
        val intrinsicsPerView: List<DoubleArray> = List(views) { intrinsics },
    )
}
