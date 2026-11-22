package io.github.sceneview.demo.locusgs

import android.content.Context
import android.util.Log
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.ln

/**
 * Orchestrates the fully on-device pipeline:
 *
 *   images+poses -> LocusGS forward -> bridge -> Brush warm-start -> export
 *
 * Each stage is timed and the inference sessions are explicitly released before
 * Brush allocates its own GPU buffers, which is what keeps the combined peak
 * inside a phone's memory budget rather than summing both peaks.
 */
class LocusGSReconstructionService(
    private val context: Context,
    private val modelDir: String = "locusgs",
) {
    companion object {
        private const val TAG = "LocusGS"
        /**
         * Must match the installed export: 4 views at 448x256 with 4096 Gaussian
         * tokens. LocusGSRunner carries the same values as its constructor defaults,
         * so these stay correct as long as the service uses that runner unchanged.
         */
        const val IMG_H = LocusGSRunner.DEFAULT_IMG_H
        const val IMG_W = LocusGSRunner.DEFAULT_IMG_W
        /** ImageNet normalisation, matching LocusGS's Provider. */
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val cancelled = AtomicBoolean(false)

    data class StageResult(
        val stage: String,
        val millis: Long,
        val detail: String = "",
    )

    data class Result(
        val ok: Boolean,
        val gaussianCount: Int,
        val initPly: File?,
        val stages: List<StageResult>,
        val bridgeStats: GaussianBridgeAndroid.Stats?,
        val error: String? = null,
    )

    fun cancel() = cancelled.set(true)

    /**
     * @param images four RGB images, each [H*W*3] float in [0,1], H=256 W=448
     * @param intrinsics four fx,fy,cx,cy
     * @param c2w four row-major 4x4 camera-to-world matrices (OpenCV convention,
     *            i.e. what LocusGS's Provider produces after normalisation)
     * @param workDir directory for the produced init.ply
     */
    fun run(
        images: List<FloatArray>,
        intrinsics: List<FloatArray>,
        c2w: List<FloatArray>,
        workDir: File,
    ): Result {
        val stages = ArrayList<StageResult>()
        require(images.size == 4) { "LocusGS needs exactly 4 context views" }

        var runner: LocusGSRunner? = null
        try {
            val prepWin = System.nanoTime()
            runner = LocusGSRunner(context.assets, modelDir).apply { setCacheDir(context.cacheDir) }
            val modelMb = runner.prepare()
            stages += StageResult("prepare_models", ms(prepWin), "${modelMb}MB extracted")

            // Inputs in the exact layout the exporter traced.
            val fwdWin = System.nanoTime()
            val normalized = FloatArray(4 * 3 * IMG_H * IMG_W)
            var o = 0
            for (v in 0 until 4) {
                for (c in 0 until 3) {
                    val channel = images[v]
                    for (i in 0 until IMG_H * IMG_W) {
                        normalized[o++] = (channel[i * 3 + c] - MEAN[c]) / STD[c]
                    }
                }
            }
            val plucker = buildPlucker(intrinsics, c2w)
            val gauss = runner.forward(normalized, plucker)
            stages += StageResult("locusgs_forward", ms(fwdWin),
                gauss.timings.entries.joinToString(", ") { "${it.key}=${it.value}ms" })

            if (cancelled.get()) return Result(false, 0, null, stages, null, "cancelled")

            // Release inference before Brush allocates, so peaks do not stack.
            val relWin = System.nanoTime()
            runner.release()
            runner = null
            stages += StageResult("release_inference", ms(relWin))

            val bridgeWin = System.nanoTime()
            val ply = File(workDir, "init.ply")
            val stats = GaussianBridgeAndroid.writePly(gauss.data, gauss.count, ply)
            stages += StageResult("bridge", ms(bridgeWin),
                "n=${stats.count} opacityMed=${stats.opacityMedian} scaleAtCap=${stats.scaleAtCapFraction}")
            Log.i(TAG, "wrote ${ply.absolutePath} (${ply.length() / 1024}KB)")
            return Result(true, gauss.count, ply, stages, stats)
        } catch (t: Throwable) {
            Log.e(TAG, "reconstruction failed", t)
            return Result(false, 0, null, stages, null, "$t")
        } finally {
            runner?.release()
        }
    }

    /**
     * Plücker coordinates (m = o x d, d) at every pixel, the encoding LocusGS
     * was trained with (`locusgs/utils/data.py::ray_condition`).
     */
    private fun buildPlucker(intrinsics: List<FloatArray>, c2w: List<FloatArray>): FloatArray {
        val out = FloatArray(4 * 6 * IMG_H * IMG_W)
        for (v in 0 until 4) {
            val (fx, fy, cx, cy) = intrinsics[v]
            val m = c2w[v]
            for (y in 0 until IMG_H) {
                for (x in 0 until IMG_W) {
                    val dx = (x + 0.5f - cx) / fx
                    val dy = (y + 0.5f - cy) / fy
                    val dz = 1f
                    // direction = R * (dx,dy,dz)
                    val wx = m[0] * dx + m[1] * dy + m[2] * dz
                    val wy = m[4] * dx + m[5] * dy + m[6] * dz
                    val wz = m[8] * dx + m[9] * dy + m[10] * dz
                    val len = kotlin.math.sqrt(wx * wx + wy * wy + wz * wz)
                    val ndx = wx / len; val ndy = wy / len; val ndz = wz / len
                    val ox = m[3]; val oy = m[7]; val oz = m[11]
                    // m = o x d
                    val mx = oy * ndz - oz * ndy
                    val my = oz * ndx - ox * ndz
                    val mz = ox * ndy - oy * ndx
                    val idx = ((v * 6) * IMG_H * IMG_W) + y * IMG_W + x
                    out[idx] = mx
                    out[idx + IMG_H * IMG_W] = my
                    out[idx + 2 * IMG_H * IMG_W] = mz
                    out[idx + 3 * IMG_H * IMG_W] = ndx
                    out[idx + 4 * IMG_H * IMG_W] = ndy
                    out[idx + 5 * IMG_H * IMG_W] = ndz
                }
            }
        }
        return out
    }

    private fun ms(t0: Long) = (System.nanoTime() - t0) / 1_000_000
}
