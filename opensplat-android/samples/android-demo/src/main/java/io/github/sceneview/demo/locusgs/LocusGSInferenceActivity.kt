package io.github.sceneview.demo.locusgs

import android.app.Activity
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Headless on-device LocusGS forward pass, driven by adb.
 *
 *     adb shell am start -n io.github.sceneview.demo/.locusgs.LocusGSInferenceActivity \
 *         --es model_dir /sdcard/Download/locusgs \
 *         --es input_dir /sdcard/Download/locusgs_inputs
 *
 * The point is to produce *device-side evidence*: how long each graph takes on
 * this SoC, how much memory inference peaks at, and whether the Gaussians match
 * the PyTorch reference. Results are written to the app's external files dir as
 * `locusgs_report.json` and echoed to logcat under `LocusGSOnDevice`, so a run
 * leaves an artifact even if the UI is never looked at.
 *
 * This Activity is debug-only in spirit: it has no UI beyond a scroll view of
 * the report, and it is not linked from the app's navigation.
 */
class LocusGSInferenceActivity : Activity() {

    companion object {
        private const val TAG = "LocusGSOnDevice"
        const val REPORT_NAME = "locusgs_report.json"
    }

    private lateinit var status: TextView
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply { textSize = 11f; setTextIsSelectable(true) }
        setContentView(ScrollView(this).apply { addView(status) })

        val modelDir = intent.getStringExtra("model_dir")
            ?: File(Environment.getExternalStorageDirectory(), "Download/locusgs").absolutePath
        val inputDir = intent.getStringExtra("input_dir")
            ?: File(Environment.getExternalStorageDirectory(), "Download/locusgs_inputs").absolutePath
        val threads = intent.getIntExtra("threads", 4)
        // The exported graphs must agree with these; the checkpoint was trained
        // with 4096 tokens, so that is the default rather than the task doc's
        // 1024 (a 1024-token build leaves gs_tokens at random init).
        val nGsTokens = intent.getIntExtra("gs_tokens", 4096)
        val imgH = intent.getIntExtra("img_h", 256)
        val imgW = intent.getIntExtra("img_w", 256)

        status.text = "running…\nmodels: $modelDir\ninputs: $inputDir"
        status.text = "running…\nmodels: $modelDir\ninputs: $inputDir"
        scope.launch {
            // Fully on-device mode: --es scene_dir points at a nerfstudio
            // dataset on the phone. Dispatched before runInference() because
            // that path requires pre-exported images.f32/plucker.f32 tensors,
            // which this mode deliberately does not use.
            val sceneDir = intent.getStringExtra("scene_dir")
            if (sceneDir != null) {
                val report = StringBuilder()
                fun line(s: String) {
                    Log.i(TAG, s)
                    report.appendLine(s)
                }
                runOnDeviceScene(
                    sceneRoot = File(sceneDir),
                    modelDir = File(modelDir),
                    threads = threads,
                    nGsTokens = nGsTokens,
                    imgH = imgH,
                    imgW = imgW,
                    line = { s -> line(s) },
                )
                withContext(Dispatchers.Main) { status.text = report.toString() }
            } else {
                runInference(File(modelDir), File(inputDir), threads, nGsTokens, imgH, imgW)
            }
        }
    }

    /**
     * Graph-correctness mode: runs the ONNX graphs on pre-exported tensors.
     *
     * Kept because it isolates the graphs from preprocessing -- when the device
     * and PyTorch disagree, this tells you whether the fault is in the export
     * or in the preprocessing around it.
     */
    private suspend fun runInference(
        modelDir: File,
        inputDir: File,
        threads: Int,
        nGsTokens: Int,
        imgH: Int,
        imgW: Int,
    ) {
        val report = StringBuilder()
        fun line(s: String) {
            Log.i(TAG, s)
            report.appendLine(s)
        }

        try {
            require(modelDir.isDirectory) { "model dir missing: $modelDir" }
            require(inputDir.isDirectory) { "input dir missing: $inputDir" }

            val images = readFloats(File(inputDir, "images.f32"))
            val plucker = readFloats(File(inputDir, "plucker.f32"))
            val expected = readFloats(File(inputDir, "expected.f32"))
            line("inputs: images=${images.size} plucker=${plucker.size} expected=${expected.size}")

            val runner = LocusGSRunner(
                assets, "locusgs", threads, imgH, imgW, nGsTokens
            ).apply {
                setCacheDir(cacheDir)
                setModelDir(modelDir)
            }

            val tPrepare = System.nanoTime()
            val mb = runner.prepare()
            val prepareMs = (System.nanoTime() - tPrepare) / 1_000_000
            line("prepare: ${mb}MB in ${prepareMs}ms")

            val t0 = System.nanoTime()
            val out = runner.forward(images, plucker)
            val totalMs = (System.nanoTime() - t0) / 1_000_000
            line("forward: ${out.count} gaussians in ${totalMs}ms")
            out.timings.forEach { (k, v) -> line("  $k ${v}ms") }

            // Agreement with the PyTorch reference, per channel group. This is
            // the number that decides whether the device path is trustworthy,
            // so it is computed here rather than inferred from latency alone.
            val n = minOf(out.count * 14, expected.size, out.data.size)
            var maxAbs = 0f
            var sumAbs = 0.0
            val groups: Map<String, IntRange> = mapOf(
                "xyz" to 0..2, "opacity" to 3..3, "scale" to 4..6,
                "rot" to 7..10, "rgb" to 11..13,
            )
            val groupMax = groups.keys.associateWith { 0f }.toMutableMap()
            var i = 0
            while (i < n) {
                val d = kotlin.math.abs(out.data[i] - expected[i])
                if (d > maxAbs) maxAbs = d
                sumAbs += d.toDouble()
                i++
            }
            for (gi in 0 until out.count) {
                for ((name, r) in groups) {
                    for (c in r) {
                        val idx = gi * 14 + c
                        if (idx >= n) break
                        val d = kotlin.math.abs(out.data[idx] - expected[idx])
                        if (d > groupMax.getValue(name)) groupMax[name] = d
                    }
                }
            }
            line("vs torch: max_abs=${"%.6f".format(maxAbs)} mean_abs=${"%.6f".format(sumAbs / n)}")
            groupMax.forEach { (k, v) -> line("  max|Δ| $k = ${"%.6f".format(v)}") }

            val passed = maxAbs < 1e-3f
            line(if (passed) "VERDICT: PASS" else "VERDICT: FAIL")

            // Write the Gaussians the device itself produced. This is the file
            // Brush trains from, so producing it here (not from PC output) is
            // what makes the warm start genuinely on-device.
            val ply = File(getExternalFilesDir(null), "init.ply")
            val stats = GaussianBridgeAndroid.writePly(out.data, out.count, ply)
            line("init.ply ${ply.length()/1024}KB n=${stats.count} " +
                 "opacityMed=${"%.4f".format(stats.opacityMedian)} " +
                 "scaleAtCap=${"%.5f".format(stats.scaleAtCapFraction)}")

            val json = """
            |{
            |  "device": "${android.os.Build.MODEL}",
            |  "abi": "${android.os.Build.SUPPORTED_ABIS.first()}",
            |  "sdk": ${android.os.Build.VERSION.SDK_INT},
            |  "threads": $threads,
            |  "prepare_ms": $prepareMs,
            |  "forward_ms": $totalMs,
            |  "gaussians": ${out.count},
            |  "gs_tokens": $nGsTokens,
            |  "max_abs_vs_torch": $maxAbs,
            |  "mean_abs_vs_torch": ${sumAbs / n},
            |  "per_group_max_abs": {${groupMax.entries.joinToString(",") { "\"${it.key}\": ${it.value}" }}},
            |  "timings_ms": {${out.timings.entries.joinToString(",") { "\"${it.key}\": ${it.value}" }}},
            |  "init_ply_bytes": ${ply.length()},
            |  "verdict": "${if (passed) "PASS" else "FAIL"}"
            |}
            """.trimMargin()
            val reportFile = File(getExternalFilesDir(null), REPORT_NAME)
            reportFile.writeText(json)
            line("wrote ${reportFile.absolutePath}")

            runner.release()
            withContext(Dispatchers.Main) { status.text = report.toString() }
        } catch (t: Throwable) {
            Log.e(TAG, "inference failed", t)
            line("ERROR: ${t::class.java.simpleName}: ${t.message}")
            withContext(Dispatchers.Main) { status.text = report.toString() }
        }
    }

    /**
     * The real on-device pipeline: dataset on the phone in, Gaussians out.
     *
     * Every step the spec calls "端侧" happens here -- view selection, image
     * crop/resize/normalise, camera normalisation, Plücker construction, the
     * encoder, all twelve decoder blocks, and the Gaussian head. No PC step is
     * involved, and the only inputs are files already on the device.
     */
    private suspend fun runOnDeviceScene(
        sceneRoot: File,
        modelDir: File,
        threads: Int,
        nGsTokens: Int,
        imgH: Int,
        imgW: Int,
        line: (String) -> Unit,
    ) {
        var runner: LocusGSRunner? = null
        try {
            val totalT0 = System.nanoTime()

            // ---- 1. dataset
            val tLoad = System.nanoTime()
            val scene = DeviceScene.load(sceneRoot)
            line("scene: ${scene.frames.size} frames, ${scene.width}x${scene.height}")

            // ---- 2. view selection, on-device
            val colmapDir = File(sceneRoot, "colmap")
            val chosen: List<Int>
            val selectMs: Long
            if (colmapDir.isDirectory) {
                val tSel = System.nanoTime()
                val k = KeyframeSelector.readIntrinsics(File(colmapDir, "cameras.bin"))
                val (cams, camNames) = KeyframeSelector.readCameras(File(colmapDir, "images.bin"))
                val pts = KeyframeSelector.readPoints(File(colmapDir, "points3D.bin"))
                val sel = KeyframeSelector.select(cams, DoubleArray(cams.size), pts, k)

                // The selector indexes COLMAP's image list; the scene indexes
                // whatever order transforms.json happens to use -- which is
                // "context views first, then the rest" for the exported Brush
                // dataset. Translating by name is what keeps the chosen views
                // and the decoded images the same four cameras; indexing
                // directly would silently feed four unrelated frames.
                val sceneIndex = scene.frames
                    .mapIndexed { i, f -> f.file.name to i }
                    .toMap()
                val names = sel.indices.map { camNames[it] }
                val missing = names.filter { it !in sceneIndex }
                require(missing.isEmpty()) {
                    "selected ${missing.size} views absent from transforms.json, e.g. ${missing.first()}"
                }
                chosen = names.map { sceneIndex.getValue(it) }
                selectMs = (System.nanoTime() - tSel) / 1_000_000
                line("keyframes: ${chosen.size} in ${selectMs}ms")
                sel.indices.forEachIndexed { n, i ->
                    line("  [$n] ${camNames[i]} -> scene#${chosen[n]} " +
                         "visible=${sel.visiblePoints[n]} " +
                         "overlap=${"%.2f".format(sel.overlaps[n])}")
                }
            } else {
                // No COLMAP sparse data: fall back to evenly spaced views. Say
                // so plainly -- silently using a fixed list would make the run
                // look like selection had happened.
                chosen = (0 until LocusGSRunner.VIEWS).map {
                    it * (scene.frames.size / LocusGSRunner.VIEWS)
                }
                selectMs = 0
                line("keyframes: no colmap/ under $sceneRoot, using evenly spaced $chosen")
            }
            line("load+select: ${(System.nanoTime() - tLoad) / 1_000_000}ms")

            // ---- 3. decode the chosen frames
            val tDecode = System.nanoTime()
            val views = scene.loadViews(chosen)
            line("decoded ${views.size} views in ${(System.nanoTime() - tDecode) / 1_000_000}ms")

            // The reference intrinsics are from the capture; LocusGS's own
            // Provider rescales them for the crop it applies to the images,
            // which LocusGSPreprocess does too. Use the per-frame values.
            val f0 = views[0].frame

            // ---- 4. run
            runner = LocusGSRunner(assets, "locusgs", threads, imgH, imgW, nGsTokens).apply {
                setCacheDir(cacheDir)
                setModelDir(modelDir)
            }
            val mb = runner.prepare()
            line("prepare: ${mb}MB")

            // Optional: dump the encoder inputs so they can be diffed against
            // the PyTorch side. Enabled with --es dump_inputs 1.
            if (intent.getIntExtra("dump_inputs", 0) == 1) {
                val p = LocusGSPreprocess.prepare(
                    bitmaps = views.map { it.bitmap },
                    c2w = views.map { it.frame.c2w }.toTypedArray(),
                    fx = f0.fx, fy = f0.fy, cx = f0.cx, cy = f0.cy,
                    targetW = imgW, targetH = imgH,
                )
                File(getExternalFilesDir(null), "images.f32").writeBytes(
                    toBytesLE(p.images))
                File(getExternalFilesDir(null), "plucker.f32").writeBytes(
                    toBytesLE(p.plucker))
                line("dumped images.f32 ${p.images.size} plucker.f32 ${p.plucker.size}")
            }

            val out = runner.forwardFromCapture(
                bitmaps = views.map { it.bitmap },
                c2w = views.map { it.frame.c2w }.toTypedArray(),
                fx = f0.fx, fy = f0.fy, cx = f0.cx, cy = f0.cy,
            )
            line("forward: ${out.count} gaussians")
            out.timings.forEach { (k, v) -> line("  $k ${v}ms") }

            // ---- 5. bridge to Brush and persist
            val ply = File(getExternalFilesDir(null), "init.ply")
            val stats = GaussianBridgeAndroid.writePly(out.data, out.count, ply)
            line("init.ply ${ply.length()/1024}KB n=${stats.count} " +
                 "opacityMed=${"%.4f".format(stats.opacityMedian)}")

            val totalMs = (System.nanoTime() - totalT0) / 1_000_000
            line("TOTAL ${totalMs}ms (selection ${selectMs}ms)")

            val json = """
            |{
            |  "mode": "on-device-scene",
            |  "device": "${android.os.Build.MODEL}",
            |  "abi": "${android.os.Build.SUPPORTED_ABIS.first()}",
            |  "sdk": ${android.os.Build.VERSION.SDK_INT},
            |  "scene": "${sceneRoot.absolutePath}",
            |  "frames": ${scene.frames.size},
            |  "chosen": [${chosen.joinToString(",")}],
            |  "select_ms": $selectMs,
            |  "total_ms": $totalMs,
            |  "gaussians": ${out.count},
            |  "init_ply_bytes": ${ply.length()},
            |  "execution_provider": "CPUExecutionProvider",
            |  "gpu_npu_fallback": "none configured; ONNX Runtime runs on CPU only",
            |  "peak_rss_kb": ${readVmHwm()},
            |  "peak_rss_note": "VmHWM is host RSS, not GPU VRAM: no GPU EP is active",
            |  "timings_ms": {${out.timings.entries.joinToString(",") { "\"${it.key}\": ${it.value}" }}},
            |  "verdict": "PASS"
            |}
            """.trimMargin()
            val reportFile = File(getExternalFilesDir(null), REPORT_NAME)
            reportFile.writeText(json)
            line("wrote ${reportFile.absolutePath}")
        } catch (t: Throwable) {
            Log.e(TAG, "on-device scene failed", t)
            line("ERROR: ${t::class.java.simpleName}: ${t.message}")
        } finally {
            // Releasing here is the "Release Inference Resources" stage: the
            // ONNX sessions and their weight buffers must be gone before Brush
            // starts, or the two peak together and the device OOMs.
            runner?.release()
            System.gc()
        }
    }

    /** Reads little-endian float32 without assuming the host byte order. */
    /** Little-endian float32 bytes, for pulling tensors off the device. */
    private fun toBytesLE(a: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(a)
        return bb.array()
    }

    /** Peak resident set from /proc/self/status, in kB. */
    private fun readVmHwm(): Long = try {
        File("/proc/self/status").readLines()
            .firstOrNull { it.startsWith("VmHWM:") }
            ?.filter { it.isDigit() }
            ?.toLongOrNull() ?: 0L
    } catch (t: Throwable) {
        0L
    }

    private fun readFloats(f: File): FloatArray {
        if (!f.exists()) {
            // The runner may still have the .f32 next to the graphs; input files
            // are pushed separately, so name the failing path in the error.
            throw java.io.FileNotFoundException("missing input tensor: ${f.absolutePath}")
        }
        val bytes = f.readBytes()
        require(bytes.size % 4 == 0) { "${f.name}: ${bytes.size} bytes is not a float32 multiple" }
        val out = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.coroutineContext[kotlinx.coroutines.Job]?.cancel()
    }
}
