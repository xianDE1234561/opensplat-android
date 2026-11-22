package io.github.sceneview.demo.locusgs

import android.content.res.AssetManager
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Runs LocusGS inference entirely on-device, block by block.
 *
 * The model is exported as 14 separate graphs (encoder, 12 decoder blocks, head)
 * rather than one 1 GB graph because:
 *  - the monolithic decoder exported incorrectly (max abs error 4.8 vs PyTorch);
 *    the block-chained path measures 3.8e-4,
 *  - one block is ~66 MB resident instead of ~788 MB,
 *  - buffers are released between blocks, which is what keeps peak memory
 *    inside an 8 GB phone's budget.
 *
 * This class is not thread-safe and is meant to be driven from one background thread.
 */
class LocusGSRunner(
    private val assets: AssetManager,
    private val modelDir: String = "locusgs",
    private val threads: Int = 4,
    imgH_: Int = DEFAULT_IMG_H,
    imgW_: Int = DEFAULT_IMG_W,
    nGsTokens_: Int = DEFAULT_N_GS_TOKENS,
) {
    companion object {
        const val VIEWS = 4
        const val PATCH = 8

        /**
         * 4 views at 448×256, with 4096 Gaussian tokens; see [DEFAULT_IMG_W] for why
         * that shape and why it is the only one shipped.
         *
         * 4096 is not arbitrary: the shipped checkpoint stores `gs_tokens` as
         * [4096,1024], `gs_anchor_xyz_raw` as [4096,3] and
         * `gs_anchor_radius_raw` as [4096,1]. Building the model with any other
         * token count leaves those three tensors at their random initial values,
         * so "pretrained" becomes partly random and results stop being
         * reproducible. `load_weights` now reports that mismatch instead of
         * hiding it.
         */
        /**
         * 448×256 is the one shipped shape.
         *
         * A phone capture is 16:9 (or 4:3 at 480p), and [LocusGSPreprocess] centre-crops
         * to the model's aspect before resizing. Measured on a 1280×720 frame, this
         * shape keeps 98.4 % of the width; the old 256×256 export kept 56.2 %, which
         * cost 12 dB of initial PSNR on the fixed-scene comparison. Other aspects are
         * served by cropping here, not by shipping more ~1 GB exports.
         */
        const val DEFAULT_IMG_H = 256
        const val DEFAULT_IMG_W = 448
        const val DEFAULT_N_GS_TOKENS = 4096

        const val GAUSSIANS_PER_TOKEN = 64
        const val GAUSSIAN_CHANNELS = 14
        const val TOKEN_DIM = 1024
        const val HEADS = 16
        const val HEAD_DIM = 64
    }

    private val imgH = imgH_
    private val imgW = imgW_
    private val nGsTokens = nGsTokens_

    val gaussianCount = nGsTokens * GAUSSIANS_PER_TOKEN

    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    var backendName: String = "unloaded"
        private set

    /**
     * Wall time spent creating ORT sessions in the last [forward].
     *
     * The twelve decoder blocks are separate graphs whose weights differ, so they
     * cannot share a session, and holding all of them at once would be ~900 MB of
     * fp32 weights. The device measurement on the 256×256 model is load ~2-3 s of
     * a ~23 s forward (about 10 %), so this is reported for transparency — the
     * remaining ~90 % is arithmetic in the blocks, not graph loading.
     */
    var sessionSetupMs: Long = 0
        private set

    /**
     * Verifies the model files are present and copies them out of the APK.
     *
     * Sessions are *not* created here: each fp32 block is ~73 MB and creating
     * all twelve at once would hold ~900 MB of weights resident. See [forward],
     * which opens one block, runs it, and closes it before the next.
     */
    fun prepare(): Int {
        var bytes = 0L
        for (name in LocusGSExport.graphFiles) {
            bytes += ensureLocal(name).length()
        }
        backendName = "cpu"      // replaced by the measured provider after first run
        return (bytes / (1024 * 1024)).toInt()
    }

    /** Resolves a model file: external directory if set, else copied from APK. */
    private fun ensureLocal(name: String): File {
        externalDir?.let { ext ->
            val f = File(ext, name)
            require(f.isFile && f.length() > 0L) { "model file missing or empty: ${f.absolutePath}" }
            return f
        }
        val dir = File(cacheRoot(), modelDir).apply { mkdirs() }
        val f = File(dir, name)
        // A .part file guards against a half-copied asset after a crash: a
        // truncated ONNX would fail session creation with a confusing error.
        if (!f.exists() || f.length() == 0L) {
            val tmp = File(dir, "$name.part")
            assets.open("$modelDir/$name").use { input ->
                tmp.outputStream().use { input.copyTo(it) }
            }
            tmp.renameTo(f)
        }
        return f
    }

    private inline fun <T> withSession(name: String, run: (OrtSession) -> T): T =
        withLocusGSSessionOptions(threads) { options ->
            val opened = System.nanoTime()
            val session = env.createSession(ensureLocal(name).absolutePath, options)
            // Only the load is attributed to setup; the graph itself runs in `run`.
            sessionSetupMs += (System.nanoTime() - opened) / 1_000_000
            session.use(run)
        }

    private lateinit var cacheDir: File
    fun setCacheDir(dir: File) {
        cacheDir = dir
    }

    /**
     * Load the graphs from a real directory instead of APK assets.
     *
     * The fp32 export is ~1 GB, which cannot ship inside an APK (Play's limit is
     * 100 MB and a debug APK that size is impractical to install). For device
     * testing the files are pushed with ``adb push`` and read from here; the
     * asset path remains for a future quantised build.
     */
    private var externalDir: File? = null
    fun setModelDir(dir: File) {
        externalDir = dir
    }

    private fun cacheRoot(): File =
        if (::cacheDir.isInitialized) cacheDir else File(System.getProperty("java.io.tmpdir")!!)

    data class GaussianOutput(
        val data: FloatArray,
        val count: Int,
        /** milliseconds per stage, for the performance report */
        val timings: Map<String, Long>,
    )

    /**
     * End-to-end entry: takes raw capture data and does the preprocessing here.
     *
     * The other [forward] overload starts after Plücker and normalisation,
     * which meant the PC had to run `ray_condition` first -- fine for
     * verifying the graphs, but it is not an on-device pipeline. This one
     * accepts what a phone actually has: decoded bitmaps plus camera poses.
     */
    fun forwardFromCapture(
        bitmaps: List<android.graphics.Bitmap>,
        c2w: Array<DoubleArray>,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        sceneScale: Double = 1.0,
        onStage: (String) -> Unit = {},
    ): GaussianOutput = forwardFromPrepared(onStage) {
        LocusGSPreprocess.prepare(
            bitmaps = bitmaps,
            c2w = c2w,
            fx = fx, fy = fy, cx = cx, cy = cy,
            targetW = imgW, targetH = imgH,
            sceneScale = sceneScale,
        )
    }

    /**
     * Same as [forwardFromCapture] but decodes one view at a time from disk.
     *
     * Decoding all four frames before preprocessing keeps four full-resolution
     * bitmaps alive at once; this path holds one. Use it whenever the frames are
     * files, which is every real pipeline run.
     */
    fun forwardFromFiles(
        files: List<File>,
        c2w: Array<DoubleArray>,
        intrinsics: List<DoubleArray>,
        sceneScale: Double = 1.0,
        onStage: (String) -> Unit = {},
    ): GaussianOutput = forwardFromPrepared(onStage) {
        LocusGSPreprocess.prepareDecoding(
            files = files,
            c2w = c2w,
            intrinsics = intrinsics,
            targetW = imgW, targetH = imgH,
            sceneScale = sceneScale,
        )
    }

    /**
     * Times [prepare] separately from the graphs.
     *
     * [prepare] is passed as a lambda rather than a finished result so the timer
     * covers the real work: decode, crop, resize, normalise, Plücker. Timing a
     * function that only receives the already-built buffers measured nothing.
     */
    private fun forwardFromPrepared(
        onStage: (String) -> Unit,
        prepare: () -> LocusGSPreprocess.Prepared,
    ): GaussianOutput {
        val t0 = System.nanoTime()
        val pre = prepare()
        require(pre.views == VIEWS) {
            "LocusGS expects $VIEWS views, got ${pre.views}"
        }
        val prepMs = (System.nanoTime() - t0) / 1_000_000
        val out = forwardFromBuffers(pre.images, pre.plucker, onStage)
        // Report preprocessing as its own stage rather than hiding it in the
        // total: it is the part that used to run on the PC.
        val merged = LinkedHashMap<String, Long>()
        merged["preprocess"] = prepMs
        merged.putAll(out.timings)
        return out.copy(timings = merged)
    }

    /**
     * @param images [4,3,H,W], in planar order, already ImageNet-normalized
     * @param plucker [4,6,H,W], in planar order
     * @param onStage called once just before each graph, synchronously on the caller's thread.
     * Stages are `encoder`, `block_0` through `block_11`, then `head`. A callback may
     * throw to cancel between graphs; it cannot interrupt an in-flight ORT operation.
     */
    @JvmOverloads
    fun forward(images: FloatArray, plucker: FloatArray, onStage: (String) -> Unit = {}): GaussianOutput =
        forwardFromBuffers(images, plucker, onStage)

    private fun forwardFromBuffers(
        images: FloatArray,
        plucker: FloatArray,
        onStage: (String) -> Unit,
    ): GaussianOutput {
        require(images.size == VIEWS * 3 * imgH * imgW) {
            "images must be [$VIEWS,3,$imgH,$imgW], got ${images.size} floats"
        }
        require(plucker.size == VIEWS * 6 * imgH * imgW) {
            "plucker must be [$VIEWS,6,$imgH,$imgW], got ${plucker.size} floats"
        }
        val timings = LinkedHashMap<String, Long>()
        sessionSetupMs = 0
        val t0 = System.nanoTime()

        // CPU arena allocation is disabled: ORT output tensors can outlive the
        // graph sessions. The owning Results keep them alive without copying.
        var encoded: OrtSession.Result? = null
        var previous: OrtSession.Result? = null
        try {
            onStage("encoder")
            withSession(LocusGSExport.ENCODER) { session ->
                logRss("after encoder session create")
                floatTensor(images, imageShape(3)).use { imageTensor ->
                    floatTensor(plucker, imageShape(6)).use { pluckerTensor ->
                        encoded = session.run(mapOf("images" to imageTensor, "plucker" to pluckerTensor))
                    }
                }
                logRss("after encoder run")
            }
            timings["encoder"] = (System.nanoTime() - t0) / 1_000_000

            val context = encoded!!
            for (i in 0 until LocusGSExport.BLOCK_COUNT) {
                val stage = "block_$i"
                onStage(stage)
                val tb = System.nanoTime()
                withSession(LocusGSExport.blockFile(i)) { session ->
                    val prior = previous
                    if (prior == null) {
                        withSeeds { tokens, anchor, radius ->
                            previous = runBlock(session, context, tokens, anchor, radius)
                        }
                    } else {
                        previous = runBlock(session, context, prior.tensor(0), prior.tensor(1), prior.tensor(2))
                        // Only after this block has consumed the old state.
                        prior.close()
                    }
                }
                timings[stage] = (System.nanoTime() - tb) / 1_000_000
            }
            encoded = null
            context.close()

            onStage("head")
            val th = System.nanoTime()
            val last = previous!!
            val gauss = withSession(LocusGSExport.HEAD) { session ->
                session.run(mapOf(
                    "tokens" to last.tensor(0),
                    "anchor_raw" to last.tensor(1),
                    "anchor_radius_raw" to last.tensor(2),
                )).use { result -> toArray(result.tensor(0)) }
            }
            timings["head"] = (System.nanoTime() - th) / 1_000_000
            val total = (System.nanoTime() - t0) / 1_000_000
            timings["total"] = total
            timings["session_setup"] = sessionSetupMs
            return GaussianOutput(gauss, gaussianCount, timings)
        } finally {
            try {
                previous?.close()
            } finally {
                encoded?.close()
            }
        }
    }

    private fun imageShape(channels: Long) =
        longArrayOf(1, VIEWS.toLong(), channels, imgH.toLong(), imgW.toLong())

    private fun OrtSession.Result.tensor(index: Int): OnnxTensor = get(index) as OnnxTensor

    private fun runBlock(
        session: OrtSession,
        encoded: OrtSession.Result,
        tokens: OnnxTensor,
        anchor: OnnxTensor,
        radius: OnnxTensor,
    ): OrtSession.Result = session.run(mapOf(
        "tokens" to tokens,
        "anchor_raw" to anchor,
        "anchor_radius_raw" to radius,
        "keys" to encoded.tensor(0),
        "values" to encoded.tensor(1),
        "patch_rays" to encoded.tensor(2),
    ))

    private inline fun withSeeds(run: (OnnxTensor, OnnxTensor, OnnxTensor) -> Unit) {
        val n = nGsTokens.toLong()
        floatTensor(loadConst(LocusGSExport.TOKEN_INIT, nGsTokens * TOKEN_DIM),
            longArrayOf(1, n, TOKEN_DIM.toLong())).use { tokens ->
            floatTensor(loadConst(LocusGSExport.ANCHOR_INIT, nGsTokens * 3),
                longArrayOf(1, n, 3)).use { anchor ->
                floatTensor(loadConst(LocusGSExport.RADIUS_INIT, nGsTokens),
                    longArrayOf(1, n, 1)).use { radius ->
                    run(tokens, anchor, radius)
                }
            }
        }
    }

    /**
     * Logs actual resident set size, not JVM heap.
     *
     * The heap numbers Android reports are useless here: the weights live in
     * native memory owned by ORT, so only VmRSS shows the real cost.
     */
    private fun logRss(stage: String) {
        try {
            val status = File("/proc/self/status").readLines()
            val rss = status.firstOrNull { it.startsWith("VmRSS:") }?.trim() ?: "?"
            val hwm = status.firstOrNull { it.startsWith("VmHWM:") }?.trim() ?: "?"
            android.util.Log.i("LocusGSOnDevice", "$stage | $rss | peak $hwm")
        } catch (t: Throwable) {
            android.util.Log.w("LocusGSOnDevice", "rss unreadable at $stage: $t")
        }
    }

    /** Reads a raw little-endian float32 blob of [count] values. */
    /**
     * Seed tensors never change between forward passes, but they are ~4 MB
     * together and used to be re-read and re-decoded from the filesystem on
     * every call. Cached per runner: [release] drops them.
     */
    private val constants = HashMap<String, FloatArray>()

    private fun loadConst(name: String, count: Int): FloatArray =
        constants.getOrPut(name) {
            val bytes = externalDir
                ?.let { File(it, name).readBytes() }
                ?: assets.open("$modelDir/$name").use { it.readBytes() }
            require(bytes.size == count * 4) {
                "$name: expected ${count * 4} bytes for $count floats, got ${bytes.size}"
            }
            FloatArray(count).also {
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it)
            }
        }

    /**
     * Wraps [data] without copying.
     *
     * [FloatBuffer.wrap] is already a view over the array; ORT performs the one
     * copy into native memory itself. A hand-built direct buffer looks faster
     * but adds a copy and must also satisfy ORT's element-count check exactly.
     */
    private fun floatTensor(data: FloatArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)

    private fun toArray(t: OnnxTensor): FloatArray {
        // ORT 1.20 returns a copied heap buffer for float tensors. Reuse its
        // backing array when it covers the entire output instead of copying again.
        val fb = t.floatBuffer
        if (fb.hasArray() && fb.arrayOffset() == 0 && fb.position() == 0 &&
            fb.remaining() == fb.array().size) return fb.array()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    /** Removes any APK-extracted graphs; externally supplied graphs are not owned here. */
    fun release() {
        constants.clear()
        if (externalDir == null) {
            val dir = File(cacheRoot(), modelDir)
            dir.listFiles()?.forEach { it.delete() }
        }
        backendName = "released"
    }
}

/**
 * One session at a time, without the arena accumulating native allocations between graphs.
 *
 * `BASIC_OPT` is deliberate, not a default that was never revisited. EXTENDED was
 * tried and showed no benefit: one run each gave 24.7 s (4092 ms session setup) vs
 * 22.8-23.9 s (2109-3060 ms) for BASIC, i.e. the difference sits inside this
 * emulator's run-to-run spread of roughly a second. The graphs are exported with
 * their constants already folded, so there is little left for EXTENDED to do and
 * some extra load work. Re-test with repeated runs (not singles) before changing it.
 */
internal inline fun <T> withLocusGSSessionOptions(
    threads: Int,
    metadataOnly: Boolean = false,
    action: (OrtSession.SessionOptions) -> T,
): T = OrtSession.SessionOptions().use { options ->
    options.setIntraOpNumThreads(threads)
    options.setOptimizationLevel(if (metadataOnly) OrtSession.SessionOptions.OptLevel.NO_OPT
        else OrtSession.SessionOptions.OptLevel.BASIC_OPT)
    options.setMemoryPatternOptimization(false)
    options.setCPUArenaAllocator(false)
    options.addConfigEntry("session.use_env_allocators", "0")
    options.addConfigEntry("session.use_ort_model_bytes_for_initializers", "1")
    action(options)
}
