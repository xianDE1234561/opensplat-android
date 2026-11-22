package io.github.sceneview.demo.locusgs

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.TensorInfo
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FilterInputStream
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.zip.ZipInputStream

/**
 * The single shipped LocusGS graph set: 4 views at 448×256, 4096 Gaussian tokens.
 *
 * One shape only. A phone capture is 16:9, this shape keeps 98.4 % of its width, and
 * every other aspect is served by [LocusGSPreprocess.cropGeometry] centre-cropping
 * into it — so there is no per-aspect ~1 GB export to install, pick, or keep in sync.
 */
internal object LocusGSModel {
    const val WIDTH = LocusGSRunner.DEFAULT_IMG_W
    const val HEIGHT = LocusGSRunner.DEFAULT_IMG_H
    val label: String get() = "${WIDTH}×${HEIGHT}"

    /** Directory under `noBackupFilesDir`; versioned because the export format changed. */
    const val DIRECTORY = "locusgs-4x256x448-4096-fp32-v2"
}

/** Installs a local SAF ZIP into app-private, non-backed-up storage. No network or APK assets. */
object LocusGSModelStore {
    // The inlined fp32 export is ~1 GB. Bound expanded input even for entries with no ZIP size.
    private const val MAX_GRAPH_BYTES = 256L * 1024 * 1024
    private const val MAX_PACKAGE_BYTES = 1280L * 1024 * 1024
    private val installLock = Mutex()

    /** Returns only a previously published installation; does not open ORT sessions. */
    fun installed(context: Context): File? =
        File(context.noBackupFilesDir, LocusGSModel.DIRECTORY).takeIf { hasModelFiles(it) }

    private fun hasModelFiles(dir: File): Boolean =
        LocusGSExport.graphFiles.all { name ->
            val file = File(dir, name)
            file.isFile && file.length() in 1..MAX_GRAPH_BYTES
        } && LocusGSExport.constantBytes.all { (name, bytes) ->
            val file = File(dir, name)
            file.isFile && file.length() == bytes
        }

    /**
     * Imports exactly the 17 flat files from [LocusGSExport]. The stream is caller-owned.
     * A complete installation is immutable; a failed or cancelled import is never published.
     */
    suspend fun install(context: Context, input: InputStream): File = withContext(Dispatchers.IO) {
        installLock.withLock {
            val root = context.noBackupFilesDir
            val staging = File(root, ".locusgs-install.part")
            // Clean up staging left behind if a previous process died during import.
            check(staging.deleteRecursively() && staging.mkdirs()) { "Cannot stage LocusGS model" }
            try {
                extract(input, staging) { ensureActive() }
                validateGraphs(staging) { ensureActive() }
                check(installed(context) == null) { "A LocusGS model is already installed" }
                val dest = File(root, LocusGSModel.DIRECTORY)
                ensureActive()
                check(dest.deleteRecursively()) { "Cannot remove incomplete LocusGS installation" }
                // Same-volume atomic publish: installed() never sees a half-populated directory.
                Files.move(staging.toPath(), dest.toPath(), StandardCopyOption.ATOMIC_MOVE)
                dest
            } finally {
                staging.deleteRecursively()
            }
        }
    }

    private inline fun extract(input: InputStream, directory: File, checkCancellation: () -> Unit) {
        val seen = HashSet<String>()
        val buffer = ByteArray(64 * 1024)
        var packageBytes = 0L
        // Close the inflater promptly without closing the SAF stream owned by the caller.
        val source = object : FilterInputStream(input) { override fun close() = Unit }
        ZipInputStream(source.buffered()).use { zip ->
            while (true) {
                checkCancellation()
                val entry = zip.nextEntry ?: break
                val name = entry.name
                val constantSize = LocusGSExport.constantBytes[name]
                require(!entry.isDirectory && (constantSize != null || name in LocusGSExport.graphFiles)) {
                    "Unexpected ZIP entry '$name'; expected a flat LocusGS model package"
                }
                require(seen.add(name)) { "Duplicate ZIP entry '$name'" }
                val limit = constantSize ?: MAX_GRAPH_BYTES
                require(entry.size == -1L || entry.size in 1..limit) { "Invalid size for '$name'" }
                var bytes = 0L
                File(directory, name).outputStream().use { output ->
                    while (true) {
                        checkCancellation()
                        val count = zip.read(buffer)
                        if (count < 0) break // EOF also verifies the entry CRC in ZipInputStream.
                        bytes += count
                        packageBytes += count
                        require(bytes <= limit && packageBytes <= MAX_PACKAGE_BYTES) {
                            "LocusGS package is too large at '$name'"
                        }
                        output.write(buffer, 0, count)
                    }
                }
                require(bytes > 0L && (constantSize == null || bytes == constantSize)) {
                    "Invalid size for '$name': got $bytes bytes" +
                        (constantSize?.let { ", expected $it" } ?: "")
                }
            }
        }
        val missing = LocusGSExport.graphFiles.filterNot(seen::contains) +
            LocusGSExport.constantBytes.keys.filterNot(seen::contains)
        require(missing.isEmpty()) { "LocusGS ZIP is missing: ${missing.joinToString()}" }
    }

    private inline fun validateGraphs(directory: File, checkCancellation: () -> Unit) {
        val env = OrtEnvironment.getEnvironment()
        withLocusGSSessionOptions(threads = 1, metadataOnly = true) { options ->
            for (name in LocusGSExport.graphFiles) {
                checkCancellation()
                try {
                    env.createSession(File(directory, name).absolutePath, options).use { session ->
                        if (name == LocusGSExport.ENCODER) {
                            // Fail here, not mid-inference: a package exported for another
                            // shape would otherwise only surface as a shape error deep in
                            // the run, after the graph load cost was already paid.
                            val shape = (session.inputInfo["images"]?.info as? TensorInfo)?.shape
                            val expected = LocusGSExport.signature(name).inputs.getValue("images")
                            require(shape != null && shape.contentEquals(expected)) {
                                "$name input 'images': expected ${expected.contentToString()}, got ${shape?.contentToString()}. " +
                                    "This package is not the ${LocusGSModel.label} export."
                            }
                        }
                        val signature = LocusGSExport.signature(name)
                        val inputs = session.inputInfo
                        require(inputs.keys == signature.inputs.keys) {
                            "$name: expected inputs ${signature.inputs.keys}, got ${inputs.keys}"
                        }
                        for ((input, shape) in signature.inputs) {
                            checkTensor("$name input '$input'", inputs[input], shape)
                        }
                        val outputs = session.outputInfo
                        // Exporter does not name outputs; the runner binds them by index in graph order.
                        val names = session.outputNames.toList()
                        require(names.size == signature.outputs.size) {
                            "$name: expected ${signature.outputs.size} outputs, got ${names.size}"
                        }
                        names.forEachIndexed { i, output ->
                            checkTensor("$name output #$i ('$output')", outputs[output], signature.outputs[i])
                        }
                    }
                } catch (e: OrtException) {
                    throw IllegalArgumentException("Cannot read LocusGS graph '$name': ${e.message}", e)
                }
            }
        }
    }

    private fun checkTensor(label: String, node: NodeInfo?, shape: LongArray) {
        val tensor = node?.info as? TensorInfo
        require(tensor != null && tensor.type == OnnxJavaType.FLOAT && tensor.shape.contentEquals(shape)) {
            "$label: expected float32 ${shape.contentToString()}, got ${node?.info}"
        }
    }
}

/**
 * Pinned from `android_infer/export_blocks.py` (FP32, 448×256, 4096 Gaussian
 * tokens) and `android_infer/export_init_state.py`.
 * Batch 1, 4 views, 8×8 patches, 12 decoder blocks.
 * Output order is keys/values/patch rays, then tokens/raw anchors/raw radii,
 * then Gaussians. Seeds are little-endian float32, with pre-softplus radii.
 * The graph checks establish runtime compatibility, not checkpoint provenance.
 */
internal object LocusGSExport {
    const val ENCODER = "encoder.onnx"
    const val HEAD = "head.onnx"
    const val BLOCK_COUNT = 12
    const val TOKEN_INIT = "gs_token_init.f32"
    const val ANCHOR_INIT = "anchor_init.f32"
    const val RADIUS_INIT = "anchor_radius_raw_init.f32"

    private val V = LocusGSRunner.VIEWS.toLong()
    private val n = LocusGSRunner.DEFAULT_N_GS_TOKENS.toLong()
    private val d = LocusGSRunner.TOKEN_DIM.toLong()
    private val state = linkedMapOf(
        "tokens" to longArrayOf(1, n, d),
        "anchor_raw" to longArrayOf(1, n, 3),
        "anchor_radius_raw" to longArrayOf(1, n, 1),
    )
    private val head = Signature(state, listOf(longArrayOf(
        1, n * LocusGSRunner.GAUSSIANS_PER_TOKEN, LocusGSRunner.GAUSSIAN_CHANNELS.toLong(),
    )))

    val graphFiles = listOf(ENCODER) + List(BLOCK_COUNT, ::blockFile) + HEAD
    val constantBytes = mapOf(TOKEN_INIT to n * d * 4, ANCHOR_INIT to n * 3 * 4, RADIUS_INIT to n * 4)

    fun blockFile(i: Int) = "block_${i.toString().padStart(2, '0')}.onnx"

    fun signature(name: String): Signature {
        val h = LocusGSModel.HEIGHT.toLong()
        val w = LocusGSModel.WIDTH.toLong()
        val patchCount = V * (h / LocusGSRunner.PATCH) * (w / LocusGSRunner.PATCH)
        val kv = longArrayOf(1, LocusGSRunner.HEADS.toLong(), patchCount, LocusGSRunner.HEAD_DIM.toLong())
        val patchRays = longArrayOf(1, patchCount, 6)
        return when (name) {
            ENCODER -> Signature(
                mapOf("images" to longArrayOf(1, V, 3, h, w), "plucker" to longArrayOf(1, V, 6, h, w)),
                listOf(kv, kv, patchRays),
            )
            HEAD -> head
            else -> Signature(state + mapOf("keys" to kv, "values" to kv, "patch_rays" to patchRays), state.values.toList())
        }
    }

    data class Signature(val inputs: Map<String, LongArray>, val outputs: List<LongArray>)
}
