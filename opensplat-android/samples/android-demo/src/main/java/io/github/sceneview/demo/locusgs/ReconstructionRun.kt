package io.github.sceneview.demo.locusgs

import android.util.AtomicFile
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
internal enum class ReconstructionStage {
    SCENE_READY, SELECTING_VIEWS, PREPROCESSING, LOCUSGS_INFERENCE, GAUSSIAN_READY,
    BRUSH_REFINING, COMPLETED, FAILED, CANCELLED, POCKETGS_INITIALIZING;

    val active: Boolean
        get() = this in setOf(SELECTING_VIEWS, PREPROCESSING, LOCUSGS_INFERENCE, BRUSH_REFINING, POCKETGS_INITIALIZING)

    fun allows(next: ReconstructionStage): Boolean = when (this) {
        SCENE_READY, GAUSSIAN_READY, COMPLETED, FAILED, CANCELLED ->
            next == SELECTING_VIEWS || next == BRUSH_REFINING || next == POCKETGS_INITIALIZING
        SELECTING_VIEWS -> next in setOf(PREPROCESSING, FAILED, CANCELLED)
        PREPROCESSING -> next in setOf(LOCUSGS_INFERENCE, FAILED, CANCELLED)
        LOCUSGS_INFERENCE -> next in setOf(GAUSSIAN_READY, FAILED, CANCELLED)
        POCKETGS_INITIALIZING -> next in setOf(GAUSSIAN_READY, FAILED, CANCELLED)
        BRUSH_REFINING -> next in setOf(COMPLETED, FAILED, CANCELLED)
    }
}

@Serializable
internal data class ReconstructionRun(
    val stage: ReconstructionStage = ReconstructionStage.SCENE_READY,
    val contextIds: List<String> = emptyList(),
    val trainingIds: List<String> = emptyList(),
    val validationIds: List<String> = emptyList(),
    val sourceWorldToFirstCamera: List<Double> = emptyList(),
    val translationScale: Double = 1.0,
    val imageTransforms: List<ContextImageTransform> = emptyList(),
    val gaussianCount: Int = 0,
    val selectionMs: Long = 0,
    val preprocessingMs: Long = 0,
    val forwardMs: Long = 0,
    val graphTimingsMs: Map<String, Long> = emptyMap(),
    val totalToInitMs: Long = 0,
    val inferenceExecutionProvider: String? = null,
    val peakProcessRssKiB: Long? = null,
    val brushBackendRequested: String? = null,
    val brushSteps: Int = 0,
    val completedSteps: Int = 0,
    val brushElapsedMs: Long = 0,
    val psnr: Float? = null,
    val ssim: Float? = null,
    val initialPly: String? = null,
    val finalPly: String? = null,
    val error: String? = null,
    val initialization: InitializationMethod = InitializationMethod.LOCUSGS,
) {
    fun advance(next: ReconstructionStage, error: String? = null): ReconstructionRun {
        check(stage.allows(next)) { "Invalid reconstruction transition: $stage → $next" }
        return copy(stage = next, error = error)
    }

    // Native training can outlive the foreground UI, but the current AAR does not
    // expose reattachment by job ID after UI process death.
    fun afterProcessRestart(): ReconstructionRun = if (stage.active) {
        advance(ReconstructionStage.FAILED, "App process interrupted; open the project and retry.")
    } else this
}

@Serializable
internal enum class InitializationMethod { LOCUSGS, POCKETGS }

@Serializable
internal data class ContextImageTransform(
    val id: String, val sourceWidth: Int, val sourceHeight: Int,
    val targetWidth: Int, val targetHeight: Int,
    val shiftX: Double, val shiftY: Double, val scaleX: Double, val scaleY: Double,
)

internal object ReconstructionRunStore {
    private val json = Json { prettyPrint = true; encodeDefaults = true; ignoreUnknownKeys = true }
    const val FILE_NAME = "reconstruction.json"

    fun load(root: File): ReconstructionRun {
        val file = AtomicFile(File(root, FILE_NAME))
        return try {
            json.decodeFromString<ReconstructionRun>(file.openRead().bufferedReader().use { it.readText() })
        } catch (_: java.io.FileNotFoundException) {
            ReconstructionRun()
        }
    }

    fun save(root: File, run: ReconstructionRun) {
        val file = AtomicFile(File(root, FILE_NAME))
        val out = file.startWrite()
        try {
            out.write(json.encodeToString(run).toByteArray(Charsets.UTF_8))
            file.finishWrite(out)
        } catch (t: Throwable) {
            file.failWrite(out)
            throw t
        }
    }
}
