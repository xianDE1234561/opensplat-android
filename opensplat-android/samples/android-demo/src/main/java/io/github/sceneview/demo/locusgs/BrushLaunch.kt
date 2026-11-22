package io.github.sceneview.demo.locusgs

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

internal object BrushLaunch {
    /** LocusGS uses the first camera's OpenCV world; PocketGS keeps the scene world. */
    fun writeBundle(
        scene: ReconstructionScene, run: ReconstructionRun, initial: File, output: File,
        checkCancelled: () -> Unit = {},
    ) {
        require(initial.isFile && initial.length() > 0 && run.gaussianCount > 0)
        val frames = scene.frames.associateBy { it.id }
        val train = run.trainingIds.map(frames::getValue)
        val validation = run.validationIds.map(frames::getValue)
        require(train.isNotEmpty() && run.trainingIds.toSet().intersect(run.validationIds.toSet()).isEmpty())
        require(train.size + validation.size == frames.size)
        val worldToModel = if (run.initialization == InitializationMethod.LOCUSGS) {
            run.sourceWorldToFirstCamera.toDoubleArray()
        } else null

        fun calibration(views: List<SceneFrame>): ByteArray = JSONObject().apply {
            put("camera_model", "PERSPECTIVE")
            put("frames", JSONArray().apply {
                for (frame in views) {
                    val pose = worldToModel?.let { transform ->
                        LocusGSPreprocess.multiply4x4(transform, frame.c2w).also {
                            for (r in 0..2) it[4 * r + 3] *= run.translationScale
                        }
                    } ?: frame.c2w
                    put(JSONObject().apply {
                        put("file_path", "./${frame.id}")
                        put("w", frame.width); put("h", frame.height)
                        put("fl_x", frame.fx); put("fl_y", frame.fy)
                        put("cx", frame.cx); put("cy", frame.cy)
                        put("transform_matrix", JSONArray().apply {
                            for (r in 0..3) put(JSONArray().apply {
                                for (c in 0..3) put(pose[4 * r + c])
                            })
                        })
                    })
                }
            })
        }.toString().toByteArray(Charsets.UTF_8)

        ZipOutputStream(output.outputStream().buffered()).use { zip ->
            // Most entries are already compressed images. Avoid recompressing them.
            zip.setLevel(0)
            fun entry(name: String, write: () -> Unit) {
                checkCancelled()
                zip.putNextEntry(ZipEntry(name))
                write()
                zip.closeEntry()
            }
            entry("init.ply") { initial.inputStream().use { it.copyTo(zip) } }
            entry("transforms.json") { zip.write(calibration(train)) }
            if (validation.isNotEmpty()) entry("transforms_val.json") { zip.write(calibration(validation)) }
            for (frame in train + validation) {
                entry(frame.id) { frame.file.inputStream().use { it.copyTo(zip) } }
            }
        }
    }
}
