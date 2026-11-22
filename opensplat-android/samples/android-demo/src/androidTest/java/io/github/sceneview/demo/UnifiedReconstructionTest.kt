package io.github.sceneview.demo

import android.content.Context
import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import io.github.sceneview.demo.locusgs.ReconstructionRunStore
import io.github.sceneview.demo.locusgs.LocusGSModel
import io.github.sceneview.demo.locusgs.ReconstructionStage
import io.github.sceneview.demo.state.LocusGSStateHolder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class UnifiedReconstructionTest {
    @Test fun importForwardAndSaveInitialPly() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("Set -e unified_fixture true to run the local model integration test",
            args.getString("unified_fixture") == "true")
        val context = instrumentation.targetContext
        val device = UiDevice.getInstance(instrumentation)
        device.pressHome()
        context.startActivity(Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            .putExtra("demo", "locusgs-pipeline"))
        assertTrue("Reconstruction route did not open", device.wait(
            Until.hasObject(By.text("Import dataset ZIP")), 30_000,
        ))
        pick(device, "Import dataset ZIP", args.getString("scene_zip") ?: "truck-scene.zip")
        await(60_000) { onMain { LocusGSStateHolder.scene != null } }
        clickText(device, "LocusGS", scroll = true)
        if (onMain { LocusGSStateHolder.model } == null) {
            pick(device, "Install LocusGS model ZIP", args.getString("model_zip") ?: "locusgs-448-chunked.zip")
            await(180_000) { onMain { LocusGSStateHolder.model != null } }
        }
        clickText(device, "Select keyframes and run LocusGS", scroll = true)
        await(240_000) { onMain { !LocusGSStateHolder.busy } }
        val run = onMain { LocusGSStateHolder.run }
        assertEquals(run.error, ReconstructionStage.GAUSSIAN_READY, run.stage)
        assertEquals(262144, run.gaussianCount)
        assertEquals(4, run.contextIds.toSet().size)
        assertTrue(run.imageTransforms.size == 4 && run.imageTransforms.all {
            it.targetWidth == LocusGSModel.WIDTH && it.targetHeight == LocusGSModel.HEIGHT
        })
        assertTrue(run.contextIds.all { it in run.trainingIds && it !in run.validationIds })
        assertEquals(14, run.graphTimingsMs.keys.count { it == "encoder" || it == "head" || it.startsWith("block_") })
        val ply = onMain { LocusGSStateHolder.artifact(run.initialPly) }
        assertNotNull(ply)
        assertTrue(ply!!.length() > 262144L * 56)
        val root = onMain { LocusGSStateHolder.scene!!.root }
        assertEquals(run, ReconstructionRunStore.load(root))
        copyEvidence(context, root, ply, "initial.ply")
        val verifyViewer = args.getString("verify_viewer") == "true"
        if (verifyViewer) preview(device, context, "initial", run.gaussianCount)
        if (args.getString("verify_saf_export") == "true") {
            // The exported document must come from the actual App action, not copyEvidence.
            clickText(device, "Initial PLY", scroll = true)
            clickDocumentSave(device)
            await(30_000) { onMain { LocusGSStateHolder.message == "Saved ${ply.name}" } }
        }
        val refine = args.getString("brush_steps")?.toIntOrNull() ?: return
        instrumentation.runOnMainSync { LocusGSStateHolder.refine(context, refine) }
        await(600_000) { onMain { !LocusGSStateHolder.busy } }
        val trained = onMain { LocusGSStateHolder.run }
        copyEvidence(context, root, ply, "initial.ply")
        if (args.getString("expect_brush_failure") == "true") {
            assertEquals(ReconstructionStage.FAILED, trained.stage)
            assertEquals(0, trained.completedSteps)
        } else {
            assertEquals(trained.error, ReconstructionStage.COMPLETED, trained.stage)
            assertEquals(refine, trained.completedSteps)
            val final = onMain { LocusGSStateHolder.artifact(trained.finalPly) }
            assertNotNull(final)
            copyEvidence(context, root, final!!, "refined.ply")
            if (verifyViewer) preview(device, context, "refined", trained.gaussianCount)
        }
    }

    private fun preview(device: UiDevice, context: Context, result: String, count: Int) {
        clickText(device, "View $result Gaussians", scroll = true)
        assertTrue("$result viewport did not render", device.wait(
            Until.hasObject(By.text("$count / $count Gaussians")), 30_000,
        ))
        val out = File(context.getExternalFilesDir(null), "unified-validation")
        assertTrue("Could not capture $result viewport", device.takeScreenshot(File(out, "viewer-$result.png")))
        val x = device.displayWidth / 2
        val y = device.displayHeight / 2
        assertTrue(device.swipe(x, y, x + device.displayWidth / 5, y, 20))
        assertTrue(device.takeScreenshot(File(out, "viewer-$result-orbit.png")))
        device.pressBack()
        assertTrue("Back did not return to the project", device.wait(
            Until.hasObject(By.text("Local reconstruction")), 10_000,
        ))
    }

    private fun copyEvidence(context: Context, root: File, ply: File, name: String) {
        val out = File(context.getExternalFilesDir(null), "unified-validation").apply { mkdirs() }
        ply.copyTo(File(out, name), overwrite = true)
        File(root, ReconstructionRunStore.FILE_NAME).copyTo(File(out, "reconstruction.json"), overwrite = true)
    }

    private fun pick(device: UiDevice, action: String, file: String) {
        clickText(device, action)
        if (!device.wait(Until.hasObject(By.text(file)), 4_000)) {
            device.findObject(By.desc("Show roots"))?.let { root ->
                root.visibleCenter.let { device.click(it.x, it.y) }
            }
            clickText(device, "Downloads")
        }
        clickText(device, file)
    }

    private fun clickDocumentSave(device: UiDevice) {
        val save = device.wait(Until.findObject(By.res("android:id/button1")), 15_000)
        assertNotNull("Missing DocumentsUI save button", save)
        save.visibleCenter.let { device.click(it.x, it.y) }
    }

    private fun clickText(device: UiDevice, label: String, scroll: Boolean = false) {
        val limit = android.os.SystemClock.elapsedRealtime() + 15_000
        while (android.os.SystemClock.elapsedRealtime() < limit) {
            val objectToClick = device.findObject(By.text(label))
            try {
                if (objectToClick != null) {
                    val point = objectToClick.visibleCenter
                    if (device.click(point.x, point.y)) return
                }
            } catch (_: androidx.test.uiautomator.StaleObjectException) {
                // DocumentsUI can replace the node during its opening animation.
            }
            if (scroll && objectToClick == null) {
                val x = device.displayWidth / 2
                device.swipe(x, device.displayHeight * 3 / 4, x, device.displayHeight / 3, 14)
            }
            Thread.sleep(250)
        }
        error("Could not click '$label'; foreground=${device.currentPackageName}")
    }

    private fun await(timeoutMs: Long, condition: () -> Boolean) {
        val limit = android.os.SystemClock.elapsedRealtime() + timeoutMs
        while (!condition() && android.os.SystemClock.elapsedRealtime() < limit) Thread.sleep(250)
        assertTrue(onMain { LocusGSStateHolder.message ?: LocusGSStateHolder.run.error }, condition())
    }

    private fun <T> onMain(block: () -> T): T {
        var result: Result<T>? = null
        InstrumentationRegistry.getInstrumentation().runOnMainSync { result = runCatching(block) }
        return result!!.getOrThrow()
    }
}
