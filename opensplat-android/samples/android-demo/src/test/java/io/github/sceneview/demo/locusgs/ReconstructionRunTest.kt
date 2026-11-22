package io.github.sceneview.demo.locusgs

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ReconstructionRunTest {
    @Test fun cannotReportTrainingCompleteBeforeStartingBrush() {
        val ready = ReconstructionRun()
            .advance(ReconstructionStage.SELECTING_VIEWS)
            .advance(ReconstructionStage.PREPROCESSING)
            .advance(ReconstructionStage.LOCUSGS_INFERENCE)
            .advance(ReconstructionStage.GAUSSIAN_READY)
        assertThrows(IllegalStateException::class.java) { ready.advance(ReconstructionStage.COMPLETED) }
        assertEquals(ReconstructionStage.COMPLETED,
            ready.advance(ReconstructionStage.BRUSH_REFINING).advance(ReconstructionStage.COMPLETED).stage)
    }

    @Test fun interruptedRefinementRetainsInitialExport() {
        val run = ReconstructionRun(stage = ReconstructionStage.BRUSH_REFINING,
            initialPly = "init.ply", gaussianCount = 262144, completedSteps = 4, brushSteps = 10)
        val restarted = run.afterProcessRestart()
        assertEquals(ReconstructionStage.FAILED, restarted.stage)
        assertEquals("init.ply", restarted.initialPly)
        assertEquals(null, restarted.finalPly)
        assertEquals(ReconstructionStage.BRUSH_REFINING, restarted.advance(ReconstructionStage.BRUSH_REFINING).stage)
    }
}
