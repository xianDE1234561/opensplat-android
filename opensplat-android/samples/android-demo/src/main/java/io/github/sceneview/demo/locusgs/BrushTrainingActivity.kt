package io.github.sceneview.demo.locusgs

import android.app.Activity
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import io.github.sceneview.demo.service.SplatTrainingService
import java.io.File

/**
 * ADB entry point for paired ZIP training through the same service as the UI.
 * Extras: bundle, iters, export_name, initialization (existing/direct),
 * frame_selection (original/coverage; selects a supplied paired ZIP), seed (long).
 * Existing bundles keep Brush's native initialization; Direct adds PR PocketGS init.ply.
 */
class BrushTrainingActivity : Activity() {
    companion object {
        private const val TAG = "BrushWarmStart"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val status = TextView(this).apply { textSize = 11f; setTextIsSelectable(true) }
        setContentView(ScrollView(this).apply { addView(status) })

        val bundlePath = intent.getStringExtra("bundle")
            ?: File(Environment.getExternalStorageDirectory(), "Download/dev_bundle.zip").absolutePath
        val iters = intent.getIntExtra("iters", 200)
        val exportName = intent.getStringExtra("export_name") ?: "warmstart_$iters"

        val f = File(bundlePath)
        if (!f.exists() || f.length() == 0L) {
            val msg = "bundle missing: $bundlePath"
            Log.e(TAG, msg)
            status.text = msg
            return
        }

        Log.i(TAG, "starting Brush: bundle=$bundlePath bytes=${f.length()} iters=$iters")
        status.text = "starting Brush\n$bundlePath\niters=$iters\nexport=$exportName"

        // SplatTrainingService writes its own logcat tag (SplatTrainingService)
        // and also runs in the :training process, where the native library loads
        // the init.ply. Progress therefore shows up under those tags.
        SplatTrainingService.start(
            context = applicationContext,
            totalIterations = iters,
            datasetName = f.name,
            initialization = intent.getStringExtra("initialization") ?: "existing",
            frameSelection = intent.getStringExtra("frame_selection") ?: "original",
            trainingSeed = intent.getLongExtra("seed", 42L),
            datasetPath = bundlePath,
            exportName = exportName,
            splatId = exportName,
        )
    }
}
