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
 * Headless Brush warm-start, driven by adb.
 *
 *     adb shell am start -n io.github.sceneview.demo/.locusgs.BrushTrainingActivity \
 *         --es bundle /sdcard/Download/dev_bundle.zip --ei iters 200
 *
 * Starts [SplatTrainingService] from inside the app, so the service's
 * `exported=false` stays intact -- only this debug entry point is exported.
 *
 * The bundle must contain `init.ply`; Brush looks that name up explicitly
 * (`crates/brush-dataset/src/formats/mod.rs`), which is what turns the run into
 * a warm start instead of a random initialisation.
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
            datasetName = "truck",
            datasetPath = bundlePath,
            exportName = exportName,
            splatId = "ws_$iters",
        )
    }
}
