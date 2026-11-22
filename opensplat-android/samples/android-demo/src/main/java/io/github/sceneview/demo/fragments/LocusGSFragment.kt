package io.github.sceneview.demo.fragments

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.runtime.Composable
import io.github.sceneview.demo.DemoCategory
import io.github.sceneview.demo.DemoEntry
import io.github.sceneview.demo.R
import io.github.sceneview.demo.demos.LocusGSDemo

/**
 * Feed-forward LocusGS initialisation followed by Brush refinement, all on the
 * device. Registered through the append-only fragment mechanism so it appears
 * in the demo list and the deep-link router like every other demo.
 */
object LocusGSFragment : DemoFragment {
    override val entry: DemoEntry = DemoEntry(
        id = "locusgs-pipeline",
        titleRes = R.string.demo_locusgs,
        subtitleRes = R.string.demo_locusgs_subtitle,
        category = DemoCategory.CONTENT,
        icon = Icons.Filled.AutoAwesome,
    )

    @Composable
    override fun Screen(onBack: () -> Unit) {
        LocusGSDemo(onBack)
    }
}
