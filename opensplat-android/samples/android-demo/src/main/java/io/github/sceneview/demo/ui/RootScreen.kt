package io.github.sceneview.demo.ui

import androidx.activity.compose.BackHandler
import androidx.annotation.StringRes
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.FolderOpen
import androidx.compose.material.icons.outlined.Layers
import androidx.compose.material.icons.outlined.ViewInAr
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import io.github.sceneview.demo.R
import io.github.sceneview.demo.demos.ArSplatCaptureDemo
import io.github.sceneview.demo.demos.LocusGSDemo
import io.github.sceneview.demo.demos.SplatViewer
import io.github.sceneview.demo.state.LocusGSStateHolder
import io.github.sceneview.demo.storage.CaptureProject
import java.io.File

@Composable
internal fun RootScreen() {
    var selected by rememberSaveable { mutableStateOf(RootTab.Capture) }
    var viewerOrigin by rememberSaveable { mutableStateOf<RootTab?>(null) }
    var modelPath by rememberSaveable { mutableStateOf<String?>(null) }
    var modelTitle by rememberSaveable { mutableStateOf<String?>(null) }
    var modelUsesCameraFrame by rememberSaveable { mutableStateOf(false) }
    val tabs = rememberSaveableStateHolder()

    fun reconstruct(capture: CaptureProject) {
        LocusGSStateHolder.openCaptureForReconstruction(capture)
        selected = RootTab.Reconstruction
    }

    fun openModel(file: File, title: String, cameraFrame: Boolean) {
        viewerOrigin = selected
        modelPath = file.absolutePath
        modelTitle = title
        modelUsesCameraFrame = cameraFrame
        selected = RootTab.Viewer
    }

    val returnToSource = viewerOrigin?.let { origin ->
        { selected = origin; viewerOrigin = null }
    }
    BackHandler(enabled = selected == RootTab.Viewer && returnToSource != null) {
        returnToSource?.invoke()
    }

    Scaffold(
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.surfaceContainer) {
                RootTab.entries.forEach { tab ->
                    NavigationBarItem(
                        selected = selected == tab,
                        onClick = { if (selected != tab) { selected = tab; viewerOrigin = null } },
                        icon = { Icon(tab.icon, contentDescription = null) },
                        label = {
                            Text(stringResource(tab.labelRes), textAlign = TextAlign.Center,
                                style = MaterialTheme.typography.labelSmall)
                        },
                        colors = NavigationBarItemDefaults.colors(
                            indicatorColor = MaterialTheme.colorScheme.primaryContainer,
                            selectedIconColor = MaterialTheme.colorScheme.onPrimaryContainer,
                        ),
                    )
                }
            }
        },
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(padding).consumeWindowInsets(padding)) {
            tabs.SaveableStateProvider(selected.name) {
                when (selected) {
                    RootTab.Capture -> ArSplatCaptureDemo(onSendToTraining = ::reconstruct)
                    RootTab.Reconstruction -> LocusGSDemo(onOpenViewer = ::openModel)
                    RootTab.Library -> LibraryScreen(
                        onTrainProject = ::reconstruct,
                        onViewModel = ::openModel,
                    )
                    RootTab.Viewer -> androidx.compose.runtime.key(modelPath) {
                        SplatViewer(
                            onBack = returnToSource,
                            resultFile = modelPath?.let(::File),
                            resultTitle = modelTitle,
                            reconstruction = modelUsesCameraFrame,
                            onBrowseLibrary = { selected = RootTab.Library; viewerOrigin = null },
                        )
                    }
                }
            }
        }
    }
}

private enum class RootTab(@StringRes val labelRes: Int, val icon: ImageVector) {
    Capture(R.string.tab_capture, Icons.Outlined.CameraAlt),
    Reconstruction(R.string.tab_reconstruction, Icons.Outlined.Layers),
    Library(R.string.tab_library, Icons.Outlined.FolderOpen),
    Viewer(R.string.tab_viewer, Icons.Outlined.ViewInAr),
}

