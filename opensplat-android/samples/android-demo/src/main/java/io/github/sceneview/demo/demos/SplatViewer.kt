@file:OptIn(ExperimentalMaterial3Api::class)

package io.github.sceneview.demo.demos

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.outlined.ViewInAr
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import io.github.sceneview.Aabb
import io.github.sceneview.SceneView
import io.github.sceneview.core.splat.SplatCloud
import io.github.sceneview.core.splat.SplatParser
import io.github.sceneview.demo.DemoScaffold
import io.github.sceneview.demo.LoadingScrim
import io.github.sceneview.demo.R
import io.github.sceneview.demo.common.rememberModelDemoEnvironment
import io.github.sceneview.demo.rememberFirstFrameState
import io.github.sceneview.fitDistanceForBounds
import io.github.sceneview.gesture.CameraGestureDetector.DefaultCameraManipulator
import io.github.sceneview.math.Position
import io.github.sceneview.math.Rotation
import io.github.sceneview.rememberCameraNode
import io.github.sceneview.rememberEngine
import io.github.sceneview.rememberEnvironmentLoader
import io.github.sceneview.splat.SplatBuffers
import io.github.sceneview.verticalFovDegreesForFocalLength
import java.io.ByteArrayOutputStream
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

/** The PLY parser materialises the whole file, so refuse sizes a phone cannot hold. */
private const val MAX_SPLAT_BYTES = 256L * 1024 * 1024

/** Fraction of the cloud the viewer draws, offered as one-tap display presets. */
private val VISIBLE_FRACTIONS = listOf(1f, 0.75f, 0.5f, 0.25f)

private fun previewBounds(cloud: SplatCloud, reconstruction: Boolean): Aabb {
    val full = SplatBuffers.boundingBox(cloud)
    if (!reconstruction) return Aabb(Position(full[0], full[1], full[2]), Position(full[3], full[4], full[5]))

    // Sparse, low-opacity points should not push the preview camera far away.
    val stride = (cloud.count / 8192).coerceAtLeast(1)
    val axes = Array(3) { FloatArray((cloud.count - 1) / stride + 1) }
    var size = 0
    for (i in 0 until cloud.count step stride) {
        if (cloud.opacities[i] < 0.005f) continue
        for (axis in 0..2) axes[axis][size] = cloud.positions[i * 3 + axis]
        size++
    }
    if (size < 32) return Aabb(Position(full[0], full[1], full[2]), Position(full[3], full[4], full[5]))
    axes.forEach { it.sort(0, size) }
    val low = (size - 1) / 100
    val high = size - 1 - low
    fun center(axis: Int) = (axes[axis][low] + axes[axis][high]) / 2f
    fun half(axis: Int) = ((axes[axis][high] - axes[axis][low]) / 2f).coerceAtLeast(1e-4f)
    return Aabb(Position(center(0), center(1), center(2)), Position(half(0), half(1), half(2)))
}

private sealed interface SplatLoadState {
    data object Loading : SplatLoadState
    data class Ready(val cloud: SplatCloud) : SplatLoadState
    data class Failed(val message: String) : SplatLoadState
}

/** Shows an exported result or a local document in its own coordinate system. */
@Composable
internal fun SplatViewer(
    onBack: (() -> Unit)? = null,
    resultFile: File? = null,
    resultTitle: String? = null,
    reconstruction: Boolean = resultFile != null,
    onBrowseLibrary: (() -> Unit)? = null,
) {
    val resolver = LocalContext.current.contentResolver
    // A local override belongs only to this entry. Opening a different result starts at its export.
    val entry = resultFile?.absolutePath
    var documentUri by key(entry) { rememberSaveable { mutableStateOf<String?>(null) } }
    var selection by remember { mutableIntStateOf(0) }
    var pickerEntry by rememberSaveable { mutableStateOf<String?>(null) }
    val filePickerLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val requestedEntry = pickerEntry
        pickerEntry = null
        if (uri != null && requestedEntry == entry.orEmpty()) {
            // Providers that can't persist access can still grant access for the current session.
            try {
                resolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: SecurityException) {
                // The load below reports a read error if access is no longer available.
            }
            documentUri = uri.toString()
            selection++
        }
    }
    // Providers often publish PLY/SPZ as application/octet-stream or with no useful MIME type.
    val openDocument = {
        pickerEntry = entry.orEmpty()
        filePickerLauncher.launch(arrayOf("*/*"))
    }
    val localUri = documentUri
    val result = resultFile.takeIf { localUri == null }
    val load = remember(localUri, result, selection) {
        mutableStateOf<SplatLoadState?>(
            if (localUri != null || result != null) SplatLoadState.Loading else null
        )
    }
    LaunchedEffect(load) {
        if (load.value != SplatLoadState.Loading) return@LaunchedEffect
        load.value = try {
            val bytes = withContext(Dispatchers.IO) {
                if (localUri != null) {
                    val uri = Uri.parse(localUri)
                    resolver.openAssetFileDescriptor(uri, "r")?.let { descriptor ->
                        descriptor.createInputStream().use { input ->
                            val length = descriptor.length
                            checkSplatSize(length)
                            val buffer = ByteArray(64 * 1024)
                            val output = ByteArrayOutputStream(length.takeIf { it > 0 }?.toInt() ?: buffer.size)
                            while (true) {
                                ensureActive()
                                val read = input.read(buffer)
                                if (read == -1) break
                                checkSplatSize(output.size().toLong() + read)
                                output.write(buffer, 0, read)
                            }
                            output.toByteArray()
                        }
                    } ?: error("Could not read the selected file.")
                } else {
                    val file = checkNotNull(result)
                    val length = file.length()
                    check(file.isFile && length > 0) { "The exported model is missing or empty." }
                    checkSplatSize(length)
                    file.readBytes().also { checkSplatSize(it.size.toLong()) }
                }
            }
            val cloud = withContext(Dispatchers.Default) { SplatParser.parse(bytes) }
            if (cloud.count > 0) SplatLoadState.Ready(cloud)
            else SplatLoadState.Failed("That file contains no Gaussians.")
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            SplatLoadState.Failed("Could not open splat: ${e.localizedMessage ?: e.javaClass.simpleName}")
        }
    }

    val state = load.value
    val cloud = (state as? SplatLoadState.Ready)?.cloud
    val orientAsReconstruction = result != null && reconstruction
    val bounds = remember(cloud, orientAsReconstruction) {
        cloud?.let { previewBounds(it, orientAsReconstruction) }
    }
    var preset by remember(cloud) { mutableIntStateOf(0) }
    val splatCount = cloud?.let { (it.count * VISIBLE_FRACTIONS[preset]).toInt().coerceIn(1, it.count) } ?: 0
    var resetCamera by remember(cloud, orientAsReconstruction) { mutableIntStateOf(0) }
    val firstFrame = if (cloud != null) key(cloud) { rememberFirstFrameState() } else null

    val displayControls: @Composable androidx.compose.foundation.layout.ColumnScope.() -> Unit = {
        val count = checkNotNull(cloud).count
            Text(
                text = "Drawing ${"%,d".format(splatCount)} of ${"%,d".format(count)} Gaussians",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(modifier = Modifier.height(12.dp))
            SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                VISIBLE_FRACTIONS.forEachIndexed { index, fraction ->
                    SegmentedButton(
                        selected = preset == index,
                        onClick = { preset = index },
                        shape = SegmentedButtonDefaults.itemShape(index, VISIBLE_FRACTIONS.size),
                    ) {
                        Text("${(fraction * 100).toInt()}%")
                    }
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "Drag to orbit · pinch to zoom · use Reset to reframe the model.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
    }

    DemoScaffold(
        title = if (localUri != null) "Local splat" else resultTitle ?: stringResource(R.string.demo_splat_viewer),
        onBack = onBack,
        firstFrameRendered = firstFrame?.rendered,
        onReset = if (cloud != null) ({ resetCamera++ }) else null,
        controls = if (cloud != null) displayControls else null,
        topActions = {
            TextButton(onClick = openDocument) { Text("Open") }
        },
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            if (cloud != null && bounds != null && firstFrame != null) {
                val engine = rememberEngine()
                val environmentLoader = rememberEnvironmentLoader(engine)
                val cameraNode = rememberCameraNode(engine)
                var aspect by remember { mutableStateOf(0.5) }
                val radius = fitDistanceForBounds(
                    bounds, verticalFovDegreesForFocalLength(28.0), aspect, padding = 0.04f
                ).coerceAtLeast(0.02f)
                val cameraManipulator = remember(bounds, radius, resetCamera) {
                    DefaultCameraManipulator(
                        orbitHomePosition = Position(0f, 0f, radius),
                        targetPosition = Position(0f, 0f, 0f),
                    )
                }
                SceneView(
                    modifier = Modifier.fillMaxSize().onSizeChanged {
                        if (it.width > 0 && it.height > 0) aspect = it.width.toDouble() / it.height
                    },
                    onFrame = firstFrame.onFrame,
                    engine = engine,
                    cameraNode = cameraNode,
                    environmentLoader = environmentLoader,
                    environment = rememberModelDemoEnvironment(environmentLoader),
                    cameraManipulator = cameraManipulator,
                    autoCenterContent = false,
                ) {
                    SplatNode(
                        splatCloud = cloud,
                        splatCount = splatCount,
                        cameraPositionProvider = { cameraNode.worldPosition },
                        position = bounds.center.let {
                            if (orientAsReconstruction) Position(-it.x, it.y, it.z) else -it
                        },
                        rotation = Rotation(x = if (orientAsReconstruction) 180f else 0f),
                        apply = { isTouchable = false }
                    )
                }
                if (firstFrame.rendered.value) {
                    Surface(
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(12.dp),
                        shape = RoundedCornerShape(percent = 50),
                        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.78f),
                        tonalElevation = 2.dp,
                    ) {
                        Text(
                            text = "$splatCount / ${cloud.count} Gaussians",
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                }
            } else if (state == SplatLoadState.Loading) {
                LoadingScrim(loading = true, label = stringResource(R.string.demo_splat_viewer_loading))
            } else {
                EmptySplatViewer(
                    onOpen = openDocument,
                    onBrowseLibrary = onBrowseLibrary,
                    error = (state as? SplatLoadState.Failed)?.message,
                    modifier = Modifier.align(Alignment.Center),
                )
            }
        }
    }
}

private fun checkSplatSize(bytes: Long) {
    require(bytes <= MAX_SPLAT_BYTES) { "Only files up to 256 MB can be opened on this device." }
}

@Composable
private fun EmptySplatViewer(
    onOpen: () -> Unit,
    onBrowseLibrary: (() -> Unit)?,
    error: String?,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .widthIn(max = 420.dp)
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Surface(
            shape = MaterialTheme.shapes.extraLarge,
            color = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
        ) {
            Icon(Icons.Outlined.ViewInAr, contentDescription = null, modifier = Modifier.padding(18.dp).size(36.dp))
        }
        Text(
            text = if (error == null) "View a Gaussian splat" else "Couldn't open model",
            style = MaterialTheme.typography.headlineSmall,
            textAlign = TextAlign.Center,
        )
        Text(
            text = error ?: "Open a local PLY or SPZ model to orbit and zoom around it in 3D.",
            style = MaterialTheme.typography.bodyMedium,
            color = if (error == null) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error,
            textAlign = TextAlign.Center,
        )
        Button(onClick = onOpen) {
            Icon(Icons.Filled.FolderOpen, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(modifier = Modifier.width(8.dp))
            Text(stringResource(R.string.demo_splat_viewer_select_file))
        }
        if (onBrowseLibrary != null) {
            TextButton(onClick = onBrowseLibrary) { Text("Browse library") }
        }
    }
}
