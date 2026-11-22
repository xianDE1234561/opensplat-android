@file:OptIn(ExperimentalMaterial3Api::class)

package io.github.sceneview.demo.demos

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.outlined.Visibility
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.sceneview.demo.DemoScaffold
import io.github.sceneview.demo.locusgs.ReconstructionRun
import io.github.sceneview.demo.locusgs.InitializationMethod
import io.github.sceneview.demo.locusgs.LocusGSModel
import io.github.sceneview.demo.locusgs.ReconstructionRunStore
import io.github.sceneview.demo.locusgs.ReconstructionStage
import io.github.sceneview.demo.state.LocusGSStateHolder
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Older jobs stay on disk but are not all listed; the newest are the likely target. */
private const val MAX_LISTED_PROJECTS = 5

/** Human-readable stage label; the enum names are internal plumbing, not copy. */
private fun stageLabel(run: ReconstructionRun, busy: Boolean): String = when (run.stage) {
    ReconstructionStage.SCENE_READY -> if (busy) "Preparing" else "Ready"
    ReconstructionStage.SELECTING_VIEWS -> "Selecting keyframes"
    ReconstructionStage.PREPROCESSING -> "Preparing views"
    ReconstructionStage.LOCUSGS_INFERENCE -> "Running LocusGS"
    ReconstructionStage.POCKETGS_INITIALIZING -> "Estimating surface normals and scales"
    ReconstructionStage.GAUSSIAN_READY -> "Initial Gaussians ready"
    ReconstructionStage.BRUSH_REFINING -> "Refining on device"
    ReconstructionStage.COMPLETED -> "Refined"
    ReconstructionStage.FAILED -> "Failed"
    ReconstructionStage.CANCELLED -> "Cancelled"
}

@Composable
fun LocusGSDemo(
    onBack: (() -> Unit)? = null,
    onOpenViewer: ((File, String, Boolean) -> Unit)? = null,
) {
    val context = LocalContext.current
    val state = LocusGSStateHolder
    var steps by rememberSaveable { mutableStateOf("300") }
    var exportFile by remember { mutableStateOf<File?>(null) }
    var previewRefined by rememberSaveable(state.scene?.id) { mutableStateOf<Boolean?>(null) }
    var selectedMethod by rememberSaveable(state.scene?.id) {
        mutableStateOf(if (state.run.initialPly != null) state.run.initialization else InitializationMethod.POCKETGS)
    }

    val scenePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { state.importZip(context, it.lastPathSegment?.substringAfterLast('/') ?: "Imported scene", it) }
    }
    val modelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { state.installModel(context, it) }
    }
    val plyPicker = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
        val file = exportFile
        exportFile = null
        if (uri != null && file != null) state.export(context, file, uri)
    }
    val reportPicker = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        if (uri != null) state.artifact(ReconstructionRunStore.FILE_NAME)?.let { state.export(context, it, uri) }
    }
    // Rescan once on entry. Re-running on every recomposition would rebuild the
    // project list under the user's selection; the in-page action rescans on demand.
    LaunchedEffect(Unit) { state.refresh(context) }
    LaunchedEffect(state.busy) { if (!state.busy) state.consumeCapture(context) }

    val preview = previewRefined?.let { refined ->
        state.artifact(if (refined) state.run.finalPly else state.run.initialPly)
    }
    if (!state.busy && preview != null) {
        BackHandler { previewRefined = null }
        SplatViewer(
            onBack = { previewRefined = null },
            resultFile = preview,
            resultTitle = if (previewRefined == true) "Refined Gaussians" else "Initial Gaussians",
            reconstruction = state.run.initialization == InitializationMethod.LOCUSGS,
        )
        return
    }

    val scene = state.scene
    val run = state.run
    val initial = state.artifact(run.initialPly)
    val final = state.artifact(run.finalPly)

    DemoScaffold(title = "Reconstruction", onBack = onBack) {
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("Build a 3D scene", style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold)
            Text("Choose a scene, initialize Gaussians, then refine with Brush on this device.",
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    if (state.projects.isEmpty()) "No saved projects" else "${state.projects.size} saved",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f),
                )
                TextButton(onClick = { state.refresh(context) }, enabled = !state.busy) {
                    Text("Rescan")
                }
            }
            if (state.busy && !run.stage.active) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                TextButton(onClick = { state.cancel(context) }) {
                    Text("Cancel · ${state.operation ?: "Working"}")
                }
            }

            if (scene == null) {
                EmptyStateCard(
                    onImport = { scenePicker.launch(arrayOf("application/zip", "application/octet-stream")) },
                    canImport = !state.busy,
                    captures = state.captures.map { it.name },
                    onPickCapture = { index ->
                        state.importCapture(context, state.captures[index])
                    },
                )
                // Previous runs are the most likely thing to want after a restart, so the
                // empty state lists them instead of hiding them behind an import.
                if (state.projects.isNotEmpty()) {
                    ProjectsCard(
                        projects = state.projects.map { it.name to it.frames.size },
                        projectTimes = state.projects.map { it.importedAt },
                        projectIds = state.projects.map { it.id },
                        activeId = null,
                        enabled = !state.busy,
                        onSelect = { index -> state.selectProject(state.projects[index]) },
                        captures = emptyList(),
                        onPickCapture = {},
                        onImportZip = { scenePicker.launch(arrayOf("application/zip", "application/octet-stream")) },
                    )
                }
            } else {
                ResultCard(
                    sceneName = scene.name,
                    run = run,
                    busy = state.busy,
                    initial = initial,
                    final = final,
                    onViewInitial = {
                        if (onOpenViewer == null) previewRefined = false
                        else initial?.let { onOpenViewer(it, "${scene.name} · Initial", run.initialization == InitializationMethod.LOCUSGS) }
                    },
                    onViewRefined = {
                        if (onOpenViewer == null) previewRefined = true
                        else final?.let { onOpenViewer(it, "${scene.name} · Refined", run.initialization == InitializationMethod.LOCUSGS) }
                    },
                )
            }

            if (scene != null) {
                RunCard(
                    method = selectedMethod,
                    onMethodChange = { selectedMethod = it },
                    run = run,
                    busy = state.busy,
                    modelInstalled = state.model != null,
                    onInstallModel = { modelPicker.launch(arrayOf("application/zip", "application/octet-stream")) },
                    onForward = { state.forward(context) },
                    onCancel = { state.cancel(context) },
                    onDirect = state::initializeDirectly,
                )

                if (initial != null) {
                    RefineCard(
                        steps = steps,
                        onStepsChange = { steps = it },
                        busy = state.busy,
                        run = run,
                        hasFinal = final != null,
                        onRefine = { count -> state.refine(context, count) },
                        onCancel = { state.cancel(context) },
                        onExportInitial = { exportFile = initial; plyPicker.launch("initial.ply") },
                        onExportFinal = { file -> exportFile = file; plyPicker.launch("refined.ply") },
                        refinedFile = final,
                        onExportReport = { reportPicker.launch("reconstruction.json") },
                    )
                }

                ProjectsCard(
                    projects = state.projects.map { it.name to it.frames.size },
                    projectTimes = state.projects.map { it.importedAt },
                    activeId = scene.id,
                    projectIds = state.projects.map { it.id },
                    enabled = !state.busy,
                    onSelect = { index -> state.selectProject(state.projects[index]) },
                    captures = state.captures.map { it.name },
                    onPickCapture = { index -> state.importCapture(context, state.captures[index]) },
                    onImportZip = { scenePicker.launch(arrayOf("application/zip", "application/octet-stream")) },
                )
            }

            state.message?.takeIf { it.isNotBlank() }?.let { message ->
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Text(
                        message,
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

@Composable
private fun EmptyStateCard(
    onImport: () -> Unit,
    canImport: Boolean,
    captures: List<String>,
    onPickCapture: (Int) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Icon(
                Icons.Filled.AutoAwesome,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(40.dp),
            )
            Text(
                "Start from a scan",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                "Pick one of your captures, or import a Nerfstudio / COLMAP ZIP. Frames stay on device.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (captures.isNotEmpty()) {
                Text(
                    "Saved captures",
                    style = MaterialTheme.typography.labelLarge,
                    modifier = Modifier.fillMaxWidth(),
                )
                captures.forEachIndexed { index, name ->
                    OutlinedButton(
                        onClick = { onPickCapture(index) },
                        enabled = canImport,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Icon(Icons.Filled.FolderOpen, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(name, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
            Spacer(Modifier.height(4.dp))
            Button(onClick = onImport, enabled = canImport, modifier = Modifier.fillMaxWidth()) {
                Icon(Icons.Filled.CloudUpload, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text("Import dataset ZIP")
            }
        }
    }
}

@Composable
private fun ResultCard(
    sceneName: String,
    run: ReconstructionRun,
    busy: Boolean,
    initial: File?,
    final: File?,
    onViewInitial: () -> Unit,
    onViewRefined: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = when (run.stage) {
                ReconstructionStage.FAILED -> MaterialTheme.colorScheme.errorContainer
                ReconstructionStage.COMPLETED -> MaterialTheme.colorScheme.primaryContainer
                else -> MaterialTheme.colorScheme.surfaceContainer
            }
        ),
    ) {
        Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        sceneName,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Text(
                        stageLabel(run, busy),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (busy) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                }
            }

            run.error?.let { error ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Filled.ErrorOutline,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.error,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        error,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }

            val metrics = buildList {
                if (run.gaussianCount > 0) add("${run.gaussianCount} Gaussians")
                if (initial != null) add(if (run.initialization == InitializationMethod.POCKETGS) "PocketGS" else "LocusGS")
                if (run.forwardMs > 0) add("forward ${run.forwardMs / 1000} s")
                run.psnr?.let { add("PSNR %.2f dB".format(it)) }
                run.ssim?.let { add("SSIM %.3f".format(it)) }
            }
            // Splitting graph load out of the forward pass is the difference between
            // "the phone is slow" and "the phone reloads 12 graphs every run".
            val setupMs = run.graphTimingsMs["session_setup"] ?: 0L
            val split = if (run.forwardMs > 0 && setupMs > 0) {
                "graph load %.1f s · compute %.1f s".format(setupMs / 1000f, (run.forwardMs - setupMs).coerceAtLeast(0) / 1000f)
            } else null
            if (metrics.isNotEmpty()) {
                Text(
                    metrics.joinToString(" · "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            split?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (final != null) {
                Button(onClick = onViewRefined, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                    Icon(Icons.Outlined.Visibility, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("View refined Gaussians")
                }
            }
            if (initial != null) {
                if (final != null) {
                    OutlinedButton(onClick = onViewInitial, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                        Text("View initial Gaussians")
                    }
                } else {
                    Button(onClick = onViewInitial, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                        Icon(Icons.Outlined.Visibility, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("View initial Gaussians")
                    }
                }
            }
        }
    }
}

@Composable
private fun RunCard(
    method: InitializationMethod,
    onMethodChange: (InitializationMethod) -> Unit,
    run: ReconstructionRun,
    busy: Boolean,
    modelInstalled: Boolean,
    onInstallModel: () -> Unit,
    onForward: () -> Unit,
    onCancel: () -> Unit,
    onDirect: () -> Unit,
) {
    val direct = method == InitializationMethod.POCKETGS
    Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("1 · Initialize", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(direct, { onMethodChange(InitializationMethod.POCKETGS) },
                    label = { Text("Direct · PocketGS") }, enabled = !busy)
                FilterChip(!direct, { onMethodChange(InitializationMethod.LOCUSGS) },
                    label = { Text("LocusGS") }, enabled = !busy)
            }
            Text(
                if (direct) "Seed surface-aligned Gaussians from the scene’s colored point cloud. No model download."
                else "Initialize from four keyframes using the ${LocusGSModel.label} model. Inputs are center-cropped.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
            )
            Button(
                onClick = if (direct) onDirect else if (modelInstalled) onForward else onInstallModel,
                enabled = !busy,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (direct) "Initialize from point cloud" else if (modelInstalled)
                    "Select keyframes and run LocusGS" else "Install LocusGS model ZIP")
            }
            if (run.stage.active && run.stage != ReconstructionStage.BRUSH_REFINING) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                TextButton(onClick = onCancel) { Text("Cancel initialization") }
            }
        }
    }
}

@Composable
private fun RefineCard(
    steps: String,
    onStepsChange: (String) -> Unit,
    busy: Boolean,
    run: ReconstructionRun,
    hasFinal: Boolean,
    onRefine: (Int) -> Unit,
    onCancel: () -> Unit,
    onExportInitial: () -> Unit,
    onExportFinal: (File) -> Unit,
    onExportReport: () -> Unit,
    refinedFile: File?,
) {
    val count = steps.toIntOrNull()?.takeIf { it in 1..10_000 }
    val refining = run.stage == ReconstructionStage.BRUSH_REFINING

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer),
    ) {
        Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("2 · Refine on device", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = steps == "50", enabled = !busy,
                    onClick = { onStepsChange("50") }, label = { Text("Quick · 50") },
                )
                FilterChip(
                    selected = steps == "300", enabled = !busy,
                    onClick = { onStepsChange("300") }, label = { Text("Quality · 300") },
                )
            }
            OutlinedTextField(
                value = steps,
                onValueChange = { onStepsChange(it.filter(Char::isDigit).take(5)) },
                label = { Text("Steps (1–10,000)") },
                singleLine = true,
                enabled = !busy,
                isError = count == null,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth(),
            )
            Button(
                onClick = { count?.let(onRefine) },
                enabled = !busy && count != null,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(Icons.Filled.AutoFixHigh, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text(if (hasFinal) "Refine again" else "Refine locally with Brush")
            }

            if (refining && run.brushSteps > 0) {
                LinearProgressIndicator(
                    progress = { (run.completedSteps.toFloat() / run.brushSteps).coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    "${run.completedSteps} / ${run.brushSteps} steps · ${run.brushElapsedMs} ms",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                TextButton(onClick = onCancel) { Text("Cancel refinement") }
            }

            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = onExportInitial, enabled = !busy, modifier = Modifier.weight(1f)) {
                    Icon(Icons.Filled.SaveAlt, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Initial PLY")
                }
                OutlinedButton(
                    onClick = { refinedFile?.let(onExportFinal) },
                    enabled = !busy && refinedFile != null,
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(Icons.Filled.SaveAlt, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Refined PLY")
                }
            }
            TextButton(onClick = onExportReport, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                Text("Export performance report")
            }
        }
    }
}

@Composable
private fun ProjectsCard(
    projects: List<Pair<String, Int>>,
    projectTimes: List<Long>,
    projectIds: List<String>,
    activeId: String?,
    enabled: Boolean,
    onSelect: (Int) -> Unit,
    captures: List<String>,
    onPickCapture: (Int) -> Unit,
    onImportZip: () -> Unit,
) {
    // Long histories collapse to the newest few; expanding is local to the card so it
    // does not fight the screen's own scroll state.
    var showAllProjects by rememberSaveable { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
    ) {
        Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Projects", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            // Repeated imports of one ZIP share a name; show the import time and keep the
            // list short so the actions below stay reachable without scrolling a wall of rows.
            val shown = if (showAllProjects) projects.size else minOf(projects.size, MAX_LISTED_PROJECTS)
            projects.take(shown).forEachIndexed { index, (name, frameCount) ->
                val active = projectIds.getOrNull(index) == activeId
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(enabled = enabled && !active) { onSelect(index) }
                        .padding(vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            name,
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = if (active) FontWeight.SemiBold else FontWeight.Normal,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        Text(
                            buildString {
                                append("$frameCount frames")
                                projectTimes.getOrNull(index)?.let { millis ->
                                    append(" · ")
                                    append(
                                        SimpleDateFormat("MMM d, HH:mm", Locale.getDefault())
                                            .format(Date(millis))
                                    )
                                }
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    if (active) {
                        Text(
                            "Open",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
            }
            if (projects.size > MAX_LISTED_PROJECTS) {
                TextButton(onClick = { showAllProjects = !showAllProjects }) {
                    Text(
                        if (showAllProjects) "Show fewer"
                        else "Show all ${projects.size} projects"
                    )
                }
            }
            if (captures.isNotEmpty()) {
                HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                Text("Saved captures", style = MaterialTheme.typography.labelLarge)
                captures.forEachIndexed { index, name ->
                    TextButton(onClick = { onPickCapture(index) }, enabled = enabled) {
                        Text(name, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
            OutlinedButton(onClick = onImportZip, enabled = enabled, modifier = Modifier.fillMaxWidth()) {
                Text("Import another ZIP")
            }
        }
    }
}
