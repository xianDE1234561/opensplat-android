package io.github.sceneview.demo.state

import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.ServiceConnection
import android.net.Uri
import android.os.IBinder
import android.os.SystemClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import io.github.sceneview.demo.locusgs.BrushLaunch
import io.github.sceneview.demo.locusgs.ContextImageTransform
import io.github.sceneview.demo.locusgs.GaussianBridgeAndroid
import io.github.sceneview.demo.locusgs.InitializationMethod
import io.github.sceneview.demo.locusgs.PocketGsInit
import io.github.sceneview.demo.locusgs.LocusGSModelStore
import io.github.sceneview.demo.locusgs.LocusGSPreprocess
import io.github.sceneview.demo.locusgs.LocusGSModel
import io.github.sceneview.demo.locusgs.LocusGSRunner
import io.github.sceneview.demo.locusgs.ReconstructionRun
import io.github.sceneview.demo.locusgs.ReconstructionRunStore
import io.github.sceneview.demo.locusgs.ReconstructionScene
import io.github.sceneview.demo.locusgs.ReconstructionStage
import io.github.sceneview.demo.locusgs.SceneRepository
import io.github.sceneview.demo.locusgs.selectContext
import io.github.sceneview.demo.service.SplatTrainingService
import io.github.sceneview.demo.storage.CaptureProject
import io.github.sceneview.demo.storage.CaptureStorageManager
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.InputStream
import java.util.UUID

/** Retains the selected project and running task across navigation and configuration changes. */
internal object LocusGSStateHolder {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var worker: Job? = null
    private var training: Training? = null
    private var receiverRegistered = false
    var projects by mutableStateOf<List<ReconstructionScene>>(emptyList()); private set
    var captures by mutableStateOf<List<CaptureProject>>(emptyList()); private set
    var scene by mutableStateOf<ReconstructionScene?>(null); private set
    /** The installed model file, or null when the 448×256 package has not been imported yet. */
    var model by mutableStateOf<File?>(null); private set
    var run by mutableStateOf(ReconstructionRun()); private set
    var operation by mutableStateOf<String?>(null); private set
    var message by mutableStateOf<String?>(null); private set
    var graph by mutableStateOf<String?>(null); private set
    private var pendingCapture: CaptureProject? = null
    val busy: Boolean get() = operation != null || run.stage.active

    private class Training(val id: String, val bundle: File, val export: File) {
        var connection: ServiceConnection? = null
        var watchdog: Job? = null
    }

    fun openCaptureForReconstruction(capture: CaptureProject) {
        pendingCapture = capture
    }

    fun consumeCapture(context: Context) {
        val capture = pendingCapture ?: return
        if (busy) return
        pendingCapture = null
        importCapture(context, capture)
    }

    fun refresh(context: Context) {
        if (busy) return
        val app = context.applicationContext
        scope.launch {
            val (saved, captured, installed) = withContext(Dispatchers.IO) {
                Triple(SceneRepository.list(app), CaptureStorageManager.getCaptureProjects(app),
                    LocusGSModelStore.installed(app))
            }
            // A new import can commit while the disk scan is in flight.
            val current = scene
            projects = if (current != null && saved.none { it.id == current.id }) listOf(current) + saved else saved
            captures = captured
            if (installed != null) model = installed
        }
    }

    fun importZip(context: Context, name: String, uri: Uri) = importScene(context, name) { app ->
        app.contentResolver.openInputStream(uri)
    }

    fun importCapture(context: Context, capture: CaptureProject) = importScene(context, capture.name) { app ->
        CaptureStorageManager.openInputStream(app, capture)
    }

    private fun importScene(context: Context, name: String, open: (Context) -> InputStream?) {
        val app = context.applicationContext
        startWork("Importing scene") {
            val imported = withContext(Dispatchers.IO) {
                requireNotNull(open(app)) { "Unable to open scene archive" }.use {
                    SceneRepository.importZip(app, name, it)
                }
            }
            projects = listOf(imported) + projects.filter { it.id != imported.id }
            openProject(imported)
        }
    }

    fun installModel(context: Context, uri: Uri) {
        val app = context.applicationContext
        startWork("Installing LocusGS model") {
            model = withContext(Dispatchers.IO) {
                requireNotNull(app.contentResolver.openInputStream(uri)) { "Unable to open model package" }.use {
                    LocusGSModelStore.install(app, it)
                }
            }
        }
    }

    fun selectProject(project: ReconstructionScene) {
        if (!busy) openProject(project)
    }

    private fun openProject(project: ReconstructionScene) {
        val restored = try {
            ReconstructionRunStore.load(project.root).afterProcessRestart()
        } catch (e: Exception) {
            message = "Cannot read reconstruction history: ${e.message}"
            ReconstructionRun(stage = ReconstructionStage.FAILED, error = e.message)
        }
        scene = project
        publish(restored)
    }

    fun forward(context: Context) {
        val project = scene ?: return
        val app = context.applicationContext
        val model = model ?: return
        startWork("Reconstructing") {
            val parent = coroutineContext
            val start = SystemClock.elapsedRealtime()
            publish(ReconstructionRun(stage = run.stage).advance(ReconstructionStage.SELECTING_VIEWS))
            val views = withContext(Dispatchers.IO) { project.selectContext() }
            val ids = views.map { it.id }
            require(ids.size == LocusGSRunner.VIEWS && ids.toSet().size == ids.size && ids.all { it in project.trainIds })
            publish(run.advance(ReconstructionStage.PREPROCESSING).copy(
                contextIds = ids, trainingIds = project.trainIds, validationIds = project.validationIds,
                selectionMs = SystemClock.elapsedRealtime() - start,
            ))
            val prepStart = SystemClock.elapsedRealtime()
            // Crop geometry per view is known from the calibration, so the report can
            // describe the transform before the pixels are decoded.
            val imageTransforms = views.map { frame ->
                val g = LocusGSPreprocess.cropGeometry(frame.width, frame.height, LocusGSModel.WIDTH, LocusGSModel.HEIGHT)
                ContextImageTransform(frame.id, frame.width, frame.height, LocusGSModel.WIDTH, LocusGSModel.HEIGHT, g[0], g[1], g[2], g[3])
            }
            // Same transform used by normalizeCameras(): inverse(first OpenCV c2w).
            val firstCv = views.first().c2w.copyOf()
            for (r in 0..3) for (c in 1..2) firstCv[4 * r + c] *= -1.0
            val modelFromWorld = LocusGSPreprocess.invert4x4(firstCv)
            publish(run.advance(ReconstructionStage.LOCUSGS_INFERENCE).copy(
                sourceWorldToFirstCamera = modelFromWorld.toList(), translationScale = project.sceneScale,
                imageTransforms = imageTransforms, preprocessingMs = SystemClock.elapsedRealtime() - prepStart,
                inferenceExecutionProvider = "CPUExecutionProvider",
            ))
            val initName = "init-${UUID.randomUUID()}.ply"
            val actual = File(project.root, initName)
            val temp = File(project.root, "$initName.part")
            try {
                val (out, peak) = withContext(Dispatchers.Default) {
                    val runner = LocusGSRunner(app.assets).apply {
                        setCacheDir(app.cacheDir)
                        setModelDir(model)
                    }
                    val output = try {
                        runner.prepare()
                        runner.forwardFromFiles(
                            files = views.map { it.file },
                            c2w = views.map { it.c2w }.toTypedArray(),
                            intrinsics = views.map { doubleArrayOf(it.fx, it.fy, it.cx, it.cy) },
                            sceneScale = project.sceneScale,
                        ) { stage ->
                            parent.ensureActive()
                            scope.launch { if (run.stage == ReconstructionStage.LOCUSGS_INFERENCE) graph = stage }
                        }
                    } finally {
                        runner.release()
                    }
                    ensureActive()
                    require(output.data.all { it.isFinite() }) { "LocusGS produced non-finite Gaussians" }
                    GaussianBridgeAndroid.writePly(output.data, output.count, temp)
                    require(temp.renameTo(actual)) { "Could not publish initial PLY" }
                    output to readPeakRss()
                }
                publish(run.advance(ReconstructionStage.GAUSSIAN_READY).copy(
                    initialPly = initName, gaussianCount = out.count, forwardMs = out.timings.getValue("total"),
                    graphTimingsMs = out.timings, totalToInitMs = SystemClock.elapsedRealtime() - start,
                    peakProcessRssKiB = peak,
                ))
            } finally {
                temp.delete()
                if (run.initialPly != initName) actual.delete()
            }
        }
    }

    fun initializeDirectly() {
        val project = scene ?: return
        startWork("Seeding surface-aligned Gaussians") {
            publish(run.advance(ReconstructionStage.POCKETGS_INITIALIZING))
            val start = SystemClock.elapsedRealtime()
            val name = "init-pocketgs-${UUID.randomUUID()}.ply"
            val output = File(project.root, name)
            val temporary = File(project.root, "$name.part")
            try {
                val count = withContext(Dispatchers.Default) {
                    PocketGsInit.write(project, temporary) { ensureActive() }.also {
                        ensureActive()
                        check(temporary.renameTo(output)) { "Could not publish initial PLY" }
                    }
                }
                publish(ReconstructionRun(
                    stage = run.advance(ReconstructionStage.GAUSSIAN_READY).stage,
                    initialization = InitializationMethod.POCKETGS,
                    trainingIds = project.trainIds,
                    validationIds = project.validationIds,
                    gaussianCount = count,
                    initialPly = name,
                    totalToInitMs = SystemClock.elapsedRealtime() - start,
                    peakProcessRssKiB = readPeakRss(),
                ))
            } finally {
                temporary.delete()
                if (run.initialPly != name) output.delete()
            }
        }
    }

    fun refine(context: Context, steps: Int) {
        val app = context.applicationContext
        val project = scene ?: return
        val initial = artifact(run.initialPly) ?: return
        if (SplatTrainingStateHolder.isTraining) { message = "Stop the other local training job first."; return }
        require(steps in 1..10_000)
        startWork("Preparing Brush dataset") {
            val job = UUID.randomUUID().toString()
            val bundle = File(app.cacheDir, "brush-$job.zip")
            val export = File(app.filesDir, "refined-$job.ply")
            val snapshot = run
            try {
                withContext(Dispatchers.IO) { BrushLaunch.writeBundle(project, snapshot, initial, bundle) { ensureActive() } }
                if (SplatTrainingStateHolder.isTraining) error("Another local training job is running")
                registerTrainingReceiver(app)
                publish(run.advance(ReconstructionStage.BRUSH_REFINING).copy(
                    brushBackendRequested = "Vulkan (local wgpu default)",
                    brushSteps = steps, completedSteps = 0, brushElapsedMs = 0, psnr = null, ssim = null, finalPly = null,
                ))
                val active = Training(job, bundle, export)
                training = active
                SplatTrainingService.start(app, steps, project.name, bundle.absolutePath, export.name, job, snapshot.gaussianCount)
                monitorTraining(app, active)
            } catch (e: Exception) {
                if (training?.id == job) {
                    finishTraining(app, ReconstructionStage.FAILED, e.message ?: "Failed to start Brush")
                }
                throw e
            } finally {
                if (training?.id != job) bundle.delete()
            }
        }
    }

    private fun monitorTraining(app: Context, active: Training) {
        val connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, service: IBinder) = Unit
            override fun onServiceDisconnected(name: ComponentName) {
                scope.launch {
                    // Completion is broadcast before stopSelf; let the main process consume it first.
                    delay(750)
                    if (training === active) finishTraining(app, ReconstructionStage.FAILED, "Native Brush process exited. See logcat.")
                }
            }
            override fun onBindingDied(name: ComponentName) = onServiceDisconnected(name)
        }
        if (app.bindService(Intent(app, SplatTrainingService::class.java), connection, 0)) active.connection = connection
        resetWatchdog(app, active)
    }

    private fun resetWatchdog(app: Context, active: Training) {
        active.watchdog?.cancel()
        active.watchdog = scope.launch {
            delay(300_000)
            if (training === active) finishTraining(app, ReconstructionStage.FAILED, "No progress from local Brush for five minutes.")
        }
    }

    private fun registerTrainingReceiver(app: Context) {
        if (receiverRegistered) return
        val filter = IntentFilter().apply {
            addAction(SplatTrainingService.BROADCAST_TRAINING_PROGRESS)
            addAction(SplatTrainingService.BROADCAST_TRAINING_EVAL)
            addAction(SplatTrainingService.BROADCAST_TRAINING_COMPLETE)
            addAction(SplatTrainingService.BROADCAST_TRAINING_ERROR)
        }
        ContextCompat.registerReceiver(app, object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val active = training ?: return
                if (intent.getStringExtra(SplatTrainingService.EXTRA_SPLAT_ID) != active.id) return
                when (intent.action) {
                    SplatTrainingService.BROADCAST_TRAINING_PROGRESS -> {
                        run = run.copy(completedSteps = intent.getIntExtra("iteration", 0),
                            brushElapsedMs = intent.getLongExtra("elapsed_ms", 0))
                        resetWatchdog(app, active)
                    }
                    SplatTrainingService.BROADCAST_TRAINING_EVAL -> run = run.copy(
                        psnr = intent.getFloatExtra("psnr", Float.NaN).takeIf(Float::isFinite),
                        ssim = intent.getFloatExtra("ssim", Float.NaN).takeIf(Float::isFinite),
                    )
                    SplatTrainingService.BROADCAST_TRAINING_ERROR -> finishTraining(app, ReconstructionStage.FAILED,
                        intent.getStringExtra("error_message") ?: "Native training failed")
                    SplatTrainingService.BROADCAST_TRAINING_COMPLETE -> {
                        if (run.completedSteps == run.brushSteps && active.export.isFile && active.export.length() > 0) {
                            val target = File(requireNotNull(scene).root, active.export.name)
                            if (active.export.renameTo(target)) {
                                run = run.copy(finalPly = target.name)
                                finishTraining(app, ReconstructionStage.COMPLETED)
                            } else finishTraining(app, ReconstructionStage.FAILED, "Cannot save trained PLY")
                        } else finishTraining(app, ReconstructionStage.FAILED, "Brush returned without confirmed steps and a final export")
                    }
                }
            }
        }, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
        receiverRegistered = true
    }

    private fun finishTraining(app: Context, stage: ReconstructionStage, error: String? = null) {
        val active = training ?: return
        training = null
        active.watchdog?.cancel()
        active.connection?.let(app::unbindService)
        if (stage != ReconstructionStage.COMPLETED) SplatTrainingService.stop(app)
        active.bundle.delete()
        active.export.delete()
        publish(run.advance(stage, error))
    }

    fun cancel(context: Context) {
        if (training != null) finishTraining(context.applicationContext, ReconstructionStage.CANCELLED)
        else worker?.cancel()
    }

    fun artifact(name: String?): File? = name?.takeIf { File(it).name == it }?.let { n ->
        scene?.let { File(it.root, n) }?.takeIf { it.isFile && it.length() > 0 }
    }

    fun export(context: Context, file: File, destination: Uri) {
        val app = context.applicationContext
        startWork("Exporting ${file.name}") {
            withContext(Dispatchers.IO) {
                requireNotNull(app.contentResolver.openOutputStream(destination)) { "Cannot open export destination" }.use { output ->
                    file.inputStream().use { it.copyTo(output) }
                }
            }
            message = "Saved ${file.name}"
        }
    }

    private fun startWork(label: String, work: suspend CoroutineScope.() -> Unit) {
        if (busy) return
        operation = label
        message = null
        worker = scope.launch {
            try { work() }
            catch (e: CancellationException) {
                withContext(NonCancellable) {
                    if (run.stage.active) publish(run.advance(ReconstructionStage.CANCELLED))
                    message = "Cancelled"
                }
            } catch (e: Exception) {
                val detail = e.message ?: e.javaClass.simpleName
                if (run.stage.active) publish(run.advance(ReconstructionStage.FAILED, detail))
                message = detail
            } finally {
                graph = null
                operation = null
            }
        }
    }

    private fun publish(snapshot: ReconstructionRun) {
        run = snapshot
        val project = scene ?: return
        try { ReconstructionRunStore.save(project.root, snapshot) }
        catch (e: Exception) { message = "Cannot persist project status: ${e.message}" }
    }

    private fun readPeakRss(): Long? = runCatching {
        File("/proc/self/status").useLines { lines ->
            lines.firstOrNull { it.startsWith("VmHWM:") }?.substringAfter(':')?.trim()?.substringBefore(' ')?.toLongOrNull()
        }
    }.getOrNull()
}
