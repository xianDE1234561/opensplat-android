package io.github.sceneview.demo.locusgs

import android.content.Context
import android.content.ContextWrapper
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** Uses real ORT sessions with tiny synthetic graphs, without shipping checkpoint weights. */
@RunWith(AndroidJUnit4::class)
class LocusGSRunnerTest {
    @get:Rule val files = TemporaryFolder()

    private companion object {
        /** The single shipped input width; every other aspect is cropped into it. */
        const val MODEL_W = 448L
    }

    private fun context() = object : ContextWrapper(ApplicationProvider.getApplicationContext<Context>()) {
        private val root = files.newFolder()
        override fun getNoBackupFilesDir() = root
    }

    @Test
    fun validPackageRunsAllGraphsAndCanBeRetriedAfterCancellation() {
        val context = context()
        runBlocking { modelZip().inputStream().use { LocusGSModelStore.install(context, it) } }
        val model = checkNotNull(LocusGSModelStore.installed(context))

        val runner = LocusGSRunner(context.assets, threads = 1).apply { setModelDir(model) }
        val images = FloatArray(4 * 3 * 256 * MODEL_W.toInt())
        val rays = FloatArray(4 * 6 * 256 * MODEL_W.toInt())
        val cancellation = CancellationException("stop between decoder blocks")
        val caller = Thread.currentThread()
        val stages = mutableListOf<String>()
        try {
            try {
                runner.forward(images, rays) { stage ->
                    assertSame(caller, Thread.currentThread())
                    stages += stage
                    if (stage == "block_1") throw cancellation
                }
                fail("Expected cancellation")
            } catch (e: CancellationException) {
                assertSame(cancellation, e)
            }
            assertEquals(listOf("encoder", "block_0", "block_1"), stages)
            stages.clear()
            val output = runner.forward(images, rays) { stages += it }
            assertEquals(listOf("encoder") + List(12) { "block_$it" } + "head", stages)
            assertEquals(stages + "total", output.timings.keys.toList())
            assertEquals(262_144, output.count)
            assertEquals(262_144 * 14, output.data.size)
            assertEquals(0f, output.data.last(), 0f)
        } finally {
            runner.release()
        }
        assertEquals(model, LocusGSModelStore.installed(context))
    }

    @Test
    fun secondInstallIsRejectedBecauseOnlyOneShapeIsShipped() {
        val context = context()
        runBlocking { modelZip().inputStream().use { LocusGSModelStore.install(context, it) } }
        val error = runCatching {
            runBlocking { modelZip().inputStream().use { LocusGSModelStore.install(context, it) } }
        }.exceptionOrNull()
        assertTrue(error is IllegalStateException && error.message.orEmpty().contains("already installed"))
        assertNotNull(LocusGSModelStore.installed(context))
    }

    @Test
    fun rejectsEncoderForAnotherAspectDespiteCompleteZip() {
        val context = context()
        try {
            runBlocking { modelZip(width = 256, decoderWidth = 256).inputStream().use { LocusGSModelStore.install(context, it) } }
            fail("Expected encoder shape mismatch")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message, e.message.orEmpty().contains("not the ${LocusGSModel.label} export"))
        }
        assertNull(LocusGSModelStore.installed(context))
        assertTrue(context.noBackupFilesDir.list().orEmpty().isEmpty())
    }

    @Test
    fun calibratedViewsUseEachBitmapAndKeepCallerBitmapsAlive() {
        // The square bitmap's full-image crop may return the source itself.
        val wide = Bitmap.createBitmap(8, 4, Bitmap.Config.ARGB_8888)
        wide.eraseColor(android.graphics.Color.RED)
        val square = Bitmap.createBitmap(4, 4, Bitmap.Config.ARGB_8888).let { bitmap ->
            try {
                bitmap.eraseColor(android.graphics.Color.BLACK)
                checkNotNull(bitmap.copy(Bitmap.Config.ARGB_8888, false))
            } finally {
                bitmap.recycle()
            }
        }
        try {
            val identity = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            val views = LocusGSPreprocess.prepareViews(
                listOf(wide, square), arrayOf(identity, identity),
                listOf(doubleArrayOf(16.0, 24.0, 3.0, 2.0), doubleArrayOf(4.0, 6.0, 2.0, 1.0)),
                targetW = 2, targetH = 2, sceneScale = 1.0,
            )
            assertFalse(wide.isRecycled)
            assertFalse(square.isRecycled)
            assertArrayEquals(doubleArrayOf(8.0, 12.0, 0.5, 1.0), views.intrinsicsPerView[0], 0.0)
            assertArrayEquals(doubleArrayOf(2.0, 3.0, 1.0, 0.5), views.intrinsicsPerView[1], 0.0)
            assertEquals(0f, views.plucker[3 * 4], 0f) // first pixel: first view looks straight in x
            assertTrue(views.plucker[6 * 4 + 3 * 4] < 0f) // second view has a different cx/fx
            assertEquals(2 * 3 * 2 * 2, views.images.size)
            assertEquals((1f - LocusGSPreprocess.MEAN[0]) / LocusGSPreprocess.STD[0], views.images[0], 0f)
            assertEquals(-LocusGSPreprocess.MEAN[0] / LocusGSPreprocess.STD[0], views.images[3 * 4], 0f)
        } finally {
            wide.recycle()
            square.recycle()
        }
    }

    private fun modelZip(width: Long = MODEL_W, decoderWidth: Long = width): File {
        val file = files.newFile()
        ZipOutputStream(file.outputStream().buffered()).use { zip ->
            fun keys(w: Long) = longArrayOf(1, 16, 4 * 32 * (w / 8), 64)
            fun rays(w: Long) = longArrayOf(1, 4 * 32 * (w / 8), 6)
            val state = linkedMapOf(
                "tokens" to longArrayOf(1, 4096, 1024),
                "anchor_raw" to longArrayOf(1, 4096, 3),
                "anchor_radius_raw" to longArrayOf(1, 4096, 1),
            )
            fun write(name: String, model: ByteArray) {
                zip.putNextEntry(ZipEntry(name))
                zip.write(model)
                zip.closeEntry()
            }
            write("encoder.onnx", model(
                mapOf("images" to longArrayOf(1, 4, 3, 256, width), "plucker" to longArrayOf(1, 4, 6, 256, width)),
                listOf(keys(width), keys(width), rays(width)),
            ))
            val block = model(state + mapOf("keys" to keys(decoderWidth), "values" to keys(decoderWidth),
                "patch_rays" to rays(decoderWidth)), state.values.toList(), sources = state.keys.toList())
            repeat(12) { write("block_${it.toString().padStart(2, '0')}.onnx", block) }
            write("head.onnx", model(state, listOf(longArrayOf(1, 262_144, 14))))
            val zeroes = ByteArray(16 * 1024)
            for ((name, bytes) in mapOf("gs_token_init.f32" to 16 * 1024 * 1024,
                "anchor_init.f32" to 48 * 1024, "anchor_radius_raw_init.f32" to 16 * 1024)) {
                zip.putNextEntry(ZipEntry(name))
                repeat(bytes / zeroes.size) { zip.write(zeroes) }
                zip.closeEntry()
            }
        }
        return file
    }

    // Minimal ONNX protobuf writer used only here. ConstantOfShape produces the large
    // test tensors on demand; decoder Add uses input state so chaining is exercised.
    private fun model(inputs: Map<String, LongArray>, outputs: List<LongArray>, sources: List<String>? = null) =
        message {
            number(1, 8) // ModelProto.ir_version
            msg(8) { number(2, 17) } // OperatorSetIdProto.version
            msg(7) { // GraphProto
                text(2, "fixture")
                for ((name, shape) in inputs) valueInfo(11, name, shape)
                if (sources != null) msg(5) { // TensorProto: scalar float32 zero
                    number(2, 1)
                    text(8, "zero")
                    bytes(9, ByteArray(4))
                }
                outputs.forEachIndexed { i, shape ->
                    val name = "out_$i"
                    valueInfo(12, name, shape)
                    msg(1) { // NodeProto
                        if (sources == null) {
                            text(1, "shape_$i")
                            text(4, "ConstantOfShape")
                        } else {
                            text(1, sources[i])
                            text(1, "zero")
                            text(4, "Add")
                        }
                        text(2, name)
                    }
                    if (sources == null) msg(5) { // TensorProto: int64 shape initializer
                        number(1, shape.size.toLong())
                        number(2, 7)
                        shape.forEach { number(7, it) }
                        text(8, "shape_$i")
                    }
                }
            }
        }

    private fun message(body: Proto.() -> Unit) = Proto().apply(body).toByteArray()

    private class Proto : ByteArrayOutputStream() {
        fun number(field: Int, value: Long) {
            varint((field shl 3).toLong())
            varint(value)
        }

        fun text(field: Int, value: String) = bytes(field, value.encodeToByteArray())
        fun msg(field: Int, body: Proto.() -> Unit) = bytes(field, Proto().apply(body).toByteArray())

        fun valueInfo(field: Int, name: String, shape: LongArray) = msg(field) {
            text(1, name)
            msg(2) { // TypeProto
                msg(1) { // TypeProto.Tensor
                    number(1, 1) // float32
                    msg(2) { shape.forEach { dim -> msg(1) { number(1, dim) } } }
                }
            }
        }

        fun bytes(field: Int, value: ByteArray) {
            varint(((field shl 3) or 2).toLong())
            varint(value.size.toLong())
            write(value)
        }

        private fun varint(value: Long) {
            var n = value
            while (n > 0x7f) {
                write(((n and 0x7f) or 0x80).toInt())
                n = n ushr 7
            }
            write(n.toInt())
        }
    }
}
