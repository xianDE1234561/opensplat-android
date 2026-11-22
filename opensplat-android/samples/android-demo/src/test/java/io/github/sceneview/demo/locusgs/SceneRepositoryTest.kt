package io.github.sceneview.demo.locusgs

import android.content.Context
import android.content.ContextWrapper
import android.graphics.Bitmap
import android.graphics.Color
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder.LITTLE_ENDIAN
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

@RunWith(RobolectricTestRunner::class)
@Config(manifest = Config.NONE, sdk = [34])
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class SceneRepositoryTest {
    @get:Rule val temp = TemporaryFolder()

    @Test
    fun `capture intrinsics work without global values or COLMAP and persist`() = runBlocking {
        val files = temp.newFolder()
        val names = listOf("a.png", "b.png", "c.png", "d.png", "e.png", "f.png")
        val entries = names.associate { "capture/images/$it" to png(it == "c.png" || it == "e.png") }
        val archive = zip(entries + ("capture/transforms.json" to transforms(names.reversed())))
        val scene = SceneRepository.importZip(context(files), "AR capture", archive.inputStream())

        assertEquals(File(files, "reconstruction/scenes/${scene.id}"), scene.root)
        assertEquals(names.reversed().map { "images/$it" }, scene.frames.map { it.id })
        assertEquals(1.0, scene.sceneScale, 0.0)
        assertEquals(80.0, scene.frames.first().fx, 0.0)
        assertEquals(85.0, scene.frames.last().fx, 0.0)
        assertEquals(32, scene.frames.first().width)
        assertEquals(listOf("images/e.png"), scene.validationIds)
        val contextIds = scene.selectContext().map { it.id }
        assertEquals("images/c.png", contextIds.first())
        assertEquals(4, contextIds.toSet().size)
        assertTrue(contextIds.all { it in scene.trainIds && it !in scene.validationIds })

        // A fresh Context with only the same filesDir can rediscover the import.
        val restored = SceneRepository.list(context(files)).single()
        assertEquals(scene.id, restored.id)
        assertEquals("AR capture", restored.name)
        assertEquals(scene.trainIds, restored.trainIds)
        assertEquals(scene.validationIds, restored.validationIds)
        assertEquals(contextIds, restored.selectContext().map { it.id })
        assertTrue(restored.frames.all { it.file == File(scene.root, it.id) && it.file.isFile })
    }

    @Test
    fun `reordered transforms join real COLMAP records by full path`() = runBlocking {
        val ctx = context(temp.newFolder())
        val paths = listOf("A/shared.png", "B/shared.png", "C view.png", "D.png", "E.png", "F.png")
        val flat = png()
        val centers = listOf(0, 2, 6, 7, 8, 12)
        val entries = paths.associate { "images/$it" to flat } + mapOf(
            "colmap/cameras.bin" to cameras(),
            "colmap/images.bin" to images(paths.zip(centers).reversed()),
            "scene_metadata.json" to """{"scene_scale":0.5}""".toByteArray(),
        )
        suspend fun importScene(order: List<String>) = SceneRepository.importZip(ctx, "Nerfstudio",
            zip(entries + ("transforms.json" to transforms(order, global = true))).inputStream())

        val scene = importScene(listOf(paths[2], paths[5], paths[0], paths[4], paths[1], paths[3]))
        val reordered = importScene(paths)
        assertEquals(0.5, scene.sceneScale, 0.0)
        assertEquals(105.0, scene.frames.first().fy, 0.0)
        assertEquals(scene.trainIds, reordered.trainIds)
        assertEquals(scene.validationIds, reordered.validationIds)
        val picks = scene.selectContext()
        // Equal-sharpness seed is A; COLMAP's farthest training center is F,
        // then C, then B. In transforms order the poses have different centers.
        assertEquals(listOf(paths[0], paths[5], paths[2], paths[1]).map { "images/$it" }, picks.map { it.id })
        assertEquals(picks.map { it.id }, reordered.selectContext().map { it.id })
        assertEquals(listOf(2.0, 1.0, 0.0, 4.0), picks.map { it.c2w[3] })
        assertTrue(picks.all { it.file == File(scene.root, it.id) })
    }

    @Test
    fun `COLMAP sparse archive uses exact names and the referenced cameras`() = runBlocking {
        val ctx = context(temp.newFolder())
        val names = listOf("room 1/照片.png", "room 2/frame.png")
        val archive = zip(names.associate { "wrapper/images/$it" to png() } + mapOf(
            "wrapper/sparse/0/cameras.bin" to cameras(),
            "wrapper/sparse/0/images.bin" to images(listOf(names[0] to 3, names[1] to 4)),
        ))
        val scene = SceneRepository.importZip(ctx, "COLMAP", archive.inputStream())
        val (simple, pinhole) = scene.frames
        assertEquals(names.map { "images/$it" }, scene.frames.map { it.id })
        assertEquals(140.0, simple.fx, 0.0)
        assertEquals(140.0, simple.fy, 0.0)
        assertEquals(160.0, pinhole.fx, 0.0)
        assertEquals(165.0, pinhole.fy, 0.0)
        assertEquals(3.0, simple.c2w[3], 1e-12)
        assertEquals(-1.0, simple.c2w[5], 1e-12)
        assertEquals(-1.0, simple.c2w[10], 1e-12)
        assertEquals(1.0, simple.c2w[15], 0.0)
        assertEquals(scene.trainIds.toSet(), scene.selectContext().map { it.id }.toSet())
    }

    @Test
    fun `invalid archives never publish a job`() = runBlocking {
        val ctx = context(temp.newFolder())
        val flat = png()
        val valid = mapOf("images/a.png" to flat, "transforms.json" to transforms(listOf("a.png")))
        val cases = listOf(
            zip(valid + ("../escape.png" to flat)),
            zip(valid + ("..\\escape.png" to flat)),
            zip(valid + ("./images/a.png" to flat)),
            zip(valid).dropLast(22).toByteArray(),
            zip(valid + ("transforms.json" to transforms(listOf("../../escape.png")))),
            zip(valid + ("transforms.json" to transforms(listOf("a.png", "./a.png")))),
            zip(valid + ("scene_metadata.json" to """{"scene_scale":0}""".toByteArray())),
            zip(valid + mapOf("init.ply" to flat,
                "transforms.json" to transforms(listOf("init.ply"), prefix = ""))),
            zip(valid + ("images/a.png" to "not an image".toByteArray())),
            zip(valid + mapOf("colmap/cameras.bin" to cameras(),
                "colmap/images.bin" to images(listOf("a.png" to 2, "images/a.png" to 7)))),
        )
        for ((i, archive) in cases.withIndex()) {
            try {
                SceneRepository.importZip(ctx, "invalid $i", archive.inputStream())
                fail("invalid archive $i accepted")
            } catch (_: Exception) {
                assertTrue(SceneRepository.list(ctx).isEmpty())
                assertTrue(File(ctx.filesDir, "reconstruction/scenes").listFiles().orEmpty().isEmpty())
                assertTrue(ctx.filesDir.walkTopDown().none { it.name == "escape.png" })
            }
        }
    }

    @Test
    fun `distorted COLMAP model is explicitly rejected`() = runBlocking {
        val ctx = context(temp.newFolder())
        val archive = zip(mapOf(
            "images/a.png" to png(),
            "sparse/0/cameras.bin" to cameras(firstModel = 2),
            "sparse/0/images.bin" to images(listOf("a.png" to 3)),
        ))
        try {
            SceneRepository.importZip(ctx, "distorted", archive.inputStream())
            fail("unsupported distortion accepted")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message.orEmpty().contains("Unsupported COLMAP camera model 2"))
        }
    }

    private fun context(directory: File): Context = object : ContextWrapper(RuntimeEnvironment.getApplication()) {
        override fun getFilesDir(): File = directory
    }

    private fun transforms(names: List<String>, global: Boolean = false, prefix: String = "./images/"): ByteArray = JSONObject().apply {
        put("camera_model", "PERSPECTIVE")
        if (global) {
            put("fl_x", 100.0); put("fl_y", 105.0); put("cx", 16.0); put("cy", 16.0)
            put("w", 32); put("h", 32); put("scene_scale", 0.25)
        }
        put("frames", JSONArray().apply {
            for ((i, name) in names.withIndex()) put(JSONObject().apply {
                put("file_path", if (name.startsWith("../")) name else "$prefix$name")
                if (!global) {
                    put("fl_x", 80.0 + i); put("fl_y", 70.0); put("cx", 16.0); put("cy", 15.0)
                    put("w", 32); put("h", 32)
                }
                put("transform_matrix", JSONArray().apply {
                    for (r in 0..3) put(JSONArray().apply {
                        for (c in 0..3) put(if (r == 0 && c == 3) i else if (r == c) 1 else 0)
                    })
                })
            })
        })
    }.toString().toByteArray()

    private fun cameras(firstModel: Int = 0): ByteArray = binary {
        putLong(2)
        putInt(215); putInt(firstModel); putLong(32); putLong(32)
        putDouble(140.0); putDouble(16.0); putDouble(15.0)
        putInt(17); putInt(1); putLong(32); putLong(32)
        putDouble(160.0); putDouble(165.0); putDouble(14.0); putDouble(13.0)
    }

    private fun images(records: List<Pair<String, Int>>): ByteArray = binary {
        putLong(records.size.toLong())
        for ((name, x) in records) {
            putInt(1_000 + x)
            putDouble(1.0); repeat(3) { putDouble(0.0) }
            putDouble(-x.toDouble()); putDouble(0.0); putDouble(0.0)
            putInt(if (x % 2 != 0) 215 else 17)
            put(name.toByteArray(Charsets.UTF_8)); put(0)
            putLong(1); putDouble(5.0); putDouble(6.0); putLong(-1)
        }
    }

    private fun binary(block: ByteBuffer.() -> Unit): ByteArray = ByteBuffer.allocate(4096).order(LITTLE_ENDIAN)
        .apply(block).let { it.array().copyOf(it.position()) }

    private fun png(checker: Boolean = false): ByteArray {
        val pixels = IntArray(32 * 32) { i ->
            if (checker && ((i / 32 / 2 + i % 32 / 2) % 2 == 0)) Color.WHITE else Color.BLACK
        }
        val bitmap = Bitmap.createBitmap(pixels, 32, 32, Bitmap.Config.ARGB_8888)
        return ByteArrayOutputStream().also { assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)) }
            .toByteArray().also { bitmap.recycle() }
    }

    private fun zip(entries: Map<String, ByteArray>): ByteArray = ByteArrayOutputStream().also { bytes ->
        ZipOutputStream(bytes).use { out ->
            for ((path, data) in entries) {
                out.putNextEntry(ZipEntry(path)); out.write(data); out.closeEntry()
            }
        }
    }.toByteArray()
}
