package io.github.sceneview.demo.locusgs

import android.content.ContextWrapper
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

@RunWith(RobolectricTestRunner::class)
@Config(manifest = Config.NONE, sdk = [34])
class LocusGSModelStoreTest {
    @get:Rule val files = TemporaryFolder()

    @Test
    fun incompleteZipDoesNotInstall() = reject("encoder.onnx", "missing: block_00.onnx")

    @Test
    fun nestedZipPathIsRejected() = reject("graphs/encoder.onnx", "flat LocusGS model package")

    @Test
    fun traversalEntryCannotEscapeStaging() = reject("../encoder.onnx", "flat LocusGS model package")

    @Test
    fun truncatedSeedIsRejected() = reject("anchor_init.f32", "expected 49152")

    private fun reject(entryName: String, expectedError: String) {
        val root = files.newFolder()
        val context = object : ContextWrapper(RuntimeEnvironment.getApplication()) {
            override fun getNoBackupFilesDir() = root
        }
        val bytes = ByteArrayOutputStream()
        ZipOutputStream(bytes).use { zip ->
            zip.putNextEntry(ZipEntry(entryName))
            zip.write(1)
            zip.closeEntry()
        }
        try {
            runBlocking { LocusGSModelStore.install(context, bytes.toByteArray().inputStream()) }
            fail("Expected the ZIP to be rejected")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message, e.message.orEmpty().contains(expectedError))
        }
        assertNull(LocusGSModelStore.installed(context))
        assertTrue(root.list().orEmpty().isEmpty())
    }
}
