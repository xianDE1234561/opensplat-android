package io.github.sceneview.demo.locusgs

import android.content.Context
import kotlinx.coroutines.runBlocking
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

/** Only replaces initialization; original calibration, train and holdout bytes survive. */
internal object ZipTrainingInit {
    fun prepare(context: Context, source: File, initial: File, output: File): Int {
        val scene = runBlocking { SceneRepository.importZip(context, source.name, source.inputStream()) }
        try {
            val count = PocketGsInit.write(scene, initial) { check(!Thread.currentThread().isInterrupted) }
            ZipFile(source).use { input ->
                ZipOutputStream(output.outputStream().buffered()).use { zip ->
                    zip.setLevel(0)
                    for (entry in input.entries()) {
                        if (entry.name == "init.ply") continue
                        zip.putNextEntry(ZipEntry(entry.name))
                        if (!entry.isDirectory) input.getInputStream(entry).use { it.copyTo(zip) }
                        zip.closeEntry()
                    }
                    zip.putNextEntry(ZipEntry("init.ply"))
                    initial.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
            }
            return count
        } finally { scene.root.deleteRecursively() }
    }
}
