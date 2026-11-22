package io.github.sceneview.demo.locusgs
import org.junit.Test
import org.junit.Assert.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
class PocketGsInitTest {
    @Test fun planarPriorProducesThinDiscsWithoutMovingPoints() {
        val root = java.nio.file.Files.createTempDirectory("pocket-test").toFile()
        try {
            File(root, "points.ply").writeText(buildString {
                append("ply\nformat ascii 1.0\nelement vertex 25\n")
                for (p in listOf("x", "y", "z")) append("property float $p\n")
                for (p in listOf("red", "green", "blue")) append("property uchar $p\n")
                append("end_header\n")
                for (y in 0..4) for (x in 0..4) append("$x $y 0 255 128 0\n")
            })
            val scene = ReconstructionScene("test", "test", root, emptyList(), emptyList(), emptyList(), 1.0)
            val output = File(root, "init.ply")
            assertEquals(25, PocketGsInit.write(scene, output) {})
            val bytes = output.readBytes()
            val end = String(bytes, Charsets.ISO_8859_1).indexOf("end_header\n") + 11
            val b = ByteBuffer.wrap(bytes, end, bytes.size-end).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until 25) {
                assertEquals((i % 5).toFloat(), b.float, 0f)
                assertEquals((i / 5).toFloat(), b.float, 0f)
                assertEquals(0f, b.float, 0f)
                val sx=b.float; val sy=b.float; val sz=b.float
                assertEquals(sx, sy, 1e-6f)
                assertEquals(kotlin.math.ln(0.1f), sz-sx, 1e-6f)
                assertEquals(kotlin.math.ln(0.1f/0.9f), b.float, 1e-6f)
                assertEquals(1f, b.float, 1e-6f)
                repeat(3) { assertEquals(0f, b.float, 1e-6f) }
                repeat(3) { assertTrue(b.float.isFinite()) }
            }
        } finally { root.deleteRecursively() }
    }
}
