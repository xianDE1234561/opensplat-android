package io.github.sceneview.demo.locusgs

import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardOpenOption.CREATE_NEW
import java.nio.file.StandardOpenOption.WRITE
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sqrt

/** PocketGS v6's initialization operator I, using the scene's existing colored SfM prior. */
internal object PocketGsInit {
    private const val NORMAL_NEIGHBORS = 16
    private const val SCALE_NEIGHBORS = 3
    // v6 specifies r_normal but not a numerical value. A tenth of the tangent scale
    // gives a thin disc while retaining nonzero normal thickness.
    private const val NORMAL_SCALE_RATIO = 0.1
    private const val INITIAL_OPACITY = 0.1
    // Bound mobile memory without subsampling and changing the KNN neighborhoods.
    private const val MAX_POINTS = 1_000_000
    private const val MAX_FILE_BYTES = 512L * 1024 * 1024
    private const val MAX_HEADER_BYTES = 64 * 1024
    private val shDc = FloatArray(256) { (it / 255f - 0.5f) / GaussianBridgeAndroid.SH_C0 }
    private const val CHECKPOINT_MASK = 1023

    /** Writes a new Brush PLY in the original scene world. The caller owns publication/cleanup. */
    fun write(scene: ReconstructionScene, output: File, checkCancelled: () -> Unit): Int {
        checkCancelled()
        val cloud = readPrior(scene, checkCancelled)
        val neighbors = Neighbors(cloud.xyz, cloud.colors.size, checkCancelled)
        val rotation = SurfaceRotation(cloud.xyz)
        val count = cloud.colors.size
        val header = buildString {
            append("ply\nformat binary_little_endian 1.0\nelement vertex $count\n")
            for (field in listOf("x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity",
                "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")) {
                append("property float $field\n")
            }
            append("end_header\n")
        }
        val opacity = ln(INITIAL_OPACITY / (1.0 - INITIAL_OPACITY)).toFloat()
        val normalLogRatio = ln(NORMAL_SCALE_RATIO)
        val minTangent = java.lang.Float.MIN_NORMAL / NORMAL_SCALE_RATIO
        // Leave room for rounding the stored float32 logarithm before exponentiation.
        val maxTangent = Float.MAX_VALUE.toDouble() * (1.0 - 1e-5)
        val record = ByteBuffer.allocate(14 * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        checkCancelled()
        Files.newOutputStream(output.toPath(), CREATE_NEW, WRITE).buffered().use { out ->
            out.write(header.toByteArray(Charsets.US_ASCII))
            for (i in 0 until count) {
                if ((i and CHECKPOINT_MASK) == 0) checkCancelled()
                neighbors.find(i)
                var tangent = 0.0
                for (k in 0 until SCALE_NEIGHBORS) tangent += sqrt(neighbors.distances[k])
                tangent /= SCALE_NEIGHBORS
                // Keep decoded scales finite and positive at the limits of float32,
                // including a zero mean when the three nearest samples coincide.
                val logTangent = ln(tangent.coerceIn(minTangent, maxTangent))
                record.clear()
                for (axis in 0..2) record.putFloat(cloud.xyz[3 * i + axis].toFloat())
                record.putFloat(logTangent.toFloat())
                record.putFloat(logTangent.toFloat())
                record.putFloat((logTangent + normalLogRatio).toFloat())
                record.putFloat(opacity)
                rotation.put(i, neighbors.indices, record)
                val color = cloud.colors[i]
                record.putFloat(shDc[color ushr 16])
                record.putFloat(shDc[(color ushr 8) and 255])
                record.putFloat(shDc[color and 255])
                out.write(record.array())
            }
            checkCancelled()
        }
        return count
    }

    private fun readPrior(scene: ReconstructionScene, checkpoint: () -> Unit): Cloud {
        val root = scene.root.canonicalFile
        val transforms = File(root, "transforms.json")
        val doc = if (transforms.exists()) readSceneJson(checkedFile(root, transforms)) else null
        if (doc?.has("ply_file_path") == true) {
            val path = scenePath(doc.get("ply_file_path") as? String
                ?: throw IllegalArgumentException("ply_file_path must be a scene-relative path"))
            return readPly(checkedFile(root, File(root, path)), checkpoint)
        }
        val standardPly = File(root, "points.ply")
        if (standardPly.exists()) return readPly(checkedFile(root, standardPly), checkpoint)
        val missingPrior = "PocketGS needs colored SfM points: ply_file_path, points.ply, or COLMAP points3D.bin"
        val model = requireNotNull(findColmapModel(root)) { missingPrior }
        val points = File(model, "points3D.bin")
        require(points.exists()) { missingPrior }
        val file = checkedFile(root, points)
        if (doc != null) {
            // COLMAP inversion changes only the camera axes. A transforms.json may also
            // change the world (e.g. Nerfstudio's applied_transform), so confirm that raw
            // COLMAP points are already in the scene world before using them.
            checkedFile(root, File(model, "cameras.bin"))
            checkedFile(root, File(model, "images.bin"))
            val colmapPoses = colmapPosesByFrame(scene.frames, readColmap(model, checkpoint))
            require(scene.frames.isNotEmpty() && scene.frames.all { frame ->
                val pose = colmapPoses.getValue(frame.id)
                (0 until 12).all { k ->
                    abs(frame.c2w[k] - pose[k]) <= 1e-6 * maxOf(1.0, abs(frame.c2w[k]), abs(pose[k]))
                }
            }) { "COLMAP points are not in the scene world; supply an aligned colored PLY" }
        }
        return readColmapPoints(file, checkpoint)
    }

    private fun checkedFile(root: File, file: File): File {
        val canonical = file.canonicalFile
        require(canonical.toPath().startsWith(root.toPath()) && canonical.isFile) {
            "Missing or unsafe scene file: ${file.name}"
        }
        require(canonical.length() in 1..MAX_FILE_BYTES) { "Invalid scene file size: ${file.name}" }
        return canonical
    }

    private class Cloud(count: Int) {
        val xyz = DoubleArray(3 * count)
        val colors = IntArray(count)

        fun put(i: Int, x: Double, y: Double, z: Double, r: Int, g: Int, b: Int) {
            val bound = Float.MAX_VALUE.toDouble()
            require(x >= -bound && x <= bound && y >= -bound && y <= bound && z >= -bound && z <= bound) {
                "Invalid position in SfM point ${i + 1}"
            }
            xyz[3 * i] = x
            xyz[3 * i + 1] = y
            xyz[3 * i + 2] = z
            colors[i] = (r shl 16) or (g shl 8) or b
        }
    }

    private fun pointCount(count: Long): Int {
        require(count > NORMAL_NEIGHBORS) { "PocketGS requires at least 17 colored SfM points; found $count" }
        require(count <= MAX_POINTS) { "SfM prior exceeds PocketGS's $MAX_POINTS-point limit" }
        return count.toInt()
    }

    private fun readColmapPoints(file: File, checkpoint: () -> Unit): Cloud =
        DataInputStream(file.inputStream().buffered()).use { input ->
            var remaining = file.length() - 8L
            val count = pointCount(input.readLongLE())
            // ID, XYZ, RGB, error and track length occupy 51 bytes; each track pair is 8 more.
            require(count * 51L <= remaining) { "Truncated COLMAP points3D.bin" }
            val cloud = Cloud(count)
            repeat(count) { i ->
                if ((i and CHECKPOINT_MASK) == 0) checkpoint()
                input.readLongLE() // Point ID
                val x = Double.fromBits(input.readLongLE())
                val y = Double.fromBits(input.readLongLE())
                val z = Double.fromBits(input.readLongLE())
                cloud.put(i, x, y, z, input.readUnsignedByte(), input.readUnsignedByte(), input.readUnsignedByte())
                input.readLongLE() // Reprojection error
                val trackLength = input.readLongLE()
                remaining -= 51
                require(trackLength in 0..remaining / 8) { "Invalid COLMAP track length at point ${i + 1}" }
                val trackBytes = trackLength * 8
                remaining -= trackBytes
                var unread = trackBytes
                while (unread > 0) {
                    if (unread > 1024 * 1024) checkpoint()
                    val skipped = input.skip(minOf(unread, 1024L * 1024))
                    require(skipped > 0) { "Truncated COLMAP track at point ${i + 1}" }
                    unread -= skipped
                }
            }
            require(remaining == 0L && input.read() == -1) { "Trailing data in COLMAP points3D.bin" }
            cloud
        }

    private fun DataInputStream.readLongLE(): Long = java.lang.Long.reverseBytes(readLong())

    private enum class Scalar(val bytes: Int) {
        I8(1), U8(1), I16(2), U16(2), I32(4), U32(4), F32(4), F64(8);

        fun read(buffer: ByteBuffer, offset: Int): Double = when (this) {
            I8 -> buffer.get(offset).toDouble()
            U8 -> (buffer.get(offset).toInt() and 0xff).toDouble()
            I16 -> buffer.getShort(offset).toDouble()
            U16 -> (buffer.getShort(offset).toInt() and 0xffff).toDouble()
            I32 -> buffer.getInt(offset).toDouble()
            U32 -> (buffer.getInt(offset).toLong() and 0xffffffffL).toDouble()
            F32 -> buffer.getFloat(offset).toDouble()
            F64 -> buffer.getDouble(offset)
        }
    }

    private fun scalar(name: String): Scalar = when (name) {
        "char", "int8" -> Scalar.I8
        "uchar", "uint8" -> Scalar.U8
        "short", "int16" -> Scalar.I16
        "ushort", "uint16" -> Scalar.U16
        "int", "int32" -> Scalar.I32
        "uint", "uint32" -> Scalar.U32
        "float", "float32" -> Scalar.F32
        "double", "float64" -> Scalar.F64
        else -> throw IllegalArgumentException("Unsupported PLY scalar type: $name")
    }

    private class Field(val type: Scalar, val offset: Int, val column: Int)
    private val plyComponents = listOf("x", "y", "z", "red", "green", "blue")
    private val whitespace = Regex("\\s+")

    private class PlyHeader(
        val count: Int, val order: ByteOrder?, val stride: Int, val columns: Int, val fields: List<Field>,
    )

    private fun readPlyHeader(input: DataInputStream): PlyHeader {
        val buffer = ByteArray(4096)
        var bytes = 0
        fun line(): List<String> {
            var size = 0
            while (true) {
                val char = input.read()
                require(char >= 0 && ++bytes <= MAX_HEADER_BYTES) { "Missing or oversized PLY header" }
                if (char == '\n'.code) break
                require(size < buffer.size) { "Oversized PLY header line" }
                buffer[size++] = char.toByte()
            }
            return String(buffer, 0, size, Charsets.US_ASCII).trim().split(whitespace)
        }
        require(line() == listOf("ply")) { "Not a PLY point cloud" }
        val format = line()
        require(format.size == 3 && format[0] == "format" && format[2] == "1.0") { "Invalid PLY format line" }
        val order = when (format[1]) {
            "ascii" -> null
            "binary_little_endian" -> ByteOrder.LITTLE_ENDIAN
            "binary_big_endian" -> ByteOrder.BIG_ENDIAN
            else -> throw IllegalArgumentException("Unsupported PLY format: ${format[1]}")
        }
        var count = -1
        var inVertex = false
        var earlierData = false
        var stride = 0
        var columns = 0
        val fields = arrayOfNulls<Field>(plyComponents.size)
        header@ while (true) {
            val words = line()
            when (words[0]) {
                "end_header" -> break@header
                "element" -> {
                    val size = words.getOrNull(2)?.toLongOrNull()
                    require(words.size == 3 && size != null && size >= 0L) { "Invalid PLY element" }
                    inVertex = words[1] == "vertex"
                    if (inVertex) {
                        require(count == -1 && !earlierData) { "PLY vertices must be its first data element" }
                        count = pointCount(size)
                    } else if (count == -1 && size > 0L) {
                        earlierData = true
                    }
                }
                "property" -> if (inVertex) {
                    require(words.size == 3) { "PLY vertex properties must be scalars" }
                    val type = scalar(words[1])
                    val component = plyComponents.indexOf(words[2])
                    if (component >= 0) {
                        require(fields[component] == null) { "Duplicate PLY vertex property: ${words[2]}" }
                        require(component < 3 || type == Scalar.U8) { "PLY RGB properties must be uchar/uint8" }
                        fields[component] = Field(type, stride, columns)
                    }
                    stride += type.bytes
                    columns++
                }
            }
        }
        require(count > 0) { "PLY has no vertices" }
        return PlyHeader(count, order, stride, columns, fields.mapIndexed { i, field ->
            requireNotNull(field) { "Colored PLY is missing vertex property ${plyComponents[i]}" }
        })
    }

    private fun readPly(file: File, checkpoint: () -> Unit): Cloud =
        DataInputStream(file.inputStream().buffered()).use { input ->
            val header = readPlyHeader(input)
            val minBytes = if (header.order == null) header.columns * 2L else header.stride.toLong()
            require(header.count * minBytes - 1 <= file.length()) { "Truncated PLY vertex data" }
            val cloud = Cloud(header.count)
            if (header.order != null) {
                val row = ByteBuffer.allocate(header.stride).order(header.order)
                val fields = header.fields
                repeat(header.count) { i ->
                    if ((i and CHECKPOINT_MASK) == 0) checkpoint()
                    input.readFully(row.array())
                    cloud.put(i, fields[0].type.read(row, fields[0].offset),
                        fields[1].type.read(row, fields[1].offset), fields[2].type.read(row, fields[2].offset),
                        row.get(fields[3].offset).toInt() and 0xff, row.get(fields[4].offset).toInt() and 0xff,
                        row.get(fields[5].offset).toInt() and 0xff)
                }
            } else {
                val componentByColumn = IntArray(header.columns) { -1 }
                header.fields.forEachIndexed { component, field -> componentByColumn[field.column] = component }
                val positionTypes = header.fields.take(3).map { it.type }
                val token = ByteArray(128)
                val vertex = DoubleArray(6)
                repeat(header.count) { i ->
                    if ((i and CHECKPOINT_MASK) == 0) checkpoint()
                    for (component in componentByColumn) {
                        val size = readPlyToken(input, token)
                        if (component in 0..2) {
                            val number = String(token, 0, size, Charsets.US_ASCII)
                            vertex[component] = if (positionTypes[component] == Scalar.F32) {
                                number.toFloat().toDouble()
                            } else number.toDouble()
                        } else if (component >= 3) {
                            vertex[component] = readAsciiColor(token, size, i).toDouble()
                        }
                    }
                    cloud.put(i, vertex[0], vertex[1], vertex[2],
                        vertex[3].toInt(), vertex[4].toInt(), vertex[5].toInt())
                }
            }
            cloud
        }

    private fun readAsciiColor(token: ByteArray, size: Int, vertex: Int): Int {
        var value = 0
        val firstDigit = if (token[0].toInt() == '+'.code) 1 else 0
        require(firstDigit < size) { "Invalid PLY RGB at vertex ${vertex + 1}" }
        for (i in firstDigit until size) {
            val digit = token[i] - '0'.code
            value = 10 * value + digit
            require(digit in 0..9 && value <= 255) { "Invalid PLY RGB at vertex ${vertex + 1}" }
        }
        return value
    }

    private fun readPlyToken(input: DataInputStream, token: ByteArray): Int {
        var char: Int
        do { char = input.read() } while (char in 0..32)
        require(char >= 0) { "Truncated ASCII PLY vertex data" }
        var size = 0
        while (char > 32) {
            require(size < token.size) { "Oversized ASCII PLY scalar" }
            token[size++] = char.toByte()
            char = input.read()
        }
        return size
    }

    /** Balanced, in-place k-d tree; a query reuses one sorted 16-entry neighbor list. */
    private class Neighbors(
        private val xyz: DoubleArray, count: Int, private val checkpoint: () -> Unit,
    ) {
        private val order = IntArray(count) { it }
        private val splitAxes = ByteArray(count)
        val indices = IntArray(NORMAL_NEIGHBORS)
        val distances = DoubleArray(NORMAL_NEIGHBORS)
        private var size = 0
        private var visited = 0

        init { build(0, count) }

        private fun build(start: Int, end: Int) {
            if (end - start <= 1) return
            if ((++visited and CHECKPOINT_MASK) == 0) checkpoint()
            // The widest axis avoids cycling through collapsed dimensions on a thin surface.
            var minX = Double.POSITIVE_INFINITY; var maxX = Double.NEGATIVE_INFINITY
            var minY = minX; var maxY = maxX
            var minZ = minX; var maxZ = maxX
            for (i in start until end) {
                val base = order[i] * 3
                val x = xyz[base]; val y = xyz[base + 1]; val z = xyz[base + 2]
                minX = minOf(minX, x); maxX = maxOf(maxX, x)
                minY = minOf(minY, y); maxY = maxOf(maxY, y)
                minZ = minOf(minZ, z); maxZ = maxOf(maxZ, z)
            }
            val xWidth = maxX - minX; val yWidth = maxY - minY; val zWidth = maxZ - minZ
            val axis = if (xWidth >= yWidth && xWidth >= zWidth) 0 else if (yWidth >= zWidth) 1 else 2
            val middle = (start + end) ushr 1
            selectMedian(start, end - 1, middle, axis)
            splitAxes[middle] = axis.toByte()
            build(start, middle)
            build(middle + 1, end)
        }

        private fun selectMedian(first: Int, last: Int, middle: Int, axis: Int) {
            var low = first
            var high = last
            while (low < high) {
                if (high - low > CHECKPOINT_MASK) checkpoint()
                val pivot = xyz[order[(low + high) ushr 1] * 3 + axis]
                var left = low
                var right = high
                while (left <= right) {
                    while (xyz[order[left] * 3 + axis] < pivot) left++
                    while (xyz[order[right] * 3 + axis] > pivot) right--
                    if (left <= right) {
                        val swap = order[left]
                        order[left++] = order[right]
                        order[right--] = swap
                    }
                }
                when {
                    middle <= right -> high = right
                    middle >= left -> low = left
                    else -> return
                }
            }
        }

        fun find(index: Int) {
            size = 0
            search(index * 3, 0, order.size)
        }

        private fun search(query: Int, start: Int, end: Int) {
            if (start == end) return
            if ((++visited and CHECKPOINT_MASK) == 0) checkpoint()
            val middle = (start + end) ushr 1
            val point = order[middle]
            val base = point * 3
            val axis = splitAxes[middle].toInt()
            val delta = xyz[query + axis] - xyz[base + axis]
            if (delta <= 0.0) search(query, start, middle) else search(query, middle + 1, end)
            val splitDistance = delta * delta
            if (size == NORMAL_NEIGHBORS && splitDistance >= distances[NORMAL_NEIGHBORS - 1]) return
            if (base != query) {
                val x = xyz[query] - xyz[base]
                val y = xyz[query + 1] - xyz[base + 1]
                val z = xyz[query + 2] - xyz[base + 2]
                val distance = x * x + y * y + z * z
                if (size < NORMAL_NEIGHBORS || distance < distances[NORMAL_NEIGHBORS - 1]) {
                    var k = minOf(size, NORMAL_NEIGHBORS - 1)
                    if (size < NORMAL_NEIGHBORS) size++
                    while (k > 0 && distance < distances[k - 1]) {
                        distances[k] = distances[k - 1]
                        indices[k] = indices[k - 1]
                        k--
                    }
                    distances[k] = distance
                    indices[k] = point
                }
            }
            // Equal-distance ties retain the first points visited, deterministically.
            if (size < NORMAL_NEIGHBORS || splitDistance < distances[NORMAL_NEIGHBORS - 1]) {
                if (delta <= 0.0) search(query, middle + 1, end) else search(query, start, middle)
            }
        }
    }

    private class SurfaceRotation(private val xyz: DoubleArray) {
        private val covariance = DoubleArray(9)
        private val vectors = DoubleArray(9)

        fun put(point: Int, neighbors: IntArray, out: ByteBuffer) {
            val origin = point * 3
            val x = xyz[origin]; val y = xyz[origin + 1]; val z = xyz[origin + 2]
            // Compute the centroid relative to the query to preserve local differences
            // when the scene is far from the world origin.
            var mx = 0.0; var my = 0.0; var mz = 0.0
            for (neighbor in neighbors) {
                val base = neighbor * 3
                mx += xyz[base] - x; my += xyz[base + 1] - y; mz += xyz[base + 2] - z
            }
            mx /= NORMAL_NEIGHBORS; my /= NORMAL_NEIGHBORS; mz /= NORMAL_NEIGHBORS
            var xx = 0.0; var xy = 0.0; var xz = 0.0; var yy = 0.0; var yz = 0.0; var zz = 0.0
            for (neighbor in neighbors) {
                val base = neighbor * 3
                val dx = (xyz[base] - x) - mx
                val dy = (xyz[base + 1] - y) - my
                val dz = (xyz[base + 2] - z) - mz
                xx += dx * dx; xy += dx * dy; xz += dx * dz
                yy += dy * dy; yz += dy * dz; zz += dz * dz
            }
            covariance[0] = xx / NORMAL_NEIGHBORS; covariance[1] = xy / NORMAL_NEIGHBORS
            covariance[2] = xz / NORMAL_NEIGHBORS; covariance[4] = yy / NORMAL_NEIGHBORS
            covariance[5] = yz / NORMAL_NEIGHBORS; covariance[8] = zz / NORMAL_NEIGHBORS
            require(maxOf(xx, yy, zz) > 0.0) { "PocketGS point ${point + 1} has no spatial extent among its 16 nearest neighbors" }
            // Only the upper triangle is used in pivot selection. Keep both halves in sync.
            covariance[3] = covariance[1]; covariance[6] = covariance[2]; covariance[7] = covariance[5]
            vectors.fill(0.0)
            vectors[0] = 1.0; vectors[4] = 1.0; vectors[8] = 1.0
            diagonalize()
            var smallest = 0
            if (covariance[4] < covariance[0]) smallest = 1
            if (covariance[8] < covariance[smallest * 4]) smallest = 2
            val nx = vectors[smallest]; val ny = vectors[3 + smallest]; val nz = vectors[6 + smallest]
            // A covariance eigenvector is sign-ambiguous. Choose +Z's hemisphere so the
            // shortest-arc rotation Z -> n has no antipodal singularity. Disc covariance
            // is unchanged by this sign choice because its two tangent scales are equal.
            val sign = if (nz < 0.0) -1.0 else 1.0
            val length = sqrt(nx * nx + ny * ny + nz * nz)
            val w = 1.0 + nz * sign / length
            val inverse = 1.0 / sqrt(2.0 * w)
            out.putFloat((w * inverse).toFloat())
            out.putFloat((-ny * sign / length * inverse).toFloat())
            out.putFloat((nx * sign / length * inverse).toFloat())
            out.putFloat(0f)
        }

        private fun diagonalize() {
            // Symmetric Jacobi eigensolver; columns of vectors track the eigenvectors.
            val a = covariance
            val tolerance = 1e-14 * maxOf(a[0], a[4], a[8])
            repeat(32) {
                var p = 0; var q = 1
                if (abs(a[2]) > abs(a[1])) q = 2
                if (abs(a[5]) > abs(a[q])) { p = 1; q = 2 }
                val pp = 3 * p + p; val qq = 3 * q + q; val pq = 3 * p + q
                val off = a[pq]
                if (abs(off) <= tolerance) return
                val tau = (a[qq] - a[pp]) / (2.0 * off)
                val t = (if (tau < 0.0) -1.0 else 1.0) / (abs(tau) + sqrt(1.0 + tau * tau))
                val c = 1.0 / sqrt(1.0 + t * t)
                val s = t * c
                a[pp] -= t * off; a[qq] += t * off
                a[pq] = 0.0; a[3 * q + p] = 0.0
                val k = 3 - p - q
                val kp = a[3 * k + p]; val kq = a[3 * k + q]
                val newP = c * kp - s * kq
                val newQ = s * kp + c * kq
                a[3 * k + p] = newP; a[3 * p + k] = newP
                a[3 * k + q] = newQ; a[3 * q + k] = newQ
                for (row in 0..2) {
                    val vp = 3 * row + p; val vq = 3 * row + q
                    val nextP = c * vectors[vp] - s * vectors[vq]
                    vectors[vq] = s * vectors[vp] + c * vectors[vq]
                    vectors[vp] = nextP
                }
            }
        }
    }
}
