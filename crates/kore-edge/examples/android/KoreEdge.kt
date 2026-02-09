/**
 * Kore Edge — Android JNI wrapper for on-device inference.
 *
 * Usage:
 *   val session = KoreEdge.load("/data/local/tmp/model.koref")
 *   val tokens = session.generate(intArrayOf(1, 2, 3), maxTokens = 32)
 *   session.close()
 *
 * Build the native library:
 *   cargo make build-android
 *   cp target/aarch64-linux-android/release/libkore_edge.so app/src/main/jniLibs/arm64-v8a/
 */
package com.kore.edge

class KoreEdge private constructor(private var handle: Long) : AutoCloseable {

    companion object {
        init {
            System.loadLibrary("kore_edge")
        }

        /** Load a .koref model from a file path. */
        fun load(path: String): KoreEdge {
            val handle = nativeLoad(path)
            if (handle == 0L) throw RuntimeException("Failed to load model: $path")
            return KoreEdge(handle)
        }

        /** Load a .koref model from raw bytes. */
        fun fromBytes(data: ByteArray): KoreEdge {
            val handle = nativeLoadBytes(data)
            if (handle == 0L) throw RuntimeException("Failed to load model from bytes")
            return KoreEdge(handle)
        }

        @JvmStatic private external fun nativeLoad(path: String): Long
        @JvmStatic private external fun nativeLoadBytes(data: ByteArray): Long
    }

    /** Run a forward pass, returning logits for the last token. */
    fun forward(inputIds: IntArray): FloatArray {
        check(handle != 0L) { "Session already closed" }
        return nativeForward(handle, inputIds)
    }

    /** Generate tokens autoregressively. */
    fun generate(inputIds: IntArray, maxTokens: Int = 32): IntArray {
        check(handle != 0L) { "Session already closed" }
        return nativeGenerate(handle, inputIds, maxTokens)
    }

    /** Reset the session (clear KV cache). */
    fun reset() {
        check(handle != 0L) { "Session already closed" }
        nativeReset(handle)
    }

    override fun close() {
        if (handle != 0L) {
            nativeFree(handle)
            handle = 0L
        }
    }

    private external fun nativeForward(handle: Long, inputIds: IntArray): FloatArray
    private external fun nativeGenerate(handle: Long, inputIds: IntArray, maxTokens: Int): IntArray
    private external fun nativeReset(handle: Long)
    private external fun nativeFree(handle: Long)
}
