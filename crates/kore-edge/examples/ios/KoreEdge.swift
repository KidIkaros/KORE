/**
 * Kore Edge — Swift wrapper for on-device inference on iOS.
 *
 * Usage:
 *   let session = try KoreEdge(path: Bundle.main.path(forResource: "model", ofType: "koref")!)
 *   let tokens = session.generate(inputIds: [1, 2, 3], maxTokens: 32)
 *   session.reset()
 *
 * Build the native library:
 *   cargo build --target aarch64-apple-ios -p kore-edge --release --features ffi
 *   # Create XCFramework from the .a static library
 */
import Foundation

/// Swift wrapper around the kore-edge C FFI.
public class KoreEdge {
    private var handle: OpaquePointer?

    /// Load a .koref model from a file path.
    public init(path: String) throws {
        handle = path.withCString { cStr in
            kore_edge_load(cStr)
        }
        guard handle != nil else {
            throw KoreEdgeError.loadFailed(path)
        }
    }

    /// Load a .koref model from raw bytes (e.g., from app bundle).
    public init(data: Data) throws {
        handle = data.withUnsafeBytes { ptr in
            kore_edge_load_bytes(ptr.baseAddress?.assumingMemoryBound(to: UInt8.self), ptr.count)
        }
        guard handle != nil else {
            throw KoreEdgeError.loadFromBytesFailed
        }
    }

    deinit {
        if let h = handle {
            kore_edge_free(h)
        }
    }

    /// Run a forward pass, returning logits for the last token.
    public func forward(inputIds: [UInt32]) -> [Float] {
        guard let h = handle else { return [] }
        let output = inputIds.withUnsafeBufferPointer { buf in
            kore_edge_run(h, buf.baseAddress, buf.count)
        }
        guard let out = output else { return [] }
        defer { kore_edge_free_output(out) }

        let logits = out.pointee.logits
        let count = out.pointee.logit_count
        guard let ptr = logits else { return [] }
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Generate tokens autoregressively.
    public func generate(inputIds: [UInt32], maxTokens: Int = 32) -> [UInt32] {
        guard let h = handle else { return [] }
        let output = inputIds.withUnsafeBufferPointer { buf in
            kore_edge_generate(h, buf.baseAddress, buf.count, maxTokens)
        }
        guard let out = output else { return [] }
        defer { kore_edge_free_output(out) }

        let tokens = out.pointee.tokens
        let count = out.pointee.token_count
        guard let ptr = tokens else { return [] }
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Reset the session (clear KV cache).
    public func reset() {
        guard let h = handle else { return }
        kore_edge_reset(h)
    }
}

public enum KoreEdgeError: Error, LocalizedError {
    case loadFailed(String)
    case loadFromBytesFailed

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let path):
            return "Failed to load .koref model at: \(path)"
        case .loadFromBytesFailed:
            return "Failed to load .koref model from bytes"
        }
    }
}
