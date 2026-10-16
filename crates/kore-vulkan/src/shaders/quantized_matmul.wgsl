struct QuantizedUniforms {
    M: u32,
    N: u32,
    K: u32,
    // bit 0: 0 = Ternary (1.58-bit, 5 trits/byte), 1 = Quaternary (2-bit, 4 quats/byte)
    // bit 1 (value 2): 1 = row-packed [N, row_bytes] (BitLinear/QuatLinear layout), 0 = flat [K, N]
    format: u32,
    alpha: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: QuantizedUniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B_packed: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> C: array<f32>;

fn get_packed_byte(byte_idx: u32) -> u32 {
    let word_idx = byte_idx >> 2u;
    let byte_off = (byte_idx & 3u) << 3u;
    return (B_packed[word_idx] >> byte_off) & 255u;
}

fn decode_quat(bits: u32) -> f32 {
    return f32(i32(bits) * 2 - 3);
}

fn decode_flat_weight(flat_idx: u32, is_quat: bool) -> f32 {
    if (!is_quat) {
        let byte_idx = flat_idx / 5u;
        let trit_idx = flat_idx % 5u;
        var val = get_packed_byte(byte_idx);
        if (trit_idx == 1u) { val = val / 3u; }
        else if (trit_idx == 2u) { val = val / 9u; }
        else if (trit_idx == 3u) { val = val / 27u; }
        else if (trit_idx == 4u) { val = val / 81u; }
        return f32(val % 3u) - 1.0;
    } else {
        let byte_idx = flat_idx >> 2u;
        let quat_idx = flat_idx & 3u;
        let b = get_packed_byte(byte_idx);
        let bits = (b >> (quat_idx * 2u)) & 3u;
        return decode_quat(bits);
    }
}

// Fast Quantized GEMV / Small-Batch GEMM (@workgroup_size(64))
// Each thread computes output column `col` across up to 4 rows per pass (`row_base = wg_id.y * 4`).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row_base = global_id.y * 4u;
    if (col >= params.N || row_base >= params.M) {
        return;
    }

    let m_rem = min(4u, params.M - row_base);
    let k_len = params.K;
    let n_len = params.N;
    let is_quat = (params.format & 1u) != 0u;
    let is_row_packed = (params.format & 2u) != 0u;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    let a_base0 = (row_base + 0u) * k_len;
    let a_base1 = (row_base + 1u) * k_len;
    let a_base2 = (row_base + 2u) * k_len;
    let a_base3 = (row_base + 3u) * k_len;

    if (is_row_packed && !is_quat) {
        // Row-packed Ternary [N, ceil(K / 5)]: division-free multiply-shift base-3 decoding (5 trits/byte)
        let row_bytes = (k_len + 4u) / 5u;
        let w_row_base = col * row_bytes;
        let full_bytes = k_len / 5u;
        for (var b_idx: u32 = 0u; b_idx < full_bytes; b_idx = b_idx + 1u) {
            let b = get_packed_byte(w_row_base + b_idx);
            let k = b_idx * 5u;
            // Exact division by 3 for b in 0..255 via (x * 171) >> 9
            let q1 = (b * 171u) >> 9u;
            let q2 = (q1 * 171u) >> 9u;
            let q3 = (q2 * 171u) >> 9u;
            let q4 = (q3 * 171u) >> 9u;
            let w0 = f32(b - q1 * 3u) - 1.0;
            let w1 = f32(q1 - q2 * 3u) - 1.0;
            let w2 = f32(q2 - q3 * 3u) - 1.0;
            let w3 = f32(q3 - q4 * 3u) - 1.0;
            let w4 = f32(q4) - 1.0;

            sum0 = sum0 + A[a_base0 + k] * w0 + A[a_base0 + k + 1u] * w1 + A[a_base0 + k + 2u] * w2 + A[a_base0 + k + 3u] * w3 + A[a_base0 + k + 4u] * w4;
            if (m_rem > 1u) {
                sum1 = sum1 + A[a_base1 + k] * w0 + A[a_base1 + k + 1u] * w1 + A[a_base1 + k + 2u] * w2 + A[a_base1 + k + 3u] * w3 + A[a_base1 + k + 4u] * w4;
            }
            if (m_rem > 2u) {
                sum2 = sum2 + A[a_base2 + k] * w0 + A[a_base2 + k + 1u] * w1 + A[a_base2 + k + 2u] * w2 + A[a_base2 + k + 3u] * w3 + A[a_base2 + k + 4u] * w4;
            }
            if (m_rem > 3u) {
                sum3 = sum3 + A[a_base3 + k] * w0 + A[a_base3 + k + 1u] * w1 + A[a_base3 + k + 2u] * w2 + A[a_base3 + k + 3u] * w3 + A[a_base3 + k + 4u] * w4;
            }
        }
        let rem_start = full_bytes * 5u;
        if (rem_start < k_len) {
            var b = get_packed_byte(w_row_base + full_bytes);
            for (var k: u32 = rem_start; k < k_len; k = k + 1u) {
                let q = (b * 171u) >> 9u;
                let w = f32(b - q * 3u) - 1.0;
                b = q;
                sum0 = sum0 + A[a_base0 + k] * w;
                if (m_rem > 1u) { sum1 = sum1 + A[a_base1 + k] * w; }
                if (m_rem > 2u) { sum2 = sum2 + A[a_base2 + k] * w; }
                if (m_rem > 3u) { sum3 = sum3 + A[a_base3 + k] * w; }
            }
        }
    } else if (is_row_packed && is_quat) {
        // Row-packed Quaternary [N, ceil(K / 4)]: unpack 4 quats per byte in a single byte read
        let row_bytes = (k_len + 3u) / 4u;
        let w_row_base = col * row_bytes;
        let full_bytes = k_len / 4u;
        for (var b_idx: u32 = 0u; b_idx < full_bytes; b_idx = b_idx + 1u) {
            let b = get_packed_byte(w_row_base + b_idx);
            let k = b_idx * 4u;
            let w0 = decode_quat(b & 3u);
            let w1 = decode_quat((b >> 2u) & 3u);
            let w2 = decode_quat((b >> 4u) & 3u);
            let w3 = decode_quat((b >> 6u) & 3u);

            sum0 = sum0 + A[a_base0 + k] * w0 + A[a_base0 + k + 1u] * w1 + A[a_base0 + k + 2u] * w2 + A[a_base0 + k + 3u] * w3;
            if (m_rem > 1u) {
                sum1 = sum1 + A[a_base1 + k] * w0 + A[a_base1 + k + 1u] * w1 + A[a_base1 + k + 2u] * w2 + A[a_base1 + k + 3u] * w3;
            }
            if (m_rem > 2u) {
                sum2 = sum2 + A[a_base2 + k] * w0 + A[a_base2 + k + 1u] * w1 + A[a_base2 + k + 2u] * w2 + A[a_base2 + k + 3u] * w3;
            }
            if (m_rem > 3u) {
                sum3 = sum3 + A[a_base3 + k] * w0 + A[a_base3 + k + 1u] * w1 + A[a_base3 + k + 2u] * w2 + A[a_base3 + k + 3u] * w3;
            }
        }
        let rem_start = full_bytes * 4u;
        if (rem_start < k_len) {
            var b = get_packed_byte(w_row_base + full_bytes);
            for (var k: u32 = rem_start; k < k_len; k = k + 1u) {
                let w = decode_quat(b & 3u);
                b = b >> 2u;
                sum0 = sum0 + A[a_base0 + k] * w;
                if (m_rem > 1u) { sum1 = sum1 + A[a_base1 + k] * w; }
                if (m_rem > 2u) { sum2 = sum2 + A[a_base2 + k] * w; }
                if (m_rem > 3u) { sum3 = sum3 + A[a_base3 + k] * w; }
            }
        }
    } else {
        // Flat [K, N] layout
        for (var k: u32 = 0u; k < k_len; k = k + 1u) {
            let w = decode_flat_weight(k * n_len + col, is_quat);
            sum0 = sum0 + A[a_base0 + k] * w;
            if (m_rem > 1u) { sum1 = sum1 + A[a_base1 + k] * w; }
            if (m_rem > 2u) { sum2 = sum2 + A[a_base2 + k] * w; }
            if (m_rem > 3u) { sum3 = sum3 + A[a_base3 + k] * w; }
        }
    }

    let scale = params.alpha * scales[col];
    C[(row_base + 0u) * n_len + col] = sum0 * scale;
    if (m_rem > 1u) { C[(row_base + 1u) * n_len + col] = sum1 * scale; }
    if (m_rem > 2u) { C[(row_base + 2u) * n_len + col] = sum2 * scale; }
    if (m_rem > 3u) { C[(row_base + 3u) * n_len + col] = sum3 * scale; }
}
