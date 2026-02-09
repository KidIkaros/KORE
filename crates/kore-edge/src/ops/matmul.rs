//! Matrix multiplication operators — f32, ternary, and quaternary.
//!
//! All operate on raw slices. C = A[M,K] @ B[K,N] → C[M,N].

use kore_btes::encoder::decode_trits;

/// f32 matmul: C[M,N] = A[M,K] @ B[K,N]
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// f32 matmul with bias: C[M,N] = A[M,K] @ B[K,N] + bias[N]
pub fn matmul_f32_bias(a: &[f32], b: &[f32], bias: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = bias[j];
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// Ternary matmul: C[M,N] = A_ternary[M,K] @ B_f32[K,N] * scales[M]
///
/// `a_packed`: base-243 packed ternary weights, row-major [M, ceil(K/5)]
/// `a_scales`: per-row scale factors [M]
/// `b`: dense f32 activations [K,N]
///
/// Uses A-stationary accumulation: iterates over K, scatters into N outputs.
/// Better cache behavior for small N (typical for autoregressive decode).
pub fn matmul_ternary(
    a_packed: &[u8],
    a_scales: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let k_packed = (k + 4) / 5;

    for row in 0..m {
        let scale = a_scales[row];
        let a_row = &a_packed[row * k_packed..(row + 1) * k_packed];
        let c_row = &mut c[row * n..(row + 1) * n];

        // Unpack trits
        let mut trits = Vec::with_capacity(k + 4);
        for &byte in a_row {
            let decoded = decode_trits(byte);
            for &t in &decoded {
                trits.push(t as i8);
            }
        }
        trits.truncate(k);

        // A-stationary: iterate K, scatter into N outputs
        for ki in 0..k {
            let t = trits[ki];
            if t == 0 { continue; }
            let t_f32 = t as f32;
            let b_row = &b[ki * n..ki * n + n];
            for col in 0..n {
                c_row[col] += t_f32 * b_row[col];
            }
        }

        // Apply scale
        for col in 0..n {
            c_row[col] *= scale;
        }
    }
}

/// Quaternary matmul: C[M,N] = A_quat[M,K] @ B_f32[K,N] * scales[M]
///
/// `a_packed`: 2-bit packed quaternary weights, 4 values per byte [M, ceil(K/4)]
/// `a_scales`: per-row scale factors [M]
/// Quaternary values: {-3, -1, +1, +3} mapped from 2-bit indices {0,1,2,3}
/// Uses A-stationary accumulation: pre-unpack row, iterate K, scatter into N.
pub fn matmul_quaternary(
    a_packed: &[u8],
    a_scales: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    const QUAT_VALUES: [f32; 4] = [-3.0, -1.0, 1.0, 3.0];
    let k_packed = (k + 3) / 4;

    for row in 0..m {
        let scale = a_scales[row];
        let a_row = &a_packed[row * k_packed..(row + 1) * k_packed];
        let c_row = &mut c[row * n..(row + 1) * n];

        // Pre-unpack quaternary values
        let mut quats = Vec::with_capacity(k + 3);
        for &byte in a_row {
            quats.push(QUAT_VALUES[((byte) & 0x3) as usize]);
            quats.push(QUAT_VALUES[((byte >> 2) & 0x3) as usize]);
            quats.push(QUAT_VALUES[((byte >> 4) & 0x3) as usize]);
            quats.push(QUAT_VALUES[((byte >> 6) & 0x3) as usize]);
        }
        quats.truncate(k);

        // A-stationary: iterate K, scatter into N outputs
        for ki in 0..k {
            let w = quats[ki];
            let b_row = &b[ki * n..ki * n + n];
            for col in 0..n {
                c_row[col] += w * b_row[col];
            }
        }

        // Apply scale
        for col in 0..n {
            c_row[col] *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_f32_identity() {
        let a = [1.0, 2.0, 3.0, 4.0]; // [2,2]
        let b = [1.0, 0.0, 0.0, 1.0]; // identity
        let mut c = [0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_f32_bias() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let bias = [10.0, 20.0];
        let mut c = [0.0f32; 4];
        matmul_f32_bias(&a, &b, &bias, &mut c, 2, 2, 2);
        assert_eq!(c, [11.0, 20.0, 10.0, 21.0]);
    }

    #[test]
    fn test_matmul_quaternary_basic() {
        // Single row, K=4: all +1 (index 2)
        let packed = [0b10101010u8]; // 4 values of index 2 = +1
        let scales = [1.0f32];
        let b = [1.0, 2.0, 3.0, 4.0]; // [4,1]
        let mut c = [0.0f32; 1];
        matmul_quaternary(&packed, &scales, &b, &mut c, 1, 1, 4);
        assert!((c[0] - 10.0).abs() < 1e-5, "got {}", c[0]);
    }

    #[test]
    fn test_matmul_ternary_identity() {
        use kore_btes::encoder::{Trit, encode_trits};
        // A = [[+1, 0], [0, +1]] (ternary identity), B = [[2, 0], [0, 3]]
        // C = A @ B = [[2, 0], [0, 3]]
        let row0 = encode_trits(&[Trit::Pos, Trit::Zero, Trit::Zero, Trit::Zero, Trit::Zero]);
        let row1 = encode_trits(&[Trit::Zero, Trit::Pos, Trit::Zero, Trit::Zero, Trit::Zero]);
        let packed = [row0, row1];
        let scales = [1.0f32, 1.0];
        let b = [2.0, 0.0, 0.0, 3.0]; // [2,2]
        let mut c = [0.0f32; 4];
        matmul_ternary(&packed, &scales, &b, &mut c, 2, 2, 2);
        assert!((c[0] - 2.0).abs() < 0.1, "c[0]={}", c[0]);
        assert!((c[1]).abs() < 0.1, "c[1]={}", c[1]);
        assert!((c[2]).abs() < 0.1, "c[2]={}", c[2]);
        assert!((c[3] - 3.0).abs() < 0.1, "c[3]={}", c[3]);
    }

    #[test]
    fn test_matmul_ternary_with_scale() {
        use kore_btes::encoder::{Trit, encode_trits};
        // A = [[+1, -1]] with scale=2.0, B = [[3], [1]] → C = (1*3 + (-1)*1) * 2 = 4
        let row0 = encode_trits(&[Trit::Pos, Trit::Neg, Trit::Zero, Trit::Zero, Trit::Zero]);
        let packed = [row0];
        let scales = [2.0f32];
        let b = [3.0, 1.0]; // [2,1]
        let mut c = [0.0f32; 1];
        matmul_ternary(&packed, &scales, &b, &mut c, 1, 1, 2);
        assert!((c[0] - 4.0).abs() < 0.1, "c[0]={}", c[0]);
    }

    #[test]
    fn test_matmul_quaternary_non_aligned_k() {
        // K=5 is not a multiple of 4
        // Row: [-3, -1, +1, +3, -3] with scale=1.0
        // Pack: byte0 = indices [0,1,2,3] = 0b11_10_01_00 = 0xE4
        //       byte1 = index [0,pad,pad,pad] = 0b10_10_10_00 = 0xAA (pad with +1=index 2)
        let packed = [0xE4u8, 0x00u8]; // byte1: index 0 = -3, rest padding
        let scales = [1.0f32];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0]; // [5,1]
        let mut c = [0.0f32; 1];
        matmul_quaternary(&packed, &scales, &b, &mut c, 1, 1, 5);
        // -3 + -1 + 1 + 3 + (-3) = -3
        assert!((c[0] - (-3.0)).abs() < 1e-5, "got {}", c[0]);
    }

    #[test]
    fn test_matmul_quaternary_with_scale() {
        // All +3 (index 3), scale=2.0, K=4, N=2
        let packed = [0b11111111u8]; // 4 values of index 3 = +3
        let scales = [2.0f32];
        let b = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // [4,2]
        let mut c = [0.0f32; 2];
        matmul_quaternary(&packed, &scales, &b, &mut c, 1, 2, 4);
        // col0: (3+3+3+3)*2 = 24, col1: (6+6+6+6)*2 = 48
        assert!((c[0] - 24.0).abs() < 1e-5, "c[0]={}", c[0]);
        assert!((c[1] - 48.0).abs() < 1e-5, "c[1]={}", c[1]);
    }
}
