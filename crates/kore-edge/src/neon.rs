//! ARM NEON SIMD kernels for Android, iOS, and Raspberry Pi.
//!
//! These are only compiled on `aarch64` targets.
//! Each function has a scalar fallback in `ops/` for other architectures.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use kore_btes::encoder::decode_trits;

/// NEON-accelerated f32 matmul: C[M,N] = A[M,K] @ B[K,N]
///
/// Uses 4×4 tiling with `vfmaq_f32` for the inner loop.
#[cfg(target_arch = "aarch64")]
pub fn neon_matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // Process 4 columns at a time using NEON
    for i in 0..m {
        let mut j = 0;
        while j + 4 <= n {
            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                for p in 0..k {
                    let a_val = vdupq_n_f32(a[i * k + p]);
                    let b_vec = vld1q_f32(b.as_ptr().add(p * n + j));
                    acc = vfmaq_f32(acc, a_val, b_vec);
                }
                vst1q_f32(c.as_mut_ptr().add(i * n + j), acc);
            }
            j += 4;
        }
        // Scalar tail
        while j < n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
            j += 1;
        }
    }
}

/// NEON-accelerated ternary matmul.
#[cfg(target_arch = "aarch64")]
pub fn neon_matmul_ternary(
    a_packed: &[u8], a_scales: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    let k_packed = (k + 4) / 5;

    for row in 0..m {
        let scale = a_scales[row];
        let a_row = &a_packed[row * k_packed..(row + 1) * k_packed];

        let mut trits = Vec::with_capacity(k);
        for &byte in a_row {
            let decoded = decode_trits(byte);
            for &t in &decoded {
                if trits.len() >= k { break; }
                trits.push(t as i8);
            }
        }
        trits.truncate(k);

        for col in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                let t = trits[ki];
                if t != 0 {
                    acc += t as f32 * b[ki * n + col];
                }
            }
            c[row * n + col] = acc * scale;
        }
    }
}

/// NEON-accelerated quaternary matmul with A-stationary accumulation.
///
/// Pre-unpacks quaternary values, iterates K, and scatters into N outputs
/// using NEON f32x4 SIMD for 4 columns at a time.
#[cfg(target_arch = "aarch64")]
pub fn neon_matmul_quaternary(
    a_packed: &[u8], a_scales: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
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

        // A-stationary: iterate K, scatter into N outputs using NEON
        for ki in 0..k {
            let w = quats[ki];
            let b_off = ki * n;

            let mut col = 0;
            unsafe {
                let w_vec = vdupq_n_f32(w);
                while col + 4 <= n {
                    let b_vec = vld1q_f32(b.as_ptr().add(b_off + col));
                    let c_vec = vld1q_f32(c_row.as_ptr().add(col));
                    let result = vfmaq_f32(c_vec, w_vec, b_vec);
                    vst1q_f32(c_row.as_mut_ptr().add(col), result);
                    col += 4;
                }
            }
            while col < n {
                c_row[col] += w * b[b_off + col];
                col += 1;
            }
        }

        // Apply scale with NEON
        let mut col = 0;
        unsafe {
            let scale_vec = vdupq_n_f32(scale);
            while col + 4 <= n {
                let c_vec = vld1q_f32(c_row.as_ptr().add(col));
                let result = vmulq_f32(c_vec, scale_vec);
                vst1q_f32(c_row.as_mut_ptr().add(col), result);
                col += 4;
            }
        }
        while col < n {
            c_row[col] *= scale;
            col += 1;
        }
    }
}

/// NEON-accelerated RMS normalization.
#[cfg(target_arch = "aarch64")]
pub fn neon_rms_norm(data: &mut [f32], gamma: &[f32], dim: usize, eps: f32) {
    let batch = data.len() / dim;
    for b_idx in 0..batch {
        let row = &mut data[b_idx * dim..(b_idx + 1) * dim];

        // Sum of squares using NEON
        let mut sum_sq = 0.0f32;
        let mut i = 0;
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            while i + 4 <= dim {
                let v = vld1q_f32(row.as_ptr().add(i));
                acc = vfmaq_f32(acc, v, v);
                i += 4;
            }
            sum_sq = vaddvq_f32(acc);
        }
        while i < dim {
            sum_sq += row[i] * row[i];
            i += 1;
        }

        let rms = (sum_sq / dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Scale using NEON
        i = 0;
        unsafe {
            let inv_rms_v = vdupq_n_f32(inv_rms);
            while i + 4 <= dim {
                let v = vld1q_f32(row.as_ptr().add(i));
                let g = vld1q_f32(gamma.as_ptr().add(i));
                let result = vmulq_f32(vmulq_f32(v, inv_rms_v), g);
                vst1q_f32(row.as_mut_ptr().add(i), result);
                i += 4;
            }
        }
        while i < dim {
            row[i] = row[i] * inv_rms * gamma[i];
            i += 1;
        }
    }
}

/// NEON-accelerated softmax.
#[cfg(target_arch = "aarch64")]
pub fn neon_softmax(data: &mut [f32], dim: usize) {
    // Delegate to scalar for now — NEON exp() requires libm or approximation
    crate::ops::activation::softmax(data, dim);
}

// Non-aarch64 stubs (should never be called due to cfg gates in lib.rs)
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_matmul_f32(_a: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _n: usize, _k: usize) {
    unreachable!("NEON not available on this architecture");
}
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_matmul_ternary(_a: &[u8], _s: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _n: usize, _k: usize) {
    unreachable!("NEON not available on this architecture");
}
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_matmul_quaternary(_a: &[u8], _s: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _n: usize, _k: usize) {
    unreachable!("NEON not available on this architecture");
}
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_rms_norm(_d: &mut [f32], _g: &[f32], _dim: usize, _eps: f32) {
    unreachable!("NEON not available on this architecture");
}
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_softmax(_d: &mut [f32], _dim: usize) {
    unreachable!("NEON not available on this architecture");
}
