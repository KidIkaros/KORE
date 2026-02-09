//! WASM SIMD128 kernels for browser and Node.js targets.
//!
//! Only compiled on `wasm32` targets. Uses `core::arch::wasm32` SIMD intrinsics.
//! Each function has a scalar fallback in `ops/` for other architectures.

use kore_btes::encoder::decode_trits;

/// WASM SIMD128 f32 matmul: C[M,N] = A[M,K] @ B[K,N]
#[cfg(target_arch = "wasm32")]
pub fn wasm_matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use core::arch::wasm32::*;

    for i in 0..m {
        let mut j = 0;
        while j + 4 <= n {
            let mut acc = f32x4_splat(0.0);
            for p in 0..k {
                let a_val = f32x4_splat(a[i * k + p]);
                let b_vec = unsafe {
                    v128_load(b.as_ptr().add(p * n + j) as *const v128)
                };
                acc = f32x4_add(acc, f32x4_mul(a_val, b_vec));
            }
            unsafe {
                v128_store(c.as_mut_ptr().add(i * n + j) as *mut v128, acc);
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

/// WASM SIMD128 ternary matmul: C[M,N] = A_ternary[M,K] @ B_f32[K,N] * scales[M]
///
/// Uses A-stationary accumulation with SIMD128 for the inner scatter loop.
/// For each non-zero trit, broadcasts trit*scale and accumulates across 4 output
/// columns at a time using f32x4 SIMD.
#[cfg(target_arch = "wasm32")]
pub fn wasm_matmul_ternary(
    a_packed: &[u8], a_scales: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    use core::arch::wasm32::*;

    let k_packed = (k + 4) / 5;

    for row in 0..m {
        let scale = a_scales[row];
        let a_row = &a_packed[row * k_packed..(row + 1) * k_packed];
        let c_row = &mut c[row * n..(row + 1) * n];

        // Unpack trits for this row
        let mut trits = Vec::with_capacity(k + 4);
        for &byte in a_row {
            let decoded = decode_trits(byte);
            for &t in &decoded {
                trits.push(t as i8);
            }
        }
        trits.truncate(k);

        // A-stationary: iterate K, scatter into N outputs using SIMD
        for ki in 0..k {
            let t = trits[ki];
            if t == 0 { continue; }
            let t_f32 = t as f32;
            let t_vec = f32x4_splat(t_f32);
            let b_off = ki * n;

            // SIMD: 4 columns at a time
            let mut col = 0;
            while col + 4 <= n {
                let b_vec = unsafe { v128_load(b.as_ptr().add(b_off + col) as *const v128) };
                let c_vec = unsafe { v128_load(c_row.as_ptr().add(col) as *const v128) };
                let result = f32x4_add(c_vec, f32x4_mul(t_vec, b_vec));
                unsafe { v128_store(c_row.as_mut_ptr().add(col) as *mut v128, result); }
                col += 4;
            }
            // Scalar tail
            while col < n {
                c_row[col] += t_f32 * b[b_off + col];
                col += 1;
            }
        }

        // Apply scale with SIMD
        let scale_vec = f32x4_splat(scale);
        let mut col = 0;
        while col + 4 <= n {
            let c_vec = unsafe { v128_load(c_row.as_ptr().add(col) as *const v128) };
            let result = f32x4_mul(c_vec, scale_vec);
            unsafe { v128_store(c_row.as_mut_ptr().add(col) as *mut v128, result); }
            col += 4;
        }
        while col < n {
            c_row[col] *= scale;
            col += 1;
        }
    }
}

/// WASM SIMD128 quaternary matmul: C[M,N] = A_quat[M,K] @ B_f32[K,N] * scales[M]
///
/// Uses A-stationary accumulation with SIMD128 for the inner scatter loop.
/// Pre-unpacks quaternary values, then broadcasts w*1.0 and accumulates across
/// 4 output columns at a time using f32x4 SIMD.
#[cfg(target_arch = "wasm32")]
pub fn wasm_matmul_quaternary(
    a_packed: &[u8], a_scales: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    use core::arch::wasm32::*;

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

        // A-stationary: iterate K, scatter into N outputs using SIMD
        for ki in 0..k {
            let w = quats[ki];
            let w_vec = f32x4_splat(w);
            let b_off = ki * n;

            // SIMD: 4 columns at a time
            let mut col = 0;
            while col + 4 <= n {
                let b_vec = unsafe { v128_load(b.as_ptr().add(b_off + col) as *const v128) };
                let c_vec = unsafe { v128_load(c_row.as_ptr().add(col) as *const v128) };
                let result = f32x4_add(c_vec, f32x4_mul(w_vec, b_vec));
                unsafe { v128_store(c_row.as_mut_ptr().add(col) as *mut v128, result); }
                col += 4;
            }
            // Scalar tail
            while col < n {
                c_row[col] += w * b[b_off + col];
                col += 1;
            }
        }

        // Apply scale with SIMD
        let scale_vec = f32x4_splat(scale);
        let mut col = 0;
        while col + 4 <= n {
            let c_vec = unsafe { v128_load(c_row.as_ptr().add(col) as *const v128) };
            let result = f32x4_mul(c_vec, scale_vec);
            unsafe { v128_store(c_row.as_mut_ptr().add(col) as *mut v128, result); }
            col += 4;
        }
        while col < n {
            c_row[col] *= scale;
            col += 1;
        }
    }
}

// Non-wasm32 stubs
#[cfg(not(target_arch = "wasm32"))]
pub fn wasm_matmul_f32(_a: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _n: usize, _k: usize) {
    unreachable!("WASM SIMD not available on this architecture");
}
#[cfg(not(target_arch = "wasm32"))]
pub fn wasm_matmul_ternary(_a: &[u8], _s: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _n: usize, _k: usize) {
    unreachable!("WASM SIMD not available on this architecture");
}
#[cfg(not(target_arch = "wasm32"))]
pub fn wasm_matmul_quaternary(_a: &[u8], _s: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _n: usize, _k: usize) {
    unreachable!("WASM SIMD not available on this architecture");
}
