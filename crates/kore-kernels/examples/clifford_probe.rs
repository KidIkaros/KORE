//! Falsification probe: does writing a linear operator in Clifford-algebra form
//! make it faster?
//!
//! The claim under test is "Clifford algebra can act as a better adapter for
//! tensor algebra and raise tok/s". An orthogonal transform is the fairest
//! possible test case, because it is the one place where the GA literature
//! genuinely has something to say (Cartan-Dieudonne: every orthogonal map is a
//! product of reflections, and rotors are the even subalgebra that generates
//! them). So we implement the *same operator* -- the model's normalized
//! block-Hadamard -- three ways and time them:
//!
//! 1. `butterfly` - the Fast Walsh-Hadamard Transform: O(n log n) additions only.
//! 2. `reflections` - the same matrix as a product of Householder reflections (the GA expression).
//! 3. `rotor8` - one quaternion sandwich per 8-vector (the Cl(3) even subalgebra).
//!
//! Arms 1 and 2 compute the same matrix, so they are compared on time only
//! (with a numerical agreement check). Arm 3 computes a *different* operator --
//! only an 8-dimensional family of maps -- which is measured for cost and for
//! how far it is from the operator the weights were actually folded through.
//!
//! Everything below is f32, single-threaded, one 512-wide block at a time, which
//! is the shape the qwen35 fold uses (`hadamard_block = 512`).

use kore_kernels::hadamard::fwht_inplace;
use std::time::Instant;

// The fold asks for 512-wide blocks, but `fwht_block_size` caps a butterfly at
// 256 threads, so the operator actually executed is two independent 256-wide
// transforms. Compare against what runs, not against what is requested.
const N: usize = 256;
const BLOCKS: usize = 2048;

/// Synchronous-block Hadamard matrix, normalized (`H / sqrt(n)`).
fn hadamard_matrix(n: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; n * n];
    h[0] = 1.0;
    let mut w = 1;
    while w < n {
        for i in 0..w {
            for j in 0..w {
                let v = h[i * n + j];
                h[(i + w) * n + j] = v;
                h[i * n + j + w] = v;
                h[(i + w) * n + j + w] = -v;
            }
        }
        w *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in h.iter_mut() {
        *v *= s;
    }
    h
}

/// Householder factorization `H = H_1 H_2 ... H_{n-1} R`, with `R` diagonal
/// (which it must be: a symmetric orthogonal matrix has R = diag(+-1)).
///
/// This is the Cartan-Dieudonne form of the operator: nothing but reflections.
fn householder_factors(h: &[f32], n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut a = h.to_vec();
    let mut vs: Vec<Vec<f32>> = Vec::with_capacity(n.saturating_sub(1));
    for k in 0..n.saturating_sub(1) {
        let mut v: Vec<f32> = (k..n).map(|r| a[r * n + k]).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            vs.push(vec![0.0; n - k]);
            continue;
        }
        let alpha = if v[0] >= 0.0 { -norm } else { norm };
        v[0] -= alpha;
        let vnorm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if vnorm > 0.0 {
            for x in v.iter_mut() {
                *x /= vnorm;
            }
        }
        // a[k.., k..] -= 2 v (v^T a[k.., k..])
        for c in k..n {
            let mut dot = 0.0f32;
            for r in k..n {
                dot += v[r - k] * a[r * n + c];
            }
            let twice = 2.0 * dot;
            for r in k..n {
                a[r * n + c] -= twice * v[r - k];
            }
        }
        vs.push(v);
    }
    let mut r = vec![0.0f32; n];
    for i in 0..n {
        r[i] = if a[i * n + i] >= 0.0 { 1.0 } else { -1.0 };
    }
    (vs, r)
}

fn apply_reflections(x: &[f32], vs: &[Vec<f32>], r: &[f32]) -> Vec<f32> {
    let mut y: Vec<f32> = x.iter().zip(r.iter()).map(|(a, b)| a * b).collect();
    for (k, v) in vs.iter().enumerate().rev() {
        let mut dot = 0.0f32;
        for (i, vi) in v.iter().enumerate() {
            dot += vi * y[k + i];
        }
        let twice = 2.0 * dot;
        for (i, vi) in v.iter().enumerate() {
            y[k + i] -= twice * vi;
        }
    }
    y
}

/// One quaternion (Cl(3) even subalgebra) sandwich on an 8-vector: the first
/// four components are rotated by `q`, the last four by `q2`. This is the shape
/// a GA-native layer takes: `y = q x q~`, two products per 4-vector.
fn rotor8(x: &[f32], q: &[f32; 4], q2: &[f32; 4], out: &mut [f32]) {
    for (half, qq) in [(0usize, q), (4usize, q2)] {
        let (a, b, c, d) = (qq[0], qq[1], qq[2], qq[3]);
        let (v0, v1, v2, v3) = (x[half], x[half + 1], x[half + 2], x[half + 3]);
        // t = q * v, then out = t * conj(q): a 4x4 map with 4 parameters
        // instead of 16, which is the whole point of the rotor form.
        let t0 = a * v0 - b * v1 - c * v2 - d * v3;
        let t1 = a * v1 + b * v0 + c * v3 - d * v2;
        let t2 = a * v2 - b * v3 + c * v0 + d * v1;
        let t3 = a * v3 + b * v2 - c * v1 + d * v0;
        out[half] = a * t0 + b * t1 + c * t2 + d * t3;
        out[half + 1] = a * t1 - b * t0 - c * t3 + d * t2;
        out[half + 2] = a * t2 + b * t3 - c * t0 - d * t1;
        out[half + 3] = a * t3 - b * t2 + c * t1 - d * t0;
    }
}

fn fill_block(seed: usize) -> Vec<f32> {
    (0..N)
        .map(|i| {
            let t = (seed * N + i) as f32 * 0.017;
            t.sin() * 0.9 + if i == 7 { 12.0 } else { 0.0 }
        })
        .collect()
}

fn main() {
    let h = hadamard_matrix(N);
    let (vs, r) = householder_factors(&h, N);

    // ---- fidelity: the reflection product must be the same operator ----
    let mut worst = 0.0f32;
    for s in 0..8 {
        let x = fill_block(s);
        let mut a = x.clone();
        fwht_inplace(&mut a, N);
        let b = apply_reflections(&x, &vs, &r);
        for (p, q) in a.iter().zip(b.iter()) {
            worst = worst.max((p - q).abs());
        }
    }
    println!("same-operator check (butterfly vs reflections): max |diff| = {worst:.3e}");

    // ---- arm 3 fidelity: how far a rotor8 map is from the model's operator ----
    let q = [0.6f32, 0.5, 0.5, 0.4];
    let q2 = [0.5f32, -0.5, 0.6, -0.4];
    let qn = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let q = q.map(|v| v / qn);
    let q2n = (q2.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let q2 = q2.map(|v| v / q2n);
    let mut rotor_err = 0.0f32;
    let mut rotor_norm = 0.0f32;
    {
        let x = fill_block(0);
        let mut y = vec![0.0f32; N];
        for b in 0..N / 8 {
            rotor8(&x[b * 8..b * 8 + 8], &q, &q2, &mut y[b * 8..b * 8 + 8]);
        }
        let mut a = x.clone();
        fwht_inplace(&mut a, N);
        for (p, s) in a.iter().zip(y.iter()) {
            rotor_err += (p - s) * (p - s);
            rotor_norm += p * p;
        }
        rotor_err = (rotor_err / rotor_norm).sqrt();
    }
    println!("rotor8 vs the model's fold: relative L2 error = {rotor_err:.3e}");

    // ---- timing ----
    let blocks: Vec<Vec<f32>> = (0..BLOCKS).map(fill_block).collect();

    let t = Instant::now();
    let mut sink = 0.0f32;
    let mut acc = vec![0.0f32; N];
    for b in &blocks {
        acc.copy_from_slice(b);
        fwht_inplace(&mut acc, N);
        sink += acc[0];
    }
    let butterfly = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for b in &blocks {
        let y = apply_reflections(b, &vs, &r);
        sink += y[0];
    }
    let reflections = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut out = vec![0.0f32; N];
    for b in &blocks {
        for k in 0..N / 8 {
            rotor8(&b[k * 8..k * 8 + 8], &q, &q2, &mut out[k * 8..k * 8 + 8]);
        }
        sink += out[0];
    }
    let rotor = t.elapsed().as_secs_f64();

    let per = |s: f64| s * 1e9 / BLOCKS as f64;
    println!();
    println!("{BLOCKS} blocks of {N} f32, single thread:");
    println!("  butterfly    {:8.0} ns/block", per(butterfly));
    println!(
        "  reflections  {:8.0} ns/block   ({:.1}x butterfly)",
        per(reflections),
        reflections / butterfly
    );
    println!(
        "  rotor8       {:8.0} ns/block   ({:.1}x butterfly)",
        per(rotor),
        rotor / butterfly
    );
    println!();
    println!("(sink {sink:.3} so nothing is optimized away)");
    println!("butterfly f32 error vs exact H x: {:.3e}", {
        // f64 reference for the first block
        let x = fill_block(0);
        let mut a = x.clone();
        fwht_inplace(&mut a, N);
        let mut worst = 0.0f32;
        for (i, ai) in a.iter().enumerate() {
            let mut acc = 0f64;
            for (j, xj) in x.iter().enumerate() {
                acc += *xj as f64 * h[i * N + j] as f64;
            }
            worst = worst.max((*ai as f64 - acc).abs() as f32);
        }
        worst
    });
}
