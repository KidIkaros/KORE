//! Benchmark: kore-kernels tiled AVX2 matmul vs kore-core naive matmul.

use kore_core::Tensor;
use kore_kernels::cpu_matmul::matmul_f32;
use std::time::Instant;

fn bench_naive(a: &Tensor, b: &Tensor, iters: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let _ = a.matmul(b).unwrap();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn bench_tiled(a: &Tensor, b: &Tensor, iters: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let _ = matmul_f32(a, b).unwrap();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn bench_quat(weights: &[f32], b: &Tensor, m: usize, n: usize, k: usize, iters: usize) -> f64 {
    let (packed, scales) = kore_kernels::cpu_quat_matmul::pack_weights_quaternary(weights, m, k);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = kore_kernels::cpu_quat_matmul::quat_matmul(&packed, &scales, b, m, n, k).unwrap();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn gflops(m: usize, n: usize, k: usize, secs: f64) -> f64 {
    (2.0 * m as f64 * n as f64 * k as f64) / secs / 1e9
}

fn main() {
    let simd = kore_kernels::SimdCapability::detect();
    println!("=== Kore Matmul Benchmark ===");
    println!("SIMD: {} (avx2={}, avx512={}, fma={})\n",
        simd.best_tier(), simd.avx2, simd.avx512f, simd.fma);

    let sizes: &[(usize, usize, usize)] = &[
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];

    println!("{:<12} {:>12} {:>12} {:>10} {:>12} {:>10}",
        "Size", "Naive (ms)", "Tiled (ms)", "Speedup", "Quat (ms)", "Quat GF/s");
    println!("{}", "-".repeat(72));

    for &(m, n, k) in sizes {
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i * 11 + 5) % 17) as f32 * 0.1 - 0.8).collect();

        let a = Tensor::from_f32(&a_data, &[m, k]);
        let b = Tensor::from_f32(&b_data, &[k, n]);

        let iters = if m <= 128 { 1000 } else if m <= 256 { 100 } else if m <= 512 { 20 } else { 5 };

        let naive_s = bench_naive(&a, &b, iters);
        let tiled_s = bench_tiled(&a, &b, iters);
        let quat_s = bench_quat(&a_data, &b, m, n, k, iters);

        let speedup = naive_s / tiled_s;
        let quat_gflops = gflops(m, n, k, quat_s);

        println!("{:<12} {:>10.3}ms {:>10.3}ms {:>9.1}x {:>10.3}ms {:>9.2}",
            format!("{}x{}x{}", m, n, k),
            naive_s * 1000.0,
            tiled_s * 1000.0,
            speedup,
            quat_s * 1000.0,
            quat_gflops,
        );
    }

    // Flash Attention benchmark
    println!("\n=== Flash Attention Benchmark ===\n");
    println!("{:<12} {:>14} {:>14} {:>10}",
        "SeqLen", "Standard (ms)", "Flash (ms)", "Speedup");
    println!("{}", "-".repeat(54));

    for &seq_len in &[64, 128, 256, 512, 1024] {
        let d = 64;
        let data: Vec<f32> = (0..seq_len * d).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6).collect();
        let q = Tensor::from_f32(&data, &[seq_len, d]);
        let k = Tensor::from_f32(&data, &[seq_len, d]);
        let v = Tensor::from_f32(&data, &[seq_len, d]);

        let iters = if seq_len <= 128 { 100 } else if seq_len <= 256 { 20 } else { 5 };

        let mask = kore_attention::mask::causal_mask(seq_len);

        let start = Instant::now();
        for _ in 0..iters {
            let _ = kore_attention::scaled_dot::scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).unwrap();
        }
        let std_s = start.elapsed().as_secs_f64() / iters as f64;

        let start = Instant::now();
        for _ in 0..iters {
            let _ = kore_attention::flash::flash_attention(&q, &k, &v, true).unwrap();
        }
        let flash_s = start.elapsed().as_secs_f64() / iters as f64;

        println!("{:<12} {:>12.3}ms {:>12.3}ms {:>9.1}x",
            format!("seq={}", seq_len),
            std_s * 1000.0,
            flash_s * 1000.0,
            std_s / flash_s,
        );
    }
}
