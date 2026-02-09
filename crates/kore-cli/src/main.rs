use clap::Parser;
use std::time::Instant;

use kore_core::Tensor;

#[derive(Parser)]
#[command(name = "kore", about = "Kore ML Framework CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Show system info (GPU, SIMD capabilities)
    Info,
    /// Run performance benchmarks
    Bench {
        /// Matrix sizes to benchmark (comma-separated)
        #[arg(long, default_value = "64,128,256,512,1024")]
        sizes: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info => cmd_info(),
        Commands::Bench { sizes } => cmd_bench(&sizes),
    }
}

fn cmd_info() {
    println!("Kore ML Framework v{}", env!("CARGO_PKG_VERSION"));
    println!("Platform: {} / {}", std::env::consts::OS, std::env::consts::ARCH);

    let simd = kore_kernels::SimdCapability::detect();
    println!("SIMD tier: {}", simd.best_tier());
    println!("  AVX2:    {}", if simd.avx2 { "yes" } else { "no" });
    println!("  AVX-512: {}", if simd.avx512f { "yes" } else { "no" });
    println!("  FMA:     {}", if simd.fma { "yes" } else { "no" });
    println!("  NEON:    {}", if simd.neon { "yes" } else { "no" });

    println!("\nDTypes: f16, bf16, f32, f64, i8, u8, i32, i64, ternary, quaternary");
    println!("Crates: core, autograd, nn, optim, btes, kernels, clifford, attention, serve, python, cli");
}

fn cmd_bench(sizes_str: &str) {
    let sizes: Vec<usize> = sizes_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let simd = kore_kernels::SimdCapability::detect();
    println!("=== Kore Matmul Benchmark ===");
    println!("SIMD: {} (avx2={}, avx512={}, fma={})\n",
        simd.best_tier(), simd.avx2, simd.avx512f, simd.fma);

    println!("{:<14} {:>12} {:>12} {:>9} {:>12} {:>10}",
        "Size", "Naive (ms)", "Tiled (ms)", "Speedup", "Quat (ms)", "Quat GF/s");
    println!("{}", "-".repeat(73));

    for &sz in &sizes {
        let (m, n, k) = (sz, sz, sz);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i * 11 + 5) % 17) as f32 * 0.1 - 0.8).collect();

        let a = Tensor::from_f32(&a_data, &[m, k]);
        let b = Tensor::from_f32(&b_data, &[k, n]);

        let iters = if sz <= 128 { 500 } else if sz <= 256 { 50 } else if sz <= 512 { 10 } else { 3 };

        // Warmup
        let _ = a.matmul(&b);
        let _ = kore_kernels::cpu_matmul::matmul_f32(&a, &b);

        let naive_s = time_it(iters, || { let _ = a.matmul(&b); });
        let tiled_s = time_it(iters, || { let _ = kore_kernels::cpu_matmul::matmul_f32(&a, &b); });

        let (packed, scales) = kore_kernels::cpu_quat_matmul::pack_weights_quaternary(&a_data, m, k);
        let quat_s = time_it(iters, || {
            let _ = kore_kernels::cpu_quat_matmul::quat_matmul(&packed, &scales, &b, m, n, k);
        });

        let speedup = naive_s / tiled_s;
        let quat_gflops = (2.0 * m as f64 * n as f64 * k as f64) / quat_s / 1e9;

        println!("{:<14} {:>10.3}ms {:>10.3}ms {:>8.1}x {:>10.3}ms {:>9.2}",
            format!("{}x{}", sz, sz),
            naive_s * 1000.0,
            tiled_s * 1000.0,
            speedup,
            quat_s * 1000.0,
            quat_gflops,
        );
    }

    // Flash Attention benchmark
    println!("\n=== Flash Attention Benchmark ===\n");
    println!("{:<14} {:>14} {:>14} {:>10}",
        "SeqLen", "Standard (ms)", "Flash (ms)", "Speedup");
    println!("{}", "-".repeat(56));

    for &seq_len in &[64, 128, 256, 512] {
        let d = 64;
        let data: Vec<f32> = (0..seq_len * d).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6).collect();
        let q = Tensor::from_f32(&data, &[seq_len, d]);
        let mask = kore_attention::mask::causal_mask(seq_len);

        let iters = if seq_len <= 128 { 50 } else if seq_len <= 256 { 10 } else { 3 };

        let std_s = time_it(iters, || {
            let _ = kore_attention::scaled_dot::scaled_dot_product_attention(
                &q, &q, &q, Some(&mask), None,
            );
        });

        let flash_s = time_it(iters, || {
            let _ = kore_attention::flash::flash_attention(&q, &q, &q, true);
        });

        println!("{:<14} {:>12.3}ms {:>12.3}ms {:>9.1}x",
            format!("seq={}", seq_len),
            std_s * 1000.0,
            flash_s * 1000.0,
            std_s / flash_s,
        );
    }
}

fn time_it(iters: usize, mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() / iters as f64
}
