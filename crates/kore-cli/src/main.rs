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
    /// Start the inference server
    Serve {
        /// Address to bind to
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: String,
        /// Path to model directory (with config.json + .safetensors)
        #[arg(long)]
        model: Option<String>,
    },
    /// Generate text with a tiny random model (demo)
    Generate {
        /// Prompt text
        #[arg(long, default_value = "Hello world")]
        prompt: String,
        /// Number of tokens to generate
        #[arg(long, default_value = "32")]
        max_tokens: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info => cmd_info(),
        Commands::Bench { sizes } => cmd_bench(&sizes),
        Commands::Serve { addr, model } => cmd_serve(&addr, model.as_deref()),
        Commands::Generate { prompt, max_tokens } => cmd_generate(&prompt, max_tokens),
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

fn cmd_serve(addr: &str, model_path: Option<&str>) {
    tracing_subscriber::fmt::init();

    let state = if let Some(path) = model_path {
        let model_dir = std::path::Path::new(path);
        println!("Loading model from {}...", path);
        match kore_transformer::loader::load_model(model_dir) {
            Ok(model) => {
                println!("Model loaded: {} params", model.param_count());
                kore_serve::state::AppState::with_model(model, path.to_string())
            }
            Err(e) => {
                eprintln!("Failed to load model: {}. Starting in placeholder mode.", e);
                kore_serve::state::AppState::empty()
            }
        }
    } else {
        println!("No --model specified. Starting with random tiny model for demo.");
        let config = kore_transformer::TransformerConfig::tiny();
        let model = kore_transformer::Transformer::new(config);
        println!("Tiny model: {} params", model.param_count());
        kore_serve::state::AppState::with_model(model, "kore-tiny-random".to_string())
    };

    println!("Starting Kore inference server on {}", addr);
    println!("  POST /v1/completions");
    println!("  POST /v1/chat/completions");
    println!("  GET  /health");

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(async {
        if let Err(e) = kore_serve::server::serve_with_state(addr, state).await {
            eprintln!("Server error: {}", e);
        }
    });
}

fn cmd_generate(prompt: &str, max_tokens: usize) {
    let config = kore_transformer::TransformerConfig::tiny();
    let mut model = kore_transformer::Transformer::new(config.clone());

    println!("=== Kore Generate (tiny random model) ===");
    println!("Config: vocab={}, d_model={}, n_heads={}, n_layers={}, params={}",
        config.vocab_size, config.d_model, config.n_heads, config.n_layers, model.param_count());
    println!("Prompt: {:?}", prompt);
    println!();

    // Byte-level tokenization
    let prompt_ids: Vec<usize> = prompt.bytes().map(|b| b as usize).collect();

    let start = Instant::now();
    match model.generate(&prompt_ids, max_tokens) {
        Ok(full_seq) => {
            let elapsed = start.elapsed();
            let generated = &full_seq[prompt_ids.len()..];
            let text: String = generated.iter()
                .map(|&id| if id < 128 { id as u8 as char } else { '?' })
                .collect();

            println!("Generated {} tokens in {:.1}ms ({:.1} tok/s)",
                generated.len(),
                elapsed.as_secs_f64() * 1000.0,
                generated.len() as f64 / elapsed.as_secs_f64(),
            );
            println!("Output: {:?}", text);
            println!("Token IDs: {:?}", &generated[..generated.len().min(20)]);
        }
        Err(e) => {
            eprintln!("Generation error: {}", e);
        }
    }
}
