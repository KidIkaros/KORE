use clap::Parser;
use std::time::Instant;
use serde::Deserialize;

use kore_core::Tensor;

const BANNER: &str = r#"
 _  _____  ____  _____
| |/ / _ \|  _ \| ____|
| ' / | | | |_) |  _|
| . \ |_| |  _ <| |___
|_|\_\___/|_| \_\_____|"#;

#[derive(Parser)]
#[command(
    name = "kore",
    about = "Kore ML Engine CLI",
    long_about = "A pure-Rust ML engine — from training to edge inference.\n\nBuild any architecture on top of KORE primitives: tensors, autograd,\nquantization, attention, and more. Model implementations live in Xura.",
    version,
)]
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
    /// Start the inference server (placeholder mode — load models via downstream crates)
    Serve {
        /// Address to bind to
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: String,
    },
    /// Export a model to .koref format for edge inference
    Export {
        /// Path to model directory (with config.json + .safetensors)
        #[arg(long)]
        model: String,
        /// Output .koref file path
        #[arg(long, default_value = "model.koref")]
        output: String,
        /// Quantization: f32, ternary, quaternary
        #[arg(long, default_value = "f32")]
        quantize: String,
    },
    /// Train a model (demo with synthetic data)
    Train {
        /// Number of training steps
        #[arg(long, default_value = "100")]
        steps: usize,
        /// Learning rate
        #[arg(long, default_value = "0.001")]
        lr: f32,
        /// LR scheduler: cosine, warmup_cosine, one_cycle, step, none
        #[arg(long, default_value = "warmup_cosine")]
        scheduler: String,
        /// Warmup fraction (for warmup_cosine and one_cycle)
        #[arg(long, default_value = "0.1")]
        warmup_pct: f32,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info => cmd_info(),
        Commands::Bench { sizes } => cmd_bench(&sizes),
        Commands::Serve { addr } => cmd_serve(&addr),
        Commands::Train { steps, lr, scheduler, warmup_pct } => cmd_train(steps, lr, &scheduler, warmup_pct),
        Commands::Export { model, output, quantize } => cmd_export(&model, &output, &quantize),
    }
}

fn cmd_info() {
    println!("{}", BANNER);
    println!("  v{}  —  A pure-Rust ML engine\n", env!("CARGO_PKG_VERSION"));

    println!("Platform");
    println!("  OS:   {}", std::env::consts::OS);
    println!("  Arch: {}", std::env::consts::ARCH);

    let simd = kore_kernels::SimdCapability::detect();
    println!("\nSIMD (tier: {})", simd.best_tier());
    println!("  AVX2:    {}", if simd.avx2 { "[x]" } else { "[ ]" });
    println!("  AVX-512: {}", if simd.avx512f { "[x]" } else { "[ ]" });
    println!("  FMA:     {}", if simd.fma { "[x]" } else { "[ ]" });
    println!("  NEON:    {}", if simd.neon { "[x]" } else { "[ ]" });

    println!("\nDTypes");
    println!("  float:    f16, bf16, f32, f64");
    println!("  int:      i8, u8, i32, i64");
    println!("  quantized: ternary (1.58-bit), quaternary (2-bit)");

    println!("\nCrates (13)");
    let crates = [
        ("core",      "Tensor, DType, Device, Storage"),
        ("autograd",  "Computation graph, backward pass"),
        ("nn",        "Layers, modules, sampler"),
        ("optim",     "Adam, SGD, LR schedulers"),
        ("btes",      "Ternary/quaternary encoding"),
        ("kernels",   "CUDA + CPU SIMD kernels"),
        ("clifford",  "Geometric algebra"),
        ("attention", "Flash attention, KV-cache"),
        ("edge",      "No-std inference runtime"),
        ("data",      "Dataset utilities"),
        ("serve",     "Inference server (OpenAI API)"),
        ("python",    "PyO3 bindings"),
        ("cli",       "This CLI"),
    ];
    for (name, desc) in crates {
        println!("  kore-{:<10} {}", name, desc);
    }
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

fn cmd_serve(addr: &str) {
    tracing_subscriber::fmt::init();

    let state = kore_serve::state::AppState::empty();

    println!("{}", BANNER);
    println!("  v{}  —  Inference Server\n", env!("CARGO_PKG_VERSION"));
    println!("  Listening on {}", addr);
    println!("  Mode: placeholder (no model loaded)\n");
    println!("  Endpoints:");
    println!("    POST /v1/completions");
    println!("    POST /v1/chat/completions");
    println!("    GET  /v1/models");
    println!("    GET  /health\n");
    println!("  To serve a model, implement kore_serve::InferenceModel:");
    println!("    use kore_serve::{{InferenceModel, state::AppState}};");
    println!("    let state = AppState::with_model(my_model, name);");
    println!("    kore_serve::server::serve_with_state(addr, state).await;\n");

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(async {
        if let Err(e) = kore_serve::server::serve_with_state(addr, state).await {
            eprintln!("Server error: {}", e);
        }
    });
}

fn cmd_train(steps: usize, lr: f32, scheduler_name: &str, warmup_pct: f32) {
    use kore_optim::{LrScheduler, CosineAnnealing, WarmupCosine, OneCycle, StepDecay};

    println!("=== Kore Training Demo ===");
    println!("Steps: {}, LR: {}, Scheduler: {}", steps, lr, scheduler_name);

    // Build scheduler
    let sched: Box<dyn LrScheduler> = match scheduler_name {
        "cosine" => Box::new(CosineAnnealing::new(lr, lr * 0.01, steps)),
        "warmup_cosine" => {
            let warmup = (steps as f32 * warmup_pct) as usize;
            Box::new(WarmupCosine::new(lr * 0.01, lr, lr * 0.01, warmup, steps))
        }
        "one_cycle" => Box::new(OneCycle::new(lr, steps, warmup_pct)),
        "step" => Box::new(StepDecay::new(lr, 0.5, steps / 5, steps)),
        _ => {
            println!("No scheduler, using constant LR");
            Box::new(CosineAnnealing::new(lr, lr, steps)) // constant
        }
    };

    // Create a tiny model and synthetic data
    let in_dim = 16;
    let out_dim = 4;
    let layer = kore_nn::Linear::new(in_dim, out_dim, true);

    // Synthetic input/target
    let x_data: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.1) - 0.8).collect();
    let target: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let x = Tensor::from_f32(&x_data, &[1, in_dim]);
    let t = Tensor::from_f32(&target, &[1, out_dim]);

    println!("\n{:<8} {:>10} {:>10}", "Step", "Loss", "LR");
    println!("{}", "-".repeat(30));

    let _opt = kore_optim::Adam::default_with_lr(lr);

    for step in 0..steps {
        let current_lr = sched.get_lr(step);

        // Forward
        let y = kore_nn::Module::forward(&layer, &x).expect("forward failed");

        // MSE loss
        let diff = y.sub(&t).expect("sub failed");
        let sq = diff.mul(&diff).expect("mul failed");
        let loss_val = sq.as_f32_slice().unwrap().iter().sum::<f32>() / out_dim as f32;

        // Compute gradient (manual for demo: d_loss/d_y = 2*(y-t)/n)
        let _grad_y = diff.mul_scalar(2.0 / out_dim as f32).unwrap();

        // Approximate param gradients via finite differences would be complex;
        // for this demo, just show the training loop structure
        if step % (steps / 10).max(1) == 0 || step == steps - 1 {
            println!("{:<8} {:>10.6} {:>10.6}", step, loss_val, current_lr);
        }
    }

    println!("\nTraining complete.");
}

/// Generic HuggingFace config.json fields used for .koref export.
#[derive(Deserialize, Default)]
struct HfExportConfig {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    intermediate_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
}

fn cmd_export(model_path: &str, output_path: &str, quantize: &str) {
    use kore_edge::format::{KorefBuilder, EdgeDType};
    use std::path::Path;

    println!("=== Kore Export → .koref ===");
    println!("Model:    {}", model_path);
    println!("Output:   {}", output_path);
    println!("Quantize: {}", quantize);
    println!();

    let model_dir = Path::new(model_path);

    // Load config.json
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        eprintln!("Error: config.json not found at {}", config_path.display());
        eprintln!("Expected a HuggingFace model directory with config.json + *.safetensors");
        return;
    }

    let config: HfExportConfig = match std::fs::read_to_string(&config_path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            return;
        }
    };

    let vocab_size = config.vocab_size.unwrap_or(32000);
    let d_model = config.hidden_size.unwrap_or(4096);
    let n_heads = config.num_attention_heads.unwrap_or(32);
    let n_kv_heads = config.num_key_value_heads.unwrap_or(n_heads);
    let n_layers = config.num_hidden_layers.unwrap_or(32);
    let d_ff = config.intermediate_size.unwrap_or(11008);
    let max_seq_len = config.max_position_embeddings.unwrap_or(2048);
    let norm_eps = config.rms_norm_eps.unwrap_or(1e-5) as f32;
    let rope_base = config.rope_theta.unwrap_or(10000.0) as f32;

    println!("Config: vocab={} d={} heads={} kv_heads={} layers={} ff={} max_seq={}",
        vocab_size, d_model, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len);

    let _target_dtype = match quantize {
        "f32" => EdgeDType::F32,
        "f16" => EdgeDType::F16,
        "ternary" => EdgeDType::Ternary,
        "quaternary" => EdgeDType::Quaternary,
        other => {
            eprintln!("Unknown quantization: {}. Use f32, f16, ternary, or quaternary.", other);
            return;
        }
    };

    let mut builder = KorefBuilder::new(
        "llama", vocab_size, d_model, n_heads, n_kv_heads,
        n_layers, d_ff, max_seq_len, norm_eps, rope_base,
    );

    // Find safetensors files
    let st_files: Vec<_> = std::fs::read_dir(model_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();

    if st_files.is_empty() {
        eprintln!("No .safetensors files found in {}", model_path);
        eprintln!("Creating demo .koref with random weights instead...");

        // Demo: create a tiny model
        let demo_builder = KorefBuilder::new(
            "demo", 256, 64, 4, 4, 2, 128, 128, 1e-5, 10000.0,
        );
        let model = demo_builder.build();
        let bytes = model.to_bytes();
        std::fs::write(output_path, &bytes).expect("Failed to write .koref");
        println!("Wrote demo .koref ({} bytes) to {}", bytes.len(), output_path);
        return;
    }

    println!("Found {} safetensors file(s)", st_files.len());

    let mut tensor_count = 0usize;
    let mut total_bytes = 0usize;

    for st_path in &st_files {
        let data = std::fs::read(st_path).expect("Failed to read safetensors");
        let tensors = safetensors::SafeTensors::deserialize(&data).expect("Failed to parse safetensors");

        for (name, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let raw = view.data();

            // For now, store as f32 (quantization conversion would go here)
            builder.add_tensor(&name, EdgeDType::F32, &shape, raw);
            total_bytes += raw.len();
            tensor_count += 1;
        }
    }

    let model = builder.build();
    let bytes = model.to_bytes();
    std::fs::write(output_path, &bytes).expect("Failed to write .koref");

    println!();
    println!("Export complete:");
    println!("  Tensors:    {}", tensor_count);
    println!("  Weight data: {:.1} MB", total_bytes as f64 / (1024.0 * 1024.0));
    println!("  .koref size: {:.1} MB", bytes.len() as f64 / (1024.0 * 1024.0));
    println!("  Output:     {}", output_path);

    // Print estimated edge memory
    let plan = kore_edge::plan::ExecutionPlan::from_header(&model.header);
    println!("  Est. runtime memory: {:.1} MB", plan.peak_memory_mb());
}

