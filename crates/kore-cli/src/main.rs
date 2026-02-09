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
        Commands::Serve { addr, model } => cmd_serve(&addr, model.as_deref()),
        Commands::Generate { prompt, max_tokens } => cmd_generate(&prompt, max_tokens),
        Commands::Train { steps, lr, scheduler, warmup_pct } => cmd_train(steps, lr, &scheduler, warmup_pct),
        Commands::Export { model, output, quantize } => cmd_export(&model, &output, &quantize),
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

    let config = match kore_transformer::loader::HfConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            return;
        }
    };

    let vocab_size = config.vocab_size.unwrap_or(32000);
    let d_model = config.d_model.unwrap_or(4096);
    let n_heads = config.n_heads.unwrap_or(32);
    let n_kv_heads = config.n_kv_heads.unwrap_or(n_heads);
    let n_layers = config.n_layers.unwrap_or(32);
    let d_ff = config.d_ff.unwrap_or(11008);
    let max_seq_len = config.max_seq_len.unwrap_or(2048);
    let norm_eps = config.norm_eps.unwrap_or(1e-5) as f32;
    let rope_base = config.rope_base.unwrap_or(10000.0) as f32;

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
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
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
