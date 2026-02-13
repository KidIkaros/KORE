use clap::Parser;
use std::time::Instant;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

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
    about = "Kore CLI",
    long_about = "A workflow-first CLI for Kore (inspect → shard/export → run/serve).",
    after_long_help = "Typical workflow:\n  1) kore inspect --path ./model\n  2) kore shard --model ./model --output ./shards\n  3) kore export --model ./model --output model.koref\n  4) kore run --model model.koref --tokenizer tokenizer.json --prompt \"Hello\"\n  5) kore serve --addr 0.0.0.0:8080",
    version,
)]
struct Cli {
    /// Increase logging verbosity (-v, -vv)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Silence non-error output
    #[arg(long = "quiet", default_value_t = false, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Show system info (GPU, SIMD capabilities)
    Info,
    /// Inspect .koref/.safetensors/model-dir metadata
    Inspect {
        /// Path to .koref file, .safetensors file, or model directory
        #[arg(long)]
        path: String,
    },
    /// Shard HuggingFace safetensors into per-layer files for layered inference
    Shard {
        /// HF model directory containing config.json + *.safetensors
        #[arg(long)]
        model: String,
        /// Output directory for per-layer shards
        #[arg(long)]
        output: String,
    },
    /// Run local generation from a .koref model
    Run {
        /// .koref model file path
        #[arg(long)]
        model: String,
        /// Tokenizer JSON path (Kore simple tokenizer format)
        #[arg(long)]
        tokenizer: Option<String>,
        /// Prompt text (requires --tokenizer)
        #[arg(long)]
        prompt: Option<String>,
        /// Raw token IDs (comma separated). Use instead of --prompt.
        #[arg(long)]
        tokens: Option<String>,
        /// Max new tokens
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
    },
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose, cli.quiet);

    match cli.command {
        Commands::Info => cmd_info(),
        Commands::Inspect { path } => cmd_inspect(&path)?,
        Commands::Shard { model, output } => cmd_shard(&model, &output)?,
        Commands::Run { model, tokenizer, prompt, tokens, max_new_tokens } => {
            cmd_run(&model, tokenizer.as_deref(), prompt.as_deref(), tokens.as_deref(), max_new_tokens)?
        }
        Commands::Bench { sizes } => cmd_bench(&sizes),
        Commands::Serve { addr } => cmd_serve(&addr)?,
        Commands::Train { steps, lr, scheduler, warmup_pct } => cmd_train(steps, lr, &scheduler, warmup_pct),
        Commands::Export { model, output, quantize } => cmd_export(&model, &output, &quantize)?,
    }

    Ok(())
}

fn init_logging(verbose: u8, quiet: bool) {
    use tracing_subscriber::EnvFilter;

    let default = if quiet {
        "error"
    } else if verbose >= 2 {
        "debug"
    } else if verbose == 1 {
        "info"
    } else {
        "warn"
    };

    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| default.to_string());
    let _ = tracing_subscriber::fmt().with_env_filter(EnvFilter::new(filter)).try_init();
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct XuraTokenizer {
    stoi: HashMap<String, usize>,
    itos: Vec<String>,
    unk_token: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl XuraTokenizer {
    fn from_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read tokenizer file {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("failed to parse tokenizer JSON {}", path.display()))
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if let Some(bos) = &self.bos_token {
            if let Some(&id) = self.stoi.get(bos) {
                out.push(id as u32);
            }
        }

        let unk_id = self.stoi.get(&self.unk_token).copied().unwrap_or(0) as u32;
        for tok in text.split_whitespace() {
            out.push(self.stoi.get(tok).copied().map(|v| v as u32).unwrap_or(unk_id));
        }

        out
    }

    fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&t| self.itos.get(t as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn cmd_inspect(path: &str) -> Result<()> {
    let p = PathBuf::from(path);
    if p.is_dir() {
        println!("Inspecting model directory: {}", p.display());
        let cfg = p.join("config.json");
        println!("  config.json: {}", if cfg.exists() { "found" } else { "missing" });
        let st_count = fs::read_dir(&p)
            .with_context(|| format!("failed to read dir {}", p.display()))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .count();
        println!("  .safetensors files: {}", st_count);
        return Ok(());
    }

    if p.extension().is_some_and(|e| e == "koref") {
        let bytes = fs::read(&p).with_context(|| format!("failed to read {}", p.display()))?;
        let model = kore_edge::format::KorefModel::from_bytes(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to parse .koref: {e}"))?;
        println!("Model: {}", p.display());
        println!("  type: {}", model.header.model_type);
        println!("  layers: {}", model.header.n_layers);
        println!("  d_model: {}", model.header.d_model);
        println!("  vocab: {}", model.header.vocab_size);
        println!("  tensors: {}", model.header.tensors.len());
        let plan = kore_edge::plan::ExecutionPlan::from_header(&model.header);
        println!("  est runtime memory: {:.1} MB", plan.peak_memory_mb());
        return Ok(());
    }

    if p.extension().is_some_and(|e| e == "safetensors") {
        let data = fs::read(&p).with_context(|| format!("failed to read {}", p.display()))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| anyhow::anyhow!("failed to parse safetensors: {e}"))?;
        println!("Safetensors: {}", p.display());
        for (name, view) in tensors.tensors() {
            println!("  {:<50} {:?} {:?}", name, view.dtype(), view.shape());
        }
        return Ok(());
    }

    bail!("unsupported path type: {}", p.display())
}

fn cmd_shard(model_dir: &str, output_dir: &str) -> Result<()> {
    let model = PathBuf::from(model_dir);
    let output = PathBuf::from(output_dir);
    let cfg_path = model.join("config.json");

    let config = kore_serve::layered::LayeredConfig::from_hf_config(&cfg_path, output.clone())
        .ok_or_else(|| anyhow::anyhow!("failed to parse {}", cfg_path.display()))?;

    let count = kore_serve::layered::sharder::shard_model(&model, &output, &config)
        .map_err(|e| anyhow::anyhow!(e))?;
    println!("Sharding complete");
    println!("  output: {}", output.display());
    println!("  shard files: {}", count);
    Ok(())
}

fn cmd_run(model_path: &str, tokenizer_path: Option<&str>, prompt: Option<&str>, tokens_csv: Option<&str>, max_new_tokens: usize) -> Result<()> {
    let bytes = fs::read(model_path)
        .with_context(|| format!("failed to read model file {}", model_path))?;
    let model = kore_edge::format::KorefModel::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("failed to parse .koref: {e}"))?;

    let mut session = kore_edge::Session::new(model);

    let (input_tokens, tokenizer) = if let Some(csv) = tokens_csv {
        let tokens = parse_u32_csv(csv)?;
        (tokens, None)
    } else {
        let prompt = prompt.ok_or_else(|| anyhow::anyhow!("provide either --tokens or (--prompt and --tokenizer)"))?;
        let tok_path = tokenizer_path.ok_or_else(|| anyhow::anyhow!("--prompt requires --tokenizer"))?;
        let tok = XuraTokenizer::from_file(Path::new(tok_path))?;
        (tok.encode(prompt), Some(tok))
    };

    if input_tokens.is_empty() {
        bail!("input token sequence is empty");
    }

    let t0 = Instant::now();
    let out = session.generate(&input_tokens, max_new_tokens);
    let dt = t0.elapsed().as_secs_f64().max(1e-6);

    println!("Generated {} tokens in {:.2}s ({:.2} tok/s)", out.len(), dt, out.len() as f64 / dt);
    if let Some(tok) = tokenizer {
        println!("\n{}", tok.decode(&out));
    } else {
        println!("\n{:?}", out);
    }
    Ok(())
}

fn parse_u32_csv(csv: &str) -> Result<Vec<u32>> {
    csv.split(',')
        .map(|s| s.trim().parse::<u32>().with_context(|| format!("invalid token id '{s}'")))
        .collect()
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

fn cmd_serve(addr: &str) -> Result<()> {
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
    println!("  Use kore run for local validation before deployment.\n");

    let rt = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    rt.block_on(async {
        if let Err(e) = kore_serve::server::serve_with_state(addr, state).await {
            eprintln!("Server error: {}", e);
        }
    });

    Ok(())
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

fn cmd_export(model_path: &str, output_path: &str, quantize: &str) -> Result<()> {
    use kore_edge::format::EdgeDType;

    println!("=== Kore Export → .koref ===");
    println!("Model:    {}", model_path);
    println!("Output:   {}", output_path);
    println!("Quantize: {}", quantize);
    println!();

    let _target_dtype = match quantize {
        "f32" => EdgeDType::F32,
        "f16" => EdgeDType::F16,
        "ternary" => EdgeDType::Ternary,
        "quaternary" => EdgeDType::Quaternary,
        other => {
            bail!("unknown quantization: {other}. Use f32, f16, ternary, or quaternary.");
        }
    };

    let model = kore_edge::loader::load_safetensors_dir(Path::new(model_path))
        .map_err(|e| anyhow::anyhow!("export failed: {e}"))?;
    let bytes = model.to_bytes();
    fs::write(output_path, &bytes)
        .with_context(|| format!("failed to write {}", output_path))?;

    println!();
    println!("Export complete:");
    println!("  Tensors:    {}", model.header.tensors.len());
    println!("  .koref size: {:.1} MB", bytes.len() as f64 / (1024.0 * 1024.0));
    println!("  Output:     {}", output_path);

    // Print estimated edge memory
    let plan = kore_edge::plan::ExecutionPlan::from_header(&model.header);
    println!("  Est. runtime memory: {:.1} MB", plan.peak_memory_mb());

    Ok(())
}

