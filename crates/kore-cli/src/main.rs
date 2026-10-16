mod commands;

use clap::Parser;
use commands::*;

#[derive(Parser)]
#[command(
    name = "kore",
    about = "Kore ML Engine CLI",
    long_about = "A pure-Rust ML engine — from training to edge inference.\n\nBuild any architecture on top of KORE primitives: tensors, autograd,\nquantization, attention, and more. Model implementations live in Xura.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InferArch {
    /// Automatically select transformer if prompt/tokenizer/mtp is provided, otherwise transformer
    #[default]
    Auto,
    /// Autoregressive Transformer with KV-cache and MTP speculative decoding
    Transformer,
    /// SqueezeNet v1.1 CNN image classifier
    Squeezenet,
}

impl std::fmt::Display for InferArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferArch::Auto => write!(f, "auto"),
            InferArch::Transformer => write!(f, "transformer"),
            InferArch::Squeezenet => write!(f, "squeezenet"),
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QuantizeMode {
    #[default]
    F32,
    F16,
    Ternary,
    Quaternary,
}

impl std::fmt::Display for QuantizeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizeMode::F32 => write!(f, "f32"),
            QuantizeMode::F16 => write!(f, "f16"),
            QuantizeMode::Ternary => write!(f, "ternary"),
            QuantizeMode::Quaternary => write!(f, "quaternary"),
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvCacheCliMode {
    #[default]
    F32,
    Q8,
    K8v4,
}

impl std::fmt::Display for KvCacheCliMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheCliMode::F32 => write!(f, "f32"),
            KvCacheCliMode::Q8 => write!(f, "q8"),
            KvCacheCliMode::K8v4 => write!(f, "k8v4"),
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SchedulerCli {
    #[default]
    WarmupCosine,
    Cosine,
    OneCycle,
    Step,
    None,
}

impl std::fmt::Display for SchedulerCli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerCli::WarmupCosine => write!(f, "warmup_cosine"),
            SchedulerCli::Cosine => write!(f, "cosine"),
            SchedulerCli::OneCycle => write!(f, "one_cycle"),
            SchedulerCli::Step => write!(f, "step"),
            SchedulerCli::None => write!(f, "none"),
        }
    }
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Show system info (GPU backends, SIMD capabilities, dtypes)
    Info,
    /// Run performance benchmarks across CPU and GPU
    Bench {
        /// Matrix sizes to benchmark (comma-separated)
        #[arg(long, default_value = "64,128,256,512,1024")]
        sizes: String,
        /// Target device: auto, cpu, vulkan, cuda
        #[arg(long, default_value = "auto")]
        device: String,
    },
    /// Start the inference server with OpenAI-compatible REST API
    Serve {
        /// Address to bind to
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: String,
        /// Optional path to model (.safetensors file or layered shard directory)
        #[arg(long)]
        model: Option<String>,
        /// Target device: auto, cpu, vulkan[:idx], cuda[:idx]
        #[arg(long, default_value = "auto")]
        device: String,
    },
    /// Export a model to .koref format for edge inference
    Export {
        /// Path to model directory (with config.json + .safetensors)
        #[arg(long)]
        model: String,
        /// Output .koref file path
        #[arg(long, default_value = "model.koref")]
        output: String,
        /// Quantization: f32, f16, ternary (1.58-bit), quaternary (2-bit)
        #[arg(long, value_enum, default_value_t = QuantizeMode::F32)]
        quantize: QuantizeMode,
    },
    /// Run a forward pass or generate tokens — the research-loop smoke test
    Infer {
        /// Architecture: auto (infer from flags), transformer, squeezenet
        #[arg(long, value_enum, default_value_t = InferArch::Auto)]
        arch: InferArch,
        /// Target device: auto, gpu, cpu, vulkan[:idx], cuda[:idx]
        #[arg(long, default_value = "auto")]
        device: String,
        /// Path to a .safetensors model file. If omitted, runs a random-init model.
        #[arg(long)]
        model: Option<String>,
        /// Number of classes (for squeezenet) or vocabulary size (for transformer)
        #[arg(long, visible_alias = "vocab-size")]
        classes: Option<usize>,
        /// Input image size (square, for squeezenet) [default: 224]
        #[arg(long, default_value = "224")]
        size: usize,
        /// Hidden dimension (for transformer) [default: 64]
        #[arg(long, default_value = "64")]
        hidden_dim: usize,
        /// Attention head dimension (for transformer) [default: 16]
        #[arg(long, default_value = "16")]
        head_dim: usize,
        /// Initial sequence length (if no prompt) [default: 8]
        #[arg(long, default_value = "8")]
        seq_len: usize,
        /// Number of new tokens to generate (for transformer) [default: 10]
        #[arg(long, short = 'n', visible_aliases = ["max-tokens", "max_tokens"], default_value = "10")]
        max_new_tokens: usize,
        /// Sampling temperature (0.0 for greedy) [default: 0.7]
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Sampling top-k [default: 40]
        #[arg(long, default_value = "40")]
        top_k: usize,
        /// Sampling top-p [default: 0.9]
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Text prompt or comma-separated token IDs for transformer
        #[arg(long, short = 'p')]
        prompt: Option<String>,
        /// Optional path to a tokenizer.json file for encoding prompt and decoding output
        #[arg(long)]
        tokenizer: Option<String>,
        /// Enable Multi-Token Prediction (MTP) speculative decoding
        #[arg(
            long,
            help = "Enable Multi-Token Prediction (MTP) speculative decoding"
        )]
        mtp: bool,
        /// Number of transformer layers to load (useful for bounded memory or quick verification)
        #[arg(long)]
        layers: Option<usize>,
        /// Number of layers to offload to GPU in hybrid GPU+CPU mode (default: all layers when on GPU)
        #[arg(long, visible_alias = "ngl")]
        gpu_layers: Option<usize>,
        /// KV-cache quantization mode: f32, q8 (INT8), k8v4 (K INT8, V Hadamard INT4)
        #[arg(long, value_enum, default_value_t = KvCacheCliMode::F32)]
        kv_cache: KvCacheCliMode,
    },
    /// Train a model (demo with synthetic data)
    Train {
        /// Number of training steps
        #[arg(long, default_value = "100")]
        steps: usize,
        /// Learning rate
        #[arg(long, default_value = "0.001")]
        lr: f32,
        /// LR scheduler: warmup_cosine, cosine, one_cycle, step, none
        #[arg(long, value_enum, default_value_t = SchedulerCli::WarmupCosine)]
        scheduler: SchedulerCli,
        /// Warmup fraction (for warmup_cosine and one_cycle)
        #[arg(long, default_value = "0.1")]
        warmup_pct: f32,
    },
}

fn main() {
    // Register available GPU backends
    let _ = kore_vulkan::register_vulkan_backend();
    #[cfg(feature = "cuda")]
    kore_kernels::cuda::register_cuda_backend();

    let cli = Cli::parse();

    match cli.command {
        Commands::Info => info::cmd_info(),
        Commands::Bench { sizes, device } => bench::cmd_bench(&sizes, &device),
        Commands::Serve {
            addr,
            model,
            device,
        } => serve::cmd_serve(&addr, model, &device),
        Commands::Train {
            steps,
            lr,
            scheduler,
            warmup_pct,
        } => train::cmd_train(steps, lr, &scheduler.to_string(), warmup_pct),
        Commands::Export {
            model,
            output,
            quantize,
        } => export::cmd_export(&model, &output, &quantize.to_string()),
        Commands::Infer {
            arch,
            device,
            model,
            classes,
            size,
            hidden_dim,
            head_dim,
            seq_len,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            prompt,
            tokenizer,
            mtp,
            layers,
            gpu_layers,
            kv_cache,
        } => infer::cmd_infer(
            &arch.to_string(),
            &device,
            model,
            classes,
            size,
            hidden_dim,
            head_dim,
            seq_len,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            prompt,
            tokenizer,
            mtp,
            layers,
            gpu_layers,
            &kv_cache.to_string(),
        ),
    }
}
