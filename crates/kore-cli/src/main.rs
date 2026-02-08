use clap::Parser;

#[derive(Parser)]
#[command(name = "kore", about = "Kore ML Framework CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Show system info (GPU, SIMD capabilities)
    Info,
    /// Run benchmarks
    Bench,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info => {
            println!("Kore ML Framework v{}", env!("CARGO_PKG_VERSION"));
            println!("Platform: {}", std::env::consts::OS);
            println!("Arch: {}", std::env::consts::ARCH);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    println!("SIMD: AVX2 ✓");
                }
                if is_x86_feature_detected!("avx512f") {
                    println!("SIMD: AVX-512 ✓");
                }
            }
        }
        Commands::Bench => {
            println!("Benchmarks not yet implemented. Use `cargo bench` for criterion benchmarks.");
        }
    }
}
