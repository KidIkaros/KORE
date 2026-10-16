use super::BANNER;

pub fn cmd_info() {
    println!("{}", BANNER);
    println!(
        "  v{}  —  A pure-Rust ML engine\n",
        env!("CARGO_PKG_VERSION")
    );

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

    println!("\nCompute Devices");
    let devices = kore_core::device::available_devices();
    let backends = kore_core::device::registered_backend_names();
    for dev in &devices {
        println!("  {:<10} (available)", dev);
    }
    if backends.is_empty() {
        println!("  GPU:       none detected (CPU fallback only)");
    } else {
        println!("  Backends:  {}", backends.join(", "));
    }

    println!("\nCrates (14)");
    let crates = [
        ("core", "Tensor, DType, Device, Storage"),
        ("autograd", "Computation graph, backward pass"),
        ("nn", "Layers, modules, sampler"),
        ("optim", "Adam, SGD, LR schedulers"),
        ("btes", "Ternary/quaternary encoding"),
        ("kernels", "CUDA + CPU SIMD kernels"),
        ("vulkan", "Pure-Rust WGSL GPU compute backend"),
        ("clifford", "Geometric algebra"),
        ("attention", "Flash attention, KV-cache"),
        ("edge", "No-std inference runtime"),
        ("data", "Dataset utilities"),
        ("serve", "Inference server (OpenAI API)"),
        ("python", "PyO3 bindings"),
        ("cli", "This CLI"),
    ];
    for (name, desc) in crates {
        println!("  kore-{:<10} {}", name, desc);
    }
}
