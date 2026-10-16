use kore_core::Tensor;
use std::time::Instant;

pub fn cmd_bench(sizes_str: &str, device_str: &str) {
    let sizes: Vec<usize> = sizes_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let target_device: kore_core::Device = device_str.parse().unwrap_or(kore_core::Device::Cpu);
    let run_gpu = (device_str.eq_ignore_ascii_case("auto") || target_device.is_vulkan())
        && kore_core::device::get_backend(kore_core::Device::Vulkan(0)).is_some();

    let simd = kore_kernels::SimdCapability::detect();
    println!("=== Kore Matmul Benchmark ===");
    println!(
        "SIMD: {} (avx2={}, avx512={}, fma={}) | GPU: {}\n",
        simd.best_tier(),
        simd.avx2,
        simd.avx512f,
        simd.fma,
        if run_gpu { "Vulkan(0) active" } else { "none" }
    );

    if run_gpu {
        println!(
            "{:<14} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Size", "CPU (ms)", "Quat (ms)", "GPU (ms)", "GPU GFLOPs", "Speedup"
        );
        println!("{}", "-".repeat(70));
    } else {
        println!(
            "{:<14} {:>12} {:>12} {:>9} {:>12} {:>10}",
            "Size", "Naive (ms)", "Tiled (ms)", "Speedup", "Quat (ms)", "Quat GF/s"
        );
        println!("{}", "-".repeat(73));
    }

    for &sz in &sizes {
        let (m, n, k) = (sz, sz, sz);
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 11 + 5) % 17) as f32 * 0.1 - 0.8)
            .collect();

        let a = Tensor::from_f32(&a_data, &[m, k]);
        let b = Tensor::from_f32(&b_data, &[k, n]);

        let iters = if sz <= 128 {
            500
        } else if sz <= 256 {
            50
        } else if sz <= 512 {
            10
        } else {
            3
        };

        // Warmup
        let _ = a.matmul(&b);
        let _ = kore_kernels::cpu_matmul::matmul_f32(&a, &b);

        let naive_s = time_it(iters, || {
            let _ = a.matmul(&b);
        });
        let tiled_s = time_it(iters, || {
            let _ = kore_kernels::cpu_matmul::matmul_f32(&a, &b);
        });

        let (packed, scales) =
            kore_kernels::cpu_quat_matmul::pack_weights_quaternary(&a_data, m, k);
        let quat_s = time_it(iters, || {
            let _ = kore_kernels::cpu_quat_matmul::quat_matmul(&packed, &scales, &b, m, n, k);
        });

        let speedup = naive_s / tiled_s;
        let quat_gflops = (2.0 * m as f64 * n as f64 * k as f64) / quat_s / 1e9;

        let gpu_s = if run_gpu {
            let a_gpu = a.to(kore_core::Device::Vulkan(0)).unwrap();
            let b_gpu = b.to(kore_core::Device::Vulkan(0)).unwrap();
            let _ = a_gpu.matmul(&b_gpu);
            Some(time_it(iters, || {
                let _ = a_gpu.matmul(&b_gpu);
            }))
        } else {
            None
        };

        if let Some(gs) = gpu_s {
            let gflops = (2.0 * m as f64 * n as f64 * k as f64) / gs / 1e9;
            let sp = tiled_s / gs;
            println!(
                "{:<14} {:>8.3}ms {:>8.3}ms {:>8.3}ms {:>10.1} {:>9.1}x",
                format!("{}x{}", sz, sz),
                tiled_s * 1000.0,
                quat_s * 1000.0,
                gs * 1000.0,
                gflops,
                sp
            );
        } else {
            println!(
                "{:<14} {:>10.3}ms {:>10.3}ms {:>8.1}x {:>10.3}ms {:>9.2}",
                format!("{}x{}", sz, sz),
                naive_s * 1000.0,
                tiled_s * 1000.0,
                speedup,
                quat_s * 1000.0,
                quat_gflops,
            );
        }
    }

    // Flash Attention benchmark
    println!("\n=== Flash Attention Benchmark ===\n");
    if run_gpu {
        println!(
            "{:<14} {:>12} {:>12} {:>12} {:>10}",
            "SeqLen", "Standard(CPU)", "Flash(CPU)", "Decode(GPU)", "GPU vs CPU"
        );
        println!("{}", "-".repeat(68));
    } else {
        println!(
            "{:<14} {:>14} {:>14} {:>10}",
            "SeqLen", "Standard (ms)", "Flash (ms)", "Speedup"
        );
        println!("{}", "-".repeat(56));
    }

    for &seq_len in &[64, 128, 256, 512] {
        let d = 64;
        let data: Vec<f32> = (0..seq_len * d)
            .map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6)
            .collect();
        let q = Tensor::from_f32(&data, &[seq_len, d]);
        let mask = kore_attention::mask::causal_mask(seq_len);

        let iters = if seq_len <= 128 {
            50
        } else if seq_len <= 256 {
            10
        } else {
            3
        };

        let std_s = time_it(iters, || {
            let _ = kore_attention::scaled_dot::scaled_dot_product_attention(
                &q,
                &q,
                &q,
                Some(&mask),
                None,
            );
        });

        let flash_s = time_it(iters, || {
            let _ = kore_attention::flash::flash_attention(&q, &q, &q, true);
        });

        let q_dec_data: Vec<f32> = (0..d).map(|i| ((i * 3 + 1) % 11) as f32 * 0.1).collect();
        let q_dec = Tensor::from_f32(&q_dec_data, &[1, d]);
        let k_cache = q.clone();
        let v_cache = q.clone();

        let gpu_decode_s = if run_gpu {
            let q_dec_gpu = q_dec.to(kore_core::Device::Vulkan(0)).unwrap();
            let k_gpu = k_cache.to(kore_core::Device::Vulkan(0)).unwrap();
            let v_gpu = v_cache.to(kore_core::Device::Vulkan(0)).unwrap();
            let _ = kore_attention::scaled_dot::scaled_dot_product_attention(
                &q_dec_gpu, &k_gpu, &v_gpu, None, None,
            );
            Some(time_it(iters * 5, || {
                let _ = kore_attention::scaled_dot::scaled_dot_product_attention(
                    &q_dec_gpu, &k_gpu, &v_gpu, None, None,
                );
            }))
        } else {
            None
        };

        if let Some(gs) = gpu_decode_s {
            println!(
                "{:<14} {:>10.3}ms {:>10.3}ms {:>10.3}ms {:>9.1}x",
                format!("seq={}", seq_len),
                std_s * 1000.0,
                flash_s * 1000.0,
                gs * 1000.0,
                flash_s / gs,
            );
        } else if run_gpu {
            println!(
                "{:<14} {:>10.3}ms {:>10.3}ms {:>12} {:>9.1}x",
                format!("seq={}", seq_len),
                std_s * 1000.0,
                flash_s * 1000.0,
                "—",
                std_s / flash_s,
            );
        } else {
            println!(
                "{:<14} {:>12.3}ms {:>12.3}ms {:>9.1}x",
                format!("seq={}", seq_len),
                std_s * 1000.0,
                flash_s * 1000.0,
                std_s / flash_s,
            );
        }
    }
}

pub fn time_it(iters: usize, mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() / iters as f64
}
