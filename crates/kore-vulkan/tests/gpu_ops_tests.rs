use kore_core::device::DeviceBackend;
use kore_core::Tensor;
use kore_vulkan::VulkanBackend;

#[test]
fn test_vulkan_backend_initialization() {
    match VulkanBackend::new() {
        Ok(backend) => {
            let limits = backend.wgpu_device().limits();
            println!(
                "Vulkan / GPU Backend initialized successfully: {:?}\n  max_buffer_size: {:.2} GB\n  max_storage_buffer_binding_size: {:.2} GB",
                backend.device(),
                limits.max_buffer_size as f64 / 1e9,
                limits.max_storage_buffer_binding_size as f64 / 1e9
            );
        }
        Err(e) => {
            println!("No Vulkan/GPU device available in this environment (skipped): {e}");
        }
    }
}

#[test]
fn test_vulkan_matmul_correctness() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return, // Skip test if no Vulkan GPU is available in environment
    };

    // A: 2x3, B: 3x2
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0f32, 8.0, 9.0, 1.0, 2.0, 3.0];

    let a = Tensor::from_f32(&a_data, &[2, 3]);
    let b = Tensor::from_f32(&b_data, &[3, 2]);

    // Expected C = A @ B:
    // [1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3] = [7 + 18 + 6, 8 + 2 + 9] = [31, 19]
    // [4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3] = [28 + 45 + 12, 32 + 5 + 18] = [85, 55]
    let expected = vec![31.0f32, 19.0, 85.0, 55.0];

    let c = backend.matmul(&a, &b).expect("GPU matmul failed");
    assert_eq!(c.shape().dims(), &[2, 2]);

    let c_cpu = c.to(kore_core::Device::Cpu).expect("to CPU");
    let c_slice = c_cpu.as_f32_slice().expect("f32 slice");
    for (i, (&actual, &exp)) in c_slice.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-4,
            "Element {i} mismatch: got {actual}, expected {exp}"
        );
    }
}

#[test]
fn test_vulkan_gemv_correctness() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // Vector A: 1x4, Matrix B: 4x3
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    let a = Tensor::from_f32(&a_data, &[1, 4]);
    let b = Tensor::from_f32(&b_data, &[4, 3]);

    // Expected C = A @ B:
    // col 0: 1*1 + 2*4 + 3*7 + 4*10 = 1 + 8 + 21 + 40 = 70
    // col 1: 1*2 + 2*5 + 3*8 + 4*11 = 2 + 10 + 24 + 44 = 80
    // col 2: 1*3 + 2*6 + 3*9 + 4*12 = 3 + 12 + 27 + 48 = 90
    let expected = vec![70.0f32, 80.0, 90.0];

    let c = backend.matmul(&a, &b).expect("GPU GEMV failed");
    assert_eq!(c.shape().dims(), &[1, 3]);

    let c_cpu = c.to(kore_core::Device::Cpu).expect("to CPU");
    let c_slice = c_cpu.as_f32_slice().expect("f32 slice");
    for (i, (&actual, &exp)) in c_slice.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-4,
            "Element {i} mismatch: got {actual}, expected {exp}"
        );
    }
}

#[test]
fn test_vulkan_elementwise_add_and_mul() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0];

    let a = Tensor::from_f32(&a_data, &[4]);
    let b = Tensor::from_f32(&b_data, &[4]);

    let sum = backend.add(&a, &b).expect("GPU add failed");
    let sum_cpu = sum.to(kore_core::Device::Cpu).expect("to CPU");
    let sum_slice = sum_cpu.as_f32_slice().expect("f32 slice");
    assert_eq!(sum_slice, &[11.0, 22.0, 33.0, 44.0]);

    let prod = backend.mul(&a, &b).expect("GPU mul failed");
    let prod_cpu = prod.to(kore_core::Device::Cpu).expect("to CPU");
    let prod_slice = prod_cpu.as_f32_slice().expect("f32 slice");
    assert_eq!(prod_slice, &[10.0, 40.0, 90.0, 160.0]);
}

#[test]
fn test_vulkan_rms_norm() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let x_data = vec![2.0f32, 2.0, 2.0, 2.0];
    let w_data = vec![1.0f32, 1.0, 1.0, 1.0];

    let x = Tensor::from_f32(&x_data, &[1, 4]);
    let w = Tensor::from_f32(&w_data, &[4]);

    let out = backend.rms_norm(&x, &w, 1e-5).expect("GPU rms_norm failed");
    let out_cpu = out.to(kore_core::Device::Cpu).expect("to CPU");
    let slice = out_cpu.as_f32_slice().expect("f32 slice");
    for &val in slice {
        assert!((val - 1.0).abs() < 1e-3, "Expected ~1.0, got {val}");
    }
}

#[test]
fn test_vulkan_transpose_and_contiguous() {
    let _backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // Matrix [2, 3]:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_gpu = Tensor::from_f32(&a_data, &[2, 3])
        .to(kore_core::Device::Vulkan(0))
        .expect("to vulkan");

    let a_t = a_gpu.transpose().expect("transpose");
    assert_eq!(a_t.shape().dims(), &[3, 2]);
    assert!(!a_t.is_contiguous());

    let a_cont = a_t.contiguous();
    assert_eq!(a_cont.shape().dims(), &[3, 2]);
    assert_eq!(a_cont.device(), kore_core::Device::Vulkan(0));

    let cpu = a_cont.to(kore_core::Device::Cpu).expect("to cpu");
    let slice = cpu.as_f32_slice().expect("f32 slice");
    // Transposed [3, 2]:
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    assert_eq!(slice, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_vulkan_hadamard_self_inverse_and_cpu_parity() {
    let _backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let data: Vec<f32> = (0..128)
        .map(|i| (i as f32 * 0.23).sin() + if i == 7 { 25.0 } else { 0.0 })
        .collect();
    let x_cpu = Tensor::from_f32(&data, &[2, 64]);
    let rot_cpu = x_cpu.hadamard().expect("cpu hadamard");

    let x_gpu = x_cpu.to(kore_core::Device::Vulkan(0)).expect("to vulkan");
    let rot_gpu = x_gpu.hadamard().expect("gpu hadamard");
    assert_eq!(rot_gpu.device(), kore_core::Device::Vulkan(0));

    let rot_gpu_cpu = rot_gpu.to(kore_core::Device::Cpu).expect("to cpu");
    for (a, b) in rot_cpu
        .as_f32_slice()
        .unwrap()
        .iter()
        .zip(rot_gpu_cpu.as_f32_slice().unwrap().iter())
    {
        assert!(
            (a - b).abs() < 1e-4,
            "Hadamard CPU vs GPU mismatch: {a} vs {b}"
        );
    }

    // Self-inverse on GPU: H(H(x)) == x
    let rec_gpu = rot_gpu
        .hadamard()
        .expect("gpu hadamard inverse")
        .to(kore_core::Device::Cpu)
        .expect("to cpu");
    for (a, b) in data.iter().zip(rec_gpu.as_f32_slice().unwrap().iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "Hadamard self-inverse mismatch: {a} vs {b}"
        );
    }
}

#[test]
fn test_vulkan_small_batch_gemv_batch_invariance() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // Test M = 3 (small-batch GEMV path) and compare row 0 against M = 1 (single-token GEMV)
    let k = 70;
    let n = 19;
    let a3_data: Vec<f32> = (0..(3 * k)).map(|i| (i as f32 * 0.17).cos()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32 * 0.11).sin()).collect();

    let a3 = Tensor::from_f32(&a3_data, &[3, k]);
    let a1 = Tensor::from_f32(&a3_data[..k], &[1, k]);
    let b = Tensor::from_f32(&b_data, &[k, n]);

    let c3 = backend
        .matmul(&a3, &b)
        .expect("M=3 gemv")
        .to(kore_core::Device::Cpu)
        .unwrap();
    let c1 = backend
        .matmul(&a1, &b)
        .expect("M=1 gemv")
        .to(kore_core::Device::Cpu)
        .unwrap();

    // Batch-invariance: row 0 of M=3 must match M=1 bit-for-bit
    assert_eq!(&c3.as_f32_slice().unwrap()[..n], c1.as_f32_slice().unwrap());
}

#[test]
fn test_vulkan_flash_decode_and_argmax() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let seq_q = 2;
    let seq_k = 8;
    let d = 16;
    let scale = 1.0 / (d as f32).sqrt();

    let q_data: Vec<f32> = (0..(seq_q * d)).map(|i| (i as f32 * 0.19).sin()).collect();
    let k_data: Vec<f32> = (0..(seq_k * d)).map(|i| (i as f32 * 0.23).cos()).collect();
    let v_data: Vec<f32> = (0..(seq_k * d)).map(|i| (i as f32 * 0.31).sin()).collect();

    let q = Tensor::from_f32(&q_data, &[seq_q, d]);
    let k = Tensor::from_f32(&k_data, &[seq_k, d]);
    let v = Tensor::from_f32(&v_data, &[seq_k, d]);

    // Reference attention: softmax(Q @ K^T * scale) @ V
    let kt = k.transpose().unwrap().contiguous();
    let scores = backend
        .matmul(&q, &kt)
        .unwrap()
        .to(kore_core::Device::Cpu)
        .unwrap();
    let scaled = backend
        .mul_scalar(&scores, scale)
        .unwrap()
        .to(kore_core::Device::Cpu)
        .unwrap();
    let probs = backend
        .softmax(&scaled)
        .unwrap()
        .to(kore_core::Device::Cpu)
        .unwrap();
    let expected = backend
        .matmul(&probs, &v)
        .unwrap()
        .to(kore_core::Device::Cpu)
        .unwrap();

    let actual = backend
        .fused_attention_decode(&q, &k, &v, scale)
        .expect("flash_decode")
        .to(kore_core::Device::Cpu)
        .unwrap();

    for (a, e) in actual
        .as_f32_slice()
        .unwrap()
        .iter()
        .zip(expected.as_f32_slice().unwrap().iter())
    {
        assert!(
            (a - e).abs() < 1e-4,
            "flash_decode mismatch: got {a}, expected {e}"
        );
    }

    // Test GPU argmax_tokens on a 2x512 logits tensor
    let mut logits = vec![-1.0f32; 2 * 512];
    logits[317] = 42.0;
    logits[512 + 509] = 99.0;
    let logits_gpu = Tensor::from_f32(&logits, &[2, 512])
        .to(kore_core::Device::Vulkan(0))
        .unwrap();
    let tokens = logits_gpu.argmax_tokens().expect("gpu argmax_tokens");
    assert_eq!(tokens, vec![317, 509]);

    // Test GPU quantized_matmul (row-packed Ternary: 5 trits per byte in base-3, code = trit + 1)
    // K = 5 (1 byte per row), N_out = 2, M_batch = 2
    // Row 0 trits: [+1, +1, +1, +1, +1] -> codes [2,2,2,2,2] -> 2*(1+3+9+27+81) = 242
    // Row 1 trits: [-1, -1,  0,  0,  0] -> codes [0,0,1,1,1] -> 0 + 0 + 9 + 27 + 81 = 117
    let packed_bytes = vec![242u8, 117u8];
    let w_storage =
        kore_core::Storage::from_packed_bytes(kore_core::DType::Ternary, 10, packed_bytes);
    let w_packed = Tensor::from_storage(w_storage, &[2, 5]);
    let scales = Tensor::from_f32(&[0.5, 2.0], &[2]);
    let x_q = Tensor::from_f32(&[1.0; 10], &[2, 5]);
    let y_q = backend
        .quantized_matmul(&x_q, &w_packed, &scales)
        .expect("gpu quantized_matmul ternary")
        .to(kore_core::Device::Cpu)
        .unwrap();
    let y_q_slice = y_q.as_f32_slice().unwrap();
    // Row 0 dot = 5.0 * 0.5 = 2.5; Row 1 dot = -2.0 * 2.0 = -4.0
    assert!((y_q_slice[0] - 2.5).abs() < 1e-4, "got {}", y_q_slice[0]);
    assert!((y_q_slice[1] + 4.0).abs() < 1e-4, "got {}", y_q_slice[1]);
}

#[test]
fn test_vulkan_gemv_q4_k_correctness() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // Matrix W: 2 rows, 256 cols (1 block per row). Total = 2 * 144 = 288 bytes.
    // Row 0:
    //   d = 1.0 (FP16 0x3C00), dmin = 0.0 (FP16 0x0000)
    //   scales = 1 for all 8 sub-blocks, mins = 0 for all 8 sub-blocks
    //   qs: all low nibbles = 2, all high nibbles = 3 (byte = 0x32)
    // Row 1:
    //   d = 2.0 (FP16 0x4000), dmin = 1.0 (FP16 0x3C00)
    //   scales = 2 for all 8 sub-blocks, mins = 1 for all 8 sub-blocks
    //   qs: all low nibbles = 5, all high nibbles = 7 (byte = 0x75)
    let mut raw_bytes = vec![0u8; 288];

    // Row 0:
    // Word 0: d=1.0, dmin=0.0 -> bytes [0x00, 0x3C, 0x00, 0x00]
    raw_bytes[0] = 0x00;
    raw_bytes[1] = 0x3C;
    raw_bytes[2] = 0x00;
    raw_bytes[3] = 0x00;
    // Scales: bytes 4..7 = 1 (sc 0..3), 8..11 = 0 (mn 0..3), 12..15 = 1 (sc 4..7 lo_sc=1, lo_mn=0)
    for i in 4..8 {
        raw_bytes[i] = 1;
    }
    for i in 8..12 {
        raw_bytes[i] = 0;
    }
    for i in 12..16 {
        raw_bytes[i] = 1;
    }
    // qs: bytes 16..144
    for i in 16..144 {
        raw_bytes[i] = 0x32;
    }

    // Row 1: offset 144
    // Word 0: d=2.0 (0x4000), dmin=1.0 (0x3C00) -> [0x00, 0x40, 0x00, 0x3C]
    raw_bytes[144] = 0x00;
    raw_bytes[145] = 0x40;
    raw_bytes[146] = 0x00;
    raw_bytes[147] = 0x3C;
    // Scales: sc=2, mn=1 for all
    // sc 0..3: byte 0..3 = 2
    for i in 148..152 {
        raw_bytes[i] = 2;
    }
    // mn 0..3: byte 4..7 = 1
    for i in 152..156 {
        raw_bytes[i] = 1;
    }
    // sc 4..7, mn 4..7: byte 8..11 -> lo_sc=2 (low nibble), lo_mn=1 (high nibble) -> (1 << 4) | 2 = 0x12
    for i in 156..160 {
        raw_bytes[i] = 0x12;
    }
    for i in 160..288 {
        raw_bytes[i] = 0x75;
    }

    let w_storage = kore_core::Storage::from_packed_bytes(kore_core::DType::U8, 288, raw_bytes);
    let w_tensor = Tensor::from_storage(w_storage, &[288]);

    // Input x: [1, 256], all 1.0
    let x_data = vec![1.0f32; 256];
    let x = Tensor::from_f32(&x_data, &[1, 256]);

    // Expected Row 0:
    // 128 weights = 1.0 * 1 * 2 - 0.0 * 0 = 2.0
    // 128 weights = 1.0 * 1 * 3 - 0.0 * 0 = 3.0
    // dot = 128 * 2.0 + 128 * 3.0 = 640.0
    //
    // Expected Row 1:
    // 128 weights = 2.0 * 2 * 5 - 1.0 * 1 = 20 - 1 = 19.0
    // 128 weights = 2.0 * 2 * 7 - 1.0 * 1 = 28 - 1 = 27.0
    // dot = 128 * 19.0 + 128 * 27.0 = 2432.0 + 3456.0 = 5888.0
    let y = backend
        .gemv_q4_k(&x, &w_tensor, 2, 256)
        .expect("gemv_q4_k failed");
    assert_eq!(y.shape().dims(), &[1, 2]);

    let y_cpu = y.to(kore_core::Device::Cpu).expect("to CPU");
    let y_slice = y_cpu.as_f32_slice().expect("f32 slice");
    assert!(
        (y_slice[0] - 640.0).abs() < 1e-3,
        "Row 0 mismatch: got {}",
        y_slice[0]
    );
    assert!(
        (y_slice[1] - 5888.0).abs() < 1e-3,
        "Row 1 mismatch: got {}",
        y_slice[1]
    );

    // Test with N=3 rows (odd row count) and K=512 (2 blocks per row)
    // Total bytes = 3 rows * 2 blocks * 144 bytes = 864 bytes.
    let mut raw_bytes_3x512 = vec![0u8; 864];
    // Block 0 of row 0: d=1.0, qs=0x11 (all nibbles = 1, sc=1, mn=0)
    raw_bytes_3x512[1] = 0x3C;
    for i in 4..8 {
        raw_bytes_3x512[i] = 1;
    }
    for i in 12..16 {
        raw_bytes_3x512[i] = 1;
    }
    for i in 16..144 {
        raw_bytes_3x512[i] = 0x11;
    }
    // Block 1 of row 0 (offset 144): d=1.0, qs=0x11
    raw_bytes_3x512[144 + 1] = 0x3C;
    for i in 148..152 {
        raw_bytes_3x512[i] = 1;
    }
    for i in 156..160 {
        raw_bytes_3x512[i] = 1;
    }
    for i in 160..288 {
        raw_bytes_3x512[i] = 0x11;
    }

    // Row 2 (offset 2 * 288 = 576): block 0 has d=2.0 (0x4000), qs=0x22 (nibbles=2, sc=1, mn=0)
    raw_bytes_3x512[576 + 1] = 0x40;
    for i in 580..584 {
        raw_bytes_3x512[i] = 1;
    }
    for i in 588..592 {
        raw_bytes_3x512[i] = 1;
    }
    for i in 592..720 {
        raw_bytes_3x512[i] = 0x22;
    }

    let w3_storage =
        kore_core::Storage::from_packed_bytes(kore_core::DType::U8, 864, raw_bytes_3x512);
    let w3_tensor = Tensor::from_storage(w3_storage, &[864]);
    let x512 = Tensor::from_f32(&vec![1.0f32; 512], &[1, 512]);

    let y3 = backend
        .gemv_q4_k(&x512, &w3_tensor, 3, 512)
        .expect("gemv_q4_k 3x512 failed");
    assert_eq!(y3.shape().dims(), &[1, 3]);
    let y3_cpu = y3.to(kore_core::Device::Cpu).expect("to CPU");
    let y3_slice = y3_cpu.as_f32_slice().expect("f32 slice");
    // Row 0: 512 weights * 1.0 = 512.0
    // Row 1: all 0.0
    // Row 2: block 0 has 256 weights * 4.0 = 1024.0, block 1 has 0.0 -> total 1024.0
    assert!(
        (y3_slice[0] - 512.0).abs() < 1e-3,
        "Row 0: got {}",
        y3_slice[0]
    );
    assert!(
        (y3_slice[1] - 0.0).abs() < 1e-3,
        "Row 1: got {}",
        y3_slice[1]
    );
    assert!(
        (y3_slice[2] - 1024.0).abs() < 1e-3,
        "Row 2: got {}",
        y3_slice[2]
    );

    println!(
        "All Q4_K GPU GEMV tests passed! Row 0 = {}, Row 1 = {}, Row 2 = {}",
        y3_slice[0], y3_slice[1], y3_slice[2]
    );
}

#[test]
fn test_vulkan_gemv_q5_k_correctness() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // Q5_K is the Q4_K block with a 32-byte `qh` inserted after the scales:
    //   d:f16 | dmin:f16 | scales[12] | qh[32] | qs[128]  = 176 bytes.
    //
    // Matrix W: 2 rows, 256 cols (1 block per row) = 352 bytes.
    //
    // Row 0 leaves `qh` zero, so it must reproduce the Q4_K result exactly:
    //   d = 1.0, dmin = 0.0, all scales = 1, all mins = 0,
    //   qs byte = 0x32 -> lower sub-blocks 2, upper sub-blocks 3.
    //   dot with all-ones x = 128*2 + 128*3 = 640.
    //
    // Row 1 sets every high bit, which is the only thing Q5_K adds:
    //   d = 2.0, dmin = 1.0, all scales = 2, all mins = 1,
    //   qs byte = 0x75 -> nibbles 5 and 7, plus 16 from the high bit -> 21 / 23.
    //   lower: 2*2*21 - 1 = 83, upper: 2*2*23 - 1 = 91
    //   dot = 128*83 + 128*91 = 22272.
    let mut raw = vec![0u8; 352];

    // Row 0
    raw[1] = 0x3C; // d = 1.0
    for i in 4..8 {
        raw[i] = 1;
    }
    for i in 12..16 {
        raw[i] = 1;
    }
    // raw[16..48] stays 0: no high bits set.
    for i in 48..176 {
        raw[i] = 0x32;
    }

    // Row 1 at offset 176
    raw[176 + 1] = 0x40; // d = 2.0
    raw[176 + 3] = 0x3C; // dmin = 1.0
    for i in 180..184 {
        raw[i] = 2; // sc 0..3
    }
    for i in 184..188 {
        raw[i] = 1; // mn 0..3
    }
    for i in 188..192 {
        raw[i] = 0x12; // sc 4..7 = 2, mn 4..7 = 1
    }
    for i in 192..224 {
        raw[i] = 0xFF; // every 5th bit set
    }
    for i in 224..352 {
        raw[i] = 0x75;
    }

    let w_storage = kore_core::Storage::from_packed_bytes(kore_core::DType::U8, 352, raw);
    let w_tensor = Tensor::from_storage(w_storage, &[352]);

    // m = 1
    let x = Tensor::from_f32(&vec![1.0f32; 256], &[1, 256]);
    let y = backend
        .gemv_q5_k(&x, &w_tensor, 2, 256)
        .expect("gemv_q5_k failed");
    assert_eq!(y.shape().dims(), &[1, 2]);
    let y_cpu = y.to(kore_core::Device::Cpu).expect("to CPU");
    let y_slice = y_cpu.as_f32_slice().expect("f32 slice");
    assert!(
        (y_slice[0] - 640.0).abs() < 1e-3,
        "no-high-bit row: got {}",
        y_slice[0]
    );
    assert!(
        (y_slice[1] - 22272.0).abs() < 1e-3,
        "high-bit row: got {}",
        y_slice[1]
    );

    // m = 2, second column doubled: exercises the multi-column accumulate path.
    let mut x2_data = vec![1.0f32; 256];
    x2_data.extend(std::iter::repeat_n(2.0f32, 256));
    let x2 = Tensor::from_f32(&x2_data, &[2, 256]);
    let y2 = backend
        .gemv_q5_k(&x2, &w_tensor, 2, 256)
        .expect("gemv_q5_k m=2 failed");
    assert_eq!(y2.shape().dims(), &[2, 2]);
    let y2_cpu = y2.to(kore_core::Device::Cpu).expect("to CPU");
    let y2_slice = y2_cpu.as_f32_slice().expect("f32 slice");
    assert!(
        (y2_slice[0] - 640.0).abs() < 1e-3,
        "m2 row0: got {}",
        y2_slice[0]
    );
    assert!(
        (y2_slice[1] - 22272.0).abs() < 1e-3,
        "m2 row1: got {}",
        y2_slice[1]
    );
    assert!(
        (y2_slice[2] - 1280.0).abs() < 1e-3,
        "m2 row0 x2: got {}",
        y2_slice[2]
    );
    assert!(
        (y2_slice[3] - 44544.0).abs() < 1e-3,
        "m2 row1 x2: got {}",
        y2_slice[3]
    );

    println!(
        "Q5_K GPU GEMV passed: no-bit row = {}, high-bit row = {}",
        y_slice[0], y_slice[1]
    );
}

/// Q8_0 GEMV: a 32-weight block under one f16 scale, `value = q * d`.
///
/// The kernel reads a *repacked* payload (36-byte blocks, quants first), so the
/// fixture is built through `kore_kernels::q8_0::repack_q8_0` — that is the same
/// path `up_w` takes for a resident Q8_0 tensor, and it is what makes the test
/// cover the repack as well as the arithmetic.
///
/// Matrix W: 2 rows x 256 cols (8 blocks per row) = 544 file bytes.
///
/// Row 0: `d = 1.0`, every quant `2` -> each weight `2.0`, dot with all-ones
///        x = 256 * 2 = 512.
/// Row 1: `d = 0.5`, every quant `-3` -> each weight `-1.5`, dot = 256 * -1.5
///        = -384. The negative quant is the point: it exercises the signed-byte
///        bias-back in the shader, which an all-positive fixture would miss.
#[test]
fn test_vulkan_gemv_q8_0_correctness() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    const BLOCK: usize = 34;
    let mut raw = vec![0u8; 2 * 8 * BLOCK];

    // Row 0: every block d = 1.0 (0x3C00 little-endian), quants 2.
    for b in 0..8 {
        raw[b * BLOCK] = 0x00;
        raw[b * BLOCK + 1] = 0x3C;
        for i in 0..32 {
            raw[b * BLOCK + 2 + i] = 2;
        }
    }
    // Row 1 at offset 8 * BLOCK: d = 0.5 (0x3800), quants -3.
    let r1 = 8 * BLOCK;
    for b in 0..8 {
        raw[r1 + b * BLOCK] = 0x00;
        raw[r1 + b * BLOCK + 1] = 0x38;
        for i in 0..32 {
            raw[r1 + b * BLOCK + 2 + i] = (-3i8) as u8;
        }
    }

    let packed = kore_kernels::q8_0::repack_q8_0(&raw).expect("repack");
    assert_eq!(packed.len(), 2 * 8 * 36);
    let w_tensor = Tensor::from_storage(
        kore_core::Storage::from_packed_bytes(kore_core::DType::U8, packed.len(), packed),
        &[2 * 8 * 36],
    );

    // m = 1
    let x = Tensor::from_f32(&vec![1.0f32; 256], &[1, 256]);
    let y = backend
        .gemv_q8_0(&x, &w_tensor, 2, 256)
        .expect("gemv_q8_0 failed");
    assert_eq!(y.shape().dims(), &[1, 2]);
    let y_cpu = y.to(kore_core::Device::Cpu).expect("to CPU");
    let y_slice = y_cpu.as_f32_slice().expect("f32 slice");
    assert!(
        (y_slice[0] - 512.0).abs() < 1e-3,
        "positive row: got {}",
        y_slice[0]
    );
    assert!(
        (y_slice[1] + 384.0).abs() < 1e-3,
        "negative row: got {}",
        y_slice[1]
    );

    // m = 4, columns scaled 1..4: exercises the four-column accumulate path and
    // checks each column independently (a lane that leaked one column into
    // another would show up here and nowhere else).
    let mut x4_data = Vec::with_capacity(4 * 256);
    for c in 1..=4 {
        x4_data.extend(std::iter::repeat_n(c as f32, 256));
    }
    let x4 = Tensor::from_f32(&x4_data, &[4, 256]);
    let y4 = backend
        .gemv_q8_0(&x4, &w_tensor, 2, 256)
        .expect("gemv_q8_0 m=4 failed");
    assert_eq!(y4.shape().dims(), &[4, 2]);
    let y4_cpu = y4.to(kore_core::Device::Cpu).expect("to CPU");
    let y4_slice = y4_cpu.as_f32_slice().expect("f32 slice");
    for c in 0..4 {
        let scale = (c + 1) as f32;
        assert!(
            (y4_slice[c * 2] - 512.0 * scale).abs() < 1e-3,
            "m4 row0 col {c}: got {}",
            y4_slice[c * 2]
        );
        assert!(
            (y4_slice[c * 2 + 1] + 384.0 * scale).abs() < 1e-3,
            "m4 row1 col {c}: got {}",
            y4_slice[c * 2 + 1]
        );
    }

    println!(
        "Q8_0 GPU GEMV passed: positive row = {}, negative row = {}",
        y_slice[0], y_slice[1]
    );
}

/// Quantize one F32 row to offset-binary INT8 + per-row scale (mirrors the CPU Q8 KV path).
fn q8_quant_row(row: &[f32]) -> (Vec<u8>, f32) {
    let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    let bytes = row
        .iter()
        .map(|&v| {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i16;
            (q + 128) as u8
        })
        .collect();
    (bytes, scale)
}

fn q8_dequant_row(bytes: &[u8], scale: f32) -> Vec<f32> {
    bytes.iter().map(|&b| (b as f32 - 128.0) * scale).collect()
}

/// Q8 KV path regression: GPU `kv_cache_append_q8` + `fused_attention_decode_gqa_q8`
/// must match a CPU reference computed from the *same dequantized* K/V (isolating
/// the GPU realization from quantization error). The append is split across two
/// offsets to exercise the non-zero `row_offset`/scale-append path.
#[test]
fn test_vulkan_kv_append_q8_and_decode_matches_reference() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);

    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 16usize; // multiple of 4: packed-u32 requirement
    let seq_q = 1usize;
    let seq_k = 12usize;
    let max_ctx = 32usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_data: Vec<f32> = (0..(seq_q * q_dim))
        .map(|i| (i as f32 * 0.19).sin())
        .collect();
    let k_data: Vec<f32> = (0..(seq_k * kv_dim))
        .map(|i| (i as f32 * 0.23).cos() * 3.0)
        .collect();
    let v_data: Vec<f32> = (0..(seq_k * kv_dim))
        .map(|i| (i as f32 * 0.31).sin() * 2.0)
        .collect();

    // Quantize K/V per token row and build the reference dequantized tensors.
    let mut k_bytes = Vec::with_capacity(seq_k * kv_dim);
    let mut v_bytes = Vec::with_capacity(seq_k * kv_dim);
    let mut k_scales_host = vec![0.0f32; max_ctx];
    let mut v_scales_host = vec![0.0f32; max_ctx];
    let mut k_deq = vec![0.0f32; seq_k * kv_dim];
    let mut v_deq = vec![0.0f32; seq_k * kv_dim];
    for t in 0..seq_k {
        let (kb, ks) = q8_quant_row(&k_data[t * kv_dim..(t + 1) * kv_dim]);
        let (vb, vs) = q8_quant_row(&v_data[t * kv_dim..(t + 1) * kv_dim]);
        k_deq[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&q8_dequant_row(&kb, ks));
        v_deq[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&q8_dequant_row(&vb, vs));
        k_bytes.extend_from_slice(&kb);
        v_bytes.extend_from_slice(&vb);
        k_scales_host[t] = ks;
        v_scales_host[t] = vs;
    }

    // Resident caches (U8, max_ctx rows) + per-token F32 scales.
    let mut kc = Tensor::zeros(&[max_ctx, kv_dim], kore_core::DType::U8)
        .to(dev)
        .unwrap();
    let mut vc = Tensor::zeros(&[max_ctx, kv_dim], kore_core::DType::U8)
        .to(dev)
        .unwrap();
    let k_scales = Tensor::from_f32(&k_scales_host, &[max_ctx])
        .to(dev)
        .unwrap();
    let v_scales = Tensor::from_f32(&v_scales_host, &[max_ctx])
        .to(dev)
        .unwrap();

    // Append in two chunks to exercise a non-zero row_offset.
    let split = 5usize;
    for (off, len) in [(0usize, split), (split, seq_k - split)] {
        let k_new = Tensor::from_storage(
            kore_core::Storage::from_bytes(
                kore_core::DType::U8,
                len * kv_dim,
                k_bytes[off * kv_dim..(off + len) * kv_dim].to_vec(),
            )
            .unwrap(),
            &[len, kv_dim],
        )
        .to(dev)
        .unwrap();
        let v_new = Tensor::from_storage(
            kore_core::Storage::from_bytes(
                kore_core::DType::U8,
                len * kv_dim,
                v_bytes[off * kv_dim..(off + len) * kv_dim].to_vec(),
            )
            .unwrap(),
            &[len, kv_dim],
        )
        .to(dev)
        .unwrap();
        let ks_new = Tensor::from_f32(&k_scales_host[off..off + len], &[len])
            .to(dev)
            .unwrap();
        let vs_new = Tensor::from_f32(&v_scales_host[off..off + len], &[len])
            .to(dev)
            .unwrap();

        let (kv_view, vv_view) = backend
            .kv_cache_append_q8(
                &kc, &vc, &k_scales, &v_scales, &k_new, &v_new, &ks_new, &vs_new, off,
            )
            .expect("kv_cache_append_q8");
        assert_eq!(kv_view.shape().dims(), &[off + len, kv_dim]);
        assert_eq!(vv_view.shape().dims(), &[off + len, kv_dim]);
        kc = kv_view;
        vc = vv_view;
    }

    // CPU reference: softmax(Q Kᵀ * scale) V over the dequantized K/V (seq_q == 1 → full attention).
    let group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; seq_q * q_dim];
    for qr in 0..seq_q {
        for h in 0..num_heads {
            let kvh = h / group;
            let mut scores = vec![0.0f32; seq_k];
            let mut maxs = f32::NEG_INFINITY;
            for t in 0..seq_k {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[qr * q_dim + h * head_dim + d]
                        * k_deq[t * kv_dim + kvh * head_dim + d];
                }
                let s = dot * scale;
                scores[t] = s;
                maxs = maxs.max(s);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - maxs).exp();
                sum += *s;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..seq_k {
                    acc += scores[t] * v_deq[t * kv_dim + kvh * head_dim + d];
                }
                expected[qr * q_dim + h * head_dim + d] = acc / sum;
            }
        }
    }

    let q = Tensor::from_f32(&q_data, &[seq_q, q_dim]).to(dev).unwrap();
    let actual = backend
        .fused_attention_decode_gqa_q8(
            &q,
            &kc,
            &vc,
            &k_scales,
            &v_scales,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            seq_k,
        )
        .expect("fused_attention_decode_gqa_q8")
        .to(kore_core::Device::Cpu)
        .unwrap();
    let a = actual.as_f32_slice().unwrap();
    assert_eq!(a.len(), expected.len());
    for (i, (got, want)) in a.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "Q8 decode mismatch at {i}: got {got}, expected {want}"
        );
    }
}

/// Split-KV attention must reproduce the single-group kernel to float rounding
/// for any shard count, including counts that do not divide `seq_k` evenly and
/// counts far larger than `seq_k / 8`.
#[test]
fn test_vulkan_split_attention_matches_single_group() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);

    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 64usize; // the shard kernel needs head_dim % 32 == 0
    let seq_k = 700usize; // deliberately not divisible by any shard count below
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_data: Vec<f32> = (0..q_dim).map(|i| (i as f32 * 0.19).sin()).collect();
    let k_data: Vec<f32> = (0..seq_k * kv_dim)
        .map(|i| (i as f32 * 0.23).cos() * 3.0)
        .collect();
    let v_data: Vec<f32> = (0..seq_k * kv_dim)
        .map(|i| (i as f32 * 0.31).sin() * 2.0)
        .collect();

    let mut k_bytes = Vec::with_capacity(seq_k * kv_dim);
    let mut v_bytes = Vec::with_capacity(seq_k * kv_dim);
    let mut k_scales_host = vec![0.0f32; seq_k];
    let mut v_scales_host = vec![0.0f32; seq_k];
    for t in 0..seq_k {
        let (kb, ks) = q8_quant_row(&k_data[t * kv_dim..(t + 1) * kv_dim]);
        let (vb, vs) = q8_quant_row(&v_data[t * kv_dim..(t + 1) * kv_dim]);
        k_bytes.extend_from_slice(&kb);
        v_bytes.extend_from_slice(&vb);
        k_scales_host[t] = ks;
        v_scales_host[t] = vs;
    }

    let kc = Tensor::zeros(&[seq_k, kv_dim], kore_core::DType::U8)
        .to(dev)
        .unwrap();
    let vc = Tensor::zeros(&[seq_k, kv_dim], kore_core::DType::U8)
        .to(dev)
        .unwrap();
    let ks_buf = Tensor::from_f32(&k_scales_host, &[seq_k]).to(dev).unwrap();
    let vs_buf = Tensor::from_f32(&v_scales_host, &[seq_k]).to(dev).unwrap();
    // Separate source buffers: the append's elementwise copy reads `*_new` and
    // writes `*_buf`, and wgpu rejects one buffer bound both read and read-write.
    let ks_new = Tensor::from_f32(&k_scales_host, &[seq_k]).to(dev).unwrap();
    let vs_new = Tensor::from_f32(&v_scales_host, &[seq_k]).to(dev).unwrap();
    let k_new = Tensor::from_storage(
        kore_core::Storage::from_bytes(kore_core::DType::U8, seq_k * kv_dim, k_bytes).unwrap(),
        &[seq_k, kv_dim],
    )
    .to(dev)
    .unwrap();
    let v_new = Tensor::from_storage(
        kore_core::Storage::from_bytes(kore_core::DType::U8, seq_k * kv_dim, v_bytes).unwrap(),
        &[seq_k, kv_dim],
    )
    .to(dev)
    .unwrap();
    let (kc, vc) = backend
        .kv_cache_append_q8(
            &kc, &vc, &ks_buf, &vs_buf, &k_new, &v_new, &ks_new, &vs_new, 0,
        )
        .expect("kv_cache_append_q8");

    let q = Tensor::from_f32(&q_data, &[1, q_dim]).to(dev).unwrap();
    let single = backend
        .fused_attention_decode_gqa_q8(
            &q,
            &kc,
            &vc,
            &ks_buf,
            &vs_buf,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            seq_k,
        )
        .expect("single-group attention")
        .to(kore_core::Device::Cpu)
        .unwrap();
    let want = single.as_f32_slice().unwrap();

    for &n_splits in &[1usize, 3, 7, 11, 64] {
        let got = backend
            .fused_attention_decode_gqa_q8_split(
                &q,
                &kc,
                &vc,
                &ks_buf,
                &vs_buf,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                n_splits,
                seq_k,
            )
            .expect("split attention")
            .to(kore_core::Device::Cpu)
            .unwrap();
        let g = got.as_f32_slice().unwrap();
        assert_eq!(g.len(), want.len());
        for (i, (a, b)) in g.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 2e-5 * (1.0 + b.abs()),
                "n_splits={n_splits} idx {i}: split {a} vs single {b}"
            );
        }
    }

    // Shape/size guards must reject rather than silently compute something wrong.
    assert!(
        backend
            .fused_attention_decode_gqa_q8_split(
                &q,
                &kc,
                &vc,
                &ks_buf,
                &vs_buf,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                0,
                seq_k,
            )
            .is_err(),
        "n_splits == 0 must be rejected"
    );
    // A negative scale means "apply causal masking" in the single-group kernel;
    // the shard kernel cannot, so it must refuse instead of returning unmasked
    // attention.
    assert!(
        backend
            .fused_attention_decode_gqa_q8_split(
                &q,
                &kc,
                &vc,
                &ks_buf,
                &vs_buf,
                num_heads,
                num_kv_heads,
                head_dim,
                -scale,
                4,
                seq_k,
            )
            .is_err(),
        "negative scale (causal) must be rejected"
    );
    // head_dim 48 is not a multiple of 32 (the shard kernel's staging requirement).
    let hd48 = 48usize;
    let q48 = Tensor::from_f32(&vec![0.0f32; num_heads * hd48], &[1, num_heads * hd48])
        .to(dev)
        .unwrap();
    let k48 = Tensor::zeros(&[8, num_kv_heads * hd48], kore_core::DType::U8)
        .to(dev)
        .unwrap();
    let s48 = Tensor::from_f32(&vec![1.0f32; 8], &[8]).to(dev).unwrap();
    assert!(
        backend
            .fused_attention_decode_gqa_q8_split(
                &q48,
                &k48,
                &k48,
                &s48,
                &s48,
                num_heads,
                num_kv_heads,
                hd48,
                scale,
                4,
                8,
            )
            .is_err(),
        "head_dim % 32 != 0 must be rejected"
    );
}

/// GPU per-row INT8 quantization must match the exact CPU recipe:
/// `scale = absmax/127`, `q = clamp(round(x/scale), -127, 127)`, stored `q + 128`.
#[test]
fn test_vulkan_quantize_q8_rows_matches_reference() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);

    let rows = 4usize;
    let cols = 64usize; // multiple of 4
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| (((i * 13 + 7) % 101) as f32 - 50.0) * 0.37)
        .collect();

    let xt = Tensor::from_f32(&x, &[rows, cols]).to(dev).unwrap();
    let (q, s) = backend.quantize_q8_rows(&xt).expect("quantize_q8_rows");
    assert_eq!(q.shape().dims(), &[rows, cols]);
    assert_eq!(q.dtype(), kore_core::DType::U8);
    assert_eq!(s.shape().dims(), &[rows]);

    let q_cpu = q.to(kore_core::Device::Cpu).unwrap();
    let s_cpu = s.to(kore_core::Device::Cpu).unwrap();
    let qb = q_cpu.storage_ref().as_bytes();
    let sc = s_cpu.as_f32_slice().unwrap();

    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
        assert!(
            (sc[r] - scale).abs() < 1e-6,
            "row {r} scale: got {}, want {scale}",
            sc[r]
        );
        let inv = 1.0 / scale;
        for c in 0..cols {
            let want = ((row[c] * inv).round().clamp(-127.0, 127.0) as i16 + 128) as u8;
            let got = qb[r * cols + c];
            assert_eq!(got, want, "q[{r}, {c}]: got {got}, want {want}");
        }
    }
}

/// GPU Q2_0 GEMV must match the CPU reference exactly. `bpr=8` (k=512) forces
/// the multi-iteration block walk, which catches weight-skipping stride bugs.
#[test]
fn test_vulkan_gemv_q2_0_matches_cpu_reference() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);

    let k = 512usize; // in-dim; multiple of 512 (host guard) -> bpr = 8 blocks
    let n = 7usize; // out-dim
    let bpr = k / 64;
    let row_bytes = bpr * 18;
    // Valid Q2_0 blocks: fp16 scale 0.25 (0x3400 LE) + deterministic code bytes.
    let mut w = vec![0u8; n * row_bytes];
    for r in 0..n {
        for b in 0..bpr {
            let off = r * row_bytes + b * 18;
            w[off] = 0x00;
            w[off + 1] = 0x34;
            for i in 0..16 {
                w[off + 2 + i] = ((r * 7 + b * 31 + i * 17 + 3) % 256) as u8;
            }
        }
    }
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.13 - 3.7).sin()).collect();

    // CPU reference
    let mut want = vec![0.0f32; n];
    kore_kernels::q2_0::gemv_q2_0(&w, k, n, &x, &mut want).unwrap();

    // GPU path: repack + upload + run
    let packed = kore_kernels::q2_0::repack_q2_0(&w, k, n).unwrap();
    let nb = packed.len();
    let ws = kore_core::Storage::from_bytes(kore_core::DType::U8, nb, packed).unwrap();
    let wt = Tensor::from_storage(ws, &[nb]).to(dev).unwrap();
    let xt = Tensor::from_f32(&x, &[1, k]).to(dev).unwrap();

    let got = backend.gemv_q2_0(&xt, &wt, n, k).expect("gemv_q2_0");
    let got_cpu = got.to(kore_core::Device::Cpu).unwrap();
    let g = got_cpu.as_f32_slice().unwrap();
    for i in 0..n {
        assert!(
            (g[i] - want[i]).abs() <= 1e-3 * (1.0 + want[i].abs()),
            "out[{i}]: gpu {} cpu {}",
            g[i],
            want[i]
        );
    }
}

/// `had_pre` must be *bit-identical* to running the standalone kernels in
/// sequence, so the fused decode path can't perturb token selection.
#[test]
fn test_vulkan_had_pre_matches_standalone_kernels() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);
    let eps = 1e-5f32;

    let cols = 2048usize;
    let phase = |i: usize| (i as f32 * 0.017).sin();
    let a_data: Vec<f32> = (0..cols).map(|i| phase(i) * 1.7).collect();
    let b_data: Vec<f32> = (0..cols).map(|i| phase(i + 13) * 0.4).collect();
    let w_data: Vec<f32> = (0..cols).map(|i| 0.5 + phase(i + 5) * 0.5).collect();
    // Random-ish deterministic ±1 sign table.
    let signs_data: Vec<f32> = (0..cols)
        .map(|i| {
            if (i * 2654435761usize) % 3 == 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();

    let a = Tensor::from_f32(&a_data, &[1, cols]).to(dev).unwrap();
    let b = Tensor::from_f32(&b_data, &[1, cols]).to(dev).unwrap();
    let w = Tensor::from_f32(&w_data, &[cols]).to(dev).unwrap();
    let signs = Tensor::from_f32(&signs_data, &[cols]).to(dev).unwrap();

    let to_cpu = |t: &Tensor| {
        t.to(kore_core::Device::Cpu)
            .unwrap()
            .as_f32_slice()
            .unwrap()
            .to_vec()
    };

    // ---- mode 0: add + rms_norm + hadamard ----
    let (res, norm, had) = backend
        .had_pre(0, true, &a, &b, &w, &signs, eps)
        .expect("had_pre 0");
    let (ref_res, ref_norm) = backend.add_rms_norm(&a, &b, &w, eps).expect("add_rms_norm");
    let ref_had = backend
        .hadamard_signed(&ref_norm, &signs, 512, 0)
        .expect("hadamard_signed");

    assert_eq!(to_cpu(&res), to_cpu(&ref_res), "mode 0 residual differs");
    assert_eq!(to_cpu(&norm), to_cpu(&ref_norm), "mode 0 normed differs");
    assert_eq!(to_cpu(&had), to_cpu(&ref_had), "mode 0 hadamard differs");

    // ---- mode 2: swiglu + hadamard ----
    let (_r, _n, had2) = backend
        .had_pre(2, true, &a, &b, &signs, &signs, eps)
        .expect("had_pre 2");
    let act = backend.swiglu(&a, &b).expect("swiglu");
    let ref_had2 = backend.hadamard_signed(&act, &signs, 512, 0).expect("had2");
    assert_eq!(to_cpu(&had2), to_cpu(&ref_had2), "mode 2 hadamard differs");

    // ---- mode 3: sigmoid_mul + hadamard ----
    let (_r, _n, had3) = backend
        .had_pre(3, true, &a, &b, &signs, &signs, eps)
        .expect("had_pre 3");
    let gated = backend.sigmoid_mul(&a, &b).expect("sigmoid_mul");
    let ref_had3 = backend
        .hadamard_signed(&gated, &signs, 512, 0)
        .expect("had3");
    assert_eq!(to_cpu(&had3), to_cpu(&ref_had3), "mode 3 hadamard differs");

    // ---- multi-row: one workgroup per (row, 512-block). The sign table covers a
    // *single* row, so every row must use the same row-local sign pattern. This is
    // exactly the property the multi-column forward relies on. The sign table is
    // deliberately non-periodic: the previous parity-2 table made a global
    // `signs[bb + j]` index (row*cols + block*512 + j) coincide with the row-local
    // one, so it could not catch a regression. With a one-row table a global index
    // would also read out of bounds. ----
    let rows = 3usize;
    let rcols = 1024usize; // 2 blocks per row
    let n = rows * rcols;
    let ra: Vec<f32> = (0..n).map(|i| phase(i) * 1.3).collect();
    let rb: Vec<f32> = (0..n).map(|i| phase(i + 7) * 0.6).collect();
    let rs: Vec<f32> = (0..rcols)
        .map(|i| {
            if (i * 2654435761usize) % 5 < 2 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    let ra = Tensor::from_f32(&ra, &[rows, rcols]).to(dev).unwrap();
    let rb = Tensor::from_f32(&rb, &[rows, rcols]).to(dev).unwrap();
    let rs = Tensor::from_f32(&rs, &[rcols]).to(dev).unwrap();
    let rw: Vec<f32> = (0..rcols).map(|i| 0.5 + phase(i + 5) * 0.5).collect();
    let rw = Tensor::from_f32(&rw, &[rcols]).to(dev).unwrap();

    let (mres, mnorm, mhad) = backend
        .had_pre(0, true, &ra, &rb, &rw, &rs, eps)
        .expect("had_pre multi-row");

    // The authoritative check: bit-identical to `rows` independent single-row calls.
    for r in 0..rows {
        let (sres, snorm, shad) = backend
            .had_pre(
                0,
                true,
                &ra.slice_rows(r, 1).expect("slice a"),
                &rb.slice_rows(r, 1).expect("slice b"),
                &rw,
                &rs,
                eps,
            )
            .expect("had_pre single row");
        let off = r * rcols;
        let row = |t: &Tensor| to_cpu(t)[off..off + rcols].to_vec();
        assert_eq!(row(&mres), to_cpu(&sres), "row {r} residual differs");
        assert_eq!(row(&mnorm), to_cpu(&snorm), "row {r} normed differs");
        assert_eq!(row(&mhad), to_cpu(&shad), "row {r} hadamard differs");
    }

    // And in agreement with the multi-row standalone kernels.
    let (rres, rnorm) = backend
        .add_rms_norm(&ra, &rb, &rw, eps)
        .expect("add_rms_norm multi-row");
    let rhad = backend
        .hadamard_signed(&rnorm, &rs, 512, 0)
        .expect("hadamard multi-row");
    assert_eq!(to_cpu(&mres), to_cpu(&rres), "multi-row residual differs");
    assert_eq!(to_cpu(&mnorm), to_cpu(&rnorm), "multi-row normed differs");
    assert_eq!(to_cpu(&mhad), to_cpu(&rhad), "multi-row hadamard differs");
}

/// `had = false` must produce exactly the pre-op value: for mode 0, the same
/// `(residual, normed)` as the `had = true` path and a `out_had` equal to
/// `out_norm`. A model with unfolded weights reads that third output as its
/// matvec input, so it has to be the plain normalized activation, not a rotated
/// one.
#[test]
fn test_vulkan_had_pre_without_hadamard() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);
    let (rows, cols) = (2usize, 1024usize);
    let phase = |i: usize| (i as f32 * 0.011).cos();
    let a: Vec<f32> = (0..rows * cols).map(|i| phase(i) * 1.7).collect();
    let b: Vec<f32> = (0..rows * cols).map(|i| phase(i + 3) * 0.4).collect();
    let w: Vec<f32> = (0..cols).map(|i| 0.5 + phase(i + 5) * 0.5).collect();
    // Deliberately a *non-trivial* sign table: if the kernel applied it when
    // `had = false` the third output would differ from the second.
    let signs: Vec<f32> = (0..cols)
        .map(|i| {
            if (i * 2654435761usize) % 5 < 2 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    let eps = 1e-6f32;
    let a = Tensor::from_f32(&a, &[rows, cols]).to(dev).unwrap();
    let b = Tensor::from_f32(&b, &[rows, cols]).to(dev).unwrap();
    let w = Tensor::from_f32(&w, &[cols]).to(dev).unwrap();
    let signs = Tensor::from_f32(&signs, &[cols]).to(dev).unwrap();
    let to_cpu = |t: &Tensor| {
        t.to(kore_core::Device::Cpu)
            .unwrap()
            .as_f32_slice()
            .unwrap()
            .to_vec()
    };

    // mode 0, no Hadamard.
    let (res, norm, had) = backend
        .had_pre(0, false, &a, &b, &w, &signs, eps)
        .expect("had_pre had=false");
    assert_eq!(to_cpu(&had), to_cpu(&norm), "out_had must be out_norm");

    // Same residual/norm as the folded path, so the flag cannot perturb the norm.
    let (res_t, norm_t, _) = backend
        .had_pre(0, true, &a, &b, &w, &signs, eps)
        .expect("had_pre had=true");
    assert_eq!(to_cpu(&res), to_cpu(&res_t), "residual differs");
    assert_eq!(to_cpu(&norm), to_cpu(&norm_t), "normed differs");

    // And the unfolded `out_had` equals the standalone residual+norm, so the
    // unfused fallback path would agree too.
    let (sres, snorm) = backend.add_rms_norm(&a, &b, &w, eps).expect("add_rms_norm");
    assert_eq!(to_cpu(&res), to_cpu(&sres), "vs standalone residual");
    assert_eq!(to_cpu(&had), to_cpu(&snorm), "vs standalone normed");

    // mode 2 (swiglu) must equal the standalone swiglu.
    let (_, _, act) = backend
        .had_pre(2, false, &a, &b, &signs, &signs, eps)
        .expect("had_pre mode 2 had=false");
    let sw = backend.swiglu(&a, &b).expect("swiglu");
    assert_eq!(to_cpu(&act), to_cpu(&sw), "mode 2 vs swiglu");

    // mode 3 must equal a * sigmoid(b), computed here. `exp` is the one
    // transcendental in these kernels, and the GPU's differs from the host's in
    // the last ulp, so this leg is checked to a relative tolerance rather than
    // bit-exactly (the kernels' *own* arithmetic is exact elsewhere).
    let (_, _, gate) = backend
        .had_pre(3, false, &a, &b, &signs, &signs, eps)
        .expect("had_pre mode 3 had=false");
    let got = to_cpu(&gate);
    let av = to_cpu(&a);
    let bv = to_cpu(&b);
    for i in 0..rows * cols {
        let want = av[i] / (1.0 + (-bv[i]).exp());
        assert!(
            (got[i] - want).abs() <= 4.0 * f32::EPSILON * want.abs().max(1.0),
            "mode 3 element {i}: got {} want {want}",
            got[i]
        );
    }

    // The unfolded path skips the transform, so a sign table of the wrong width
    // must not be an error -- there is nothing to index it with.
    let tiny = Tensor::from_f32(&[0.0f32], &[1]).to(dev).unwrap();
    backend
        .had_pre(0, false, &a, &b, &w, &tiny, eps)
        .expect("had=false must not require a full-width sign table");
}

/// `hadamard_signed` on an `[rows, cols]` input must be *bit-identical* to `rows`
/// single-row calls: the transform and its sign pattern are per-row, and `signs`
/// is one row wide. The multi-column forward batches the DeltaNet output through
/// this op, so a global sign index would corrupt every row past the first.
#[test]
fn test_vulkan_hadamard_signed_multi_row_bit_identical() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);
    let (rows, cols) = (4usize, 1024usize); // 2 blocks per row
    let phase = |i: usize| (i as f32 * 0.013).sin();
    let x: Vec<f32> = (0..rows * cols).map(|i| phase(i) * 2.1).collect();
    // Single-row, non-periodic sign table.
    let signs: Vec<f32> = (0..cols)
        .map(|i| {
            if (i * 2246822519usize) % 7 < 3 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    let x = Tensor::from_f32(&x, &[rows, cols]).to(dev).unwrap();
    let signs = Tensor::from_f32(&signs, &[cols]).to(dev).unwrap();
    let to_cpu = |t: &Tensor| {
        t.to(kore_core::Device::Cpu)
            .unwrap()
            .as_f32_slice()
            .unwrap()
            .to_vec()
    };

    for mode in 0u32..2u32 {
        let all = backend
            .hadamard_signed(&x, &signs, 512, mode)
            .expect("hadamard_signed multi-row");
        for r in 0..rows {
            let one = backend
                .hadamard_signed(&x.slice_rows(r, 1).expect("slice"), &signs, 512, mode)
                .expect("hadamard_signed single row");
            let off = r * cols;
            assert_eq!(
                to_cpu(&all)[off..off + cols],
                to_cpu(&one)[..],
                "mode {mode} row {r} differs from the single-row call"
            );
        }
    }
}

/// The wide Q2_0 path walks one code word (16 elements) per lane per row, so a
/// warp covers an 8-block group per k-iteration: `bpr/8` iterations, plus a
/// partial group if `bpr` is not a multiple of 8. The host requires
/// `k % 512 == 0` (so `bpr` is always a multiple of 8 here), which means the
/// partial group is reachable only through `DeviceBackend` implementations that
/// skip that guard -- the shader still has to be correct for it. Sweep `k` over
/// the legal multiples of 512 and `n` across the 16-row workgroup tile, its
/// remainder, and multiple workgroups.
#[test]
fn test_vulkan_gemv_q2_0_wide_path_sweep() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);

    for &k in &[512usize, 1024, 1536, 2048, 2560, 4096, 6144] {
        let bpr = k / 64;
        let row_bytes = bpr * 18;
        for &n in &[1usize, 7, 16, 17, 33, 64, 65] {
            let mut w = vec![0u8; n * row_bytes];
            for r in 0..n {
                for b in 0..bpr {
                    let off = r * row_bytes + b * 18;
                    w[off] = 0x00;
                    w[off + 1] = 0x34; // fp16 scale 0.25
                    for i in 0..16 {
                        w[off + 2 + i] = ((r * 7 + b * 31 + i * 17 + 3) % 256) as u8;
                    }
                }
            }
            let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.13 - 3.7).sin()).collect();

            let mut want = vec![0.0f32; n];
            kore_kernels::q2_0::gemv_q2_0(&w, k, n, &x, &mut want).unwrap();

            let packed = kore_kernels::q2_0::repack_q2_0(&w, k, n).unwrap();
            let nb = packed.len();
            let ws = kore_core::Storage::from_bytes(kore_core::DType::U8, nb, packed).unwrap();
            let wt = Tensor::from_storage(ws, &[nb]).to(dev).unwrap();
            let xt = Tensor::from_f32(&x, &[1, k]).to(dev).unwrap();

            let got = backend.gemv_q2_0(&xt, &wt, n, k).expect("gemv_q2_0");
            let got_cpu = got.to(kore_core::Device::Cpu).unwrap();
            let g = got_cpu.as_f32_slice().unwrap();
            for i in 0..n {
                assert!(
                    (g[i] - want[i]).abs() <= 1e-3 * (1.0 + want[i].abs()),
                    "k={k} n={n} out[{i}]: gpu {} cpu {}",
                    g[i],
                    want[i]
                );
            }
        }
    }
}

/// A Q2_0 output row must be bit-identical whether it is evaluated at m=1 or
/// inside a batched call. That is what makes batched spec-decode verification
/// lossless, so assert equality rather than a tolerance.
#[test]
fn test_vulkan_gemv_q2_0_batch_invariance() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);

    let k = 1024usize;
    let n = 19usize;
    let m = 4usize;
    let bpr = k / 64;
    let row_bytes = bpr * 18;
    let mut w = vec![0u8; n * row_bytes];
    for r in 0..n {
        for b in 0..bpr {
            let off = r * row_bytes + b * 18;
            w[off] = 0x00;
            w[off + 1] = 0x34;
            for i in 0..16 {
                w[off + 2 + i] = ((r * 11 + b * 7 + i * 29 + 5) % 256) as u8;
            }
        }
    }
    let packed = kore_kernels::q2_0::repack_q2_0(&w, k, n).unwrap();
    let nb = packed.len();
    let ws = kore_core::Storage::from_bytes(kore_core::DType::U8, nb, packed).unwrap();
    let wt = Tensor::from_storage(ws, &[nb]).to(dev).unwrap();

    let x: Vec<f32> = (0..(m * k))
        .map(|i| (i as f32 * 0.23 - 1.1).cos())
        .collect();
    let xm = Tensor::from_f32(&x, &[m, k]).to(dev).unwrap();
    let wide = backend.gemv_q2_0(&xm, &wt, n, k).expect("m=4");
    let wide_cpu = wide.to(kore_core::Device::Cpu).unwrap();
    let wv = wide_cpu.as_f32_slice().unwrap();

    for row in 0..m {
        let x1 = Tensor::from_f32(&x[row * k..(row + 1) * k], &[1, k])
            .to(dev)
            .unwrap();
        let single = backend.gemv_q2_0(&x1, &wt, n, k).expect("m=1");
        let sc = single.to(kore_core::Device::Cpu).unwrap();
        let sv = sc.as_f32_slice().unwrap();
        assert_eq!(
            &wv[row * n..(row + 1) * n],
            sv,
            "row {row}: m={m} differs from m=1"
        );
    }
}

/// The multi-column forward's first kernel gate.
///
/// `qknorm_rope` with `rows = m` must be *bit-identical* to `m` separate
/// single-row calls, each at its own position. RoPE is position-dependent, so
/// this is the one op in a decode batch where "batched" and "m single steps"
/// have to agree exactly rather than approximately: a verify pass that wants to
/// accept a draft on identical logits cannot tolerate a reassociated rotation.
#[test]
fn test_vulkan_qknorm_rope_per_row_positions_bit_identical() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);
    let (n_heads, n_kv, head_dim, rope_dim) = (8usize, 2usize, 256usize, 64usize);
    let (theta, eps) = (1e7f32, 1e-5f32);
    let rows = 3usize;
    let pos0 = 1234usize;
    let phase = |i: usize| (i as f32 * 0.037).sin();

    let qg_dim = 2 * n_heads * head_dim;
    let kk_dim = n_kv * head_dim;
    let qg_data: Vec<f32> = (0..rows * qg_dim).map(|i| phase(i) * 1.3).collect();
    let kk_data: Vec<f32> = (0..rows * kk_dim).map(|i| phase(i + 91) * 0.7).collect();
    let qn_data: Vec<f32> = (0..head_dim).map(|i| 0.5 + phase(i + 3) * 0.5).collect();
    let kn_data: Vec<f32> = (0..head_dim).map(|i| 0.5 + phase(i + 7) * 0.5).collect();

    let qn = Tensor::from_f32(&qn_data, &[head_dim]).to(dev).unwrap();
    let kn = Tensor::from_f32(&kn_data, &[head_dim]).to(dev).unwrap();

    let to_cpu = |t: &Tensor| {
        t.to(kore_core::Device::Cpu)
            .unwrap()
            .as_f32_slice()
            .unwrap()
            .to_vec()
    };

    // Batched: one dispatch, `rows` columns, positions pos0 .. pos0 + rows - 1.
    let qg = Tensor::from_f32(&qg_data, &[rows, qg_dim]).to(dev).unwrap();
    let kk = Tensor::from_f32(&kk_data, &[rows, kk_dim]).to(dev).unwrap();
    let (q_b, k_b, g_b) = backend
        .qknorm_rope(
            &qg, &kk, &qn, &kn, n_heads, n_kv, head_dim, rope_dim, pos0, theta, eps,
        )
        .expect("rows = 3");
    let (q_b, k_b, g_b) = (to_cpu(&q_b), to_cpu(&k_b), to_cpu(&g_b));

    // Reference: the same rows one at a time, each with its own position. A
    // strided slice of the batch output must equal the standalone row exactly.
    let q_stride = n_heads * head_dim;
    let k_stride = n_kv * head_dim;
    for r in 0..rows {
        let qg1 = Tensor::from_f32(&qg_data[r * qg_dim..(r + 1) * qg_dim], &[1, qg_dim])
            .to(dev)
            .unwrap();
        let kk1 = Tensor::from_f32(&kk_data[r * kk_dim..(r + 1) * kk_dim], &[1, kk_dim])
            .to(dev)
            .unwrap();
        let (q1, k1, g1) = backend
            .qknorm_rope(
                &qg1,
                &kk1,
                &qn,
                &kn,
                n_heads,
                n_kv,
                head_dim,
                rope_dim,
                pos0 + r,
                theta,
                eps,
            )
            .expect("rows = 1");
        assert_eq!(
            &q_b[r * q_stride..(r + 1) * q_stride],
            to_cpu(&q1).as_slice(),
            "q row {r} differs from the standalone call at pos {}",
            pos0 + r
        );
        assert_eq!(
            &k_b[r * k_stride..(r + 1) * k_stride],
            to_cpu(&k1).as_slice(),
            "k row {r} differs from the standalone call at pos {}",
            pos0 + r
        );
        assert_eq!(
            &g_b[r * q_stride..(r + 1) * q_stride],
            to_cpu(&g1).as_slice(),
            "gate row {r} differs from the standalone call at pos {}",
            pos0 + r
        );
    }
}

/// The multi-column forward's KV *write* gate.
///
/// `kv_cache_append_q8` with `num_new = m` rows at `row_offset` must be
/// *bit-identical* to `m` single-row appends at `row_offset + r`. Both halves of
/// the append are contiguous range copies — the packed-INT8 K/V run and the f32
/// per-token scales (elementwise `op_type = 8`) — so this holds by construction,
/// which is precisely why it is worth pinning: the whole multi-column design
/// assumes it, and a future "optimization" that word-strides the copy by row, or
/// writes per-row offsets, would break the verify pass silently and only at m > 1.
///
/// The K/V compare is byte-for-byte over the entire resident cache (not just the
/// appended window) so a stray write outside `[row_offset, row_offset + m)` fails
/// here too.
#[test]
fn test_vulkan_kv_cache_append_q8_multi_row_bit_identical() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);

    let kv_dim = 32usize; // multiple of 4: packed-u32 requirement
    let max_ctx = 16usize;
    let rows = 3usize;
    let row_offset = 4usize;

    let k_data: Vec<f32> = (0..rows * kv_dim)
        .map(|i| (i as f32 * 0.23).cos() * 3.0)
        .collect();
    let v_data: Vec<f32> = (0..rows * kv_dim)
        .map(|i| (i as f32 * 0.31).sin() * 2.0)
        .collect();
    let mut k_all = Vec::new();
    let mut v_all = Vec::new();
    let mut k_scales = Vec::new();
    let mut v_scales = Vec::new();
    for t in 0..rows {
        let (kb, ks) = q8_quant_row(&k_data[t * kv_dim..(t + 1) * kv_dim]);
        let (vb, vs) = q8_quant_row(&v_data[t * kv_dim..(t + 1) * kv_dim]);
        k_all.extend_from_slice(&kb);
        v_all.extend_from_slice(&vb);
        k_scales.push(ks);
        v_scales.push(vs);
    }

    let u8_rows = |bytes: &[u8], n: usize| {
        Tensor::from_storage(
            kore_core::Storage::from_bytes(kore_core::DType::U8, bytes.len(), bytes.to_vec())
                .unwrap(),
            &[n, kv_dim],
        )
        .to(dev)
        .unwrap()
    };

    // A fresh full-capacity cache pair, so the whole underlying buffer is comparable.
    let fresh = || {
        (
            Tensor::zeros(&[max_ctx, kv_dim], kore_core::DType::U8)
                .to(dev)
                .unwrap(),
            Tensor::zeros(&[max_ctx, kv_dim], kore_core::DType::U8)
                .to(dev)
                .unwrap(),
            Tensor::from_f32(&vec![0.0f32; max_ctx], &[max_ctx])
                .to(dev)
                .unwrap(),
            Tensor::from_f32(&vec![0.0f32; max_ctx], &[max_ctx])
                .to(dev)
                .unwrap(),
        )
    };

    // Arm A: one dispatch, `rows` columns at `row_offset`.
    let (kc_a, vc_a, ks_a, vs_a) = fresh();
    let (kv_a, vv_a) = backend
        .kv_cache_append_q8(
            &kc_a,
            &vc_a,
            &ks_a,
            &vs_a,
            &u8_rows(&k_all, rows),
            &u8_rows(&v_all, rows),
            &Tensor::from_f32(&k_scales, &[rows]).to(dev).unwrap(),
            &Tensor::from_f32(&v_scales, &[rows]).to(dev).unwrap(),
            row_offset,
        )
        .expect("batched append");

    // Arm B: the same rows one at a time, each at its own offset.
    let (kc_b, vc_b, ks_b, vs_b) = fresh();
    let mut last: Option<(Tensor, Tensor)> = None;
    for r in 0..rows {
        let (kv, vv) = backend
            .kv_cache_append_q8(
                &kc_b,
                &vc_b,
                &ks_b,
                &vs_b,
                &u8_rows(&k_all[r * kv_dim..(r + 1) * kv_dim], 1),
                &u8_rows(&v_all[r * kv_dim..(r + 1) * kv_dim], 1),
                &Tensor::from_f32(&k_scales[r..r + 1], &[1]).to(dev).unwrap(),
                &Tensor::from_f32(&v_scales[r..r + 1], &[1]).to(dev).unwrap(),
                row_offset + r,
            )
            .expect("single-row append");
        last = Some((kv, vv));
    }
    let (kv_b, vv_b) = last.expect("rows > 0");

    // Compare raw storage bytes: U8 cache and f32 scales alike, bit for bit.
    let raw = |t: &Tensor| {
        t.to(kore_core::Device::Cpu)
            .unwrap()
            .storage_ref()
            .as_bytes()
            .to_vec()
    };
    assert_eq!(
        raw(&kv_a),
        raw(&kv_b),
        "K cache bytes differ: batched m={rows} vs {rows} single-row appends"
    );
    assert_eq!(
        raw(&vv_a),
        raw(&vv_b),
        "V cache bytes differ: batched m={rows} vs {rows} single-row appends"
    );
    assert_eq!(
        raw(&ks_a),
        raw(&ks_b),
        "K scales differ: batched m={rows} vs {rows} single-row appends"
    );
    assert_eq!(
        raw(&vs_a),
        raw(&vs_b),
        "V scales differ: batched m={rows} vs {rows} single-row appends"
    );

    // Non-vacuous: the appended window must actually contain the rows.
    let kc_bytes = raw(&kv_a);
    assert_eq!(
        &kc_bytes[row_offset * kv_dim..(row_offset + rows) * kv_dim],
        k_all.as_slice(),
        "appended K window does not hold the source rows"
    );
}

/// `scatter_row` is the inverse of `slice_rows`, and the multi-column forward's
/// only new host↔device plumbing: the DeltaNet scan has to run one row at a time
/// (its state carries across rows) but everything downstream of it is batched, so
/// the per-row outputs are written back into an `[m, cols]` staging tensor rather
/// than copied to the host and back. Out-of-order writes must land in the right
/// rows, and a slice-then-scatter round trip must be bit-exact.
#[test]
fn test_vulkan_scatter_row_is_inverse_of_slice_rows() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);
    let (rows, cols) = (5usize, 96usize);
    let phase = |i: usize| (i as f32 * 0.07).sin() * 3.5;

    // Deliberately not in row order.
    let dst = Tensor::zeros(&[rows, cols], kore_core::DType::F32)
        .to(dev)
        .unwrap();
    let mut want = vec![0.0f32; rows * cols];
    for r in [3usize, 1, 4, 0, 2] {
        let row_data: Vec<f32> = (0..cols).map(|c| phase(r * cols + c) + r as f32).collect();
        let src = Tensor::from_f32(&row_data, &[cols]).to(dev).unwrap();
        backend.scatter_row(&dst, r, &src).expect("scatter_row");
        want[r * cols..(r + 1) * cols].copy_from_slice(&row_data);
    }
    let got = dst.to(kore_core::Device::Cpu).unwrap();
    assert_eq!(
        got.as_f32_slice().unwrap(),
        want.as_slice(),
        "scatter_row placed rows in the wrong slots"
    );

    // Round trip through `slice_rows` into a fresh tensor, which is the shape the
    // forward actually uses (slice returns `[1, cols]`).
    let dst2 = Tensor::zeros(&[rows, cols], kore_core::DType::F32)
        .to(dev)
        .unwrap();
    for r in 0..rows {
        let row = backend.slice_rows(&dst, r, 1).expect("slice_rows");
        backend.scatter_row(&dst2, r, &row).expect("scatter_row");
    }
    let rt = dst2.to(kore_core::Device::Cpu).unwrap();
    assert_eq!(
        rt.as_f32_slice().unwrap(),
        want.as_slice(),
        "slice/scatter round trip is not bit-exact"
    );
}

/// `copy_buffer` is a plain device memcpy, and it has to be a *real* copy: the
/// DeltaNet recurrence updates its state buffer in place, so the only way to
/// rewind a rejected verify batch is to have kept the bytes from before it.
/// That means writing to `dst` must not touch `src`, however the driver has
/// aliased the two allocations.
#[test]
fn test_vulkan_copy_buffer_is_a_real_copy() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);
    // Deliberately not a multiple of the 256-element workgroup for the tail.
    let n = 1000usize;
    let src_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin() * 7.0).collect();

    let src = Tensor::from_f32(&src_data, &[n]).to(dev).unwrap();
    let dst = Tensor::zeros(&[n], kore_core::DType::F32).to(dev).unwrap();
    backend.copy_buffer(&dst, &src).expect("copy_buffer");

    let got = dst.to(kore_core::Device::Cpu).unwrap();
    assert_eq!(got.as_f32_slice().unwrap(), src_data.as_slice());

    // Clear `dst` on device and copy again -- if the first copy had aliased the
    // two storages, `src` would have been clobbered by this point.
    let zeros = Tensor::zeros(&[n], kore_core::DType::F32).to(dev).unwrap();
    backend
        .copy_buffer(&dst, &zeros)
        .expect("copy_buffer zeros");
    backend.copy_buffer(&dst, &src).expect("copy_buffer again");

    let got2 = dst.to(kore_core::Device::Cpu).unwrap();
    assert_eq!(got2.as_f32_slice().unwrap(), src_data.as_slice());

    // A mismatched length is a caller bug, not a silent partial copy.
    let short = Tensor::zeros(&[n - 1], kore_core::DType::F32)
        .to(dev)
        .unwrap();
    assert!(backend.copy_buffer(&dst, &short).is_err());
}

/// A 1-D activation is one row, not `k` rows.
///
/// The DeltaNet readout produces a `[n_v*d_v]` vector, and an *unfolded* model
/// passes it on unchanged, so every quantized GEMV has to read a 1-D input as
/// `m = 1`. Reading `x_shape[0]` as `m` silently produced a `[k, k]` output
/// instead of `[1, n]`, which then re-entered the next fused norm as a
/// `[k, k]` residual. Only reachable on a file with no folded Hadamard, which
/// is why the Q2_0 path (which had always done it correctly) never hit it.
#[test]
fn test_vulkan_k_quant_gemv_treats_1d_activation_as_one_row() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // One Q5_K row of 256 columns: d = 1.0, dmin = 0, scales = 1, mins = 0,
    // qs byte 0x32 -> nibbles 2 and 3, so the dot with an all-ones x is
    // 128*2 + 128*3 = 640.
    let mut raw = vec![0u8; 176];
    raw[1] = 0x3C; // d = 1.0
    for i in 4..8 {
        raw[i] = 1;
    }
    for i in 12..16 {
        raw[i] = 1;
    }
    for i in 48..176 {
        raw[i] = 0x32;
    }
    let w = Tensor::from_storage(
        kore_core::Storage::from_packed_bytes(kore_core::DType::U8, 176, raw),
        &[176],
    );

    let ones = vec![1.0f32; 256];
    let x_2d = Tensor::from_f32(&ones, &[1, 256]);
    let y_2d = backend.gemv_q5_k(&x_2d, &w, 1, 256).expect("q5_k 2-D");
    assert_eq!(y_2d.shape().dims(), &[1, 1]);

    let x_1d = Tensor::from_f32(&ones, &[256]);
    let y_1d = backend.gemv_q5_k(&x_1d, &w, 1, 256).expect("q5_k 1-D");
    assert_eq!(
        y_1d.shape().dims(),
        &[1, 1],
        "a 1-D activation must produce a [1, n] output"
    );

    let a = y_2d.to(kore_core::Device::Cpu).unwrap();
    let b = y_1d.to(kore_core::Device::Cpu).unwrap();
    assert_eq!(
        a.as_f32_slice().unwrap(),
        b.as_f32_slice().unwrap(),
        "1-D and [1, k] activations must agree bit-for-bit"
    );
    assert!((a.as_f32_slice().unwrap()[0] - 640.0).abs() < 1e-3);

    // A 1-D input of the wrong length is still a caller bug.
    let short = Tensor::from_f32(&vec![1.0f32; 128], &[128]);
    assert!(backend.gemv_q5_k(&short, &w, 1, 256).is_err());

    // The shape rule belongs to all three K-quants, not just Q5_K. The value
    // paths are covered by the per-quant correctness tests above.
    let q4k = Tensor::from_storage(
        kore_core::Storage::from_packed_bytes(kore_core::DType::U8, 144, vec![0u8; 144]),
        &[144],
    );
    let q6k = Tensor::from_storage(
        kore_core::Storage::from_packed_bytes(kore_core::DType::U8, 210, vec![0u8; 210]),
        &[210],
    );
    let y4 = backend.gemv_q4_k(&x_1d, &q4k, 1, 256).unwrap();
    assert_eq!(y4.shape().dims(), &[1, 1]);
    let y6 = backend.gemv_q6_k(&x_1d, &q6k, 1, 256).unwrap();
    assert_eq!(y6.shape().dims(), &[1, 1]);
}

/// GQA inside the linear attention: `n_v` value heads read `n_k` key heads.
///
/// Value head `hv` borrows key head `hv % n_k` -- the key heads are *tiled* up
/// to the value head count, which is what llama.cpp's `ggml_gated_delta_net`
/// does (`iq1 = iv1 % neq1`, and `ggml_repeat` in the unfused path). Grouping
/// them as `hv / (n_v / n_k)` instead is a silent, whole-model-sized error.
///
/// The oracle avoids restating the rule: explicitly tile q/k up to `n_v` heads
/// and run the kernel with `n_k == n_v`, where the mapping is the identity.
/// Output *and* recurrent state must then match bit-for-bit.
#[test]
fn test_vulkan_gdn_step_groups_key_heads_by_tiling() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let dev = kore_core::Device::Vulkan(0);

    // The kernel stages one head of k/q/v in a 128-wide workgroup, and thread
    // `j` owns state column `j`, so both head dims are fixed at 128.
    const DK: usize = 128;
    const DV: usize = 128;
    // `n_v / n_k` must be > 1 for the two conventions to disagree: tiling reads
    // `hv % n_k` = [0, 1, 0, 1] where interleaving reads `hv / 2` = [0, 0, 1, 1].
    const NK: usize = 2;
    const NV: usize = 4;
    const EPS: f32 = 1e-6;

    let mut s: u32 = 0x2545_f491;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 16_777_216.0) - 0.5
            })
            .collect()
    };

    let q = rnd(NK * DK);
    let k = rnd(NK * DK);
    let v = rnd(NV * DV);
    let z = rnd(NV * DV);
    let beta = rnd(NV);
    // Negative gates keep `exp(gate) < 1`, so the state decays rather than
    // growing without bound.
    let gate: Vec<f32> = rnd(NV).into_iter().map(|g| -0.5 - g.abs()).collect();
    let norm_w: Vec<f32> = rnd(DV).into_iter().map(|w| 0.5 + w.abs()).collect();
    let state0 = rnd(NV * DV * DV);

    // `n_k == n_k`, the layout the file actually stores.
    let mut qkv_grouped = Vec::with_capacity(2 * NK * DK + NV * DV);
    qkv_grouped.extend_from_slice(&q);
    qkv_grouped.extend_from_slice(&k);
    qkv_grouped.extend_from_slice(&v);

    // q/k tiled up to `n_v` heads: `n_k == n_v`, so the mapping is identity.
    let mut qkv_tiled = Vec::with_capacity(2 * NV * DK + NV * DV);
    for _ in 0..(NV / NK) {
        qkv_tiled.extend_from_slice(&q);
    }
    for _ in 0..(NV / NK) {
        qkv_tiled.extend_from_slice(&k);
    }
    qkv_tiled.extend_from_slice(&v);

    let up = |data: &[f32]| -> Tensor { Tensor::from_f32(data, &[data.len()]).to(dev).unwrap() };
    let z_t = up(&z);
    let beta_t = up(&beta);
    let gate_t = up(&gate);
    let nw_t = up(&norm_w);
    let state_grouped = up(&state0);
    let state_tiled = up(&state0);

    let out_grouped = VulkanBackend::gdn_step(
        &backend,
        &up(&qkv_grouped),
        &z_t,
        &state_grouped,
        &beta_t,
        &gate_t,
        &nw_t,
        NV,
        NK,
        DK,
        DV,
        EPS,
    )
    .expect("gdn_step grouped");
    let out_tiled = VulkanBackend::gdn_step(
        &backend,
        &up(&qkv_tiled),
        &z_t,
        &state_tiled,
        &beta_t,
        &gate_t,
        &nw_t,
        NV,
        NV,
        DK,
        DV,
        EPS,
    )
    .expect("gdn_step tiled");

    let og = out_grouped.to(kore_core::Device::Cpu).unwrap();
    let ot = out_tiled.to(kore_core::Device::Cpu).unwrap();
    assert_eq!(og.shape().dims(), &[NV * DV]);
    assert_eq!(
        og.as_f32_slice().unwrap(),
        ot.as_f32_slice().unwrap(),
        "value heads must read key heads by tiling"
    );

    let sg = state_grouped.to(kore_core::Device::Cpu).unwrap();
    let st = state_tiled.to(kore_core::Device::Cpu).unwrap();
    assert_eq!(
        sg.as_f32_slice().unwrap(),
        st.as_f32_slice().unwrap(),
        "the recurrent state must diverge too if the head pairing is wrong"
    );
    // Guard against the test passing because nothing happened at all.
    assert!(any_nonzero(og.as_f32_slice().unwrap()));
    assert!(any_nonzero(sg.as_f32_slice().unwrap()));
}

/// True when at least one entry is finite and non-zero -- so a comparison of
/// two "all zeros because the kernel never ran" results cannot pass silently.
fn any_nonzero(x: &[f32]) -> bool {
    x.iter().any(|v| v.is_finite() && *v != 0.0)
}
