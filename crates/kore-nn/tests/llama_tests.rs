use kore_attention::KvCacheQuantMode;
use kore_core::{Device, Tensor};
use kore_nn::llama::{LlamaAttention, LlamaConfig, LlamaDecoderLayer, LlamaMlp, LlamaModel, PromptLookupDrafter};
use kore_nn::{Embedding, Linear, Module, RMSNorm};

/// Deterministic all-zero toy model: every logits row is uniform, so greedy
/// sampling always emits token 0 — guaranteeing PLD n-gram drafts fire.
fn make_zero_toy_llama() -> LlamaModel {
    let config = LlamaConfig {
        vocab_size: 100,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        head_dim: 16,
    };
    let z = |r: usize, c: usize| Tensor::zeros(&[r, c], kore_core::DType::F32);
    let mut layers = Vec::new();
    for _ in 0..config.num_hidden_layers {
        let self_attn = LlamaAttention::new(
            Linear::from_weight(z(config.num_attention_heads * config.head_dim, config.hidden_size), None),
            Linear::from_weight(z(config.num_key_value_heads * config.head_dim, config.hidden_size), None),
            Linear::from_weight(z(config.num_key_value_heads * config.head_dim, config.hidden_size), None),
            Linear::from_weight(z(config.hidden_size, config.num_attention_heads * config.head_dim), None),
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rope_theta,
            config.rope_inv_freq(),
        );
        let mlp = LlamaMlp::new(
            Linear::from_weight(z(config.intermediate_size, config.hidden_size), None),
            Linear::from_weight(z(config.intermediate_size, config.hidden_size), None),
            Linear::from_weight(z(config.hidden_size, config.intermediate_size), None),
        );
        layers.push(LlamaDecoderLayer {
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            self_attn,
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            mlp,
        });
    }
    let embed_tokens = Embedding::from_weight(z(config.vocab_size, config.hidden_size));
    let lm_head = Linear::from_weight(z(config.vocab_size, config.hidden_size), None);
    LlamaModel {
        config: config.clone(),
        embed_tokens,
        layers,
        norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        lm_head,
        device: Device::Cpu,
        mtp_draft_head: None,
    }
}

fn make_toy_llama() -> LlamaModel {
    let config = LlamaConfig {
        vocab_size: 100,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        head_dim: 16,
    };

    let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size);
    let mut layers = Vec::new();

    for _ in 0..config.num_hidden_layers {
        let input_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps);
        let q_proj = Linear::new(config.hidden_size, config.num_attention_heads * config.head_dim, false);
        let k_proj = Linear::new(config.hidden_size, config.num_key_value_heads * config.head_dim, false);
        let v_proj = Linear::new(config.hidden_size, config.num_key_value_heads * config.head_dim, false);
        let o_proj = Linear::new(config.num_attention_heads * config.head_dim, config.hidden_size, false);

        let self_attn = LlamaAttention::new(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rope_theta,
            config.rope_inv_freq(),
        );

        let post_attention_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps);
        let gate_proj = Linear::new(config.hidden_size, config.intermediate_size, false);
        let up_proj = Linear::new(config.hidden_size, config.intermediate_size, false);
        let down_proj = Linear::new(config.intermediate_size, config.hidden_size, false);
        let mlp = LlamaMlp::new(gate_proj, up_proj, down_proj);

        layers.push(LlamaDecoderLayer {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        });
    }

    let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps);
    let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

    LlamaModel {
        config,
        embed_tokens,
        layers,
        norm,
        lm_head,
        device: Device::Cpu,
        mtp_draft_head: None,
    }
}

#[test]
fn test_llama_forward_cpu() {
    let model = make_toy_llama();
    let mut kv_caches = model.create_kv_caches(32).expect("create kv caches");

    let token_id = 42;
    let next_token = model.step(token_id, 0, &mut kv_caches).expect("cpu step");
    println!("CPU sampled next token: {}", next_token);
    assert!(next_token < 100);
}

#[test]
fn test_llama_forward_vulkan() {
    let _backend = match kore_vulkan::VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => {
            println!("Skipping Vulkan test: no Vulkan backend available in environment");
            return;
        }
    };

    let mut model = make_toy_llama();
    println!("Transferring LlamaModel to Vulkan(0)...");
    model.to(Device::Vulkan(0)).expect("transfer to Vulkan failed");
    assert_eq!(model.device, Device::Vulkan(0));

    let mut kv_caches = model.create_kv_caches(32).expect("create kv caches on GPU");

    println!("Running GPU decode step (Pillar 2 GEMV + Pillar 3 Flash-Decode + Pillar 4 Argmax)...");
    let t0 = std::time::Instant::now();
    let next_token = model.step(42, 0, &mut kv_caches).expect("gpu step");
    let elapsed = t0.elapsed();
    println!("GPU sampled next token: {} in {:.3} ms", next_token, elapsed.as_secs_f64() * 1000.0);
    assert!(next_token < 100);
}

#[test]
fn test_minicpm5_real_gguf_gpu_forward() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("../../models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf");
    if !path.exists() {
        println!("Skipping: model file does not exist at {}", path.display());
        return;
    }

    let _backend = match kore_vulkan::VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => {
            println!("Skipping: no Vulkan backend available in environment");
            return;
        }
    };

    println!("Loading 2 layers of MiniCPM5 from GGUF...");
    let t0 = std::time::Instant::now();
    let mut model = LlamaModel::from_gguf(&path, Some(2)).expect("failed to load GGUF model");
    println!("Loaded 2 layers in {:.2}s", t0.elapsed().as_secs_f64());

    println!("Transferring 2 layers to Vulkan(0)...");
    model.to(Device::Vulkan(0)).expect("transfer to Vulkan failed");

    let mut kv_caches = model.create_kv_caches(32).expect("create kv caches on GPU");

    println!("Running GPU decode step with real MiniCPM5 weights on RTX 2070...");
    let t1 = std::time::Instant::now();
    let next_token = model.step(0, 0, &mut kv_caches).expect("gpu step 0");
    let latency0 = t1.elapsed();
    println!(
        "Step 0 (warmup) sampled token: {} in {:.3} ms ({:.1} tok/s)",
        next_token,
        latency0.as_secs_f64() * 1000.0,
        1.0 / latency0.as_secs_f64()
    );

    let mut curr_token = next_token;
    for step_idx in 1..=3 {
        let t_step = std::time::Instant::now();
        curr_token = model.step(curr_token, step_idx, &mut kv_caches).expect("gpu step");
        let lat = t_step.elapsed();
        println!(
            "Step {} sampled token: {} in {:.3} ms ({:.1} tok/s)",
            step_idx,
            curr_token,
            lat.as_secs_f64() * 1000.0,
            1.0 / lat.as_secs_f64()
        );
    }
    assert!(curr_token < model.config.vocab_size);
}

#[test]
fn test_llama_pld_speculative_and_hybrid_parity() {
    let drafter = PromptLookupDrafter::new(3);
    // History ends with [10, 20]; earlier occurrence of [10, 20] was followed by [30, 40, 50]
    let history = vec![5, 10, 20, 30, 40, 50, 99, 10, 20];
    assert_eq!(drafter.draft(&history), vec![30, 40, 50]);

    let model_cpu = make_zero_toy_llama();
    // Repeated pattern in prompt so PLD triggers draft verification
    let prompt = vec![1, 2, 3, 4, 1, 2];
    let baseline = model_cpu
        .generate(&prompt, 6, None, |_| {})
        .expect("baseline generate");
    let (spec_out, _accepted, total_drafted) = model_cpu
        .generate_with_options(&prompt, 6, None, true, KvCacheQuantMode::K8V4, |_| {})
        .expect("speculative generate");
    assert_eq!(spec_out.len(), 6);
    assert!(total_drafted > 0, "Expected PLD/MTP to draft tokens");
    // Verify F32 speculative greedy output is 100% identical to non-speculative greedy output
    let (spec_f32, _, _) = model_cpu
        .generate_with_options(&prompt, 6, None, true, KvCacheQuantMode::F32, |_| {})
        .expect("speculative f32 generate");
    assert_eq!(
        spec_f32, baseline,
        "Lossless speculative decoding must match greedy baseline token-for-token"
    );

    // If Vulkan is available, test Hybrid GPU+CPU split (1 layer on Vulkan, 1 layer on CPU)
    if kore_vulkan::VulkanBackend::new().is_ok() {
        let mut hybrid_model = make_toy_llama();
        hybrid_model
            .to_hybrid(Device::Vulkan(0), 1)
            .expect("to_hybrid");
        let hybrid_out = hybrid_model
            .generate(&prompt, 4, None, |_| {})
            .expect("hybrid generate");
        assert_eq!(hybrid_out.len(), 4);
    }
}

#[test]
fn test_q6_k_and_q4_k_real_weights_parity() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("../../models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }
    let _backend = match kore_vulkan::VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let gguf = oxillama_gguf::GgufModel::load(&path).expect("load gguf");

    // 1. Test Q6_K layer: blk.0.ffn_down.weight [6144, 2048]
    println!("Testing Q6_K parity on blk.0.ffn_down.weight...");
    let in_features = 6144;
    let out_features = 2048;
    let raw = gguf.tensor_data("blk.0.ffn_down.weight").expect("read ffn_down");
    let mut layer_gpu = Linear::from_q6_k(raw.to_vec(), in_features, out_features, None);
    let layer_cpu = Linear::from_q6_k(raw.to_vec(), in_features, out_features, None);
    layer_gpu.to(Device::Vulkan(0)).expect("to vulkan");

    let x_data: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.05).sin()).collect();
    let x_cpu = Tensor::from_f32(&x_data, &[1, in_features]);
    let x_gpu = x_cpu.to(Device::Vulkan(0)).expect("x to vulkan");

    let y_cpu = layer_cpu.forward(&x_cpu).expect("cpu forward");
    let y_gpu = layer_gpu.forward(&x_gpu).expect("gpu forward").to(Device::Cpu).expect("to cpu");

    let y_cpu_slice = y_cpu.as_f32_slice().unwrap();
    let y_gpu_slice = y_gpu.as_f32_slice().unwrap();

    let mut max_diff = 0.0f32;
    for (i, (&a, &b)) in y_cpu_slice.iter().zip(y_gpu_slice.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.05 {
            panic!("Q6_K mismatch at index {}: CPU={}, GPU={}, diff={}", i, a, b, diff);
        }
    }
    println!("Q6_K parity verified! Max diff = {}", max_diff);

    // 2. Test Q4_K layer: blk.0.attn_q.weight [2048, 2048]
    println!("Testing Q4_K parity on blk.0.attn_q.weight...");
    let in_features = 2048;
    let out_features = 2048;
    let raw_q4 = gguf.tensor_data("blk.0.attn_q.weight").expect("read attn_q");
    let mut layer_q4_gpu = Linear::from_q4_k(raw_q4.to_vec(), in_features, out_features, None);
    let layer_q4_cpu = Linear::from_q4_k(raw_q4.to_vec(), in_features, out_features, None);
    layer_q4_gpu.to(Device::Vulkan(0)).expect("to vulkan");

    let x_q4_cpu = Tensor::from_f32(&x_data[..in_features], &[1, in_features]);
    let x_q4_gpu = x_q4_cpu.to(Device::Vulkan(0)).expect("x to vulkan");

    let y_q4_cpu = layer_q4_cpu.forward(&x_q4_cpu).expect("cpu forward");
    let y_q4_gpu = layer_q4_gpu.forward(&x_q4_gpu).expect("gpu forward").to(Device::Cpu).expect("to cpu");

    let y_q4_cpu_slice = y_q4_cpu.as_f32_slice().unwrap();
    let y_q4_gpu_slice = y_q4_gpu.as_f32_slice().unwrap();

    let mut max_q4_diff = 0.0f32;
    for (i, (&a, &b)) in y_q4_cpu_slice.iter().zip(y_q4_gpu_slice.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_q4_diff {
            max_q4_diff = diff;
        }
        if diff > 0.05 {
            panic!("Q4_K mismatch at index {}: CPU={}, GPU={}, diff={}", i, a, b, diff);
        }
    }
    println!("Q4_K parity verified! Max diff = {}", max_q4_diff);
}

#[test]
fn test_debug_real_forward_step() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("../../models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }
    let _backend = match kore_vulkan::VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    println!("Loading full 42 layers of MiniCPM5 from GGUF...");
    let t0 = std::time::Instant::now();
    let mut model = LlamaModel::from_gguf(&path, None).expect("load gguf");
    println!("Loaded in {:.2}s", t0.elapsed().as_secs_f64());

    model.to(Device::Vulkan(0)).expect("to vulkan");

    let prompt = vec![608, 4894, 304, 6918, 357]; // "The capital of France is"
    let mut kv_caches = model.create_kv_caches(32).expect("create kv caches");

    for (pos, &tok) in prompt.iter().enumerate() {
        let logits = model.forward_step(tok, pos, &mut kv_caches).expect("forward step");
        let cpu_logits = logits.to(Device::Cpu).expect("to cpu");
        let slice = cpu_logits.as_f32_slice().unwrap();

        // Find top 5 tokens
        let mut indexed: Vec<(usize, f32)> = slice.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("--- Step pos={} tok={} ---", pos, tok);
        println!("  Sum of logits: {:.4} (ref pos=0: -272224.09)", slice.iter().sum::<f32>());
        println!("  Logit for 8181 (' Paris'): {:.4}", slice[8181]);
        println!("  Logit for 5 ('<|fim_prefix|>'): {:.4}", slice[5]);
        println!("  Logit for 0 ('<s>'): {:.4}", slice[0]);
        println!("  Top 5: {:?}", &indexed[..5]);
    }
}

#[test]
fn test_trace_layer0_against_llamacpp() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("../../models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }
    let _backend = match kore_vulkan::VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    println!("Loading 1 layer of MiniCPM5 from GGUF...");
    let mut model = LlamaModel::from_gguf(&path, Some(1)).expect("load gguf");
    model.to(Device::Vulkan(0)).expect("to vulkan");

    let sum = |t: &Tensor| -> f32 {
        let cpu_t = t.to(Device::Cpu).expect("to cpu");
        cpu_t.as_f32_slice().unwrap().iter().sum()
    };

    let tok_t = Tensor::from_f32(&[608.0], &[1]);
    let x = model.embed_tokens.forward(&tok_t).expect("embed").to(Device::Vulkan(0)).expect("x to vulkan");
    println!("1. embd:         got sum = {:>10.6}  (ref = -0.482270)", sum(&x));

    let norm1 = model.layers[0].input_layernorm.forward(&x).expect("norm1");
    println!("2. attn_norm:    got sum = {:>10.6}  (ref = 25.716749)", sum(&norm1));

    let q = model.layers[0].self_attn.q_proj.forward(&norm1).expect("q");
    println!("3. Qcur:         got sum = {:>10.6}  (ref = -3.494100)", sum(&q));

    let k = model.layers[0].self_attn.k_proj.forward(&norm1).expect("k");
    println!("4. Kcur:         got sum = {:>10.6}  (ref =  7.812304)", sum(&k));

    let v = model.layers[0].self_attn.v_proj.forward(&norm1).expect("v");
    println!("5. Vcur:         got sum = {:>10.6}  (ref =  2.516237)", sum(&v));

    let mut kv_caches = model.create_kv_caches(32).expect("create kv caches");
    let attn_out = model.layers[0].self_attn.forward(&norm1, 0, &mut kv_caches[0], Device::Vulkan(0)).expect("attn_out");
    println!("6. attn_out:     got sum = {:>10.6}  (ref = -0.151895)", sum(&attn_out));

    let ffn_inp = x.add(&attn_out).expect("ffn_inp");
    println!("7. ffn_inp:      got sum = {:>10.6}  (ref = -0.634165)", sum(&ffn_inp));

    let norm2 = model.layers[0].post_attention_layernorm.forward(&ffn_inp).expect("norm2");
    println!("8. ffn_norm:     got sum = {:>10.6}  (ref = -14.176463)", sum(&norm2));

    let ffn_gate = model.layers[0].mlp.gate_proj.forward(&norm2).expect("gate");
    println!("9. ffn_gate:     got sum = {:>10.6}  (ref = 385.743195)", sum(&ffn_gate));

    let ffn_up = model.layers[0].mlp.up_proj.forward(&norm2).expect("up");
    println!("10. ffn_up:      got sum = {:>10.6}  (ref = 15.537988)", sum(&ffn_up));

    let swiglu = kore_nn::activations::swiglu(&ffn_gate, &ffn_up).expect("swiglu");
    println!("11. ffn_swiglu:  got sum = {:>10.6}  (ref = -1.421464)", sum(&swiglu));

    let ffn_out = model.layers[0].mlp.down_proj.forward(&swiglu).expect("down");
    println!("12. ffn_out:     got sum = {:>10.6}  (ref =  2.737209)", sum(&ffn_out));

    let l_out = ffn_inp.add(&ffn_out).expect("l_out");
    println!("13. l_out:       got sum = {:>10.6}  (ref =  2.103047)", sum(&l_out));

    // STEP 1: token 4894 at pos 1
    println!("\n=== Step 1 (pos=1, tok=4894) ===");
    let tok_t1 = Tensor::from_f32(&[4894.0], &[1]);
    let x1 = model.embed_tokens.forward(&tok_t1).expect("embed").to(Device::Vulkan(0)).expect("x to vulkan");
    let norm1_1 = model.layers[0].input_layernorm.forward(&x1).expect("norm1");
    let q1 = model.layers[0].self_attn.q_proj.forward(&norm1_1).expect("q1");
    let k1 = model.layers[0].self_attn.k_proj.forward(&norm1_1).expect("k1");
    let backend = kore_core::device::get_backend(Device::Vulkan(0)).unwrap();
    let q1_rope = backend.rope(&q1, 1, 16, 128, 5000000.0).expect("q1_rope");
    let k1_rope = backend.rope(&k1, 1, 2, 128, 5000000.0).expect("k1_rope");
    println!("Step 1 Q:        got sum = {:>10.6}  (ref = 18.241805)", sum(&q1));
    println!("Step 1 Q_rope:   got sum = {:>10.6}  (ref = 22.048499)", sum(&q1_rope));
    println!("Step 1 K:        got sum = {:>10.6}  (ref = 22.736263)", sum(&k1));
    println!("Step 1 K_rope:   got sum = {:>10.6}  (ref = 26.373453)", sum(&k1_rope));

    let attn_out1 = model.layers[0].self_attn.forward(&norm1_1, 1, &mut kv_caches[0], Device::Vulkan(0)).expect("attn_out");
    println!("Step 1 attn_out: got sum = {:>10.6}  (ref = -0.403297)", sum(&attn_out1));
    let ffn_inp1 = x1.add(&attn_out1).expect("ffn_inp");
    let norm2_1 = model.layers[0].post_attention_layernorm.forward(&ffn_inp1).expect("norm2");
    let ffn_gate1 = model.layers[0].mlp.gate_proj.forward(&norm2_1).expect("gate");
    let ffn_up1 = model.layers[0].mlp.up_proj.forward(&norm2_1).expect("up");
    let swiglu1 = kore_nn::activations::swiglu(&ffn_gate1, &ffn_up1).expect("swiglu");
    let ffn_out1 = model.layers[0].mlp.down_proj.forward(&swiglu1).expect("down");
    let l_out1 = ffn_inp1.add(&ffn_out1).expect("l_out");
    println!("Step 1 l_out:    got sum = {:>10.6}  (ref = -1.508353)", sum(&l_out1));
}



