use kore_core::Tensor;
use kore_nn::{KvCache, Module, SamplerConfig, SqueezeNet, Transformer};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

// ponytail: P0 smoke test for the research loop (R0). Loads safetensors -> forward
// -> logits, CPU only. Supports both SqueezeNet (CNN) and Transformer (autoregressive).
#[allow(clippy::too_many_arguments)]
pub fn cmd_infer(
    arch: &str,
    device: &str,
    model: Option<String>,
    classes: Option<usize>,
    size: usize,
    hidden_dim: usize,
    head_dim: usize,
    seq_len: usize,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    prompt: Option<String>,
    tokenizer: Option<String>,
    mtp: bool,
    layers: Option<usize>,
    gpu_layers: Option<usize>,
    kv_cache: &str,
) {
    let target_device: kore_core::Device = match device.parse() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    if !target_device.is_cpu() && kore_core::device::get_backend(target_device).is_none() {
        let avail: Vec<String> = kore_core::device::available_devices()
            .into_iter()
            .map(|d| d.to_string())
            .collect();
        eprintln!(
            "Error: Device '{}' is not registered or unavailable on this system.\nAvailable devices on this machine: [{}]\n(Tip: use '--device auto' or '--device vulkan:0')",
            target_device,
            avail.join(", ")
        );
        std::process::exit(1);
    }

    // Auto-detect architecture if set to 'auto' or unspecified
    let resolved_arch = if arch.eq_ignore_ascii_case("auto") {
        if prompt.is_some() || tokenizer.is_some() || mtp || hidden_dim != 64 || head_dim != 16 {
            "transformer"
        } else if size != 224 {
            "squeezenet"
        } else {
            "transformer"
        }
    } else {
        arch
    };

    println!(
        "=== Kore Infer (research-loop smoke test: arch={}, device={}) ===",
        resolved_arch, target_device
    );

    let kv_mode = match kv_cache.to_lowercase().as_str() {
        "q8" | "q8_0" => kore_attention::KvCacheQuantMode::Q8_0,
        "k8v4" => kore_attention::KvCacheQuantMode::K8V4,
        _ => kore_attention::KvCacheQuantMode::F32,
    };

    if model.as_ref().map_or(false, |m| m.ends_with(".gguf")) {
        cmd_infer_gguf(
            target_device,
            model.as_ref().unwrap(),
            max_new_tokens,
            prompt,
            tokenizer,
            layers,
            gpu_layers,
            mtp,
            kv_mode,
        );
        return;
    }

    if resolved_arch.eq_ignore_ascii_case("transformer") {
        cmd_infer_transformer(
            target_device,
            model,
            classes.unwrap_or(32000),
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
            kv_mode,
        );
    } else if resolved_arch.eq_ignore_ascii_case("squeezenet") {
        if prompt.is_some() {
            eprintln!("Note: --prompt is ignored when --arch squeezenet is selected.");
        }
        cmd_infer_squeezenet(model, classes.unwrap_or(1000), size);
    } else {
        eprintln!(
            "Error: Unknown architecture '{}'. Supported architectures: auto, transformer, squeezenet",
            arch
        );
        std::process::exit(1);
    }
}

pub fn cmd_infer_squeezenet(model: Option<String>, classes: usize, size: usize) {
    let model = match model {
        Some(path) => {
            println!("Loading:   {}", path);
            SqueezeNet::load_safetensors(Path::new(&path), classes)
                .expect("failed to load safetensors model")
        }
        None => {
            println!("No --model given; using random-init SqueezeNet v1.1 (no download)");
            SqueezeNet::v1_1(classes)
        }
    };
    println!("  {}", model.info());

    // Dummy input: [1, 3, size, size]
    let input = Tensor::randn(&[1, 3, size, size]);
    println!("  input:  {:?}", input.shape().dims());

    let t0 = Instant::now();
    let logits = model.forward(&input).expect("forward pass failed");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    let dims = logits.shape().dims().to_vec();
    let data = logits.as_f32_slice().expect("logits not f32").to_vec();
    let head: Vec<String> = data.iter().take(10).map(|x| format!("{x:.4}")).collect();

    println!("  output: {:?}  ({:.1} ms)", dims, ms);
    println!("  logits[..10]: [{}]", head.join(", "));
}

#[allow(clippy::too_many_arguments)]
pub fn cmd_infer_transformer(
    device: kore_core::Device,
    model: Option<String>,
    vocab_size: usize,
    hidden_dim: usize,
    head_dim: usize,
    seq_len: usize,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    prompt: Option<String>,
    tokenizer_path: Option<String>,
    mtp: bool,
    kv_mode: kore_attention::KvCacheQuantMode,
) {
    // 1. Try to load tokenizer if provided or adjacent to model
    let tokenizer = if let Some(ref path) = tokenizer_path {
        println!("Loading tokenizer: {}", path);
        match Tokenizer::from_file(path) {
            Ok(tok) => Some(tok),
            Err(e) => {
                eprintln!("Warning: could not load tokenizer at {}: {}", path, e);
                None
            }
        }
    } else if let Some(ref m_path) = model {
        let p = Path::new(m_path);
        let cand = p.parent().unwrap_or(Path::new(".")).join("tokenizer.json");
        if cand.exists() {
            println!("Found adjacent tokenizer: {}", cand.display());
            Tokenizer::from_file(&cand).ok()
        } else {
            None
        }
    } else {
        None
    };

    // 2. Load or instantiate Transformer
    let mut model = match model {
        Some(path) => {
            println!("Loading model:     {}", path);
            Transformer::load_safetensors(Path::new(&path), head_dim)
                .expect("failed to load safetensors transformer")
        }
        None => {
            println!("No --model given; using random-init Transformer validation rig");
            Transformer::new(vocab_size, hidden_dim, head_dim)
        }
    };

    if !device.is_cpu() {
        println!("  Allocating weights on device {}...", device);
        model
            .to(device)
            .expect("failed to transfer model to device");
    }

    let actual_vocab = model.vocab_size();
    let actual_hidden = model.hidden_dim();
    let actual_head = model.head_dim();
    let num_params = model.num_parameters();
    println!(
        "  Transformer | device={} | vocab={} | hidden={} | head_dim={} | params={:.2}M",
        device,
        actual_vocab,
        actual_hidden,
        actual_head,
        num_params as f64 / 1_000_000.0
    );

    // 3. Prepare initial token sequence
    let mut token_ids: Vec<u32> = if let Some(ref p_str) = prompt {
        if let Some(ref tok) = tokenizer {
            let enc = tok
                .encode(p_str.as_str(), true)
                .expect("failed to encode prompt");
            enc.get_ids().to_vec()
        } else if p_str.contains(',') {
            p_str
                .split(',')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect()
        } else {
            p_str
                .bytes()
                .map(|b| (b as usize % actual_vocab) as u32)
                .collect()
        }
    } else {
        (0..seq_len).map(|i| (i % actual_vocab) as u32).collect()
    };

    if token_ids.is_empty() {
        token_ids = vec![1];
    }

    println!("  Prompt tokens ({}): {:?}", token_ids.len(), token_ids);
    if let Some(ref tok) = tokenizer {
        if let Ok(decoded) = tok.decode(&token_ids, true) {
            println!("  Prompt text:      {:?}", decoded);
        }
    }

    // 4. Sampler configuration
    let sampler_config = if temperature <= 0.0 {
        SamplerConfig::greedy()
    } else {
        SamplerConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.1,
            eos_token_id: None,
            gumbel_coupled: true,
        }
    };

    let mtp_enabled = mtp || std::env::var("KORE_MTP").ok().as_deref() == Some("1");
    if mtp_enabled && !model.has_mtp_head() {
        model.enable_mtp_head().expect("failed to enable MTP head");
    }

    // 5. Prefill phase (with KV-cache)
    let max_cache_len = token_ids.len() + max_new_tokens + 64;
    let mut kv_cache =
        KvCache::with_quant_mode(actual_hidden, actual_hidden, max_cache_len, kv_mode);

    let t0 = Instant::now();
    let input_f32: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
    let input_tensor = Tensor::from_f32(&input_f32, &[input_f32.len()]);
    let (logits, hidden_last) = model
        .forward_with_cache_and_hidden(&input_tensor, &mut kv_cache)
        .expect("prefill forward pass failed");
    let mut next_token = kore_nn::sampler::sample_from_tensor(&logits, &[], &sampler_config, 42, 0)
        .expect("failed to sample prefill token") as u32;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let cur_seq_len = input_f32.len();
    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(next_token);

    println!(
        "  Prefill:          {:.1} ms ({} tokens, {:.1} tok/s)",
        prefill_ms,
        cur_seq_len,
        if prefill_ms > 0.0 {
            cur_seq_len as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        }
    );

    // 6. Autoregressive decode phase (O(1) single-token GEMV or M=2 MTP speculative verification)
    let t_decode_start = Instant::now();
    if model.has_mtp_head() {
        let mut draft_opt = model
            .draft_token(&hidden_last, next_token as usize, &sampler_config, 43, 1)
            .expect("initial MTP draft failed");
        let mut accepted_drafts = 0usize;
        let mut total_steps = 0usize;

        while generated.len() < max_new_tokens {
            let step = generated.len();
            if let Some(draft_tok) = draft_opt {
                total_steps += 1;
                let spec = model
                    .step_speculative(
                        next_token as usize,
                        draft_tok,
                        &mut kv_cache,
                        &sampler_config,
                        42 + step as u64,
                        step,
                    )
                    .expect("MTP speculative step failed");
                generated.push(spec.verified_token as u32);
                if let Some(bonus) = spec.bonus_token {
                    accepted_drafts += 1;
                    if generated.len() < max_new_tokens {
                        generated.push(bonus as u32);
                    }
                    next_token = bonus as u32;
                } else {
                    next_token = spec.verified_token as u32;
                }
                draft_opt = spec.next_draft;
            } else {
                break;
            }
        }
        if total_steps > 0 {
            println!(
                "  MTP Speculative:  {}/{} drafts accepted ({:.1}%)",
                accepted_drafts,
                total_steps,
                (accepted_drafts as f64 / total_steps as f64) * 100.0
            );
        }
    } else {
        for step in 1..max_new_tokens {
            token_ids.push(next_token);
            let step_tensor = Tensor::from_f32(&[next_token as f32], &[1]);
            let logits = model
                .forward_with_cache(&step_tensor, &mut kv_cache)
                .expect("decode forward pass failed");
            next_token = kore_nn::sampler::sample_from_tensor(
                &logits,
                &[],
                &sampler_config,
                42 + step as u64,
                step,
            )
            .expect("decode sample failed") as u32;
            generated.push(next_token);
        }
    }
    let decode_ms = t_decode_start.elapsed().as_secs_f64() * 1000.0;
    let decode_tok_per_sec = if decode_ms > 0.0 {
        (generated.len() as f64) / (decode_ms / 1000.0)
    } else {
        0.0
    };

    println!(
        "  Decode:           {:.1} ms ({} tokens, {:.2} tok/s)",
        decode_ms,
        generated.len(),
        decode_tok_per_sec
    );
    println!("  Generated tokens: {:?}", generated);

    if let Some(ref tok) = tokenizer {
        if let Ok(decoded) = tok.decode(&generated, true) {
            println!("  Generated text:   {:?}", decoded);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cmd_infer_gguf(
    device: kore_core::Device,
    model_path: &str,
    max_new_tokens: usize,
    prompt: Option<String>,
    tokenizer_path: Option<String>,
    layers: Option<usize>,
    gpu_layers: Option<usize>,
    mtp: bool,
    kv_mode: kore_attention::KvCacheQuantMode,
) {
    let path = Path::new(model_path);
    if !path.exists() {
        eprintln!("Error: model file does not exist at {}", path.display());
        std::process::exit(1);
    }

    // 1. Resolve tokenizer
    let tokenizer_file = if let Some(ref t_path) = tokenizer_path {
        Path::new(t_path).to_path_buf()
    } else {
        let parent = path.parent().unwrap_or(Path::new("."));
        let cand = parent.join("tokenizer.json");
        if cand.exists() {
            cand
        } else {
            eprintln!("Error: tokenizer.json not found adjacent to model. Specify --tokenizer <path>");
            std::process::exit(1);
        }
    };

    println!("Loading tokenizer: {}", tokenizer_file.display());
    let tokenizer = Tokenizer::from_file(&tokenizer_file)
        .unwrap_or_else(|e| {
            eprintln!("Error: failed to load tokenizer: {}", e);
            std::process::exit(1);
        });

    // arch dispatch: bounded header read, qwen35 takes the hybrid path
    if let Ok(gg) = kore_nn::gguf_min::MiniGguf::load(path) {
        if gg.get_str("general.architecture") == Some("qwen35") {
            return infer_gguf_qwen35(path, &tokenizer, prompt, max_new_tokens);
        }
    }

    // 2. Load GGUF model
    println!("Loading model from GGUF: {}", path.display());
    let t_load = Instant::now();
    let mut model = kore_nn::llama::LlamaModel::from_gguf(path, layers)
        .unwrap_or_else(|e| {
            eprintln!("Error: failed to load GGUF model: {}", e);
            std::process::exit(1);
        });
    let load_time = t_load.elapsed();
    println!("  Loaded model in {:.2}s", load_time.as_secs_f64());

    if !device.is_cpu() {
        let t_trans = Instant::now();
        if let Some(ngl) = gpu_layers {
            println!(
                "  Hybrid GPU+CPU placement: {}/{} layers on {}, {} layers on CPU SIMD...",
                ngl.min(model.layers.len()),
                model.layers.len(),
                device,
                model.layers.len().saturating_sub(ngl)
            );
            model.to_hybrid(device, ngl).unwrap_or_else(|e| {
                eprintln!("Error: failed to transfer hybrid layers to device: {}", e);
                std::process::exit(1);
            });
        } else {
            println!("  Transferring model weights to device {}...", device);
            model.to(device).unwrap_or_else(|e| {
                eprintln!("Error: failed to transfer model to device: {}", e);
                std::process::exit(1);
            });
        }
        println!("  Weights resident on {} in {:.2}s", device, t_trans.elapsed().as_secs_f64());
    }

    let mtp_enabled = mtp || std::env::var("KORE_MTP").ok().as_deref() == Some("1");
    println!(
        "  Model config: vocab={} | hidden={} | layers={} | heads={}/{} | head_dim={} | kv={:?} | spec={}",
        model.config.vocab_size,
        model.config.hidden_size,
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
        model.config.head_dim,
        kv_mode,
        mtp_enabled
    );

    // 3. Prepare prompt
    let prompt_str = prompt.unwrap_or_else(|| "Hello, my name is".to_string());
    println!("\nPrompt: \"{}\"", prompt_str);

    let encoding = tokenizer.encode(prompt_str.as_str(), false)
        .unwrap_or_else(|e| {
            eprintln!("Error: tokenization failed: {}", e);
            std::process::exit(1);
        });
    let prompt_tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
    print!("\nGenerated: ");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    // 4. Autoregressive streaming generation loop (4-token batched prefill + optional PLD/MTP speculation)
    let t_gen = Instant::now();
    let mut t_first_token: Option<std::time::Duration> = None;
    let mut generated_count = 0usize;
    let (_, accepted_drafts, total_drafts) = model
        .generate_with_options(
            &prompt_tokens,
            max_new_tokens,
            None,
            mtp_enabled,
            kv_mode,
            |tok_id| {
                if t_first_token.is_none() {
                    t_first_token = Some(t_gen.elapsed());
                }
                generated_count += 1;
                if let Ok(text) = tokenizer.decode(&[tok_id as u32], false) {
                    print!("{}", text);
                    let _ = std::io::stdout().flush();
                }
            },
        )
        .unwrap_or_else(|e| {
            eprintln!("\nError during generation: {}", e);
            std::process::exit(1);
        });

    let total_time = t_gen.elapsed();
    let prefill_time = t_first_token.unwrap_or(total_time);
    let decode_time = total_time.saturating_sub(prefill_time);
    let decode_tokens = generated_count.saturating_sub(1);
    let prefill_tok_s = if prefill_time.as_secs_f64() > 0.0 {
        prompt_tokens.len() as f64 / prefill_time.as_secs_f64()
    } else {
        0.0
    };
    let decode_tok_s = if decode_time.as_secs_f64() > 0.0 && decode_tokens > 0 {
        decode_tokens as f64 / decode_time.as_secs_f64()
    } else if total_time.as_secs_f64() > 0.0 {
        generated_count as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    println!("\n");
    println!("--------------------------------------------------");
    println!(
        "Prefill:   {} prompt tokens in {:.3}s ({:.2} tok/s)",
        prompt_tokens.len(),
        prefill_time.as_secs_f64(),
        prefill_tok_s
    );
    if mtp_enabled && total_drafts > 0 {
        println!(
            "Speculate: {}/{} draft tokens accepted ({:.1}%)",
            accepted_drafts,
            total_drafts,
            (accepted_drafts as f64 / total_drafts as f64) * 100.0
        );
    }
    println!(
        "Decode:    {} tokens in {:.3}s ({:.2} tok/s) | Total: {:.3}s",
        generated_count,
        decode_time.as_secs_f64(),
        decode_tok_s,
        total_time.as_secs_f64()
    );
}

/// qwen35 (Bonsai ternary hybrid) — CPU reference path.
fn infer_gguf_qwen35(
    path: &Path,
    tokenizer: &Tokenizer,
    prompt: Option<String>,
    max_new_tokens: usize,
) {
    use kore_nn::qwen35::Qwen35Model;

    println!("Loading qwen35 model: {}", path.display());
    let t_load = Instant::now();
    let model = Qwen35Model::from_gguf(path).unwrap_or_else(|e| {
        eprintln!("Error: failed to load qwen35 model: {}", e);
        std::process::exit(1);
    });
    println!("  Loaded in {:.2}s: {:?}", t_load.elapsed().as_secs_f64(), model.cfg);

    let prompt_str = prompt.unwrap_or_else(|| "Hello, my name is".to_string());
    println!("\nPrompt: \"{}\"", prompt_str);
    let prompt_tokens: Vec<u32> = tokenizer
        .encode(prompt_str.as_str(), false)
        .unwrap_or_else(|e| {
            eprintln!("Error: tokenization failed: {}", e);
            std::process::exit(1);
        })
        .get_ids()
        .to_vec();

    print!("\nGenerated: ");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    let t_gen = Instant::now();
    let (mut dstate, mut astate) = model.new_state();
    let mut generated = 0usize;
    let mut logits = Vec::new();
    for (i, &t) in prompt_tokens.iter().enumerate() {
        logits = model.forward(t, i, &mut dstate, &mut astate).unwrap_or_else(|e| {
            eprintln!("\nError during prefill: {}", e);
            std::process::exit(1);
        });
    }
    let prefill_time = t_gen.elapsed();
    let mut pos = prompt_tokens.len();
    for _ in 0..max_new_tokens {
        let tok = Qwen35Model::argmax(&logits);
        if tok == model.cfg.eos {
            break;
        }
        generated += 1;
        if let Ok(text) = tokenizer.decode(&[tok], false) {
            print!("{}", text);
            let _ = std::io::stdout().flush();
        }
        logits = model.forward(tok, pos, &mut dstate, &mut astate).unwrap_or_else(|e| {
            eprintln!("\nError during decode: {}", e);
            std::process::exit(1);
        });
        pos += 1;
    }
    let total = t_gen.elapsed();

    println!("\n\n--------------------------------------------------");
    println!(
        "Prefill:   {} tokens in {:.3}s ({:.2} tok/s)",
        prompt_tokens.len(),
        prefill_time.as_secs_f64(),
        prompt_tokens.len() as f64 / prefill_time.as_secs_f64()
    );
    println!(
        "Decode:    {} tokens in {:.3}s ({:.2} tok/s)",
        generated,
        total.as_secs_f64(),
        generated as f64 / total.as_secs_f64()
    );
}
