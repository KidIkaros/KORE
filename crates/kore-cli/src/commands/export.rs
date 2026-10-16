use kore_edge::format::{EdgeDType, KorefBuilder};
use serde::Deserialize;
use std::path::Path;

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

pub fn cmd_export(model_path: &str, output_path: &str, quantize: &str) {
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

    println!(
        "Config: vocab={} d={} heads={} kv_heads={} layers={} ff={} max_seq={}",
        vocab_size, d_model, n_heads, n_kv_heads, n_layers, d_ff, max_seq_len
    );

    let target_dtype = match quantize {
        "f32" => EdgeDType::F32,
        "f16" => EdgeDType::F16,
        "ternary" => EdgeDType::Ternary,
        "quaternary" => EdgeDType::Quaternary,
        other => {
            eprintln!(
                "Unknown quantization: {}. Use f32, f16, ternary, or quaternary.",
                other
            );
            return;
        }
    };

    let mut builder = KorefBuilder::new(
        "llama",
        vocab_size,
        d_model,
        n_heads,
        n_kv_heads,
        n_layers,
        d_ff,
        max_seq_len,
        norm_eps,
        rope_base,
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
        let demo_builder = KorefBuilder::new("demo", 256, 64, 4, 4, 2, 128, 128, 1e-5, 10000.0);
        let model = demo_builder.build();
        let bytes = model.to_bytes();
        std::fs::write(output_path, &bytes).expect("Failed to write .koref");
        println!(
            "Wrote demo .koref ({} bytes) to {}",
            bytes.len(),
            output_path
        );
        return;
    }

    println!("Found {} safetensors file(s)", st_files.len());

    let mut tensor_count = 0usize;
    let mut original_bytes = 0usize;
    let mut stored_bytes = 0usize;

    for st_path in &st_files {
        let data = std::fs::read(st_path).expect("Failed to read safetensors");
        let tensors =
            safetensors::SafeTensors::deserialize(&data).expect("Failed to parse safetensors");

        for (name, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            original_bytes += view.data().len();

            let f32_vals: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => bytemuck::cast_slice::<u8, f32>(view.data()).to_vec(),
                safetensors::Dtype::F16 => {
                    let u16s = bytemuck::cast_slice::<u8, u16>(view.data());
                    u16s.iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect()
                }
                safetensors::Dtype::BF16 => {
                    let u16s = bytemuck::cast_slice::<u8, u16>(view.data());
                    u16s.iter()
                        .map(|&b| half::bf16::from_bits(b).to_f32())
                        .collect()
                }
                _ => bytemuck::cast_slice::<u8, f32>(view.data()).to_vec(),
            };

            match target_dtype {
                EdgeDType::Ternary if shape.len() == 2 => {
                    let (packed, scales) = kore_kernels::cpu_ternary_matmul::pack_weights_ternary(
                        &f32_vals, shape[0], shape[1], 0.3,
                    );
                    stored_bytes += packed.len();
                    builder.add_tensor(&name, EdgeDType::Ternary, &shape, &packed);
                    let scales_bytes: &[u8] = bytemuck::cast_slice(&scales);
                    stored_bytes += scales_bytes.len();
                    builder.add_tensor(
                        &format!("{}.scales", name),
                        EdgeDType::F32,
                        &[shape[0]],
                        scales_bytes,
                    );
                }
                EdgeDType::Quaternary if shape.len() == 2 => {
                    let (packed, scales) = kore_kernels::cpu_quat_matmul::pack_weights_quaternary(
                        &f32_vals, shape[0], shape[1],
                    );
                    stored_bytes += packed.len();
                    builder.add_tensor(&name, EdgeDType::Quaternary, &shape, &packed);
                    let scales_bytes: &[u8] = bytemuck::cast_slice(&scales);
                    stored_bytes += scales_bytes.len();
                    builder.add_tensor(
                        &format!("{}.scales", name),
                        EdgeDType::F32,
                        &[shape[0]],
                        scales_bytes,
                    );
                }
                EdgeDType::F16 if shape.len() >= 2 => {
                    let f16_data: Vec<u16> = f32_vals
                        .iter()
                        .map(|&x| half::f16::from_f32(x).to_bits())
                        .collect();
                    let f16_bytes: &[u8] = bytemuck::cast_slice(&f16_data);
                    stored_bytes += f16_bytes.len();
                    builder.add_tensor(&name, EdgeDType::F16, &shape, f16_bytes);
                }
                _ => {
                    let raw_bytes: &[u8] = bytemuck::cast_slice(&f32_vals);
                    stored_bytes += raw_bytes.len();
                    builder.add_tensor(&name, EdgeDType::F32, &shape, raw_bytes);
                }
            }
            tensor_count += 1;
        }
    }

    let model = builder.build();
    let bytes = model.to_bytes();
    std::fs::write(output_path, &bytes).expect("Failed to write .koref");

    println!();
    println!("Export complete:");
    println!("  Tensors:       {}", tensor_count);
    println!(
        "  Original data: {:.2} MB",
        original_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Stored data:   {:.2} MB",
        stored_bytes as f64 / (1024.0 * 1024.0)
    );
    if stored_bytes > 0 && original_bytes > 0 {
        println!(
            "  Compression:   {:.2}x",
            original_bytes as f64 / stored_bytes as f64
        );
    }
    println!(
        "  .koref size:   {:.2} MB",
        bytes.len() as f64 / (1024.0 * 1024.0)
    );
    println!("  Output:        {}", output_path);

    // Print estimated edge memory
    let plan = kore_edge::plan::ExecutionPlan::from_header(&model.header);
    println!("  Est. runtime memory: {:.1} MB", plan.peak_memory_mb());
}
