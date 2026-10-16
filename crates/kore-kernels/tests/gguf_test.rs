use std::path::PathBuf;

#[test]
fn test_load_minicpm_gguf() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("../../models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf");
    if !path.exists() {
        println!("Skipping test: model file does not exist at {}", path.display());
        return;
    }

    let model = oxillama_gguf::GgufModel::load(&path).expect("failed to load GGUF model");
    println!("Architecture: {:?}", model.architecture());
    println!("Tensor count: {}", model.file.header.tensor_count);

    // Inspect some metadata
    let meta = &model.file.metadata;
    if let Some(arch) = meta.get("general.architecture").and_then(|v| v.as_str()) {
        println!("  arch = {}", arch);
    }
    if let Some(layers) = meta.get("llama.block_count").and_then(|v| v.as_u32()) {
        println!("  layers = {}", layers);
    }
    if let Some(dim) = meta.get("llama.embedding_length").and_then(|v| v.as_u32()) {
        println!("  hidden_dim = {}", dim);
    }
    if let Some(heads) = meta.get("llama.attention.head_count").and_then(|v| v.as_u32()) {
        println!("  head_count = {}", heads);
    }
    if let Some(kv_heads) = meta.get("llama.attention.head_count_kv").and_then(|v| v.as_u32()) {
        println!("  head_count_kv = {}", kv_heads);
    }

    // Try reading and dequantizing a tensor (e.g. blk.0.attn_norm.weight)
    let tensor_name = "blk.0.attn_norm.weight";
    if let Ok(info) = model.file.tensors.get(tensor_name) {
        let raw_bytes = model.tensor_data(tensor_name).expect("tensor_data failed");
        let n_elements: usize = info.dimensions.iter().map(|&d| d as usize).product();
        let f32_weights = oxillama_quant::dequantize_to_f32(raw_bytes, info.tensor_type, n_elements)
            .expect("dequantize failed");
        println!(
            "\nDequantized {} (len={}): [{:.4}, {:.4}, {:.4}, ...]",
            tensor_name,
            f32_weights.len(),
            f32_weights[0],
            f32_weights[1],
            f32_weights[2]
        );
        assert_eq!(f32_weights.len(), n_elements);
    }

    // Also test dequantizing a Q4_K attention projection tensor (e.g. blk.0.attn_q.weight)
    let q_proj_name = "blk.0.attn_q.weight";
    if let Ok(info) = model.file.tensors.get(q_proj_name) {
        let raw_bytes = model.tensor_data(q_proj_name).expect("tensor_data failed");
        let n_elements: usize = info.dimensions.iter().map(|&d| d as usize).product();
        println!(
            "\nTesting {} (type={:?}, numel={}, raw_bytes={}):",
            q_proj_name, info.tensor_type, n_elements, raw_bytes.len()
        );
        let f32_weights = oxillama_quant::dequantize_to_f32(raw_bytes, info.tensor_type, n_elements)
            .expect("dequantize Q4_K failed");
        println!(
            "Dequantized {} (len={}): [{:.4}, {:.4}, {:.4}, ...]",
            q_proj_name,
            f32_weights.len(),
            f32_weights[0],
            f32_weights[1],
            f32_weights[2]
        );
        assert_eq!(f32_weights.len(), n_elements);
    }

    for name in &["token_embd.weight", "output.weight", "blk.0.attn_output.weight", "blk.0.ffn_down.weight"] {
        if let Ok(info) = model.file.tensors.get(*name) {
            let n_elements: usize = info.dimensions.iter().map(|&d| d as usize).product();
            let raw_len = model.tensor_data(*name).map(|d| d.len()).unwrap_or(0);
            println!("Tensor {}: type={:?}, dims={:?}, numel={}, raw_bytes={}", name, info.tensor_type, info.dimensions, n_elements, raw_len);
        } else {
            println!("Tensor {} NOT found in GGUF", name);
        }
    }
}
