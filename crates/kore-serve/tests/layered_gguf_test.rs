use std::path::PathBuf;
use kore_serve::layered::{LayeredConfig, LayeredEngine};

#[test]
fn test_layered_engine_gguf_forward() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("../../models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf");
    if !path.exists() {
        println!("Skipping test: GGUF model file not found at {}", path.display());
        return;
    }

    println!("Loading LayeredConfig from GGUF...");
    let config = LayeredConfig::from_gguf(&path).expect("failed to parse GGUF config");
    println!(
        "Config parsed: {} layers, d_model={}, vocab={}, heads={}, kv_heads={}, intermediate={}",
        config.num_layers, config.d_model, config.vocab_size, config.num_heads, config.num_kv_heads, config.intermediate_size
    );
    assert_eq!(config.num_layers, 42);
    assert_eq!(config.d_model, 2048);
    assert_eq!(config.num_heads, 16);
    assert_eq!(config.num_kv_heads, 2);
    assert_eq!(config.intermediate_size, 6144);

    println!("Initializing LayeredEngine...");
    let mut engine = LayeredEngine::new(config).expect("failed to create LayeredEngine");

    println!("Running forward pass on single token (token_id=0)...");
    let t0 = std::time::Instant::now();
    let logits = engine.forward(&[0]).expect("forward pass failed");
    let elapsed = t0.elapsed();
    println!("Forward pass took: {:.2}s", elapsed.as_secs_f64());
    println!("Logits len: {} (vocab={})", logits.len(), engine.config().vocab_size);
    assert_eq!(logits.len(), engine.config().vocab_size);

    // Verify logits are finite (no NaNs or Infs)
    let has_nan = logits.iter().any(|x| x.is_nan() || x.is_infinite());
    assert!(!has_nan, "logits contain NaN or Inf values");

    println!("Logits sample [..10]: {:?}", &logits[..10]);
}
