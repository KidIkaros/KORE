//! BitNet b1.58 Benchmark
//!
//! Compares f32 Transformer vs BitNet (1.58-bit ternary) on:
//! - Memory usage
//! - Inference speed (tok/s)
//! - Output quality (logit correlation)
//!
//! Run: cargo run --example bitnet_bench -p kore-cli --release

use std::time::Instant;

use kore_transformer::{Transformer, TransformerConfig, BitNetTransformer};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Kore BitNet b1.58 Benchmark                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Configuration ──────────────────────────────────────────────────
    let config = TransformerConfig {
        vocab_size: 1024,
        d_model: 256,
        n_heads: 8,
        n_kv_heads: 4,
        n_layers: 4,
        d_ff: 512,
        max_seq_len: 256,
        norm_eps: 1e-5,
        use_rope: true,
        rope_base: 10000.0,
    };

    let prompt: Vec<usize> = (0..16).collect();
    let gen_tokens = 32;
    let threshold = 0.3;

    println!("Config: d_model={}, heads={}, kv_heads={}, layers={}, d_ff={}, vocab={}",
        config.d_model, config.n_heads, config.n_kv_heads, config.n_layers, config.d_ff, config.vocab_size);
    println!("Prompt: {} tokens, Generate: {} tokens", prompt.len(), gen_tokens);
    println!("Quantization threshold: {}", threshold);
    println!();

    // ── Build models ───────────────────────────────────────────────────
    println!("Building f32 Transformer...");
    let f32_model = Transformer::new(config.clone());
    let f32_params = f32_model.param_count();

    println!("Quantizing to BitNet b1.58...");
    let bitnet = BitNetTransformer::from_transformer(&f32_model, threshold);

    println!();
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Memory Comparison                                          │");
    println!("├─────────────────────────────────────────────────────────────┤");

    let f32_bytes = bitnet.f32_equivalent_bytes();
    let bitnet_bytes = bitnet.weight_memory_bytes();
    let ratio = bitnet.compression_ratio();

    println!("│ F32 model:    {:>8.2} MB  ({} params)          │",
        f32_bytes as f64 / (1024.0 * 1024.0), f32_params);
    println!("│ BitNet model: {:>8.2} MB  (ternary packed)              │",
        bitnet_bytes as f64 / (1024.0 * 1024.0));
    println!("│ Compression:  {:>8.1}×                                   │", ratio);
    println!("│ Bits/param:   {:>8.2}  (vs 32.0 for f32)                │",
        32.0 / ratio as f64);
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // ── Benchmark f32 ──────────────────────────────────────────────────
    println!("Benchmarking f32 Transformer...");
    let mut f32_model = f32_model;
    // Warmup
    let _ = f32_model.generate(&prompt, 4);

    let start = Instant::now();
    let f32_output = f32_model.generate(&prompt, gen_tokens).unwrap();
    let f32_elapsed = start.elapsed();
    let f32_toks_per_sec = gen_tokens as f64 / f32_elapsed.as_secs_f64();

    // ── Benchmark BitNet ───────────────────────────────────────────────
    println!("Benchmarking BitNet b1.58...");
    let mut bitnet = bitnet;
    // Warmup
    let _ = bitnet.generate(&prompt, 4);

    let start = Instant::now();
    let bitnet_output = bitnet.generate(&prompt, gen_tokens).unwrap();
    let bitnet_elapsed = start.elapsed();
    let bitnet_toks_per_sec = gen_tokens as f64 / bitnet_elapsed.as_secs_f64();

    println!();
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Speed Comparison                                           │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ F32:    {:>8.1} tok/s  ({:.1}ms total)                    │",
        f32_toks_per_sec, f32_elapsed.as_secs_f64() * 1000.0);
    println!("│ BitNet: {:>8.1} tok/s  ({:.1}ms total)                    │",
        bitnet_toks_per_sec, bitnet_elapsed.as_secs_f64() * 1000.0);
    println!("│ Speedup: {:>7.2}×                                         │",
        bitnet_toks_per_sec / f32_toks_per_sec);
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // ── Output comparison ──────────────────────────────────────────────
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Output Comparison                                          │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ F32 output:    {:?}│", &f32_output[prompt.len()..]);
    println!("│ BitNet output: {:?}│", &bitnet_output[prompt.len()..]);

    let matching: usize = f32_output[prompt.len()..]
        .iter()
        .zip(bitnet_output[prompt.len()..].iter())
        .filter(|(a, b)| a == b)
        .count();
    println!("│ Token match:   {}/{} ({:.0}%)                              │",
        matching, gen_tokens, matching as f64 / gen_tokens as f64 * 100.0);
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // ── Logit comparison ───────────────────────────────────────────────
    println!("Comparing logit distributions...");
    f32_model.reset_cache();
    bitnet.reset_cache();
    let f32_logits = f32_model.forward(&prompt, false).unwrap();
    let bitnet_logits = bitnet.forward(&prompt, false).unwrap();

    let f32_data = f32_logits.as_f32_slice().unwrap();
    let bn_data = bitnet_logits.as_f32_slice().unwrap();

    // Compute cosine similarity of last-token logits
    let v = config.vocab_size;
    let last = prompt.len() - 1;
    let f32_last = &f32_data[last * v..(last + 1) * v];
    let bn_last = &bn_data[last * v..(last + 1) * v];

    let dot: f64 = f32_last.iter().zip(bn_last.iter()).map(|(&a, &b)| a as f64 * b as f64).sum();
    let norm_a: f64 = f32_last.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt();
    let norm_b: f64 = bn_last.iter().map(|&b| (b as f64) * (b as f64)).sum::<f64>().sqrt();
    let cosine_sim = if norm_a > 0.0 && norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 };

    println!();
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Logit Quality                                              │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Cosine similarity (last token): {:.4}                      │", cosine_sim);
    println!("│ (1.0 = identical, 0.0 = uncorrelated)                      │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // ── Summary ────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Summary                                                    ║");
    println!("║                                                            ║");
    println!("║ BitNet b1.58 achieves {:.1}× memory compression            ║", ratio);
    println!("║ with {:.2}× speed ratio vs f32.                             ║",
        bitnet_toks_per_sec / f32_toks_per_sec);
    println!("║                                                            ║");
    println!("║ {}                                                  ║", bitnet.info().chars().take(50).collect::<String>());
    println!("╚══════════════════════════════════════════════════════════════╝");
}
