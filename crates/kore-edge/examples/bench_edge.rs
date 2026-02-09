//! Kore Edge benchmark — measures tok/s for on-device inference.
//!
//! Usage: cargo run --example bench_edge -p kore-edge --release

use kore_edge::format::KorefBuilder;
use kore_edge::runtime::Session;
use std::time::Instant;

fn main() {
    println!("=== Kore Edge Benchmark ===");
    println!("Backend: {}", kore_edge::simd_dispatch::backend_name());
    println!();

    // Build a small test model
    let configs = [
        ("tiny",  8,   4,  2, 2, 1,  8,  32),
        ("small", 256, 64, 4, 4, 2, 128, 128),
        ("medium", 1024, 256, 8, 8, 4, 512, 256),
    ];

    for (name, vocab, d, n_heads, n_kv_heads, n_layers, d_ff, max_seq) in &configs {
        let model = build_model(*vocab, *d, *n_heads, *n_kv_heads, *n_layers, *d_ff, *max_seq);
        let mut session = Session::new(model);

        // Warmup
        let _ = session.forward(&[0, 1, 2]);
        session.reset();

        // Prefill benchmark
        let input: Vec<u32> = (0..*vocab as u32).take(16.min(*max_seq)).collect();
        let start = Instant::now();
        let _logits = session.forward(&input);
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
        session.reset();

        // Generation benchmark
        let gen_tokens = 16.min(*max_seq - 4);
        let start = Instant::now();
        let output = session.generate(&[0, 1, 2, 3], gen_tokens);
        let gen_ms = start.elapsed().as_secs_f64() * 1000.0;
        let generated = output.len() - 4;
        let tok_per_sec = generated as f64 / (gen_ms / 1000.0);

        println!("{:<8} vocab={:<5} d={:<4} layers={} | prefill {:.1}ms | gen {}/{} tok in {:.1}ms ({:.0} tok/s)",
            name, vocab, d, n_layers, prefill_ms, generated, gen_tokens, gen_ms, tok_per_sec);
    }

    println!();
    println!("Done.");
}

fn build_model(vocab: usize, d: usize, n_heads: usize, n_kv_heads: usize, n_layers: usize, d_ff: usize, max_seq: usize) -> kore_edge::KorefModel {
    let mut builder = KorefBuilder::new("bench", vocab, d, n_heads, n_kv_heads, n_layers, d_ff, max_seq, 1e-5, 10000.0);

    // Embedding
    let embed: Vec<f32> = (0..vocab * d).map(|i| ((i as f32 * 0.7123).sin()) * 0.1).collect();
    builder.add_f32("model.embed_tokens.weight", &[vocab, d], &embed);

    let ones = vec![1.0f32; d];

    for layer in 0..n_layers {
        let prefix = format!("model.layers.{}", layer);

        builder.add_f32(&format!("{}.input_layernorm.weight", prefix), &[d], &ones);
        builder.add_f32(&format!("{}.post_attention_layernorm.weight", prefix), &[d], &ones);

        // Identity-ish projections
        let eye: Vec<f32> = (0..d * d).map(|i| if i / d == i % d { 1.0 } else { 0.0 }).collect();
        builder.add_f32(&format!("{}.self_attn.q_proj.weight", prefix), &[d, d], &eye);

        let kv_d = n_kv_heads * (d / n_heads);
        let kv_eye: Vec<f32> = (0..kv_d * d).map(|i| if i / d == i % d { 1.0 } else { 0.0 }).collect();
        builder.add_f32(&format!("{}.self_attn.k_proj.weight", prefix), &[kv_d, d], &kv_eye);
        builder.add_f32(&format!("{}.self_attn.v_proj.weight", prefix), &[kv_d, d], &kv_eye);
        builder.add_f32(&format!("{}.self_attn.o_proj.weight", prefix), &[d, d], &eye);

        let gate_w: Vec<f32> = (0..d_ff * d).map(|i| ((i as f32 * 0.3).sin()) * 0.01).collect();
        let up_w: Vec<f32> = (0..d_ff * d).map(|i| ((i as f32 * 0.5).cos()) * 0.01).collect();
        let down_w: Vec<f32> = (0..d * d_ff).map(|i| ((i as f32 * 0.7).sin()) * 0.01).collect();
        builder.add_f32(&format!("{}.mlp.gate_proj.weight", prefix), &[d_ff, d], &gate_w);
        builder.add_f32(&format!("{}.mlp.up_proj.weight", prefix), &[d_ff, d], &up_w);
        builder.add_f32(&format!("{}.mlp.down_proj.weight", prefix), &[d, d_ff], &down_w);
    }

    builder.add_f32("model.norm.weight", &[d], &ones);
    let lm_w: Vec<f32> = (0..vocab * d).map(|i| ((i as f32 * 0.9).sin()) * 0.02).collect();
    builder.add_f32("lm_head.weight", &[vocab, d], &lm_w);

    builder.build()
}
