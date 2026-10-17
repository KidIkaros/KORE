//! Long-context decode-attention cost curve, with a matched split-KV A/B.
//!
//! Decode attention is O(seq_k): every step re-reads the whole K/V window. This
//! times `fused_attention_decode_gqa_q8` against `seq_k`, and against
//! `fused_attention_decode_gqa_q8_split` at several shard counts, so the
//! long-context cost can be attributed and compared in-process.
//!
//! Shapes are Bonsai-2's full-attention layers: 8 query heads, 2 KV heads,
//! head_dim 256 (kv_dim 512).
//!
//! Measurement notes (both learned the hard way):
//! - Every arm for a given `seq_k` is warmed up and then timed in the *same*
//!   loop, with the arm order rotated per repetition. Timing the single-group
//!   arm in a separate pass made it look 1.8x slower than it is, because the GPU
//!   idles at 300 MHz and spends the first pass ramping to boost.
//! - Reported times are per-arm medians, not means, so one throttling spike
//!   does not move the number.

use kore_core::{Device, Tensor};
use kore_vulkan::VulkanBackend;
use std::time::Instant;

const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const KV_DIM: usize = KV_HEADS * HEAD_DIM;
const Q_DIM: usize = HEADS * HEAD_DIM;
const SCALE: f32 = 0.0625; // 1/sqrt(256)

/// One arm of the A/B: index 0 is the single-group kernel, index i is the split
/// kernel with `SPLITS[i - 1]` shards.
const SPLITS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

fn rand(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let b = VulkanBackend::new().expect("vulkan backend");
    let dev = Device::Vulkan(0);
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let warmup = 4;

    println!("fused_attention_decode_gqa: heads={HEADS} kv_heads={KV_HEADS} head_dim={HEAD_DIM} reps={reps} (medians)");
    println!(
        "{:>7}  {:>7}  {:>10}  {:>9}  {:>9}  {:>12}",
        "seq_k", "splits", "ms", "GB/s", "speedup", "max |diff|"
    );

    for &seq_k in &[64usize, 128, 256, 512, 1024, 2048, 4096, 8192] {
        let q = Tensor::from_f32(&rand(Q_DIM, 7), &[1, Q_DIM])
            .to(dev)
            .unwrap();
        let k = Tensor::from_f32(&rand(seq_k * KV_DIM, 11), &[seq_k, KV_DIM])
            .to(dev)
            .unwrap();
        let v = Tensor::from_f32(&rand(seq_k * KV_DIM, 13), &[seq_k, KV_DIM])
            .to(dev)
            .unwrap();
        let (k8, ks) = b.quantize_q8_rows(&k).unwrap();
        let (v8, vs) = b.quantize_q8_rows(&v).unwrap();

        // Arms that make sense for this seq_k (no point asking for more shards
        // than keys).
        let mut arms: Vec<usize> = vec![0];
        for (i, &s) in SPLITS.iter().enumerate() {
            if s <= seq_k {
                arms.push(i + 1);
            }
        }
        let arm_splits = |a: usize| if a == 0 { 0 } else { SPLITS[a - 1] };

        let call = |a: usize| -> Vec<f32> {
            let o = if a == 0 {
                b.fused_attention_decode_gqa_q8(
                    &q, &k8, &v8, &ks, &vs, HEADS, KV_HEADS, HEAD_DIM, SCALE,
                )
            } else {
                b.fused_attention_decode_gqa_q8_split(
                    &q,
                    &k8,
                    &v8,
                    &ks,
                    &vs,
                    HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    SCALE,
                    arm_splits(a),
                )
            }
            .unwrap();
            o.cpu().unwrap().as_f32_slice().unwrap().to_vec()
        };

        for _ in 0..warmup {
            for &a in &arms {
                let _ = call(a);
            }
        }

        let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); arms.len()];
        let mut outs: Vec<Vec<f32>> = vec![Vec::new(); arms.len()];
        for r in 0..reps {
            let off = r % arms.len();
            for i in 0..arms.len() {
                let slot = (i + off) % arms.len();
                let a = arms[slot];
                let t0 = Instant::now();
                let out = call(a);
                times[slot].push(t0.elapsed().as_secs_f64() * 1e3);
                outs[slot] = out;
            }
        }

        let want = &outs[0];
        let bytes = 2.0 * (seq_k * KV_DIM) as f64;
        let base_ms = median(&mut times[0].clone());
        for (i, &a) in arms.iter().enumerate() {
            let ms = median(&mut times[i].clone());
            let max_diff = if a == 0 {
                0.0
            } else {
                outs[i]
                    .iter()
                    .zip(want.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            };
            println!(
                "{seq_k:>7}  {:>7}  {ms:>10.3}  {:>9.1}  {:>9.2}  {max_diff:>12.2e}",
                if a == 0 {
                    "single".to_string()
                } else {
                    arm_splits(a).to_string()
                },
                bytes / (ms / 1e3) / 1e9,
                base_ms / ms
            );
        }
    }
}
