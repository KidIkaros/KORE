//! What speculative decoding would buy on this GPU, measured against the real
//! weight shapes of a Bonsai-2 decode step.
//!
//! Speculative decoding replaces several single-token decode steps with one
//! *verify* pass over `k+1` columns. It is a win only because `gemv_q2_0` reads
//! each weight block once for every row of the batch: the workgroup count is
//! `(n/16, ceil(m/4))`, which is the same for `m = 1` and `m = 4`. So a token
//! gets cheaper with the batch exactly where the token is *weight traffic*.
//!
//! Whether it actually is weight traffic is not obvious for Q2_0: the kernel
//! has to unpack 2-bit codes and apply an fp16 scale per block, so it could
//! easily be ALU-bound on the dequantisation rather than bandwidth-bound on the
//! read -- in which case four rows cost four times the arithmetic and batching
//! buys nothing. This measures which it is, at Bonsai's shapes, read out of the
//! checkpoint rather than guessed.
//!
//! Usage: `bench_spec [gguf] [reps]`. Only the GEMV half of the step is timed;
//! attention, the DeltaNet scan and the norms are accounted for elsewhere
//! (`op_counts`, `bench_context_cool`).

use kore_core::{DType, Device, Storage, Tensor};
use kore_nn::gguf_min::MiniGguf;
use kore_vulkan::VulkanBackend;
use std::path::Path;
use std::time::Instant;

fn next_u64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Raw Q2_0 bytes for an `[k, n]` matrix: a valid fp16 scale of 1.0 in every
/// 64-element block (so nothing is denormal and nothing is NaN) and random
/// codes. The kernel's cost does not depend on the values.
fn raw_q2_0(rng: &mut u64, k: usize, n: usize) -> Vec<u8> {
    let blocks = n * (k / 64);
    let mut out = Vec::with_capacity(blocks * 18);
    for _ in 0..blocks {
        out.push(0x00); // fp16 1.0 is 0x3C00, little-endian
        out.push(0x3C);
        for _ in 0..16 {
            out.push((next_u64(rng) & 0xFF) as u8);
        }
    }
    out
}

struct Op {
    w: Tensor,
    /// Output rows, and the reduction width.
    n: usize,
    k: usize,
    /// One input per batch size, pre-uploaded so the timing loop does not
    /// measure an upload.
    xs: Vec<Tensor>,
}

fn main() {
    let gguf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf".into());
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let bs = [1usize, 2, 3, 4];

    let gg = MiniGguf::load(Path::new(&gguf)).expect("load gguf");
    let arch = gg
        .get_str("general.architecture")
        .unwrap_or("qwen35")
        .to_string();
    let n_layers = gg
        .get_u64(&format!("{arch}.block_count"))
        .expect("block_count") as usize;
    let interval = gg
        .get_u64(&format!("{arch}.full_attention_interval"))
        .expect("full_attention_interval") as usize;

    // The GEMV op list of one decode step, in the order the layer loop runs it.
    let mut names: Vec<String> = Vec::new();
    for i in 0..n_layers {
        let p = |s: &str| format!("blk.{i}.{s}");
        let attn: &[&str] = if (i + 1) % interval == 0 {
            &[
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            ]
        } else {
            &["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"]
        };
        for t in attn {
            names.push(p(t));
        }
        for t in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
            names.push(p(t));
        }
    }
    names.push("token_embd.weight".into());

    let backend = VulkanBackend::new().expect("vulkan");
    let dev = Device::Vulkan(0);
    let mut rng = 0x5eed_1234_9abc_def0u64;
    let mut ops: Vec<Op> = Vec::with_capacity(names.len());
    let mut weight_bytes = 0usize;
    for name in &names {
        let t = gg.tensor(name).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            t.type_id,
            kore_kernels::q2_0::Q2_0_TYPE_ID,
            "{name} is not Q2_0"
        );
        let (k, n) = (t.dims[0], t.dims[1]);
        let raw = raw_q2_0(&mut rng, k, n);
        let packed = kore_kernels::q2_0::repack_q2_0(&raw, k, n)
            .unwrap_or_else(|e| panic!("repack {name}: {e}"));
        weight_bytes += packed.len();
        let numel = packed.len();
        let s = Storage::from_bytes(DType::U8, numel, packed).unwrap();
        let w = Tensor::from_storage(s, &[numel]).to(dev).unwrap();

        let xs = bs
            .iter()
            .map(|&b| {
                let v = vec![0.5f32; b * k];
                Tensor::from_f32(&v, &[b, k]).to(dev).unwrap()
            })
            .collect();

        ops.push(Op { w, n, k, xs });
    }

    println!(
        "{} ops, {:.1} MiB of packed Q2_0 weights per step, {} reps",
        ops.len(),
        weight_bytes as f64 / (1024.0 * 1024.0),
        reps
    );
    println!(
        "{:>7}  {:>11}  {:>11}  {:>8}  {:>10}",
        "batch", "step ms", "per-token", "vs m=1", "eff GB/s"
    );

    let time = |bi: usize, ops: &[Op], reps: usize| -> f64 {
        let t0 = Instant::now();
        let mut last = None;
        for _ in 0..reps {
            for op in ops {
                last = Some(backend.gemv_q2_0(&op.xs[bi], &op.w, op.n, op.k).unwrap());
            }
        }
        let _ = last.unwrap().cpu().unwrap();
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };

    // Warm every batch size (shaders, and the output-buffer pool for 151
    // distinct shapes) before any of them is measured.
    for (bi, _) in bs.iter().enumerate() {
        for _ in 0..2 {
            let _ = time(bi, &ops, reps);
        }
    }

    // Interleaved rounds, not a sweep: the die changes temperature and clock
    // over a run, and a sweep charges that drift to whichever batch size
    // happened to go last. Alternating makes drift common-mode, so the ratio
    // is between batch sizes rather than between positions in the sweep.
    let rounds = 8usize;
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); bs.len()];
    for r in 0..rounds {
        for (bi, _) in bs.iter().enumerate() {
            let step_ms = time(bi, &ops, reps);
            if r > 0 {
                samples[bi].push(step_ms);
            }
        }
    }

    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let low = |v: &Vec<f64>| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let base = median(samples[0].clone());
    for (bi, &b) in bs.iter().enumerate() {
        let med = median(samples[bi].clone());
        let lo = low(&samples[bi]);
        println!(
            "{b:>7}  {med:>11.3}  {:>11.3}  {:>7.2}x  {:>10.1}   (min {lo:.3})",
            med / b as f64,
            base / med,
            weight_bytes as f64 / (med / 1e3) / 1e9
        );
    }
}
