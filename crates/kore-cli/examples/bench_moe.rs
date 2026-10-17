//! Expert offload: what missing the expert cache costs, and what it buys.
//!
//! Builds a stack of MoE layers whose expert banks together run to hundreds of
//! megabytes -- far past what the card will hold -- then pumps decode steps
//! through them with a routing pattern you choose and reports the cache
//! behaviour and the decode rate.
//!
//! The point is the trade curve, not a single number: run it once with the
//! cache big enough for every expert (the "if it fit" ceiling) and once with a
//! slice small enough that the routing order churns it. The difference is the
//! offload tax, and the miss count is the size of the traffic it costs.
//!
//! ```text
//! cargo run --release -p kore-cli --example bench_moe -- [layers] [experts]
//!     [hidden] [ffn] [topk] [budget_experts] [steps] [skew]
//! ```
//!
//! Defaults: 4 layers, 32 experts, hidden 1024, ffn 2048, top-2, 4 experts
//! resident per layer, 32 timed steps, Zipf skew 1.0.
//!
//! The weights are synthetic but genuinely packed Q2_0, so the GEMVs run the
//! real kernels; nothing here is simulated on the CPU except the routing
//! decision, which is what a real engine would sample on the host too.

use std::time::Instant;

use kore_core::device::DeviceBackend;
use kore_core::{Device, Tensor};
use kore_nn::moe::{ExpertMats, HostMat, MoeConfig, MoeExpert, MoeLayer, RouterMat};

/// Deterministic byte/float source, so a run is reproducible from its seed.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// Uniform in `[0, 1)`.
    fn f32(&mut self) -> f32 {
        self.u32() as f32 / u32::MAX as f32
    }
    fn byte(&mut self) -> u8 {
        (self.u32() >> 24) as u8
    }
}

/// A packed Q2_0 matrix with valid fp16 scales (0.5) and pseudo-random codes.
///
/// The values are meaningless on purpose -- this measures the residency engine,
/// not model quality -- but the byte layout is the real thing, so `repack_q2_0`
/// and the GEMV kernels see exactly what they would see from a checkpoint.
fn synth_mat(rng: &mut Lcg, m: usize, n: usize) -> HostMat {
    assert!(m.is_multiple_of(64));
    let row_bytes = (m / 64) * 18;
    let mut bytes = vec![0u8; n * row_bytes];
    for row in 0..n {
        for b in 0..(m / 64) {
            let off = row * row_bytes + b * 18;
            bytes[off] = 0x00;
            bytes[off + 1] = 0x38; // fp16 0.5
            for k in 2..18 {
                bytes[off + k] = rng.byte();
            }
        }
    }
    HostMat::new(bytes, m, n).expect("synth mat")
}

/// Unused router: the benchmark supplies routing directly, through
/// `forward_routed`, and never asks the layer to decide.
fn idle_router(n_expert: usize, hidden: usize) -> RouterMat {
    RouterMat::new(vec![0.0; n_expert * hidden], n_expert, hidden).expect("router")
}

/// Weighted sampling without replacement (Efraimidis-Spirakis: key `u^(1/w)`),
/// returning the top `k` expert indices -- how a real router's top-k behaves,
/// with a popularity skew rather than uniform pressure on the cache.
fn sample_experts(rng: &mut Lcg, pop: &[f32], k: usize) -> Vec<usize> {
    let mut keys: Vec<(f32, usize)> = pop
        .iter()
        .enumerate()
        .map(|(e, &w)| {
            let u = rng.f32().clamp(1e-7, 1.0);
            (u.powf(1.0 / w), e)
        })
        .collect();
    keys.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    keys.truncate(k);
    keys.into_iter().map(|(_, e)| e).collect()
}

#[derive(Clone, Copy, Default)]
struct Counters {
    hits: u64,
    misses: u64,
    uploads: u64,
    evictions: u64,
    repacks: u64,
}

fn snapshot(model: &[MoeLayer]) -> Counters {
    let mut c = Counters::default();
    for l in model {
        c.hits += l.stats.hits;
        c.misses += l.stats.misses;
        c.uploads += l.stats.uploads;
        c.evictions += l.stats.evictions;
        c.repacks += l.stats.repacks;
    }
    c
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, d: usize| argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let layers = arg(0, 4);
    let n_expert = arg(1, 32);
    let hidden = arg(2, 1024);
    let ffn = arg(3, 2048);
    let topk = arg(4, 2).min(n_expert);
    let budget = arg(5, 4);
    let steps = arg(6, 32);
    let skew: f32 = argv.get(7).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    let device = Device::Vulkan(0);
    let backend: std::sync::Arc<dyn DeviceBackend> =
        kore_core::device::get_backend(device).expect("vulkan device");

    let cfg = MoeConfig {
        n_expert,
        n_expert_used: topk,
        expert_ffn: ffn,
        norm_topk_prob: true,
    };
    let per_expert = cfg.expert_bytes(hidden);
    let mut rng = Lcg::new(0x1234_5678_9abc_def0);

    let build = Instant::now();
    let mut model: Vec<MoeLayer> = (0..layers)
        .map(|_| {
            let experts: Vec<MoeExpert> = (0..n_expert)
                .map(|_| {
                    MoeExpert::offloaded(ExpertMats {
                        gate: synth_mat(&mut rng, hidden, ffn),
                        up: synth_mat(&mut rng, hidden, ffn),
                        down: synth_mat(&mut rng, ffn, hidden),
                    })
                })
                .collect();
            MoeLayer::new(cfg, hidden, idle_router(n_expert, hidden), experts, budget)
                .expect("moe layer")
        })
        .collect();
    let build_s = build.elapsed().as_secs_f64();

    let bank = per_expert * n_expert * layers;
    let cache = per_expert * model[0].cache.capacity() * layers;
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);

    println!("=== expert offload bench ===");
    println!(
        "expert bank  : {layers} layers x {n_expert} experts x {:.2} MiB = {:.1} MiB of Q2_0 weights",
        mib(per_expert),
        mib(bank)
    );
    println!(
        "vram cache   : {} experts/layer = {:.1} MiB total ({:.1}% of the bank)",
        model[0].cache.capacity(),
        mib(cache),
        cache as f64 / bank as f64 * 100.0
    );
    println!("routing      : zipf s={skew:.2}, top-{topk} of {n_expert}");
    println!("hidden/ffn   : {hidden} / {ffn}");
    println!("built in     : {build_s:.2} s");

    // Activations are small and fixed: this measures the expert bank, so the
    // same vector feeds every layer and the routing is injected per layer.
    let x: Vec<f32> = (0..hidden).map(|_| rng.f32() * 0.1).collect();
    let x_rot = Tensor::from_f32(&x, &[1, hidden]).to(device).expect("x");

    let pop: Vec<f32> = (0..n_expert)
        .map(|e| 1.0 / ((e + 1) as f32).powf(skew))
        .collect();

    let step = |model: &mut Vec<MoeLayer>, rng: &mut Lcg| {
        let mut out = None;
        for layer in model.iter_mut() {
            let ids = sample_experts(rng, &pop, topk);
            let w = vec![1.0f32 / topk as f32; topk];
            out = Some(
                layer
                    .forward_routed(backend.as_ref(), device, &ids, &w, &x_rot)
                    .expect("moe forward"),
            );
        }
        // A real decode loop reads the logits back to sample, so the step ends
        // with a host sync. Without this the loop would only be measuring how
        // fast the CPU can enqueue.
        let _ = out.expect("layer stack").cpu().expect("sync");
    };

    // Warm up long enough to touch every expert at least once, so the timed
    // window reports steady-state residency rather than first-touch loads.
    let warmup = (4 * n_expert / topk.max(1)).max(8);
    for _ in 0..warmup {
        step(&mut model, &mut rng);
    }
    let before = snapshot(&model);

    let t = Instant::now();
    for _ in 0..steps {
        step(&mut model, &mut rng);
    }
    let secs = t.elapsed().as_secs_f64();
    let after = snapshot(&model);

    let hits = after.hits - before.hits;
    let misses = after.misses - before.misses;
    let uploads = after.uploads - before.uploads;
    let evictions = after.evictions - before.evictions;
    let repacks = after.repacks - before.repacks;
    let touched = hits + misses;
    let streamed = uploads * per_expert as u64;

    println!();
    println!("warmup       : {warmup} steps (fills the cache)");
    println!(
        "decode       : {steps} steps in {secs:.3} s = {:.1} tok/s",
        steps as f64 / secs
    );
    println!(
        "cache        : {hits} hits / {misses} misses ({:.1}% hit), {evictions} evictions, \
{repacks} host repacks",
        if touched > 0 {
            hits as f64 / touched as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "offload      : {uploads} expert uploads = {:.1} MiB streamed from RAM ({:.2} MiB/token)",
        mib(streamed as usize),
        streamed as f64 / steps as f64 / (1024.0 * 1024.0)
    );
    println!(
        "per token    : {} expert FFNs ({} GEMV dispatches)",
        layers * topk,
        layers * topk * 3
    );
}
