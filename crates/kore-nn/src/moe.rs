//! Mixture-of-Experts FFN with a bounded expert-residency planner.
//!
//! An MoE decode step is: route the token to `k` of `n_expert` experts, run the
//! selected experts' FFNs, and sum them weighted by the router probability.
//! What makes this interesting on small hardware is that the model's *weights*,
//! not its activations, are what does not fit: an MoE's experts are a large,
//! sparsely-used weight bank, so a layer can keep only a hot subset in VRAM and
//! leave the rest in system RAM.
//!
//! This module owns that split. Each expert holds:
//!
//! - a **host source** (`ExpertMats`): the packed Q2_0 bytes, mmap-backed in
//!   production, always present -- this is the RAM-resident bank; and
//! - an optional **device copy** (`ExpertMatsDev`), present only while the
//!   expert is in the VRAM cache.
//!
//! `ExpertCache` is the planner: an LRU over `n_expert` slots bounded by
//! `capacity`. A miss uploads the expert (repack + copy) and evicts the least
//! recently used one. Because promotion is a *pure copy of the same bytes*, an
//! expert that thrashes in and out of the cache computes the identical result
//! as one that never leaves: residency changes *where* the weights live, not
//! the arithmetic, so a fully-resident run and a memory-starved run are
//! bit-identical. (That property is asserted in the tests below.) It is the
//! reason this design is compatible with the engine's batch-invariance
//! contract where a CPU-compute fallback would not be: the CPU twin sums a row
//! in a different order, and a different order is a different answer.
//!
//! The alternative offload policy -- compute a miss on the host CPU from the
//! same packed bytes, overlapping with the GPU -- is deliberately not wired in
//! here. It is the right choice when the CPU is otherwise idle and the expert
//! is large enough that uploading it costs more than computing it, but it
//! trades away exactness. See `docs/kore-improvement-plan.md`.

use std::collections::VecDeque;

use kore_core::device::DeviceBackend;
use kore_core::{DType, Device, KoreError, Result, Storage, Tensor};
use kore_kernels::q2_0::gemv_q2_0;

/// Routing configuration for one MoE layer.
///
/// Mirrors the GGUF metadata real MoE checkpoints carry
/// (`expert_count`, `expert_used_count`, `expert_feed_forward_length`,
/// `expert_weights_norm`); a loader fills this from those keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeConfig {
    /// Total experts in the layer.
    pub n_expert: usize,
    /// Experts a single token is routed to (`k`).
    pub n_expert_used: usize,
    /// Per-expert intermediate (FFN) width.
    pub expert_ffn: usize,
    /// Renormalise the selected top-`k` probabilities to sum to one.
    pub norm_topk_prob: bool,
}

impl MoeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.n_expert == 0 {
            return Err(KoreError::StorageError("moe: n_expert must be > 0".into()));
        }
        if self.n_expert_used == 0 || self.n_expert_used > self.n_expert {
            return Err(KoreError::StorageError(format!(
                "moe: n_expert_used {} out of range for n_expert {}",
                self.n_expert_used, self.n_expert
            )));
        }
        if self.expert_ffn == 0 {
            return Err(KoreError::StorageError(
                "moe: expert_ffn must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Bytes of packed Q2_0 weights one expert occupies (3 projections).
    pub fn expert_bytes(&self, hidden: usize) -> usize {
        let per = |m: usize, n: usize| n * (m / 64) * 18;
        per(hidden, self.expert_ffn) + per(hidden, self.expert_ffn) + per(self.expert_ffn, hidden)
    }

    /// How many experts of this shape fit in `budget_bytes`, at least one.
    pub fn capacity_for(&self, hidden: usize, budget_bytes: usize) -> usize {
        let per = self.expert_bytes(hidden).max(1);
        (budget_bytes / per).clamp(1, self.n_expert)
    }
}

/// Router decision for one token: the selected expert indices in descending
/// probability, and their (optionally renormalised) weights.
#[derive(Clone, Debug, PartialEq)]
pub struct Routing {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Softmax over `logits`, then the top `k` by probability.
///
/// Ties are broken by the *lower* expert index, so the choice does not depend
/// on sort stability or on the order the logits were produced in. That matters
/// because the routing decision is part of the model's output: a tie that
/// resolves differently on a different thread count is a different token.
pub fn route(logits: &[f32], cfg: &MoeConfig) -> Routing {
    let n = logits.len().max(1);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then_with(|| a.cmp(&b)));
    order.truncate(cfg.n_expert_used.min(n));

    let mut weights: Vec<f32> = order.iter().map(|&e| probs[e]).collect();
    if cfg.norm_topk_prob {
        let s: f32 = weights.iter().sum();
        if s > 0.0 {
            for w in weights.iter_mut() {
                *w /= s;
            }
        }
    }
    Routing {
        experts: order,
        weights,
    }
}

/// Result of asking the cache for an expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOutcome {
    /// Already resident; nothing to do.
    Hit,
    /// Promoted into the cache. If `evicted` is set, that expert's device copy
    /// was dropped to make room and must be released by the caller.
    Miss { evicted: Option<usize> },
}

/// Per-layer LRU over expert VRAM slots.
///
/// `order` is least-recently-used first, so eviction is `pop_front`. `capacity`
/// is clamped to `[1, n_expert]`: at zero an expert could never be computed
/// under this policy, so one slot is the floor.
#[derive(Clone, Debug)]
pub struct ExpertCache {
    capacity: usize,
    order: VecDeque<usize>,
    resident: Vec<bool>,
    /// Experts that have never been promoted, for `is_resident` bookkeeping.
    hits: u64,
    misses: u64,
}

impl ExpertCache {
    pub fn new(n_expert: usize, capacity: usize) -> Self {
        Self {
            capacity: capacity.clamp(1, n_expert.max(1)),
            order: VecDeque::new(),
            resident: vec![false; n_expert],
            hits: 0,
            misses: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn resident_count(&self) -> usize {
        self.order.len()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn is_resident(&self, expert: usize) -> bool {
        self.resident.get(expert).copied().unwrap_or(false)
    }

    /// Currently resident experts, least recently used first.
    pub fn resident(&self) -> impl Iterator<Item = usize> + '_ {
        self.order.iter().copied()
    }

    /// Move `expert` to the most-recently-used end, evicting the LRU if the
    /// cache is already full.
    pub fn acquire(&mut self, expert: usize) -> CacheOutcome {
        assert!(
            expert < self.resident.len(),
            "expert {expert} out of range for {} experts",
            self.resident.len()
        );
        if self.resident[expert] {
            self.hits += 1;
            if let Some(pos) = self.order.iter().position(|&e| e == expert) {
                self.order.remove(pos);
            }
            self.order.push_back(expert);
            return CacheOutcome::Hit;
        }
        self.misses += 1;
        let evicted = if self.order.len() >= self.capacity {
            self.order.pop_front()
        } else {
            None
        };
        if let Some(v) = evicted {
            self.resident[v] = false;
        }
        self.resident[expert] = true;
        self.order.push_back(expert);
        CacheOutcome::Miss { evicted }
    }
}

/// A host-resident Q2_0 matrix in the GGUF `[m, n]` layout (`m` contiguous),
/// so `out[n] = sum_m w[m, n] * x[m]`. In production the bytes are a slice of
/// the mmap'd checkpoint; here they are owned so a synthetic model can be built
/// without a file.
#[derive(Clone)]
pub struct HostMat {
    pub bytes: Vec<u8>,
    /// Reduction dimension (contiguous, multiple of 64).
    pub m: usize,
    /// Output dimension (rows).
    pub n: usize,
}

impl HostMat {
    /// Packed bytes for a matrix whose dequantised shape is `[m, n]`.
    pub fn new(bytes: Vec<u8>, m: usize, n: usize) -> Result<Self> {
        let expect = n * (m / 64) * 18;
        if !m.is_multiple_of(64) {
            return Err(KoreError::StorageError(format!(
                "moe: weight in-dim {m} is not a multiple of 64"
            )));
        }
        if bytes.len() != expect {
            return Err(KoreError::StorageError(format!(
                "moe: weight [{m}, {n}] needs {expect} packed bytes, got {}",
                bytes.len()
            )));
        }
        Ok(Self { bytes, m, n })
    }

    /// CPU mat-vec against this matrix (the RAM-side twin of `gemv_q2_0`).
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.m {
            return Err(KoreError::StorageError(format!(
                "moe: matvec in-dim {} != {}",
                x.len(),
                self.m
            )));
        }
        let mut out = vec![0.0f32; self.n];
        gemv_q2_0(&self.bytes, self.m, self.n, x, &mut out)?;
        Ok(out)
    }

    /// Repack into the GPU layout and upload. Byte count is preserved.
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        let repacked = kore_kernels::q2_0::repack_q2_0(&self.bytes, self.m, self.n)?;
        let len = repacked.len();
        let s = Storage::from_bytes(DType::U8, len, repacked)?;
        Tensor::from_storage(s, &[len]).to(device)
    }
}

/// The three projections of one expert, host-resident.
#[derive(Clone)]
pub struct ExpertMats {
    pub gate: HostMat,
    pub up: HostMat,
    pub down: HostMat,
}

/// The three projections of one expert, device-resident.
pub struct ExpertMatsDev {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

/// One expert: a RAM-resident source bank, plus its VRAM cache slot when held.
pub struct MoeExpert {
    /// Offload source. `None` for experts that are permanently device-only
    /// (e.g. a shared expert that is never evicted).
    pub host: Option<ExpertMats>,
    /// Current VRAM copy, if the cache holds it.
    pub device: Option<ExpertMatsDev>,
}

impl MoeExpert {
    /// An expert living in RAM, to be promoted on demand.
    pub fn offloaded(host: ExpertMats) -> Self {
        Self {
            host: Some(host),
            device: None,
        }
    }

    /// An expert pinned in VRAM (never evicted by the planner).
    pub fn resident(dev: ExpertMatsDev) -> Self {
        Self {
            host: None,
            device: Some(dev),
        }
    }
}

/// Counters for one layer, for telemetry and for the residency assertions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoeStats {
    pub hits: u64,
    pub misses: u64,
    pub uploads: u64,
    pub evictions: u64,
}

/// A router matrix stored as plain f32 `[n_expert, hidden]`.
///
/// Real MoE routers are f32 (DeepSeek-V3, Qwen-MoE) and are *not* folded into
/// the activation rotation the expert weights use, so they consume the
/// un-rotated post-norm activation. It is small enough (`n_expert * hidden`) to
/// run on the CPU without a device round-trip.
pub struct RouterMat {
    pub w: Vec<f32>,
    pub n_expert: usize,
    pub hidden: usize,
}

impl RouterMat {
    pub fn new(w: Vec<f32>, n_expert: usize, hidden: usize) -> Result<Self> {
        if w.len() != n_expert * hidden {
            return Err(KoreError::StorageError(format!(
                "moe: router needs {} weights, got {}",
                n_expert * hidden,
                w.len()
            )));
        }
        Ok(Self {
            w,
            n_expert,
            hidden,
        })
    }

    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        (0..self.n_expert)
            .map(|e| {
                let row = &self.w[e * self.hidden..(e + 1) * self.hidden];
                row.iter().zip(x).map(|(a, b)| a * b).sum()
            })
            .collect()
    }
}

/// One MoE layer: router, expert bank, residency cache.
pub struct MoeLayer {
    pub cfg: MoeConfig,
    pub hidden: usize,
    pub router: RouterMat,
    pub experts: Vec<MoeExpert>,
    pub cache: ExpertCache,
    pub stats: MoeStats,
}

impl MoeLayer {
    pub fn new(
        cfg: MoeConfig,
        hidden: usize,
        router: RouterMat,
        experts: Vec<MoeExpert>,
        capacity: usize,
    ) -> Result<Self> {
        cfg.validate()?;
        if experts.len() != cfg.n_expert {
            return Err(KoreError::StorageError(format!(
                "moe: {} experts supplied for n_expert {}",
                experts.len(),
                cfg.n_expert
            )));
        }
        if router.n_expert != cfg.n_expert || router.hidden != hidden {
            return Err(KoreError::StorageError(
                "moe: router shape does not match the layer".into(),
            ));
        }
        Ok(Self {
            cache: ExpertCache::new(cfg.n_expert, capacity),
            cfg,
            hidden,
            router,
            experts,
            stats: MoeStats::default(),
        })
    }

    /// Bytes of expert weights currently held in VRAM.
    pub fn resident_bytes(&self) -> usize {
        self.experts.iter().filter(|e| e.device.is_some()).count()
            * self.cfg.expert_bytes(self.hidden)
    }

    fn promote(&mut self, expert: usize, device: Device) -> Result<()> {
        if self.experts[expert].device.is_some() {
            return Ok(());
        }
        let (gate, up, down) = {
            let host = self.experts[expert].host.as_ref().ok_or_else(|| {
                KoreError::StorageError(format!("moe: expert {expert} has no host source"))
            })?;
            (
                host.gate.to_device(device)?,
                host.up.to_device(device)?,
                host.down.to_device(device)?,
            )
        };
        self.experts[expert].device = Some(ExpertMatsDev { gate, up, down });
        self.stats.uploads += 1;
        Ok(())
    }

    fn release(&mut self, expert: usize) {
        if self.experts[expert].device.take().is_some() {
            self.stats.evictions += 1;
        }
    }

    /// Route `x_norm` and run the selected experts.
    ///
    /// `x_norm` is the post-norm activation on the host (it drives the router).
    /// `x_rot` is the same activation after the model's folded Hadamard
    /// transform, resident on the device; it is what the expert GEMVs consume.
    /// For an unfolded weight set the two are the same vector.
    pub fn forward(
        &mut self,
        backend: &dyn DeviceBackend,
        device: Device,
        x_norm: &[f32],
        x_rot: &Tensor,
    ) -> Result<Tensor> {
        if x_norm.len() != self.hidden {
            return Err(KoreError::StorageError(format!(
                "moe: activation width {} != hidden {}",
                x_norm.len(),
                self.hidden
            )));
        }
        let routing = route(&self.router.logits(x_norm), &self.cfg);

        let mut acc: Option<Tensor> = None;
        for (&expert, &weight) in routing.experts.iter().zip(routing.weights.iter()) {
            match self.cache.acquire(expert) {
                CacheOutcome::Hit => {}
                CacheOutcome::Miss { evicted } => {
                    if let Some(victim) = evicted {
                        self.release(victim);
                    }
                    self.promote(expert, device)?;
                }
            }
            let mats = self.experts[expert].device.as_ref().ok_or_else(|| {
                KoreError::StorageError(format!("moe: expert {expert} not resident after acquire"))
            })?;

            let gate = backend.gemv_q2_0(x_rot, &mats.gate, self.cfg.expert_ffn, self.hidden)?;
            let up = backend.gemv_q2_0(x_rot, &mats.up, self.cfg.expert_ffn, self.hidden)?;
            let act = backend.swiglu(&gate, &up)?;
            let down = backend.gemv_q2_0(&act, &mats.down, self.hidden, self.cfg.expert_ffn)?;
            let scaled = backend.mul_scalar(&down, weight)?;

            // Sum in routing order, starting from the first expert's term so a
            // single-expert route is exactly `w * down`, with no zero added.
            acc = Some(match acc {
                None => scaled,
                Some(prev) => backend.add(&prev, &scaled)?,
            });
        }

        self.stats.hits = self.cache.hits();
        self.stats.misses = self.cache.misses();
        acc.ok_or_else(|| KoreError::StorageError("moe: layer routed no expert".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MoeConfig {
        MoeConfig {
            n_expert: 4,
            n_expert_used: 2,
            expert_ffn: 512,
            norm_topk_prob: true,
        }
    }

    #[test]
    fn route_takes_the_top_k_and_renormalises() {
        let c = cfg();
        let logits = [1.0f32, 2.0, 3.0, 0.5];
        let r = route(&logits, &c);
        assert_eq!(r.experts, vec![2, 1]);

        let p = |l: f32| (l - 3.0).exp();
        let s = p(1.0) + p(2.0) + p(3.0) + p(0.5);
        let (top, second) = (p(3.0) / s, p(2.0) / s);
        assert!((r.weights[0] - top / (top + second)).abs() < 1e-6);
        assert!((r.weights[1] - second / (top + second)).abs() < 1e-6);
        assert!((r.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn route_breaks_ties_by_lowest_index() {
        let c = cfg();
        let r = route(&[0.0, 0.0, 0.0, 0.0], &c);
        assert_eq!(r.experts, vec![0, 1]);
        assert!((r.weights[0] - 0.5).abs() < 1e-6);
        assert!((r.weights[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn route_leaves_weights_unnormalised_when_configured() {
        let mut c = cfg();
        c.norm_topk_prob = false;
        let r = route(&[0.0, 0.0, 0.0, 0.0], &c);
        assert!((r.weights[0] - 0.25).abs() < 1e-6);
        assert!((r.weights[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn cache_evicts_least_recently_used() {
        let mut c = ExpertCache::new(4, 2);
        assert_eq!(c.acquire(0), CacheOutcome::Miss { evicted: None });
        assert_eq!(c.acquire(1), CacheOutcome::Miss { evicted: None });
        // Full: 0 is oldest, so it goes.
        assert_eq!(c.acquire(2), CacheOutcome::Miss { evicted: Some(0) });
        assert!(!c.is_resident(0));
        // Touching 1 refreshes it, so 2 becomes evictable and 1 survives.
        assert_eq!(c.acquire(1), CacheOutcome::Hit);
        assert_eq!(c.acquire(3), CacheOutcome::Miss { evicted: Some(2) });
        assert!(c.is_resident(1) && c.is_resident(3));
        assert_eq!(c.hits(), 1);
        assert_eq!(c.misses(), 4);
        assert_eq!(c.resident_count(), 2);
    }

    #[test]
    fn cache_capacity_is_clamped_to_the_expert_count() {
        assert_eq!(ExpertCache::new(4, 0).capacity(), 1);
        assert_eq!(ExpertCache::new(4, 99).capacity(), 4);
    }

    #[test]
    fn config_derives_capacity_from_a_byte_budget() {
        let c = cfg();
        let per = c.expert_bytes(512);
        // Note the hidden width is the *reduction* dim of gate/up and the
        // output dim of down; 512 keeps every projection a multiple of 64.
        assert_eq!(per, {
            let one = |m: usize, n: usize| n * (m / 64) * 18;
            one(512, 512) + one(512, 512) + one(512, 512)
        });
        assert_eq!(c.capacity_for(512, per - 1), 1);
        assert_eq!(c.capacity_for(512, per * 3), 3);
        assert_eq!(c.capacity_for(512, usize::MAX), 4);
    }

    // ---- Q2_0 packing helpers, so the device path can be exercised without a
    // checkpoint on disk. Values are exact: every scale is a power of two, so
    // the fp16 scale is lossless and the codes reproduce `d * (code - 1)`.

    /// Deterministic uniform values in `[-1, 1)`, so a test failure is
    /// reproducible from the seed alone.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u32 << 23) as f32) - 1.0
        }
    }

    /// fp16 bits for a positive power of two (no subnormal, no rounding).
    fn f16_bits_pow2(d: f32) -> u16 {
        let exp = ((d.to_bits() >> 23) & 0xff) as i32 - 127 + 15;
        ((exp as u16) & 0x1f) << 10
    }

    /// Pack `n` rows of `m` f32 into the raw Q2_0 layout (64 elems / 18 bytes).
    fn encode_q2_0(w: &[f32], m: usize, n: usize) -> Vec<u8> {
        assert_eq!(m % 64, 0);
        assert_eq!(w.len(), m * n);
        let max = w.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let d = if max == 0.0 {
            1.0
        } else {
            (max / 2.0).log2().ceil().exp2()
        };
        let bits = f16_bits_pow2(d).to_le_bytes();
        let mut out = Vec::with_capacity(n * (m / 64) * 18);
        for r in 0..n {
            for b in 0..(m / 64) {
                let mut blk = [0u8; 18];
                blk[0] = bits[0];
                blk[1] = bits[1];
                for k in 0..64 {
                    let code = ((w[r * m + b * 64 + k] / d) + 1.0).round().clamp(0.0, 3.0) as u8;
                    blk[2 + k / 4] |= code << (2 * (k % 4));
                }
                out.extend_from_slice(&blk);
            }
        }
        out
    }

    fn host_mat(rng: &mut Lcg, m: usize, n: usize) -> HostMat {
        let w: Vec<f32> = (0..m * n).map(|_| rng.next()).collect();
        HostMat::new(encode_q2_0(&w, m, n), m, n).unwrap()
    }

    /// Reference FFN using the *dequantised* weights, i.e. exactly the values
    /// the kernels see, so the Q2_0 encoding cancels out of the comparison.
    fn reference(host: &ExpertMats, x: &[f32], m: usize, n: usize) -> Vec<f32> {
        let deq =
            |hm: &HostMat| kore_kernels::q2_0::dequantize_q2_0(&hm.bytes, hm.m * hm.n).unwrap();
        let rows = |hm: &HostMat| -> Vec<Vec<f32>> {
            let d = deq(hm);
            d.chunks_exact(hm.m).map(|r| r.to_vec()).collect()
        };
        let mv = |hm: &HostMat, x: &[f32]| -> Vec<f32> {
            rows(hm)
                .iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let g = mv(&host.gate, x);
        let u = mv(&host.up, x);
        let act: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| {
                let s = 1.0 / (1.0 + (-g).exp());
                g * s * u
            })
            .collect();
        assert_eq!(act.len(), n);
        assert_eq!(m, x.len());
        mv(&host.down, &act)
    }

    fn to_vec(t: &Tensor) -> Vec<f32> {
        let cpu = t.cpu().unwrap();
        cpu.as_f32_slice().unwrap().to_vec()
    }

    fn vulkan() -> Option<(std::sync::Arc<dyn DeviceBackend>, Device)> {
        let _ = kore_vulkan::register_vulkan_backend();
        let dev = Device::Vulkan(0);
        kore_core::device::get_backend(dev).map(|b| (b, dev))
    }

    /// Build `n_expert` RAM-resident experts plus a router, deterministically.
    fn synth_layer(capacity: usize) -> (MoeLayer, Vec<f32>) {
        const HIDDEN: usize = 512;
        let c = MoeConfig {
            n_expert: 4,
            n_expert_used: 2,
            expert_ffn: 512,
            norm_topk_prob: true,
        };
        let mut rng = Lcg(0x5eed_1234_abcd_ef01);
        let mut experts = Vec::new();
        for _ in 0..c.n_expert {
            experts.push(MoeExpert::offloaded(ExpertMats {
                gate: host_mat(&mut rng, HIDDEN, c.expert_ffn),
                up: host_mat(&mut rng, HIDDEN, c.expert_ffn),
                down: host_mat(&mut rng, c.expert_ffn, HIDDEN),
            }));
        }
        let router_w: Vec<f32> = (0..c.n_expert * HIDDEN).map(|_| rng.next()).collect();
        let router = RouterMat::new(router_w, c.n_expert, HIDDEN).unwrap();
        let layer = MoeLayer::new(c, HIDDEN, router, experts, capacity).unwrap();
        let x: Vec<f32> = (0..HIDDEN).map(|_| rng.next()).collect();
        (layer, x)
    }

    #[test]
    fn resident_and_starved_runs_are_bit_identical() {
        let Some((backend, device)) = vulkan() else {
            eprintln!("skipping: no Vulkan backend");
            return;
        };

        // Same expert bank, two cache sizes: everything resident, and a single
        // slot that thrashes on every route.
        let (mut roomy, x) = synth_layer(4);
        let (mut starved, _) = synth_layer(1);

        let x_rot = Tensor::from_f32(&x, &[1, 512]).to(device).unwrap();
        let a = roomy.forward(backend.as_ref(), device, &x, &x_rot).unwrap();
        let b = starved
            .forward(backend.as_ref(), device, &x, &x_rot)
            .unwrap();

        let (av, bv) = (to_vec(&a), to_vec(&b));
        assert_eq!(av.len(), 512);
        // Promotion is a copy of the same bytes, so this is exact equality,
        // not a tolerance: residency must not perturb a single bit.
        assert_eq!(av, bv);

        // Cold cache: both routed experts are loaded once, nothing evicted.
        assert_eq!(roomy.stats.misses, 2);
        assert_eq!(roomy.stats.uploads, 2, "both routed experts promoted once");
        assert_eq!(roomy.stats.evictions, 0);
        assert!(starved.stats.misses >= 2, "single slot must thrash");
        assert!(starved.stats.evictions >= 1);
    }

    #[test]
    fn device_path_numerically_matches_the_dequantised_reference() {
        let Some((backend, device)) = vulkan() else {
            eprintln!("skipping: no Vulkan backend");
            return;
        };
        let (mut layer, x) = synth_layer(4);
        let x_rot = Tensor::from_f32(&x, &[1, 512]).to(device).unwrap();
        let got = to_vec(&layer.forward(backend.as_ref(), device, &x, &x_rot).unwrap());

        // Reproduce the same routing decision on the host, then the reference.
        let routing = route(&layer.router.logits(&x), &layer.cfg);
        let mut want = vec![0.0f32; 512];
        for (&e, &w) in routing.experts.iter().zip(&routing.weights) {
            let host = layer.experts[e].host.as_ref().unwrap();
            let d = reference(host, &x, 512, layer.cfg.expert_ffn);
            for (a, b) in want.iter_mut().zip(&d) {
                *a += w * b;
            }
        }
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            let denom = b.abs().max(1e-3);
            assert!(
                (a - b).abs() / denom < 1e-3,
                "device {a} vs reference {b} diverged"
            );
        }
    }

    /// Router whose logits are exactly `x[0..n_expert]`, so a test can choose
    /// the top-k just by ordering the leading components of the activation.
    fn selection_router(n_expert: usize, hidden: usize) -> RouterMat {
        let mut w = vec![0.0f32; n_expert * hidden];
        for e in 0..n_expert {
            w[e * hidden + e] = 1.0;
        }
        RouterMat::new(w, n_expert, hidden).unwrap()
    }

    #[test]
    fn resident_bytes_stay_within_the_cache_budget() {
        const HIDDEN: usize = 512;
        let Some((backend, device)) = vulkan() else {
            eprintln!("skipping: no Vulkan backend");
            return;
        };
        let cfg = MoeConfig {
            n_expert: 4,
            n_expert_used: 2,
            expert_ffn: 512,
            norm_topk_prob: true,
        };
        let mut rng = Lcg(0xbeef_cafe_1234_5678);
        let experts: Vec<MoeExpert> = (0..cfg.n_expert)
            .map(|_| {
                MoeExpert::offloaded(ExpertMats {
                    gate: host_mat(&mut rng, HIDDEN, cfg.expert_ffn),
                    up: host_mat(&mut rng, HIDDEN, cfg.expert_ffn),
                    down: host_mat(&mut rng, cfg.expert_ffn, HIDDEN),
                })
            })
            .collect();
        let mut layer = MoeLayer::new(
            cfg,
            HIDDEN,
            selection_router(cfg.n_expert, HIDDEN),
            experts,
            2,
        )
        .unwrap();
        let per = cfg.expert_bytes(HIDDEN);
        assert_eq!(layer.resident_bytes(), 0);

        // The leading four components are the routing logits. These three
        // routes touch {3,2}, then {0,1}, then {2,3}: five distinct experts
        // through two slots.
        for lead in [
            [0.0f32, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [2.0, 0.0, 3.0, 1.0],
        ] {
            let mut x = vec![0.0f32; HIDDEN];
            x[..4].copy_from_slice(&lead);
            let x_rot = Tensor::from_f32(&x, &[1, HIDDEN]).to(device).unwrap();
            layer.forward(backend.as_ref(), device, &x, &x_rot).unwrap();
            assert!(layer.cache.resident_count() <= 2);
            assert!(layer.resident_bytes() <= per * layer.cache.capacity());
        }
        assert!(
            layer.stats.evictions > 0,
            "three routes through two slots must evict"
        );
        assert_eq!(layer.resident_bytes(), per * layer.cache.resident_count());
    }
}
