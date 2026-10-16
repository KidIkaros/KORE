//! Token sampling strategies: temperature, top-k, top-p, repetition penalty, EOS.

/// Configuration for token sampling.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// Temperature scaling (1.0 = no change, <1 = sharper, >1 = flatter).
    pub temperature: f32,
    /// Top-k filtering (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) filtering (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty (1.0 = disabled, >1 = penalize repeats).
    pub repetition_penalty: f32,
    /// EOS token ID. Generation stops when this token is sampled.
    /// `None` means no early stopping.
    pub eos_token_id: Option<usize>,
    /// Use Coupled Gumbel-Max sampling (`argmax_i(p_i / E_{seed, pos, i})`) so draft and target
    /// models share identical per-position exponential noise during speculative decoding.
    pub gumbel_coupled: bool,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            eos_token_id: None,
            gumbel_coupled: false,
        }
    }
}

impl SamplerConfig {
    /// Greedy decoding (argmax).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }
}

/// Simple xorshift64 PRNG (no external dep needed).
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_1234
            } else {
                seed
            },
        }
    }

    /// Returns a random u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Apply repetition penalty to logits in-place.
///
/// For each token in `previous_tokens`, if the logit is positive, divide by penalty;
/// if negative, multiply by penalty.
pub fn apply_repetition_penalty(logits: &mut [f32], previous_tokens: &[usize], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &tok in previous_tokens {
        if tok < logits.len() {
            if logits[tok] > 0.0 {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

/// Sample a token from a `Tensor` of logits (`[1, vocab_size]` or `[vocab_size]`),
/// using the GPU-side `argmax_tokens()` fast path (4-byte readback) when greedy.
pub fn sample_from_tensor(
    logits_tensor: &kore_core::Tensor,
    previous_tokens: &[usize],
    config: &SamplerConfig,
    seed: u64,
    pos: usize,
) -> kore_core::Result<usize> {
    if config.temperature <= 0.0
        && config.repetition_penalty == 1.0
        && !logits_tensor.device().is_cpu()
    {
        let indices = logits_tensor.argmax_tokens()?;
        if let Some(&tok) = indices.last() {
            return Ok(tok as usize);
        }
    }
    let cpu_t = logits_tensor.cpu()?;
    let slice = cpu_t
        .as_f32_slice()
        .ok_or_else(|| kore_core::KoreError::UnsupportedDType(logits_tensor.dtype()))?;
    let dims = cpu_t.shape().dims();
    let vocab_size = *dims.last().unwrap_or(&slice.len());
    let last_row = &slice[slice.len().saturating_sub(vocab_size)..];
    Ok(sample_with_pos(
        last_row,
        previous_tokens,
        config,
        seed,
        pos,
    ))
}

/// Sample a token from logits at sequence position `pos`, supporting Coupled Gumbel-Max.
pub fn sample_with_pos(
    logits: &[f32],
    previous_tokens: &[usize],
    config: &SamplerConfig,
    seed: u64,
    pos: usize,
) -> usize {
    let mut rng = Rng::new(seed.wrapping_add(pos as u64));
    sample_internal(logits, previous_tokens, config, &mut rng, seed, pos)
}

/// Sample a token from logits using the given config and RNG.
///
/// `logits`: raw logits of shape [vocab_size]
/// `previous_tokens`: tokens generated so far (for repetition penalty)
/// `config`: sampling configuration
/// `rng`: random number generator
///
/// Returns the sampled token ID.
pub fn sample(
    logits: &[f32],
    previous_tokens: &[usize],
    config: &SamplerConfig,
    rng: &mut Rng,
) -> usize {
    let seed = rng.state;
    sample_internal(logits, previous_tokens, config, rng, seed, 0)
}

fn sample_internal(
    logits: &[f32],
    previous_tokens: &[usize],
    config: &SamplerConfig,
    rng: &mut Rng,
    seed: u64,
    pos: usize,
) -> usize {
    let vocab_size = logits.len();
    if vocab_size == 0 {
        return 0;
    }

    let mut logits = logits.to_vec();

    // 1. Repetition penalty
    apply_repetition_penalty(&mut logits, previous_tokens, config.repetition_penalty);

    // 2. Greedy (temperature == 0)
    if config.temperature <= 0.0 {
        return argmax(&logits);
    }

    // 3. Temperature scaling
    if config.temperature != 1.0 {
        let inv_t = 1.0 / config.temperature;
        for v in logits.iter_mut() {
            *v *= inv_t;
        }
    }

    // 4. Top-k filtering
    if config.top_k > 0 && config.top_k < vocab_size {
        let threshold = top_k_threshold(&logits, config.top_k);
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // 5. Softmax
    let probs = softmax(&logits);

    // 6. Top-p (nucleus) filtering
    let probs = if config.top_p < 1.0 {
        top_p_filter(&probs, config.top_p)
    } else {
        probs
    };

    // 7. Coupled Gumbel-Max or Categorical sampling
    if config.gumbel_coupled {
        gumbel_coupled_sample(&probs, seed, pos)
    } else {
        categorical_sample(&probs, rng)
    }
}

/// Coupled Gumbel-Max sampling (`Strata` style):
/// Computes `argmax_i (p_i / E_{seed, pos, i})` where `E_{seed, pos, i} = -ln(U_{seed, pos, i})`
/// is a deterministic SplitMix64-derived exponential random variate shared between draft and target.
pub fn gumbel_coupled_sample(probs: &[f32], seed: u64, pos: usize) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    let base_hash = seed ^ (pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

    for (i, &p) in probs.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        // SplitMix64 finalizer per (seed, pos, token_idx)
        let mut z = base_hash.wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;

        // Map to open interval (0, 1)
        let u = ((z >> 40) as f32 + 0.5) / ((1u64 << 24) as f32);
        let exp_noise = (-u.ln()).max(1e-20);
        let score = p / exp_noise;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&v| {
            if v == f32::NEG_INFINITY {
                0.0
            } else {
                let e = (v - max_val).exp();
                sum += e;
                e
            }
        })
        .collect();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            if *p > 0.0 {
                *p *= inv_sum;
            }
        }
    }
    probs
}

/// Find the k-th largest value (threshold for top-k) in O(V) time via Quickselect.
fn top_k_threshold(logits: &[f32], k: usize) -> f32 {
    let mut scratch: Vec<f32> = logits.to_vec();
    let idx = k.min(scratch.len()).saturating_sub(1);
    let (_, nth, _) = scratch.select_nth_unstable_by(idx, |a, b| b.total_cmp(a));
    *nth
}

/// Zero out probabilities below the nucleus threshold, then renormalize.
fn top_p_filter(probs: &[f32], p: f32) -> Vec<f32> {
    // Only collect non-zero probabilities (when top_k is active, this is at most k elements!)
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, prob)| prob > 0.0)
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff_idx = i + 1;
            break;
        }
    }

    let mut filtered = vec![0.0f32; probs.len()];
    let mut sum = 0.0f32;
    for &(idx, prob) in &indexed[..cutoff_idx] {
        filtered[idx] = prob;
        sum += prob;
    }

    // Renormalize only the active subset
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for &(idx, _) in &indexed[..cutoff_idx] {
            filtered[idx] *= inv_sum;
        }
    }
    filtered
}

/// Sample from a categorical distribution.
fn categorical_sample(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    // Fallback: return last non-zero
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let config = SamplerConfig::greedy();
        let mut rng = Rng::new(42);
        let token = sample(&logits, &[], &config, &mut rng);
        assert_eq!(token, 3); // highest logit
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        let logits = vec![1.0, 5.0, 3.0];
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = Rng::new(42);
        for _ in 0..10 {
            assert_eq!(sample(&logits, &[], &config, &mut rng), 1);
        }
    }

    #[test]
    fn test_top_k() {
        let logits = vec![1.0, 10.0, 9.0, 0.5, 0.1];
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 2,
            ..Default::default()
        };
        let mut rng = Rng::new(42);
        // With top_k=2, only indices 1 and 2 should ever be sampled
        for _ in 0..50 {
            let tok = sample(&logits, &[], &config, &mut rng);
            assert!(tok == 1 || tok == 2, "got unexpected token {}", tok);
        }
    }

    #[test]
    fn test_top_p() {
        // Token 0 has overwhelming probability
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let config = SamplerConfig {
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };
        let mut rng = Rng::new(42);
        // Should almost always sample token 0
        let mut count_0 = 0;
        for _ in 0..100 {
            if sample(&logits, &[], &config, &mut rng) == 0 {
                count_0 += 1;
            }
        }
        assert!(count_0 > 95);
    }

    #[test]
    fn test_repetition_penalty() {
        let logits = vec![5.0, 5.0, 5.0];
        let mut penalized = logits.clone();
        apply_repetition_penalty(&mut penalized, &[0, 1], 2.0);
        // Tokens 0 and 1 should be penalized (divided by 2 since positive)
        assert!((penalized[0] - 2.5).abs() < 0.01);
        assert!((penalized[1] - 2.5).abs() < 0.01);
        assert!((penalized[2] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(123);
        let mut rng2 = Rng::new(123);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_sampling_distribution() {
        // With high temperature, sampling should be more uniform
        let logits = vec![1.0; 10];
        let config = SamplerConfig {
            temperature: 1.0,
            ..Default::default()
        };
        let mut rng = Rng::new(42);
        let mut counts = vec![0usize; 10];
        for _ in 0..1000 {
            let tok = sample(&logits, &[], &config, &mut rng);
            counts[tok] += 1;
        }
        // Each token should appear roughly 100 times (uniform)
        for &c in &counts {
            assert!(c > 50 && c < 200, "count {} out of expected range", c);
        }
    }

    #[test]
    fn test_gumbel_coupled_sampling() {
        let target_logits = vec![2.5, 1.2, 0.3, -1.0, 0.8];
        let draft_logits = vec![2.48, 1.22, 0.29, -0.98, 0.79];
        let config = SamplerConfig {
            temperature: 0.8,
            top_k: 4,
            top_p: 0.95,
            repetition_penalty: 1.0,
            eos_token_id: None,
            gumbel_coupled: true,
        };
        let mut agreements = 0;
        for pos in 0..100 {
            let t_tok = sample_with_pos(&target_logits, &[], &config, 777, pos);
            let d_tok = sample_with_pos(&draft_logits, &[], &config, 777, pos);
            if t_tok == d_tok {
                agreements += 1;
            }
        }
        // Coupled Gumbel-Max with nearly identical draft/target logits should agree > 95% of the time
        assert!(
            agreements >= 95,
            "Expected high coupled agreement, got {agreements}/100"
        );
    }
}
