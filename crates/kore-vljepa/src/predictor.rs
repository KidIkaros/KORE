//! Mamba-3 Predictor — replaces Llama-3.2-1B in VL-JEPA.
//!
//! Maps (visual embeddings + text query) → predicted target embedding
//! using a Mamba-3 MixerModel backbone with trapezoidal discretization,
//! complex-valued state dynamics, and MIMO multi-head structure.

use rand::Rng;

use kore_mamba::MixerModel;

use crate::config::Mamba3PredictorConfig;

/// Linear projection layer (weight + bias).
struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (in_dim + out_dim) as f32).sqrt();
        Self {
            weight: (0..out_dim * in_dim).map(|_| rng.gen_range(-std..std)).collect(),
            bias: vec![0.0f32; out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Forward: (n, in_dim) → (n, out_dim)
    fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * self.out_dim];
        for i in 0..n {
            for o in 0..self.out_dim {
                let mut acc = self.bias[o];
                for k in 0..self.in_dim {
                    acc += x[i * self.in_dim + k] * self.weight[o * self.in_dim + k];
                }
                out[i * self.out_dim + o] = acc;
            }
        }
        out
    }
}

/// Mamba-3 Predictor.
///
/// Takes visual embeddings from the X-Encoder and a text query,
/// projects them into a shared space, processes through Mamba-3 layers,
/// and outputs a predicted target embedding.
pub struct Mamba3Predictor {
    pub config: Mamba3PredictorConfig,
    /// Projects ViT visual tokens to predictor d_model.
    vision_proj: Linear,
    /// Projects query token embeddings to predictor d_model.
    query_proj: Linear,
    /// Mamba-3 backbone (uses MixerModel internally but we bypass embedding).
    backbone_layers: MixerModel,
    /// Prediction head: d_model → embed_dim.
    pred_head: Linear,
}

impl Mamba3Predictor {
    pub fn new(config: Mamba3PredictorConfig) -> Self {
        let vision_proj = Linear::new(config.vision_dim, config.d_model);
        let query_proj = Linear::new(config.query_embed_dim, config.d_model);

        // Build MixerModel — we'll use a dummy vocab since we bypass embedding
        let mamba_config = config.to_mamba_config(2); // minimal vocab, unused
        let backbone_layers = MixerModel::new(mamba_config);

        let pred_head = Linear::new(config.d_model, config.embed_dim);

        Self { config, vision_proj, query_proj, backbone_layers, pred_head }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `visual_tokens`: shape (batch, n_vis, vision_dim) — from X-Encoder
    /// - `query_embeds`: shape (batch, n_qry, query_embed_dim) — embedded query tokens
    /// - `batch`: batch size
    /// - `n_vis`: number of visual tokens per sample
    /// - `n_qry`: number of query tokens per sample
    ///
    /// # Returns
    /// Predicted embedding: shape (batch, embed_dim), L2-normalized.
    pub fn forward(
        &self,
        visual_tokens: &[f32],
        query_embeds: &[f32],
        batch: usize,
        n_vis: usize,
        n_qry: usize,
    ) -> Vec<f32> {
        let dm = self.config.d_model;
        let seq_len = n_vis + n_qry;

        // 1) Project visual tokens: (batch * n_vis, vision_dim) → (batch * n_vis, d_model)
        let vis_proj = self.vision_proj.forward(visual_tokens, batch * n_vis);

        // 2) Project query embeddings: (batch * n_qry, query_embed_dim) → (batch * n_qry, d_model)
        let qry_proj = self.query_proj.forward(query_embeds, batch * n_qry);

        // 3) Concatenate: (batch, n_vis + n_qry, d_model)
        let mut concat = vec![0.0f32; batch * seq_len * dm];
        for b in 0..batch {
            // Visual tokens
            let vis_src = b * n_vis * dm;
            let dst = b * seq_len * dm;
            concat[dst..dst + n_vis * dm]
                .copy_from_slice(&vis_proj[vis_src..vis_src + n_vis * dm]);
            // Query tokens
            let qry_src = b * n_qry * dm;
            let dst_qry = dst + n_vis * dm;
            concat[dst_qry..dst_qry + n_qry * dm]
                .copy_from_slice(&qry_proj[qry_src..qry_src + n_qry * dm]);
        }

        // 4) Feed through Mamba-3 backbone layers (bypass embedding, use raw hidden states)
        let hidden = self.forward_backbone(&concat, batch, seq_len);

        // 5) Average pool over all positions → (batch, d_model)
        let mut pooled = vec![0.0f32; batch * dm];
        for b in 0..batch {
            for pos in 0..seq_len {
                for d in 0..dm {
                    pooled[b * dm + d] += hidden[b * seq_len * dm + pos * dm + d];
                }
            }
            let inv_len = 1.0 / seq_len as f32;
            for d in 0..dm {
                pooled[b * dm + d] *= inv_len;
            }
        }

        // 6) Prediction head → L2-normalize
        let projected = self.pred_head.forward(&pooled, batch);
        l2_normalize(&projected, batch, self.config.embed_dim)
    }

    /// Run the Mamba-3 backbone layers directly on hidden states (bypassing embedding).
    fn forward_backbone(&self, hidden: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let n = batch * seq_len;
        let mut h = hidden.to_vec();

        for (i, layer) in self.backbone_layers.layers.iter().enumerate() {
            // Pre-norm
            let normed = self.backbone_layers.norms[i].forward(&h, n);
            // Mixer
            let mixed = layer.forward_train(&normed, batch, seq_len);
            // Residual
            for j in 0..h.len() {
                h[j] += mixed[j];
            }
        }

        // Final norm
        self.backbone_layers.final_norm.forward(&h, n)
    }
}

/// L2-normalize each row.
fn l2_normalize(data: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let start = r * dim;
        let norm: f32 = data[start..start + dim]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        for d in 0..dim {
            out[start + d] = data[start + d] / norm;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_forward_shape() {
        let config = Mamba3PredictorConfig::tiny();
        let predictor = Mamba3Predictor::new(config.clone());

        let batch = 2;
        let n_vis = 4;
        let n_qry = 3;
        let vis = vec![0.1f32; batch * n_vis * config.vision_dim];
        let qry = vec![0.1f32; batch * n_qry * config.query_embed_dim];

        let output = predictor.forward(&vis, &qry, batch, n_vis, n_qry);
        assert_eq!(output.len(), batch * config.embed_dim);
    }

    #[test]
    fn test_predictor_output_normalized() {
        let config = Mamba3PredictorConfig::tiny();
        let predictor = Mamba3Predictor::new(config.clone());

        let batch = 1;
        let n_vis = 4;
        let n_qry = 2;
        let vis = vec![0.5f32; batch * n_vis * config.vision_dim];
        let qry = vec![0.3f32; batch * n_qry * config.query_embed_dim];

        let output = predictor.forward(&vis, &qry, batch, n_vis, n_qry);

        // Check L2 norm ≈ 1
        let norm: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "output should be L2-normalized, got norm={}", norm
        );
    }

    #[test]
    fn test_linear_forward() {
        let lin = Linear::new(4, 2);
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let output = lin.forward(&input, 1);
        assert_eq!(output.len(), 2);
    }
}
