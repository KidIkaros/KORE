//! Configuration structs for Mamba3-JEPA.

use serde::{Deserialize, Serialize};

/// Vision Transformer (X-Encoder) configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VitConfig {
    pub patch_size: usize,
    pub image_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub in_channels: usize,
}

impl VitConfig {
    /// V-JEPA 2 ViT-L preset (304M params, frozen).
    pub fn vjepa2_vit_l() -> Self {
        Self {
            patch_size: 14,
            image_size: 224,
            d_model: 1024,
            n_heads: 16,
            n_layers: 24,
            d_ff: 4096,
            in_channels: 3,
        }
    }

    /// Tiny preset for unit tests.
    pub fn tiny() -> Self {
        Self {
            patch_size: 4,
            image_size: 16,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            d_ff: 64,
            in_channels: 3,
        }
    }

    /// Number of patches per image.
    pub fn num_patches(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }
}

/// Mamba-3 Predictor configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mamba3PredictorConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub d_state: usize,
    pub expand: usize,
    pub headdim: usize,
    pub ngroups: usize,
    pub trapezoidal_alpha: f32,
    pub max_seq_len: usize,
    /// Output embedding dimension (shared space).
    pub embed_dim: usize,
    /// Input vision dimension (from ViT).
    pub vision_dim: usize,
    /// Query embedding dimension (from tokenizer).
    pub query_embed_dim: usize,
}

impl Mamba3PredictorConfig {
    /// Small preset (~80M params).
    pub fn small() -> Self {
        Self {
            d_model: 1024,
            n_layers: 12,
            d_state: 128,
            expand: 2,
            headdim: 64,
            ngroups: 1,
            trapezoidal_alpha: 0.5,
            max_seq_len: 2048,
            embed_dim: 1536,
            vision_dim: 1024,
            query_embed_dim: 1024,
        }
    }

    /// Tiny preset for unit tests.
    pub fn tiny() -> Self {
        Self {
            d_model: 64,
            n_layers: 2,
            d_state: 16,
            expand: 2,
            headdim: 16,
            ngroups: 1,
            trapezoidal_alpha: 0.5,
            max_seq_len: 128,
            embed_dim: 32,
            vision_dim: 32,
            query_embed_dim: 32,
        }
    }

    /// Build a MambaConfig for the predictor's MixerModel.
    pub fn to_mamba_config(&self, vocab_size: usize) -> kore_mamba::MambaConfig {
        kore_mamba::MambaConfig {
            d_model: self.d_model,
            n_layer: self.n_layers,
            d_intermediate: 0,
            vocab_size,
            ssm_cfg: kore_mamba::SsmConfig {
                layer: "Mamba3".into(),
                d_state: self.d_state,
                expand: self.expand,
                headdim: self.headdim,
                ngroups: self.ngroups,
                use_rope: true,
                trapezoidal_alpha: self.trapezoidal_alpha,
                ..Default::default()
            },
            rms_norm: true,
            residual_in_fp32: false,
            fused_add_norm: false,
            pad_vocab_size_multiple: 1,
            tie_embeddings: false,
            norm_epsilon: 1e-5,
        }
    }
}

/// Mamba-3 Text Encoder (Y-Encoder) configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mamba3TextEncoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub d_state: usize,
    pub expand: usize,
    pub headdim: usize,
    pub ngroups: usize,
    pub max_seq_len: usize,
    /// Output embedding dimension (shared space).
    pub embed_dim: usize,
}

impl Mamba3TextEncoderConfig {
    /// Small preset.
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 768,
            n_layers: 12,
            d_state: 64,
            expand: 2,
            headdim: 32,
            ngroups: 1,
            max_seq_len: 512,
            embed_dim: 1536,
        }
    }

    /// Tiny preset for unit tests.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            d_model: 64,
            n_layers: 2,
            d_state: 16,
            expand: 2,
            headdim: 16,
            ngroups: 1,
            max_seq_len: 64,
            embed_dim: 32,
        }
    }

    /// Build a MambaConfig for the encoder's MixerModel.
    pub fn to_mamba_config(&self) -> kore_mamba::MambaConfig {
        kore_mamba::MambaConfig {
            d_model: self.d_model,
            n_layer: self.n_layers,
            d_intermediate: 0,
            vocab_size: self.vocab_size,
            ssm_cfg: kore_mamba::SsmConfig {
                layer: "Mamba3".into(),
                d_state: self.d_state,
                expand: self.expand,
                headdim: self.headdim,
                ngroups: self.ngroups,
                use_rope: true,
                trapezoidal_alpha: 0.5,
                ..Default::default()
            },
            rms_norm: true,
            residual_in_fp32: false,
            fused_add_norm: false,
            pad_vocab_size_multiple: 1,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }
}

/// Mamba-3 Decoder (Y-Decoder) configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mamba3DecoderConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub d_state: usize,
    pub expand: usize,
    pub headdim: usize,
    pub vocab_size: usize,
    /// Number of prefix tokens generated from the predicted embedding.
    pub prefix_len: usize,
    /// Input embedding dimension (from predictor output).
    pub embed_dim: usize,
}

impl Mamba3DecoderConfig {
    /// Small preset.
    pub fn small() -> Self {
        Self {
            d_model: 512,
            n_layers: 6,
            d_state: 64,
            expand: 2,
            headdim: 32,
            vocab_size: 32000,
            prefix_len: 8,
            embed_dim: 1536,
        }
    }

    /// Tiny preset for unit tests.
    pub fn tiny() -> Self {
        Self {
            d_model: 64,
            n_layers: 2,
            d_state: 16,
            expand: 2,
            headdim: 16,
            vocab_size: 256,
            prefix_len: 4,
            embed_dim: 32,
        }
    }

    /// Build a MambaConfig for the decoder's MambaLMHeadModel.
    pub fn to_mamba_config(&self) -> kore_mamba::MambaConfig {
        kore_mamba::MambaConfig {
            d_model: self.d_model,
            n_layer: self.n_layers,
            d_intermediate: 0,
            vocab_size: self.vocab_size,
            ssm_cfg: kore_mamba::SsmConfig {
                layer: "Mamba3".into(),
                d_state: self.d_state,
                expand: self.expand,
                headdim: self.headdim,
                ngroups: 1,
                use_rope: true,
                trapezoidal_alpha: 0.5,
                ..Default::default()
            },
            rms_norm: true,
            residual_in_fp32: false,
            fused_add_norm: false,
            pad_vocab_size_multiple: 1,
            tie_embeddings: true,
            norm_epsilon: 1e-5,
        }
    }
}

/// Top-level Mamba3-JEPA configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mamba3JepaConfig {
    pub vit: VitConfig,
    pub predictor: Mamba3PredictorConfig,
    pub y_encoder: Mamba3TextEncoderConfig,
    pub y_decoder: Mamba3DecoderConfig,
    pub shared_embed_dim: usize,
}

impl Mamba3JepaConfig {
    /// Small preset (~150M trainable params).
    pub fn small() -> Self {
        Self {
            vit: VitConfig::vjepa2_vit_l(),
            predictor: Mamba3PredictorConfig::small(),
            y_encoder: Mamba3TextEncoderConfig::small(),
            y_decoder: Mamba3DecoderConfig::small(),
            shared_embed_dim: 1536,
        }
    }

    /// Tiny preset for unit tests.
    pub fn tiny() -> Self {
        Self {
            vit: VitConfig::tiny(),
            predictor: Mamba3PredictorConfig::tiny(),
            y_encoder: Mamba3TextEncoderConfig::tiny(),
            y_decoder: Mamba3DecoderConfig::tiny(),
            shared_embed_dim: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vit_num_patches() {
        let cfg = VitConfig::vjepa2_vit_l();
        assert_eq!(cfg.num_patches(), 256); // (224/14)^2
    }

    #[test]
    fn test_tiny_config() {
        let cfg = Mamba3JepaConfig::tiny();
        assert_eq!(cfg.shared_embed_dim, 32);
        assert_eq!(cfg.vit.d_model, 32);
        assert_eq!(cfg.predictor.d_model, 64);
    }

    #[test]
    fn test_predictor_to_mamba_config() {
        let cfg = Mamba3PredictorConfig::tiny();
        let mc = cfg.to_mamba_config(256);
        assert_eq!(mc.ssm_cfg.layer, "Mamba3");
        assert!(mc.ssm_cfg.use_rope);
    }
}
