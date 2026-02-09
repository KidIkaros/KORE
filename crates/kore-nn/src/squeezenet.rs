//! SqueezeNet: compact CNN architecture using Fire modules.
//!
//! SqueezeNet achieves AlexNet-level accuracy with 50× fewer parameters
//! by using "Fire modules" that squeeze channels through 1×1 convolutions
//! before expanding with a mix of 1×1 and 3×3 convolutions.
//!
//! This implements SqueezeNet v1.1 which has 2.4× less computation than v1.0
//! while maintaining accuracy.
//!
//! # Architecture
//! ```text
//! Input → Conv2d(3→64, 3×3, stride=2) → MaxPool(3, stride=2)
//!       → Fire(64→128) → Fire(128→128) → MaxPool(3, stride=2)
//!       → Fire(128→256) → Fire(256→256) → MaxPool(3, stride=2)
//!       → Fire(256→384) → Fire(384→384) → Fire(384→512) → Fire(512→512)
//!       → Dropout → Conv2d(512→num_classes, 1×1) → AdaptiveAvgPool(1,1)
//! ```

use std::collections::HashMap;
use std::path::Path;

use kore_core::{KoreError, Tensor};
use crate::conv::Conv2d;
use crate::pool::{MaxPool2d, AdaptiveAvgPool2d};
use crate::activations::relu;
use crate::module::Module;

/// Fire module: the core building block of SqueezeNet.
///
/// Architecture:
/// ```text
/// input → squeeze(1×1) → ReLU → [expand1×1, expand3×3] → cat → ReLU
/// ```
///
/// The squeeze layer reduces channels, then two parallel expand layers
/// (1×1 and 3×3) increase them back. The outputs are concatenated
/// along the channel dimension.
pub struct Fire {
    /// Squeeze: reduce channels with 1×1 conv.
    squeeze: Conv2d,
    /// Expand path 1: 1×1 conv.
    expand1x1: Conv2d,
    /// Expand path 2: 3×3 conv with padding=1 to preserve spatial dims.
    expand3x3: Conv2d,
    /// Number of output channels (expand1x1_planes + expand3x3_planes).
    out_channels: usize,
}

impl Fire {
    /// Create a new Fire module.
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `squeeze_planes` - Channels after squeeze (bottleneck)
    /// * `expand1x1_planes` - Channels from 1×1 expand path
    /// * `expand3x3_planes` - Channels from 3×3 expand path
    pub fn new(
        in_channels: usize,
        squeeze_planes: usize,
        expand1x1_planes: usize,
        expand3x3_planes: usize,
    ) -> Self {
        Self {
            squeeze: Conv2d::new(in_channels, squeeze_planes, 1, 1, 0, true),
            expand1x1: Conv2d::new(squeeze_planes, expand1x1_planes, 1, 1, 0, true),
            expand3x3: Conv2d::new(squeeze_planes, expand3x3_planes, 3, 1, 1, true),
            out_channels: expand1x1_planes + expand3x3_planes,
        }
    }

    /// Output channels of this Fire module.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Load a Fire module from a state dict with a given key prefix.
    ///
    /// Expected keys (torchvision format):
    /// - `{prefix}.squeeze.weight`, `{prefix}.squeeze.bias`
    /// - `{prefix}.expand1x1.weight`, `{prefix}.expand1x1.bias`
    /// - `{prefix}.expand3x3.weight`, `{prefix}.expand3x3.bias`
    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str) -> Result<Self, KoreError> {
        let get = |key: &str| -> Result<Tensor, KoreError> {
            sd.get(key).cloned().ok_or_else(|| {
                KoreError::StorageError(format!("missing key: {}", key))
            })
        };
        let get_opt = |key: &str| -> Option<Tensor> { sd.get(key).cloned() };

        let sq_w = get(&format!("{}.squeeze.weight", prefix))?;
        let sq_b = get_opt(&format!("{}.squeeze.bias", prefix));
        let e1_w = get(&format!("{}.expand1x1.weight", prefix))?;
        let e1_b = get_opt(&format!("{}.expand1x1.bias", prefix));
        let e3_w = get(&format!("{}.expand3x3.weight", prefix))?;
        let e3_b = get_opt(&format!("{}.expand3x3.bias", prefix));

        let expand1x1_planes = e1_w.shape().dims()[0];
        let expand3x3_planes = e3_w.shape().dims()[0];

        Ok(Self {
            squeeze: Conv2d::from_weight(sq_w, sq_b, 1, 0),
            expand1x1: Conv2d::from_weight(e1_w, e1_b, 1, 0),
            expand3x3: Conv2d::from_weight(e3_w, e3_b, 1, 1),
            out_channels: expand1x1_planes + expand3x3_planes,
        })
    }
}

impl Module for Fire {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let squeezed = relu(&self.squeeze.forward(input)?)?;
        let e1 = relu(&self.expand1x1.forward(&squeezed)?)?;
        let e3 = relu(&self.expand3x3.forward(&squeezed)?)?;
        // Concatenate along channel dimension (axis=1)
        Tensor::cat(&[&e1, &e3], 1)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.squeeze.parameters();
        params.extend(self.expand1x1.parameters());
        params.extend(self.expand3x3.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor)> {
        let mut params = Vec::new();
        for (_name, t) in self.squeeze.named_parameters() {
            params.push(("squeeze.weight", t));
        }
        for (_name, t) in self.expand1x1.named_parameters() {
            params.push(("expand1x1.weight", t));
        }
        for (_name, t) in self.expand3x3.named_parameters() {
            params.push(("expand3x3.weight", t));
        }
        params
    }
}

/// SqueezeNet v1.1 — compact CNN for image classification.
///
/// ~1.2M parameters for 1000-class ImageNet classification.
/// Uses Fire modules to achieve high accuracy with minimal parameters.
pub struct SqueezeNet {
    // Feature extractor
    conv1: Conv2d,
    pool1: MaxPool2d,
    fire2: Fire,
    fire3: Fire,
    pool3: MaxPool2d,
    fire4: Fire,
    fire5: Fire,
    pool5: MaxPool2d,
    fire6: Fire,
    fire7: Fire,
    fire8: Fire,
    fire9: Fire,
    // Classifier
    classifier_conv: Conv2d,
    adaptive_pool: AdaptiveAvgPool2d,
    num_classes: usize,
    training: bool,
}

impl SqueezeNet {
    /// Create SqueezeNet v1.1.
    ///
    /// # Arguments
    /// * `num_classes` - Number of output classes (default: 1000 for ImageNet)
    pub fn v1_1(num_classes: usize) -> Self {
        Self {
            // Features
            conv1: Conv2d::new(3, 64, 3, 2, 0, true),
            pool1: MaxPool2d::new(3, 2, 0),
            fire2: Fire::new(64, 16, 64, 64),
            fire3: Fire::new(128, 16, 64, 64),
            pool3: MaxPool2d::new(3, 2, 0),
            fire4: Fire::new(128, 32, 128, 128),
            fire5: Fire::new(256, 32, 128, 128),
            pool5: MaxPool2d::new(3, 2, 0),
            fire6: Fire::new(256, 48, 192, 192),
            fire7: Fire::new(384, 48, 192, 192),
            fire8: Fire::new(384, 64, 256, 256),
            fire9: Fire::new(512, 64, 256, 256),
            // Classifier
            classifier_conv: Conv2d::new(512, num_classes, 1, 1, 0, true),
            adaptive_pool: AdaptiveAvgPool2d::global(),
            num_classes,
            training: true,
        }
    }

    /// Load SqueezeNet v1.1 from a safetensors file (torchvision key format).
    ///
    /// Torchvision SqueezeNet v1.1 key mapping:
    /// - `features.0.weight/bias` → conv1
    /// - `features.3` → fire2, `features.4` → fire3
    /// - `features.6` → fire4, `features.7` → fire5
    /// - `features.9` → fire6, `features.10` → fire7
    /// - `features.11` → fire8, `features.12` → fire9
    /// - `classifier.1.weight/bias` → classifier_conv
    pub fn load_safetensors(path: &Path, num_classes: usize) -> Result<Self, KoreError> {
        let sd = crate::serialization::load_state_dict(path)?;
        Self::from_state_dict(&sd, num_classes)
    }

    /// Load SqueezeNet v1.1 from a pre-loaded state dict (torchvision key format).
    pub fn from_state_dict(sd: &HashMap<String, Tensor>, num_classes: usize) -> Result<Self, KoreError> {
        let get = |key: &str| -> Result<Tensor, KoreError> {
            sd.get(key).cloned().ok_or_else(|| {
                KoreError::StorageError(format!("missing key: {}", key))
            })
        };
        let get_opt = |key: &str| -> Option<Tensor> { sd.get(key).cloned() };

        let conv1 = Conv2d::from_weight(
            get("features.0.weight")?,
            get_opt("features.0.bias"),
            2, 0,
        );

        // Fire modules: torchvision feature indices
        let fire2 = Fire::from_state_dict(sd, "features.3")?;
        let fire3 = Fire::from_state_dict(sd, "features.4")?;
        let fire4 = Fire::from_state_dict(sd, "features.6")?;
        let fire5 = Fire::from_state_dict(sd, "features.7")?;
        let fire6 = Fire::from_state_dict(sd, "features.9")?;
        let fire7 = Fire::from_state_dict(sd, "features.10")?;
        let fire8 = Fire::from_state_dict(sd, "features.11")?;
        let fire9 = Fire::from_state_dict(sd, "features.12")?;

        let classifier_conv = Conv2d::from_weight(
            get("classifier.1.weight")?,
            get_opt("classifier.1.bias"),
            1, 0,
        );

        Ok(Self {
            conv1,
            pool1: MaxPool2d::new(3, 2, 0),
            fire2,
            fire3,
            pool3: MaxPool2d::new(3, 2, 0),
            fire4,
            fire5,
            pool5: MaxPool2d::new(3, 2, 0),
            fire6,
            fire7,
            fire8,
            fire9,
            classifier_conv,
            adaptive_pool: AdaptiveAvgPool2d::global(),
            num_classes,
            training: false,
        })
    }

    /// Number of output classes.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Count total trainable parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter()
            .map(|t| t.numel())
            .sum()
    }

    /// Model info string.
    pub fn info(&self) -> String {
        format!(
            "SqueezeNet v1.1 | classes={} | params={:.1}K",
            self.num_classes,
            self.num_parameters() as f32 / 1000.0,
        )
    }
}

impl Module for SqueezeNet {
    /// Forward pass.
    ///
    /// Input: `[batch, 3, H, W]` (e.g. `[1, 3, 224, 224]`)
    /// Output: `[batch, num_classes]`
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 4 || dims[1] != 3 {
            return Err(KoreError::ShapeMismatch {
                expected: vec![0, 3, 0, 0],
                got: dims.to_vec(),
            });
        }

        // Feature extraction
        let x = relu(&self.conv1.forward(input)?)?;
        let x = self.pool1.forward(&x)?;
        let x = self.fire2.forward(&x)?;
        let x = self.fire3.forward(&x)?;
        let x = self.pool3.forward(&x)?;
        let x = self.fire4.forward(&x)?;
        let x = self.fire5.forward(&x)?;
        let x = self.pool5.forward(&x)?;
        let x = self.fire6.forward(&x)?;
        let x = self.fire7.forward(&x)?;
        let x = self.fire8.forward(&x)?;
        let x = self.fire9.forward(&x)?;

        // Classifier
        let x = relu(&self.classifier_conv.forward(&x)?)?;
        let x = self.adaptive_pool.forward(&x)?;

        // Flatten: [batch, num_classes, 1, 1] → [batch, num_classes]
        let batch = x.shape().dims()[0];
        x.reshape(&[batch as isize, self.num_classes as isize])
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.conv1.parameters();
        params.extend(self.fire2.parameters());
        params.extend(self.fire3.parameters());
        params.extend(self.fire4.parameters());
        params.extend(self.fire5.parameters());
        params.extend(self.fire6.parameters());
        params.extend(self.fire7.parameters());
        params.extend(self.fire8.parameters());
        params.extend(self.fire9.parameters());
        params.extend(self.classifier_conv.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor)> {
        // Simplified: return all parameters with generic names
        self.parameters().into_iter()
            .enumerate()
            .map(|(i, t)| {
                // Leak a string for the name — acceptable for debugging
                let name: &str = Box::leak(format!("param_{}", i).into_boxed_str());
                (name, t)
            })
            .collect()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fire_module_shape() {
        let fire = Fire::new(64, 16, 64, 64);
        assert_eq!(fire.out_channels(), 128);

        // [1, 64, 8, 8] → [1, 128, 8, 8]
        let input = Tensor::from_f32(&vec![0.1; 1 * 64 * 8 * 8], &[1, 64, 8, 8]);
        let output = fire.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 128, 8, 8]);
    }

    #[test]
    fn test_fire_module_parameters() {
        let fire = Fire::new(64, 16, 64, 64);
        let params = fire.parameters();
        // squeeze: weight + bias = 2
        // expand1x1: weight + bias = 2
        // expand3x3: weight + bias = 2
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_fire_module_batch() {
        let fire = Fire::new(32, 8, 32, 32);
        let input = Tensor::from_f32(&vec![0.1; 2 * 32 * 4 * 4], &[2, 32, 4, 4]);
        let output = fire.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 64, 4, 4]);
    }

    #[test]
    fn test_fire_module_finite_output() {
        let fire = Fire::new(16, 4, 8, 8);
        let input = Tensor::from_f32(&vec![0.5; 1 * 16 * 4 * 4], &[1, 16, 4, 4]);
        let output = fire.forward(&input).unwrap();
        let data = output.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn test_squeezenet_construction() {
        let model = SqueezeNet::v1_1(10);
        assert_eq!(model.num_classes(), 10);
        let params = model.parameters();
        assert!(!params.is_empty());
        let num_params = model.num_parameters();
        assert!(num_params > 0, "model has no parameters");
    }

    #[test]
    fn test_squeezenet_forward_small() {
        // Use small spatial dims to keep test fast
        // Input must be large enough to survive all the pooling layers
        // conv1(3→64, k=3, s=2): 32→15, pool1(k=3,s=2): 15→7
        // pool3(k=3,s=2): 7→3, pool5(k=3,s=2): 3→1
        let model = SqueezeNet::v1_1(10);
        let input = Tensor::from_f32(&vec![0.1; 1 * 3 * 32 * 32], &[1, 3, 32, 32]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 10]);
        let data = output.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_squeezenet_info() {
        let model = SqueezeNet::v1_1(1000);
        let info = model.info();
        assert!(info.contains("SqueezeNet"));
        assert!(info.contains("1000"));
    }

    #[test]
    fn test_squeezenet_parameter_count() {
        let model = SqueezeNet::v1_1(1000);
        let num_params = model.num_parameters();
        // SqueezeNet v1.1 should have roughly 1.2M params
        // Our random init may differ slightly but should be in the right ballpark
        assert!(num_params > 500_000, "too few params: {}", num_params);
        assert!(num_params < 5_000_000, "too many params: {}", num_params);
    }

    /// Helper: create a random tensor of given shape.
    fn rand_tensor(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) % 1.0).collect();
        Tensor::from_f32(&data, shape)
    }

    /// Build a synthetic state dict matching torchvision SqueezeNet v1.1 keys.
    fn build_squeezenet_state_dict() -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();

        // conv1: 3→64, k=3, s=2
        sd.insert("features.0.weight".into(), rand_tensor(&[64, 3, 3, 3]));
        sd.insert("features.0.bias".into(), rand_tensor(&[64]));

        // Fire modules: (prefix, in_ch, sq, e1, e3)
        let fires = [
            ("features.3", 64, 16, 64, 64),
            ("features.4", 128, 16, 64, 64),
            ("features.6", 128, 32, 128, 128),
            ("features.7", 256, 32, 128, 128),
            ("features.9", 256, 48, 192, 192),
            ("features.10", 384, 48, 192, 192),
            ("features.11", 384, 64, 256, 256),
            ("features.12", 512, 64, 256, 256),
        ];
        for (prefix, in_ch, sq, e1, e3) in &fires {
            sd.insert(format!("{}.squeeze.weight", prefix), rand_tensor(&[*sq, *in_ch, 1, 1]));
            sd.insert(format!("{}.squeeze.bias", prefix), rand_tensor(&[*sq]));
            sd.insert(format!("{}.expand1x1.weight", prefix), rand_tensor(&[*e1, *sq, 1, 1]));
            sd.insert(format!("{}.expand1x1.bias", prefix), rand_tensor(&[*e1]));
            sd.insert(format!("{}.expand3x3.weight", prefix), rand_tensor(&[*e3, *sq, 3, 3]));
            sd.insert(format!("{}.expand3x3.bias", prefix), rand_tensor(&[*e3]));
        }

        // classifier: 512→1000, k=1
        sd.insert("classifier.1.weight".into(), rand_tensor(&[1000, 512, 1, 1]));
        sd.insert("classifier.1.bias".into(), rand_tensor(&[1000]));

        sd
    }

    #[test]
    fn test_squeezenet_from_state_dict() {
        let sd = build_squeezenet_state_dict();
        let model = SqueezeNet::from_state_dict(&sd, 1000).unwrap();
        assert_eq!(model.num_classes(), 1000);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_squeezenet_from_state_dict_forward() {
        let sd = build_squeezenet_state_dict();
        let model = SqueezeNet::from_state_dict(&sd, 1000).unwrap();
        let input = Tensor::from_f32(&vec![0.1; 1 * 3 * 32 * 32], &[1, 3, 32, 32]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1000]);
        let data = output.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_squeezenet_from_state_dict_missing_key() {
        let mut sd = build_squeezenet_state_dict();
        sd.remove("features.3.squeeze.weight");
        let result = SqueezeNet::from_state_dict(&sd, 1000);
        assert!(result.is_err());
    }
}
