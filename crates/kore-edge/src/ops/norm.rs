//! Normalization operators — RMSNorm and LayerNorm on raw slices.

/// RMS Normalization (in-place): x[i] = x[i] / rms(x) * gamma[i]
///
/// `data`: [batch, dim] row-major, modified in place
/// `gamma`: [dim] scale weights
pub fn rms_norm(data: &mut [f32], gamma: &[f32], dim: usize, eps: f32) {
    let batch = data.len() / dim;
    for b in 0..batch {
        let row = &mut data[b * dim..(b + 1) * dim];

        let sum_sq: f32 = row.iter().map(|x| x * x).sum();
        let rms = (sum_sq / dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..dim {
            row[i] = row[i] * inv_rms * gamma[i];
        }
    }
}

/// Layer Normalization (in-place): x[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
pub fn layer_norm(data: &mut [f32], gamma: &[f32], beta: &[f32], dim: usize, eps: f32) {
    let batch = data.len() / dim;
    for b in 0..batch {
        let row = &mut data[b * dim..(b + 1) * dim];

        let mean: f32 = row.iter().sum::<f32>() / dim as f32;
        let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..dim {
            row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let gamma = [1.0, 1.0, 1.0];
        let mut data = [1.0f32, 2.0, 3.0];
        rms_norm(&mut data, &gamma, 3, 1e-5);
        // RMS = sqrt((1+4+9)/3) ≈ 2.16
        assert!(data[0] > 0.0 && data[0] < 1.0);
        assert!(data[2] > 1.0);
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        layer_norm(&mut data, &gamma, &beta, 4, 1e-5);
        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_batched() {
        let gamma = [1.0; 3];
        let beta = [0.0; 3];
        let mut data = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        layer_norm(&mut data, &gamma, &beta, 3, 1e-5);
        let m1: f32 = data[0..3].iter().sum::<f32>() / 3.0;
        let m2: f32 = data[3..6].iter().sum::<f32>() / 3.0;
        assert!(m1.abs() < 1e-5);
        assert!(m2.abs() < 1e-5);
    }
}
