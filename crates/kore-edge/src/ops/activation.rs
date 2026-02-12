//! Activation functions on raw f32 slices (in-place).

/// ReLU: x = max(0, x)
pub fn relu(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = v.max(0.0);
    }
}

/// GELU (tanh approximation): x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(data: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    for v in data.iter_mut() {
        let x = *v;
        let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
        *v = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// SiLU (Swish): x = x * sigmoid(x)
pub fn silu(data: &mut [f32]) {
    for v in data.iter_mut() {
        let x = *v;
        *v = x / (1.0 + (-x).exp());
    }
}

/// Sigmoid: x = 1 / (1 + exp(-x))
pub fn sigmoid(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Softmax over rows of shape [batch, dim] (in-place, numerically stable).
pub fn softmax(data: &mut [f32], dim: usize) {
    let batch = data.len() / dim;
    for b in 0..batch {
        let row = &mut data[b * dim..(b + 1) * dim];

        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }

        let inv_sum = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let mut data = [-1.0, 0.0, 1.0, -0.5, 2.0];
        relu(&mut data);
        assert_eq!(data, [0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_gelu() {
        let mut data = [0.0, 1.0, -1.0];
        gelu(&mut data);
        assert!((data[0]).abs() < 1e-5); // gelu(0) = 0
        assert!(data[1] > 0.8); // gelu(1) ≈ 0.841
        assert!(data[2] < 0.0); // gelu(-1) ≈ -0.159
    }

    #[test]
    fn test_silu() {
        let mut data = [0.0, 1.0, -1.0];
        silu(&mut data);
        assert!((data[0]).abs() < 1e-5); // silu(0) = 0
        assert!(data[1] > 0.7); // silu(1) ≈ 0.731
    }

    #[test]
    fn test_sigmoid() {
        let mut data = [0.0, 100.0, -100.0];
        sigmoid(&mut data);
        assert!((data[0] - 0.5).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
        assert!(data[2].abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let mut data = [1.0, 2.0, 3.0];
        softmax(&mut data, 3);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_softmax_stability() {
        let mut data = [1000.0, 1001.0, 1002.0];
        softmax(&mut data, 3);
        assert!(data.iter().all(|v| v.is_finite()));
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
