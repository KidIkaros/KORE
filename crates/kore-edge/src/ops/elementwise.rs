//! Element-wise operations on raw f32 slices.

/// Residual add: output[i] = a[i] + b[i]
pub fn residual_add(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..output.len() {
        output[i] = a[i] + b[i];
    }
}

/// In-place residual add: a[i] += b[i]
pub fn residual_add_inplace(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Element-wise multiply: output[i] = a[i] * b[i]
pub fn mul(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..output.len() {
        output[i] = a[i] * b[i];
    }
}

/// Scalar multiply in-place: a[i] *= scalar
pub fn scale_inplace(a: &mut [f32], scalar: f32) {
    for v in a.iter_mut() {
        *v *= scalar;
    }
}

/// Copy slice
pub fn copy(src: &[f32], dst: &mut [f32]) {
    dst.copy_from_slice(src);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_add() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut c = [0.0f32; 3];
        residual_add(&a, &b, &mut c);
        assert_eq!(c, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_residual_add_inplace() {
        let mut a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        residual_add_inplace(&mut a, &b);
        assert_eq!(a, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_scale() {
        let mut a = [1.0, 2.0, 3.0];
        scale_inplace(&mut a, 2.0);
        assert_eq!(a, [2.0, 4.0, 6.0]);
    }
}
