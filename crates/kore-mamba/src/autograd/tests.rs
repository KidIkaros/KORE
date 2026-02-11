//! Finite-difference gradient checks for MambaScanBackward.

#[cfg(test)]
mod scan_grad_tests {
    use std::sync::Arc;
    use kore_core::autograd::GradNode;
    use kore_core::Tensor;
    use crate::autograd::scan_forward_saved::mamba3_scan_with_grad;

    /// Run forward scan (no grad) and return sum of output.
    fn forward_sum(
        x: &[f32], batch: usize, seq_len: usize, nheads: usize, headdim: usize,
        dt: &[f32], a_real: &[f32], a_imag: &[f32],
        b: &[f32], ngroups: usize, d_state: usize, c: &[f32],
        d: Option<&[f32]>, z: Option<&[f32]>, alpha: f32,
    ) -> f32 {
        let r = crate::ssd3::mamba3_scan_combined(
            x, batch, seq_len, nheads, headdim,
            dt, a_real, a_imag, b, ngroups, d_state, c,
            None, None, d, z, None, false, alpha, false,
        );
        r.output.iter().sum()
    }

    /// Numerical gradient via central finite differences.
    fn numerical_grad(
        f: &dyn Fn(&[f32]) -> f32, param: &[f32], eps: f32,
    ) -> Vec<f32> {
        let mut buf = param.to_vec();
        let mut grad = vec![0.0f32; param.len()];
        for i in 0..param.len() {
            let orig = buf[i];
            buf[i] = orig + eps;
            let fp = f(&buf);
            buf[i] = orig - eps;
            let fm = f(&buf);
            buf[i] = orig;
            grad[i] = (fp - fm) / (2.0 * eps);
        }
        grad
    }

    fn check_close(analytic: &[f32], numerical: &[f32], tol: f32, name: &str) {
        assert_eq!(analytic.len(), numerical.len(), "{name}: length mismatch");
        for i in 0..analytic.len() {
            let abs_err = (analytic[i] - numerical[i]).abs();
            // Use absolute tolerance for near-zero values
            if abs_err < 1e-4 { continue; }
            let scale = numerical[i].abs().max(analytic[i].abs()).max(1e-7);
            let rel = abs_err / scale;
            assert!(
                rel < tol,
                "{name}[{i}]: analytic={:.6}, numerical={:.6}, rel_err={:.6}",
                analytic[i], numerical[i], rel
            );
        }
    }

    fn run_backward(
        x: &[f32], dt: &[f32], a_real: &[f32], a_imag: &[f32],
        b: &[f32], c: &[f32], d: Option<&[f32]>, z: Option<&[f32]>,
        batch: usize, seq_len: usize, nheads: usize, headdim: usize,
        ngroups: usize, d_state: usize, alpha: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
        let x_n = GradNode::leaf();
        let dt_n = GradNode::leaf();
        let b_n = GradNode::leaf();
        let c_n = GradNode::leaf();

        let mut nodes: Vec<Arc<GradNode>> = vec![
            Arc::clone(&x_n), Arc::clone(&dt_n),
            Arc::clone(&b_n), Arc::clone(&c_n),
        ];
        let z_n = if z.is_some() {
            let n = GradNode::leaf();
            nodes.push(Arc::clone(&n));
            Some(n)
        } else {
            None
        };

        let (out_t, _) = mamba3_scan_with_grad(
            x, batch, seq_len, nheads, headdim,
            dt, a_real, a_imag, b, ngroups, d_state, c,
            None, None, d, z, None, false, alpha, false, nodes,
        );

        let go = Tensor::from_f32(
            &vec![1.0f32; batch * seq_len * nheads * headdim],
            &[batch, seq_len, nheads * headdim],
        );
        kore_core::autograd::backward(out_t.grad_node().unwrap(), go);

        let dx = x_n.get_grad().unwrap().as_f32_slice().unwrap().to_vec();
        let ddt = dt_n.get_grad().unwrap().as_f32_slice().unwrap().to_vec();
        let db = b_n.get_grad().unwrap().as_f32_slice().unwrap().to_vec();
        let dc = c_n.get_grad().unwrap().as_f32_slice().unwrap().to_vec();
        let dz_out = z_n.map(|n| n.get_grad().unwrap().as_f32_slice().unwrap().to_vec());

        (dx, ddt, db, dc, dz_out)
    }

    // Test dimensions
    const B: usize = 1;
    const L: usize = 4;
    const H: usize = 2;
    const D: usize = 2;
    const NG: usize = 1;
    const NS: usize = 2;
    const ALPHA: f32 = 0.5;

    fn test_params() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..B*L*H*D).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let dt = vec![0.1f32; B * L * H];
        let a_real = vec![-0.5f32; H];
        let a_imag = vec![0.0f32; H];
        let b: Vec<f32> = (0..B*L*NG*NS).map(|i| 0.2 * ((i % 3) as f32 + 1.0)).collect();
        let c: Vec<f32> = (0..B*L*NG*NS).map(|i| 0.15 * ((i % 3) as f32 + 1.0)).collect();
        (x, dt, a_real, a_imag, b, c)
    }

    #[test]
    fn test_dx_finite_diff() {
        let (x, dt, ar, ai, b, c) = test_params();
        let (dx, _, _, _, _) = run_backward(
            &x, &dt, &ar, &ai, &b, &c, None, None,
            B, L, H, D, NG, NS, ALPHA,
        );
        let ng = numerical_grad(
            &|xp| forward_sum(xp, B, L, H, D, &dt, &ar, &ai, &b, NG, NS, &c, None, None, ALPHA),
            &x, 1e-3,
        );
        check_close(&dx, &ng, 5e-2, "dx");
    }

    #[test]
    fn test_db_finite_diff() {
        let (x, dt, ar, ai, b, c) = test_params();
        let (_, _, db, _, _) = run_backward(
            &x, &dt, &ar, &ai, &b, &c, None, None,
            B, L, H, D, NG, NS, ALPHA,
        );
        let ng = numerical_grad(
            &|bp| forward_sum(&x, B, L, H, D, &dt, &ar, &ai, bp, NG, NS, &c, None, None, ALPHA),
            &b, 1e-3,
        );
        check_close(&db, &ng, 5e-2, "dB");
    }

    #[test]
    fn test_dc_finite_diff() {
        let (x, dt, ar, ai, b, c) = test_params();
        let (_, _, _, dc, _) = run_backward(
            &x, &dt, &ar, &ai, &b, &c, None, None,
            B, L, H, D, NG, NS, ALPHA,
        );
        let ng = numerical_grad(
            &|cp| forward_sum(&x, B, L, H, D, &dt, &ar, &ai, &b, NG, NS, cp, None, None, ALPHA),
            &c, 1e-3,
        );
        check_close(&dc, &ng, 5e-2, "dC");
    }

    #[test]
    fn test_ddt_finite_diff() {
        let (x, dt, ar, ai, b, c) = test_params();
        let (_, ddt, _, _, _) = run_backward(
            &x, &dt, &ar, &ai, &b, &c, None, None,
            B, L, H, D, NG, NS, ALPHA,
        );
        let ng = numerical_grad(
            &|dtp| forward_sum(&x, B, L, H, D, dtp, &ar, &ai, &b, NG, NS, &c, None, None, ALPHA),
            &dt, 1e-3,
        );
        check_close(&ddt, &ng, 5e-2, "d_dt");
    }

    #[test]
    fn test_dz_finite_diff() {
        let (x, dt, ar, ai, b, c) = test_params();
        let z: Vec<f32> = (0..B*L*H*D).map(|i| 0.3 * (i as f32 - 4.0)).collect();
        let (_, _, _, _, dz_opt) = run_backward(
            &x, &dt, &ar, &ai, &b, &c, None, Some(&z),
            B, L, H, D, NG, NS, ALPHA,
        );
        let dz = dz_opt.unwrap();
        let ng = numerical_grad(
            &|zp| forward_sum(&x, B, L, H, D, &dt, &ar, &ai, &b, NG, NS, &c, None, Some(zp), ALPHA),
            &z, 1e-3,
        );
        check_close(&dz, &ng, 5e-2, "dz");
    }

    #[test]
    fn test_backward_with_d_skip() {
        let (x, dt, ar, ai, b, c) = test_params();
        let d_skip = vec![1.5f32; H];
        let (dx, _, _, _, _) = run_backward(
            &x, &dt, &ar, &ai, &b, &c, Some(&d_skip), None,
            B, L, H, D, NG, NS, ALPHA,
        );
        let ng = numerical_grad(
            &|xp| forward_sum(xp, B, L, H, D, &dt, &ar, &ai, &b, NG, NS, &c, Some(&d_skip), None, ALPHA),
            &x, 1e-3,
        );
        check_close(&dx, &ng, 5e-2, "dx_with_D");
    }
}
