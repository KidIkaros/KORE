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

    /// GPU-vs-CPU backward comparison test.
    /// Runs both the GPU and CPU backward paths on identical inputs and asserts
    /// element-wise closeness of all gradient outputs.
    /// Only compiled and run when the `cuda` feature is enabled.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_vs_cpu_backward() {
        use kore_kernels::cuda::context::{get_device, is_cuda_available};
        use kore_kernels::cuda::memory::CudaBuffer;
        use kore_kernels::cuda::ops::cuda_mamba3_scan_f32;
        use crate::autograd::scan_forward_saved::GpuSavedContext;

        if !is_cuda_available() {
            eprintln!("CUDA not available, skipping GPU-vs-CPU backward test");
            return;
        }

        let (x, dt, ar, ai, b, c) = test_params();
        let z: Vec<f32> = (0..B*L*H*D).map(|i| 0.3 * (i as f32 - 4.0)).collect();

        // --- CPU backward ---
        let (cpu_dx, cpu_ddt, cpu_db, cpu_dc, cpu_dz) = run_backward(
            &x, &dt, &ar, &ai, &b, &c, None, Some(&z),
            B, L, H, D, NG, NS, ALPHA,
        );

        // --- GPU backward ---
        // Run forward on GPU with save_states=true, then backward
        let dev_idx = 0;
        let dev = get_device(dev_idx).expect("get CUDA device");

        let cast = |data: &[f32]| -> Vec<u8> {
            unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * 4,
                ).to_vec()
            }
        };

        let x_gpu = dev.htod_copy(cast(&x)).expect("upload x");
        let dt_gpu = dev.htod_copy(cast(&dt)).expect("upload dt");
        let a_gpu = dev.htod_copy(cast(&ar)).expect("upload a_real");
        let ai_gpu = dev.htod_copy(cast(&ai)).expect("upload a_imag");
        let b_gpu = dev.htod_copy(cast(&b)).expect("upload b");
        let c_gpu = dev.htod_copy(cast(&c)).expect("upload c");
        let z_gpu = dev.htod_copy(cast(&z)).expect("upload z");

        let fwd = cuda_mamba3_scan_f32(
            &dev, dev_idx,
            &x_gpu, &dt_gpu, &a_gpu, &ai_gpu, &b_gpu, &c_gpu,
            None, Some(&z_gpu), None,
            B, L, H, D, NG, NS, ALPHA, false, false,
            true, // save_states
        ).expect("GPU forward");

        let h_all_gpu = fwd.h_all.expect("h_all should be Some when save_states=true");
        let bx_all_gpu = fwd.bx_all.expect("bx_all should be Some when save_states=true");

        // Build GpuSavedContext and run backward
        let gpu_ctx = GpuSavedContext {
            dev: dev.clone(),
            dev_idx,
            x: x_gpu,
            dt: dt_gpu,
            a_real: a_gpu,
            a_imag: ai_gpu,
            b: b_gpu,
            c: c_gpu,
            dt_bias: None,
            z: Some(z_gpu),
            d_skip: None,
            h_all: h_all_gpu,
            bx_all: bx_all_gpu,
        };

        // Create MambaScanSaved with GPU context
        let saved = crate::autograd::scan_forward_saved::MambaScanSaved {
            x: x.clone(), dt: dt.clone(), a_real: ar.clone(), a_imag: ai.clone(),
            b: b.clone(), c: c.clone(),
            b_bias: None, c_bias: None, d: None,
            z: Some(z.clone()), dt_bias: None,
            batch: B, seq_len: L, nheads: H, headdim: D,
            ngroups: NG, d_state: NS,
            dt_softplus: false, alpha: ALPHA, use_rope: false,
            h_all: vec![], bx_all: vec![], // empty — GPU path uses gpu_ctx
            gpu_ctx: Some(gpu_ctx),
        };

        let backward = crate::autograd::scan_backward::MambaScanBackward { saved };
        let go = Tensor::from_f32(
            &vec![1.0f32; B * L * H * D],
            &[B, L, H * D],
        );
        let grads = <crate::autograd::scan_backward::MambaScanBackward as kore_core::autograd::GradFn>::apply(&backward, &go);

        let gpu_dx = grads[0].as_ref().unwrap().as_f32_slice().unwrap().to_vec();
        let gpu_ddt = grads[1].as_ref().unwrap().as_f32_slice().unwrap().to_vec();
        let gpu_db = grads[2].as_ref().unwrap().as_f32_slice().unwrap().to_vec();
        let gpu_dc = grads[3].as_ref().unwrap().as_f32_slice().unwrap().to_vec();
        let gpu_dz = grads[4].as_ref().unwrap().as_f32_slice().unwrap().to_vec();

        // Compare GPU vs CPU
        check_close(&gpu_dx, &cpu_dx, 1e-3, "gpu_vs_cpu_dx");
        check_close(&gpu_ddt, &cpu_ddt, 1e-3, "gpu_vs_cpu_ddt");
        check_close(&gpu_db, &cpu_db, 1e-3, "gpu_vs_cpu_db");
        check_close(&gpu_dc, &cpu_dc, 1e-3, "gpu_vs_cpu_dc");
        check_close(&gpu_dz, &cpu_dz.unwrap(), 1e-3, "gpu_vs_cpu_dz");
    }
}
