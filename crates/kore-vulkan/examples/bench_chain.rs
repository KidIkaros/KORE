//! Per-dispatch latency probe for the KORE decode chain.
//!
//! Decode is a strictly serial dependency chain, so the metric that matters is
//! the *dependent* latency of each tiny op, not its throughput. This measures
//! that directly: N chained dispatches of one op, end to end, divided by N.
//!
//! `indep` rows run the same number of dispatches but with no data dependency
//! (all reading the same input, writing distinct buffers). The gap between
//! `dep` and `indep` is the exposed latency that only *fusion* can remove.

use kore_core::{Device, Tensor};
use kore_vulkan::VulkanBackend;
use std::time::Instant;

fn f32s(v: f32, n: usize) -> Vec<f32> {
    vec![v; n]
}

struct Probe<'a> {
    b: &'a VulkanBackend,
    dev: Device,
}

impl<'a> Probe<'a> {
    fn t(&self, v: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_f32(v, shape).to(self.dev).unwrap()
    }

    fn time(&self, label: &str, n: usize, mut dep: impl FnMut(&Tensor) -> Tensor, x: &Tensor) {
        let reps = 64usize;

        // ---- dependent chain: out_{i+1} = f(out_i) ----
        for _ in 0..3 {
            let mut cur = x.clone();
            for _ in 0..reps {
                cur = dep(&cur);
            }
            let _ = cur.cpu().unwrap();
        }
        let t0 = Instant::now();
        let mut cur = x.clone();
        for _ in 0..reps {
            cur = dep(&cur);
        }
        let _ = cur.cpu().unwrap();
        let dep_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

        // ---- independent: same dispatch count, no data dependency ----
        let mut keep: Vec<Tensor> = Vec::with_capacity(n);
        let t0 = Instant::now();
        for _ in 0..reps {
            keep.push(dep(x));
        }
        let _ = keep.last().unwrap().cpu().unwrap();
        let indep_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

        println!(
            "{label:<28} dep {dep_us:7.2} us   indep {indep_us:7.2} us   (exposed {:.2} us)",
            dep_us - indep_us
        );

        drop(keep);
    }
}

fn main() {
    let b = VulkanBackend::new().expect("vulkan");
    let dev = Device::Vulkan(0);
    let p = Probe { b: &b, dev };
    let eps = 1e-6f32;

    for &h in &[256usize, 2048, 6144] {
        println!("--- vector width {h} ---");
        let x = p.t(&f32s(0.5, h), &[1, h]);
        let w = p.t(&f32s(1.0, h), &[h]);
        let zero = p.t(&f32s(0.0, h), &[1, h]);
        let signs = p.t(&f32s(1.0, h), &[h]);

        p.time(&format!("add            h={h}"), h, |c| p.b.add(c, &x).unwrap(), &x);
        p.time(&format!("add_scalar     h={h}"), h, |c| p.b.add_scalar(c, 1e-3).unwrap(), &x);
        p.time(&format!("silu           h={h}"), h, |c| p.b.silu(c).unwrap(), &x);
        p.time(&format!("rms_norm       h={h}"), h, |c| p.b.rms_norm(c, &w, eps).unwrap(), &x);
        if h % 512 == 0 {
            p.time(
                &format!("hadamard       h={h}"),
                h,
                |c| p.b.hadamard_signed(c, &signs, 512, 0).unwrap(),
                &x,
            );
        }
        p.time(
            &format!("add_rms_norm   h={h}"),
            h,
            |c| p.b.add_rms_norm(c, &zero, &w, eps).unwrap().1,
            &x,
        );
        if h % 512 == 0 {
            p.time(
                &format!("add+rms+had    h={h}"),
                h,
                |c| {
                    let (_, r) = p.b.add_rms_norm(c, &zero, &w, eps).unwrap();
                    p.b.hadamard_signed(&r, &signs, 512, 0).unwrap()
                },
                &x,
            );
        }
    }
}
