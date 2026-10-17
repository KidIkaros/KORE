use kore_core::{Device, Tensor};
use kore_vulkan::VulkanBackend;
use std::time::Instant;

fn bench(b: &VulkanBackend, name: &str, q: &str, n: usize, k: usize, m: usize) {
    let (row_bytes, w, is_f32) = match q {
        "q4k" => (k / 256 * 144, vec![0x55u8; n * (k / 256) * 144], false),
        "q6k" => (k / 256 * 210, vec![0x55u8; n * (k / 256) * 210], false),
        "q20" => (k / 64 * 18, vec![0x55u8; n * (k / 64) * 18], false),
        "f32" => (k * 4, vec![0x55u8; n * k * 4], true),
        _ => unreachable!(),
    };
    let dev = Device::Vulkan(0);
    let w_len = w.len();
    let x = Tensor::from_f32(&vec![0.5f32; m * k], &[m, k]).to(dev).unwrap();
    let wt = if is_f32 {
        let f: Vec<f32> = vec![0.25f32; w_len / 4];
        Tensor::from_f32(&f, &[k, n]).to(dev).unwrap()
    } else {
        let storage = kore_core::Storage::from_packed_bytes(kore_core::DType::U8, w_len, w);
        Tensor::from_storage(storage, &[n, row_bytes]).to(dev).unwrap()
    };
    let f = |x: &Tensor, w: &Tensor| match q {
        "q4k" => b.gemv_q4_k(x, w, n, k).unwrap(),
        "q6k" => b.gemv_q6_k(x, w, n, k).unwrap(),
        "q20" => b.gemv_q2_0(x, w, n, k).unwrap(),
        "f32" => b.matmul(x, w).unwrap(),
        _ => unreachable!(),
    };
    // warmup + ramp clocks
    for _ in 0..3 { let y = f(&x, &wt); let _ = y.cpu().unwrap(); }

    // pipelined: record all reps, single sync
    let reps = 10;
    let t0 = Instant::now();
    let mut ys = Vec::new();
    for _ in 0..reps { ys.push(f(&x, &wt)); }
    let _ = ys.last().unwrap().cpu().unwrap();
    let ms_pipe = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    // serialized: sync per iter
    let t0 = Instant::now();
    for _ in 0..reps { let y = f(&x, &wt); let _ = y.cpu().unwrap(); }
    let ms_ser = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    let mb = w_len as f64 / 1e6;
    println!(
        "{name} {q} n={n} k={k} m={m}: pipe {ms_pipe:.2} ms ({:.0} GB/s) | serial {ms_ser:.2} ms ({:.0} GB/s)",
        mb / 1000.0 / (ms_pipe / 1000.0),
        mb / 1000.0 / (ms_ser / 1000.0)
    );
}

fn main() {
    let b = VulkanBackend::new().expect("vulkan");
    // f32 streaming ceiling (column-major coalesced reads)
    bench(&b, "f32_150mb", "f32", 18360, 2048, 1);
    bench(&b, "f32_219mb", "f32", 26751, 2048, 1);
    for m in [1usize, 2, 4] {
        bench(&b, "lm_head", "q6k", 130560, 2048, m);
    }
    for m in [1usize, 2, 4] {
        bench(&b, "lm_head", "q4k", 130560, 2048, m);
    }
    for m in [1usize, 2, 4] {
        bench(&b, "big", "q4k", 16384, 8192, m);
    }
    for m in [1usize, 2, 4] {
        bench(&b, "ffn_gate_up", "q4k", 12288, 2048, m);
    }
    // q2_0 shapes from Bonsai-2B
    bench(&b, "emb_head_q4k", "q4k", 248320, 2048, 1);
    bench(&b, "emb_head", "q20", 248320, 2048, 1);
    bench(&b, "qkv", "q20", 6144, 2048, 1);
    bench(&b, "ffn_down", "q20", 2048, 6144, 1);
    bench(&b, "attn_out", "q20", 2048, 2048, 1);
}
