//! Real-checkpoint gate: a released Qwen3.6-35B-A3B GGUF has to reproduce
//! llama.cpp's greedy token stream id for id.
//!
//! The file is ~13 GiB and is not vendored, so this skips unless it is present.
//! Point `KORE_QWEN36_GGUF` at it, or drop it at
//! `models/qwen36-35b-a3b/Qwen_Qwen3.6-35B-A3B-Q2_K_L.gguf`:
//!
//! ```text
//! KORE_QWEN36_GGUF=/tmp/q36.gguf cargo test --release -p kore-nn --test qwen36_real
//! ```
//!
//! Both sides are greedy, so the streams are a deterministic function of the
//! weights and any disagreement is a real defect rather than a sampling
//! difference. The expected ids were produced by llama.cpp's `ref_ids` example
//! on this exact file.
//!
//! What this covers that the synthetic fixtures cannot: bartowski's `Q2_K_L`
//! puts **Q8_0** on the shared-expert FFN, `ssm_out`, `attn_k`/`attn_v` and
//! both the LM head and the embedding, and Q2_K / Q3_K / Q4_K / Q6_K on the
//! routed experts. It also ships the 41-block layout — 40 real layers plus a
//! trailing MTP block that must be skipped — which no other fixture has.
//! So one file exercises every device format this family's releases use, on a
//! real tensor layout rather than a fixture's.

use kore_core::Device;
use kore_nn::qwen35::{Qwen35Gpu, Qwen35Model};
use std::path::{Path, PathBuf};

/// "The capital of France is", tokenised by the model's own vocabulary.
const PROMPT_IDS: [u32; 5] = [760, 6511, 314, 9338, 369];

/// llama.cpp `ref_ids` on the same file, same prompt, `n_predict = 8`.
const EXPECTED: [u32; 8] = [11751, 13, 198, 760, 6511, 314, 9338, 369];

fn checkpoint() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("KORE_QWEN36_GGUF") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
        println!(
            "Skipping: KORE_QWEN36_GGUF points at {}, which does not exist",
            p.display()
        );
        return None;
    }
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/qwen36-35b-a3b/Qwen_Qwen3.6-35B-A3B-Q2_K_L.gguf");
    if p.exists() {
        return Some(p);
    }
    println!(
        "Skipping: no checkpoint at {} (set KORE_QWEN36_GGUF)",
        p.display()
    );
    None
}

#[test]
fn released_qwen36_moe_matches_the_llama_cpp_token_stream() {
    let Some(path) = checkpoint() else { return };

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    if kore_core::device::get_backend(Device::Vulkan(0)).is_none() {
        println!("Skipping: no Vulkan device in this environment");
        return;
    }

    let model = Qwen35Model::from_gguf(&path).expect("load model");
    // The file declares 41 blocks and `nextn_predict_layers = 1`: the MTP block
    // is not part of the decode path and the loader has to drop it.
    assert_eq!(model.cfg.n_layers, 40, "the MTP block was not skipped");
    assert_eq!(model.cfg.n_experts, 256);
    assert_eq!(model.cfg.n_experts_used, 8);

    let gpu = Qwen35Gpu::from_cpu(&model, Device::Vulkan(0)).expect("materialize on device");
    let got = gpu
        .generate_greedy(&model, &PROMPT_IDS, EXPECTED.len(), 4096)
        .expect("greedy decode");

    assert_eq!(
        got, EXPECTED,
        "token stream diverged from llama.cpp:\n  engine {:?}\n  ref    {:?}",
        got, EXPECTED
    );
    println!("matched llama.cpp on all {} ids", got.len());
}
