//! Print the top-k logits of every greedy step.
//!
//! A token disagreement against llama.cpp is only actionable once it is
//! classified: a near-tie (both engines within ~1e-3, ranked oppositely) means
//! a different accumulation order, while a *large* logit gap at the split means
//! a real bug. `qwen35moe-tiny-q4km` splits exactly this way -- step 4 has
//! 110997/180412 at 1.29327/1.29252 here and 1.29206/1.29233 in llama.cpp --
//! and the cause is llama.cpp's CPU `q4_K_8x8` repack, which KORE does not do.
//!
//! usage: logit_probe <gguf> <tokenizer.json> "<prompt>" [n_steps] [top_k]
use kore_nn::qwen35::Qwen35Model;
use std::path::Path;

fn topk(logits: &[f32], k: usize, tok: &tokenizers::Tokenizer) -> String {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let mut s = String::new();
    for &i in idx.iter().take(k) {
        let t = tok.decode(&[i as u32], false).unwrap_or_default();
        s.push_str(&format!("\n    {:>7} {:>16.7} {:?}", i, logits[i], t));
    }
    s
}

fn main() {
    let gguf = std::env::args().nth(1).expect("gguf");
    let tok_path = std::env::args().nth(2).expect("tokenizer");
    let prompt = std::env::args().nth(3).expect("prompt");
    let n_new: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let top_k: usize = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
    let ids = tok
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();

    let (mut ds, mut as_) = model.new_state();
    let mut pos = 0usize;
    let mut logits = Vec::new();
    for &t in &ids {
        logits = model.forward(t, pos, &mut ds, &mut as_).expect("fwd");
        pos += 1;
    }
    println!("topk_after_prompt:{}", topk(&logits, top_k, &tok));
    for step in 0..n_new {
        let id = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .expect("vocab") as u32;
        logits = model.forward(id, pos, &mut ds, &mut as_).expect("fwd");
        pos += 1;
        println!("topk_step_{}:{}", step + 1, topk(&logits, top_k, &tok));
    }
}
