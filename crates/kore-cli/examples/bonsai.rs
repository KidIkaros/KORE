//! Bonsai-2 (qwen35/ternary) end-to-end smoke test: load GGUF + tokenizer,
//! greedy-decode a short prompt, print text.
use kore_nn::qwen35::Qwen35Model;
use std::path::Path;
use std::time::Instant;

fn main() {
    let gguf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf".into());
    let tok_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/bonsai2-2b/tokenizer.json".into());
    let prompt = std::env::args().nth(3).unwrap_or_else(|| "The capital of France is".into());
    let n_new: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(20);

    let t0 = Instant::now();
    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    eprintln!("model loaded in {:?}: {:?}", t0.elapsed(), model.cfg);

    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
    let ids = tok
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    eprintln!("prompt tokens: {ids:?}");

    // top-10 probe after prompt
    {
        let (mut ds, mut as_) = model.new_state();
        let mut logits = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            logits = model.forward(t, i, &mut ds, &mut as_).expect("fwd");
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        eprintln!("top-10 after prompt:");
        for &i in idx.iter().take(10) {
            let s = tok.decode(&[i as u32], false).unwrap_or_else(|_| "?".into());
            eprintln!("  {:>7} {:>9.3} {:?}", i, logits[i], s);
        }
    }

    let t1 = Instant::now();
    let out = model.generate_greedy(&ids, n_new, 248046).expect("generate");
    let dt = t1.elapsed();
    eprintln!("generated {} tokens in {:?} ({:.2} tok/s)", out.len(), dt, out.len() as f64 / dt.as_secs_f64());
    eprintln!("token ids: {out:?}");
    let text = tok.decode(&out, true).expect("decode");
    println!("{text}");
}
