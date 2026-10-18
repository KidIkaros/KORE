//! Dump the keys and values a model actually appends, for the KV-quantization
//! probe (plan §14.22, P6.6a).
//!
//! The rotation width of the INT4 KV cache has to be chosen against the real
//! distribution, not a proxy. `AttnState` after a prefill holds exactly what a
//! quantizer is handed -- post-RoPE keys and raw values -- so this writes them
//! verbatim and `kore-tools/clifford-quant-probe.py --kv <file>` scores the
//! rotation arms on them.
//!
//! Usage:
//!
//! ```text
//! KORE_KV_IDS=ids.txt KORE_KV_OUT=/tmp/opencode/kv.bin \
//!   cargo run --release -p kore-nn --example kv_dump -- <model.gguf> [n_tokens]
//! ```
//!
//! `ids.txt` is whitespace- or comma-separated token ids from the model's *own*
//! tokenizer (llama.cpp's `ref_ids` prints a real prompt's ids). A proxy model's
//! ids would be the wrong embedding rows and so the wrong distribution.
//!
//! File format, little-endian:
//!
//! ```text
//! magic   b"KVDU"
//! u32     version = 1
//! u32     n_layers      (full-attention layers only)
//! u32     folded        (1 if the file carries `prism.hadamard.*`)
//! per layer:
//!   u32   kv_dim
//!   u32   len
//!   len*kv_dim f32   keys    (post-RoPE)
//!   len*kv_dim f32   values  (raw)
//! ```

use kore_nn::gguf_min::MiniGguf;
use kore_nn::qwen35::Qwen35Model;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn ids_from(path: &Path) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>().map_err(|e| format!("{s:?}: {e}")))
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let gguf = args
        .next()
        .or_else(|| std::env::var("KORE_KV_GGUF").ok())
        .unwrap_or_else(|| {
            eprintln!("usage: kv_dump <model.gguf> [n_tokens]  (see the header for env vars)");
            std::process::exit(2);
        });
    let n_tokens: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let ids_path = std::env::var("KORE_KV_IDS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/opencode/kv_ids.txt"));
    let out_path = std::env::var("KORE_KV_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/opencode/kv.bin"));

    let mut ids = ids_from(&ids_path).unwrap_or_else(|e| {
        eprintln!("cannot read token ids: {e}");
        std::process::exit(2);
    });
    if ids.len() < n_tokens {
        eprintln!(
            "note: only {} ids in {}, using all of them",
            ids.len(),
            ids_path.display()
        );
    }
    ids.truncate(n_tokens);
    if ids.is_empty() {
        eprintln!("no token ids: {}", ids_path.display());
        std::process::exit(2);
    }

    // `folded` decides which basis the appended vectors live in, and that is the
    // whole point of this probe -- two files with the same geometry can be in
    // different bases, so it has to travel with the data.
    let folded = MiniGguf::load(Path::new(&gguf))
        .map(|m| m.get("prism.hadamard.sign_widths").is_some())
        .unwrap_or(false);

    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    let cfg = &model.cfg;
    let kv_dim = cfg.n_kv * cfg.head_dim;
    eprintln!(
        "{}: {} layers, {}/{} heads, head_dim {}, kv_dim {}, folded {}, {} prompt tokens",
        gguf,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv,
        cfg.head_dim,
        kv_dim,
        folded,
        ids.len()
    );

    let (mut dstate, mut astate) = model.new_state();
    let t0 = std::time::Instant::now();
    for (pos, &tok) in ids.iter().enumerate() {
        model
            .forward(tok, pos, &mut dstate, &mut astate)
            .expect("forward");
        if pos % 256 == 255 {
            eprintln!(
                "  {}/{} tokens, {:.2} tok/s",
                pos + 1,
                ids.len(),
                (pos + 1) as f64 / t0.elapsed().as_secs_f64()
            );
        }
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "prefill done: {} tokens in {:.2}s ({:.2} tok/s)",
        ids.len(),
        elapsed.as_secs_f64(),
        ids.len() as f64 / elapsed.as_secs_f64()
    );

    let file = std::fs::File::create(&out_path).expect("create output");
    let mut w = BufWriter::new(file);
    w.write_all(b"KVDU").expect("magic");
    for v in [1u32, astate.len() as u32, u32::from(folded)] {
        w.write_all(&v.to_le_bytes()).expect("header");
    }

    let mut total = 0usize;
    for (i, st) in astate.iter().enumerate() {
        let (k, v) = st.kv();
        assert_eq!(k.len(), st.len() * kv_dim, "layer {i} keys");
        assert_eq!(v.len(), st.len() * kv_dim, "layer {i} values");
        w.write_all(&(kv_dim as u32).to_le_bytes()).expect("kv_dim");
        w.write_all(&(st.len() as u32).to_le_bytes()).expect("len");
        for row in k.chunks(kv_dim) {
            for x in row {
                w.write_all(&x.to_le_bytes()).expect("k");
            }
        }
        for row in v.chunks(kv_dim) {
            for x in row {
                w.write_all(&x.to_le_bytes()).expect("v");
            }
        }
        let rms = |s: &[f32]| {
            if s.is_empty() {
                0.0
            } else {
                (s.iter().map(|x| (x * x) as f64).sum::<f64>() / s.len() as f64).sqrt() as f32
            }
        };
        let amax = |s: &[f32]| s.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        eprintln!(
            "  layer {i:2}: len {:5}  K rms {:8.4} max {:8.3}   V rms {:8.4} max {:8.3}",
            st.len(),
            rms(k),
            amax(k),
            rms(v),
            amax(v)
        );
        total += k.len() + v.len();
    }
    w.flush().expect("flush");
    drop(w);
    eprintln!(
        "wrote {} ({} layers, {} f32 values, {:.1} MB)",
        out_path.display(),
        astate.len(),
        total,
        std::fs::metadata(&out_path)
            .map(|m| m.len() as f64 / 1e6)
            .unwrap_or(0.0)
    );
}
