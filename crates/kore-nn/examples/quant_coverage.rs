//! Header probe for a GGUF file: architecture, metadata, tensor table.
//!
//! This is the P6.0 instrument (see `docs`/plan §14.8). It answers, *before* any
//! loader is written:
//!
//! - what `general.architecture` actually is (never trust an HF tag);
//! - which metadata keys exist (so a loader reads real keys, not guessed ones);
//! - the tensor table with **type ids and dims** (so the expert-bank layout,
//!   `ffn_{gate,up,down}_exps.weight`, is confirmed against reality);
//! - anything MoE-shaped (`*_exps`, `*expert*`) and the routing metadata.
//!
//! Unknown quant type ids are reported as-is rather than hidden: `nbytes` is 0
//! for a type `gguf_min` does not yet know, which is exactly the signal that the
//! reader needs the type added.
//!
//! Usage: `cargo run --release --example gguf_probe -- <file.gguf> [--tensors]`

use std::collections::BTreeMap;
use std::collections::HashMap;

use kore_nn::gguf_min::{MetaVal, MiniGguf};

fn type_name(id: u32) -> &'static str {
    match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "IQ1_M",
        28 => "BF16",
        30 => "BF16",
        42 => "Q2_0",
        _ => "?",
    }
}

/// Block geometry `(elements, bytes)` for the types the *engine* can consume,
/// kept in sync with the loader's table by construction: if this returns None the
/// engine cannot read the tensor yet.
fn known_block(id: u32) -> Option<(usize, usize)> {
    Some(match id {
        0 => (1, 4),
        1 => (1, 2),
        30 => (1, 2),
        8 => (32, 34),
        42 => (64, 18),
        _ => return None,
    })
}

fn fmt_val(v: &MetaVal) -> String {
    match v {
        MetaVal::Str(s) => {
            if s.len() > 80 {
                format!("Str({}B) {:?}...", s.len(), &s[..60])
            } else {
                format!("Str({:?})", s)
            }
        }
        MetaVal::StrArr(a) => format!("StrArr[{}]", a.len()),
        MetaVal::F32Arr(a) => {
            let head: Vec<String> = a.iter().take(4).map(|x| format!("{x:.5}")).collect();
            format!("F32Arr[{}] first={:?}", a.len(), head)
        }
        MetaVal::I64Arr(a) => format!("I64Arr[{}] {:?}", a.len(), &a[..a.len().min(8)]),
        other => format!("{other:?}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: gguf_probe <file.gguf> [--tensors]");
        std::process::exit(2);
    }
    let path = std::path::Path::new(&args[1]);
    let show_all_tensors = args.iter().any(|a| a == "--tensors");

    let fsize = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!("file      : {}", path.display());
    println!("size      : {fsize} bytes ({:.2} GB)", fsize as f64 / 1e9);

    let g = match MiniGguf::load(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "arch      : {:?}",
        g.get_str("general.architecture").unwrap_or("<none>")
    );
    println!("tensors   : {}", g.tensors.len());
    println!();

    // ---- metadata ----
    println!("== metadata ==");
    let mut keys: Vec<_> = g.meta.keys().cloned().collect();
    keys.sort();
    for k in keys {
        println!("  {k} = {}", fmt_val(&g.meta[&k]));
    }
    println!();

    // ---- tensor table aggregate ----
    let mut by_type: BTreeMap<u32, (usize, u64)> = BTreeMap::new(); // id -> (count, bytes)
    for t in &g.tensors {
        let e = by_type.entry(t.type_id).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.nbytes as u64;
    }
    println!("== tensor types ==");
    let mut payload_bytes: u64 = 0;
    for (id, (count, bytes)) in &by_type {
        let readable = if known_block(*id).is_some() {
            "readable"
        } else {
            "NOT readable yet"
        };
        println!(
            "  {:>3} {:<9} x{:<5} {:>14} bytes  [{}]",
            id,
            type_name(*id),
            count,
            bytes,
            readable
        );
        payload_bytes += *bytes;
    }
    // Reconciliation: the payload must not exceed the file, and the remainder is
    // the header + metadata block (tokenizer vocab dominates) + 32-byte padding.
    // A wrong block size shows up here immediately as a large or negative gap.
    println!(
        "  payload sum = {} bytes; file = {} bytes; header+pad = {} bytes ({:.2} MB)",
        payload_bytes,
        fsize,
        fsize as i128 - payload_bytes as i128,
        (fsize as f64 - payload_bytes as f64) / 1e6
    );
    if payload_bytes > fsize {
        println!("  !! payload exceeds file: a block size in the table is WRONG");
    }
    println!();

    // ---- anything MoE-shaped ----
    let moe_keys = [
        "expert_count",
        "expert_used_count",
        "expert_shared_count",
        "expert_feed_forward_length",
        "expert_group_count",
        "expert_group_used_count",
    ];
    println!("== MoE metadata (searched by name) ==");
    let mut any_moe = false;
    for (k, v) in &g.meta {
        let lk = k.to_lowercase();
        if lk.contains("expert") || moe_keys.iter().any(|m| lk.ends_with(m)) {
            println!("  {k} = {}", fmt_val(v));
            any_moe = true;
        }
    }
    if !any_moe {
        println!("  (none)");
    }
    let exps: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| t.name.contains("_exps") || t.name.to_lowercase().contains("expert"))
        .collect();
    println!("== tensors containing '_exps'/'expert': {} ==", exps.len());
    for t in exps.iter().take(24) {
        println!(
            "  {} type={}({}) dims={:?} nbytes={}",
            t.name,
            t.type_id,
            type_name(t.type_id),
            t.dims,
            t.nbytes
        );
    }
    if exps.len() > 24 {
        println!("  ... {} more", exps.len() - 24);
    }
    println!();

    // ---- per-suffix aggregate: how many distinct layer suffixes exist ----
    let mut suffix_counts: HashMap<String, usize> = HashMap::new();
    for t in &g.tensors {
        // strip a leading "blk.N." then take the remainder as the suffix
        let s = match t.name.strip_prefix("blk.") {
            Some(rest) => match rest.split_once('.') {
                Some((_, suffix)) => suffix.to_string(),
                None => rest.to_string(),
            },
            None => t.name.clone(),
        };
        *suffix_counts.entry(s).or_insert(0) += 1;
    }
    let mut suff: Vec<_> = suffix_counts.into_iter().collect();
    suff.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("== distinct tensor suffixes (name after 'blk.N.') ==");
    for (s, c) in suff.iter().take(40) {
        println!("  x{:<4} {}", c, s);
    }
    println!();

    if show_all_tensors {
        println!("== all tensors ==");
        for t in &g.tensors {
            println!(
                "  {:<48} {:<9} {:?} nbytes={}",
                t.name,
                format!("{}({})", type_name(t.type_id), t.type_id),
                t.dims,
                t.nbytes
            );
        }
    } else {
        println!("(pass --tensors to print all {})", g.tensors.len());
    }
}
