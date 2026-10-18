//! Token-level gate for the generic GGUF lane.
//!
//! Greedy-decodes from explicit prompt ids, so a file whose tokenizer KORE
//! cannot read is still comparable against llama.cpp's `ref_ids` oracle: the
//! oracle prints the ids it tokenized, and those ids go straight in here.
//!
//!   gguf_ids model.gguf 760,6511,314,9338,369 8 [cpu|gpu]
//!
//! Nothing is sampled -- argmax only -- because the point is to compare two
//! implementations, not two samplers.

use kore_nn::llama::LlamaModel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: gguf_ids <model.gguf> <id,id,...> <n_new> [cpu|gpu]");
        std::process::exit(2);
    }
    let path = std::path::Path::new(&args[1]);
    let ids: Vec<usize> = args[2]
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse().expect("token id"))
        .collect();
    let n_new: usize = args[3].parse().expect("n_new");
    let device: kore_core::Device = args
        .get(4)
        .map(|s| s.as_str())
        .unwrap_or("cpu")
        .parse()
        .expect("device");

    if !device.is_cpu() {
        kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    }
    let mut model = LlamaModel::from_gguf(path, None).expect("load GGUF");
    if !device.is_cpu() {
        model.to(device).expect("weights to device");
    }

    let out = model
        .generate(&ids, n_new, None, |t| print!("{t} "))
        .expect("generate");
    println!();
    println!("ids: {out:?}");
}
