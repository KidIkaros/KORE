fn main() {
    let g = kore_nn::gguf_min::MiniGguf::load(std::path::Path::new("models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf")).unwrap();
    let mut keys: Vec<_> = g.meta.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let v = &g.meta[&k];
        let s = format!("{:?}", v);
        println!("{} = {}", k, &s[..s.len().min(140)]);
    }
}
