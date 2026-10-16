use std::path::PathBuf;

fn main() {
    let path = std::env::args().nth(1).expect("usage: gguf_probe <file.gguf>");
    let model = oxillama_gguf::GgufModel::load(PathBuf::from(path)).expect("load");
    let meta = &model.file.metadata;
    let mut keys: Vec<String> = meta.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        let v = meta.get(k).unwrap();
        let s = format!("{:?}", v);
        let s = if s.len() > 220 { format!("{}...", &s[..220]) } else { s };
        println!("{k} = {s}");
    }
    println!("--- TENSORS ({}) ---", model.file.tensors.len());
    let mut names: Vec<String> = model.file.tensors.names().cloned().collect();
    names.sort();
    for n in &names {
        let info = model.file.tensors.get(n).unwrap();
        println!("{} {:?} {:?}", n, info.tensor_type, info.dimensions);
    }
}
