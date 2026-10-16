use super::BANNER;

pub fn cmd_serve(addr: &str, model_path: Option<String>, device_str: &str) {
    tracing_subscriber::fmt::init();

    let target_device: kore_core::Device = match device_str.parse() {
        Ok(d) => d,
        Err(_) => kore_core::Device::Cpu,
    };

    let (state, mode_desc) = match model_path {
        Some(path) => {
            let p = std::path::Path::new(&path);
            if p.is_dir() {
                let config_file = p.join("config.json");
                let config = if config_file.exists() {
                    kore_serve::layered::LayeredConfig::from_hf_config(
                        &config_file,
                        p.to_path_buf(),
                    )
                    .unwrap_or_else(|| {
                        kore_serve::layered::LayeredConfig::llama(
                            p.to_path_buf(),
                            32,
                            4096,
                            32000,
                            32,
                            32,
                            11008,
                        )
                    })
                } else {
                    kore_serve::layered::LayeredConfig::llama(
                        p.to_path_buf(),
                        32,
                        4096,
                        32000,
                        32,
                        32,
                        11008,
                    )
                };
                match kore_serve::layered::LayeredEngine::new(config) {
                    Ok(engine) => {
                        let name = p
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("layered")
                            .to_string();
                        (
                            kore_serve::state::AppState::with_model(engine, name.clone()),
                            format!("layered model ({})", name),
                        )
                    }
                    Err(e) => {
                        eprintln!("Error loading layered model at {}: {}", path, e);
                        std::process::exit(1);
                    }
                }
            } else {
                match kore_nn::Transformer::load_safetensors(p, 16) {
                    Ok(mut transformer) => {
                        if !target_device.is_cpu() {
                            let _ = transformer.to(target_device);
                        }
                        let name = p
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("transformer")
                            .to_string();
                        (
                            kore_serve::state::AppState::with_model(transformer, name.clone()),
                            format!("safetensors ({} on {})", name, target_device),
                        )
                    }
                    Err(e) => {
                        eprintln!("Error loading safetensors model at {}: {}", path, e);
                        std::process::exit(1);
                    }
                }
            }
        }
        None => {
            let mut transformer = kore_nn::Transformer::new(32000, 64, 16);
            if !target_device.is_cpu() {
                let _ = transformer.to(target_device);
            }
            (
                kore_serve::state::AppState::with_model(
                    transformer,
                    "demo-transformer".to_string(),
                ),
                format!("demo-transformer (active on {})", target_device),
            )
        }
    };

    println!("{}", BANNER);
    println!("  v{}  —  Inference Server\n", env!("CARGO_PKG_VERSION"));
    println!("  Listening on {}", addr);
    println!("  Model:        {}\n", mode_desc);
    println!("  Endpoints:");
    println!("    POST /v1/completions");
    println!("    POST /v1/chat/completions");
    println!("    GET  /v1/models");
    println!("    GET  /health\n");

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(async {
        if let Err(e) = kore_serve::server::serve_with_state(addr, state).await {
            eprintln!("Server error: {}", e);
        }
    });
}
