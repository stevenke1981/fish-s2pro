//! Headless Rust inference server (replaces external s2.exe HTTP process).
//!
//! ```text
//! cargo run -p fish_s2_infer --bin fish_s2_server -- \
//!   --transformer models/s2-pro-f16-transformer-only.gguf \
//!   --codec models/s2-pro-f16-codec-only.gguf \
//!   --port 8081
//! ```

use std::path::PathBuf;

use fish_s2_infer::spawn_server;
use fish_s2_infer::{
    default_tokenizer_path, models_dir, output_dir, project_root, server_workdir, EngineBackend,
    EngineConfig, InferenceEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut transformer = None;
    let mut codec = None;
    let mut backend = None;
    let mut cuda_device = None;
    let mut codec_cuda = false;
    let mut max_new_tokens = None;
    let mut port: u16 = 8081;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--transformer" | "-t" => {
                i += 1;
                transformer = Some(PathBuf::from(
                    args.get(i).ok_or("missing --transformer path")?,
                ));
            }
            "--codec" | "-c" => {
                i += 1;
                codec = Some(PathBuf::from(args.get(i).ok_or("missing --codec path")?));
            }
            "--port" | "-p" => {
                i += 1;
                port = args
                    .get(i)
                    .ok_or("missing --port")?
                    .parse()
                    .map_err(|_| "invalid port")?;
            }
            "--backend" => {
                i += 1;
                backend = Some(EngineBackend::parse(
                    args.get(i).ok_or("missing --backend")?,
                )?);
            }
            "--max-new-tokens" => {
                i += 1;
                max_new_tokens = Some(
                    args.get(i)
                        .ok_or("missing --max-new-tokens")?
                        .parse()
                        .map_err(|_| "invalid --max-new-tokens")?,
                );
            }
            "--cuda-device" => {
                i += 1;
                cuda_device = Some(
                    args.get(i)
                        .ok_or("missing --cuda-device")?
                        .parse()
                        .map_err(|_| "invalid --cuda-device")?,
                );
            }
            "--codec-cuda" => {
                codec_cuda = true;
            }
            "--print-paths" => {
                print_paths();
                return Ok(());
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                eprintln!("unknown arg: {}", args[i]);
                print_help();
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let (transformer, codec) = match (transformer, codec) {
        (Some(t), Some(c)) => (t, c),
        _ => auto_discover_pair()?,
    };

    let mut cfg = EngineConfig::new(transformer, codec)?;
    if let Some(backend) = backend {
        cfg.backend = backend;
    }
    if let Some(max_new_tokens) = max_new_tokens {
        cfg.generate_params.max_new_tokens = max_new_tokens;
    }
    if let Some(cuda_device) = cuda_device {
        cfg.cuda_device = cuda_device;
    }
    cfg.codec_cuda = codec_cuda;
    if !default_tokenizer_path().exists() {
        eprintln!(
            "warning: tokenizer missing at {} — run scripts/download_models.ps1",
            default_tokenizer_path().display()
        );
    }

    eprintln!(
        "loading models (may take a while)...\n  backend: {}\n  cuda_device: {}\n  codec_cuda: {}\n  transformer: {}\n  codec: {}",
        cfg.backend.as_str(),
        if cfg.backend.uses_cuda() {
            cfg.cuda_device.to_string()
        } else {
            "unused".to_string()
        },
        if cfg.backend.uses_cuda() && cfg.codec_cuda {
            "requested (guarded; CPU fallback unless FISH_S2_CODEC_CUDA_UNSAFE=1)"
        } else {
            "off"
        },
        cfg.transformer_gguf.display(),
        cfg.codec_gguf.display()
    );
    let engine = InferenceEngine::load(cfg)?;
    let handle = spawn_server(engine, port)?;
    eprintln!(
        "Rust S2 server listening on http://127.0.0.1:{}/v1/tts",
        handle.port()
    );
    loop {
        std::thread::park();
    }
}

fn auto_discover_pair() -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let root = models_dir();
    if !root.is_dir() {
        return Err(format!("models dir not found: {}", root.display()).into());
    }
    let mut transformers = Vec::new();
    let mut codecs = Vec::new();
    for entry in std::fs::read_dir(&root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !name.ends_with(".gguf") {
            continue;
        }
        if name.contains("transformer-only") || name.contains("transformer_only") {
            transformers.push(path);
        } else if name.contains("codec-only") || name.contains("codec_only") {
            codecs.push(path);
        }
    }
    transformers.sort();
    codecs.sort();
    let t = transformers
        .first()
        .cloned()
        .ok_or_else(|| format!("no *-transformer-only.gguf in {}", root.display()))?;
    let c = codecs
        .first()
        .cloned()
        .ok_or_else(|| format!("no *-codec-only.gguf in {}", root.display()))?;
    Ok((t, c))
}

fn print_help() {
    eprintln!(
        r#"fish_s2_server — in-process S2 Pro inference (Rust)

Usage:
  fish_s2_server [--transformer PATH] [--codec PATH] [--port PORT] [--backend {}] [--cuda-device N] [--codec-cuda] [--max-new-tokens N] [--print-paths]

If paths are omitted, picks the first transformer-only + codec-only pair in models/.
"#,
        EngineBackend::cli_values()
    );
}

fn print_paths() {
    println!("project_root={}", project_root().display());
    println!("models_dir={}", models_dir().display());
    println!("tokenizer={}", default_tokenizer_path().display());
    println!("output_dir={}", output_dir().display());
    println!("server_workdir={}", server_workdir().display());
}
