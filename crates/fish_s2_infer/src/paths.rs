use std::path::PathBuf;

pub fn project_root() -> PathBuf {
    if let Ok(root) = std::env::var("FISH_S2PRO_ROOT") {
        let p = PathBuf::from(root);
        if p.is_dir() {
            return p;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default();
        for _ in 0..6 {
            if dir.join("Cargo.toml").exists() && dir.join("crates").is_dir() {
                return dir;
            }
            if !dir.pop() {
                break;
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn models_dir() -> PathBuf {
    project_root().join("models")
}

pub fn output_dir() -> PathBuf {
    project_root().join("output")
}

pub fn server_workdir() -> PathBuf {
    project_root().join("runtime").join("s2_server")
}

pub fn default_tokenizer_path() -> PathBuf {
    models_dir().join("tokenizer.json")
}

pub fn ensure_project_dirs() -> std::io::Result<()> {
    std::fs::create_dir_all(models_dir())?;
    std::fs::create_dir_all(output_dir())?;
    std::fs::create_dir_all(server_workdir())?;
    Ok(())
}
