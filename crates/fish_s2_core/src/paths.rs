use std::path::{Path, PathBuf};

/// Project root (`fish-s2pro/`), resolved from `FISH_S2PRO_ROOT` or the current exe / cwd.
pub fn project_root() -> PathBuf {
    if let Ok(root) = std::env::var("FISH_S2PRO_ROOT") {
        let p = PathBuf::from(root);
        if p.is_dir() {
            return p;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(root) =
            discover_project_root_from(exe.parent().map(Path::to_path_buf).unwrap_or_default())
        {
            return root;
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn discover_project_root_from(mut dir: PathBuf) -> Option<PathBuf> {
    for _ in 0..6 {
        if dir.join("manifest.json").is_file()
            && dir.join("models").is_dir()
            && dir.join("bin").is_dir()
        {
            return Some(dir);
        }
        if dir.join("Cargo.toml").exists() && dir.join("models").exists() {
            return Some(dir);
        }
        if dir.join("Cargo.toml").exists() {
            let has_crates = dir.join("crates").is_dir();
            if has_crates {
                return Some(dir);
            }
        }
        if !dir.pop() {
            break;
        }
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_dir_is_under_project_root() {
        let root = project_root();
        let models = models_dir();
        assert_eq!(models, root.join("models"));
        assert!(models.ends_with("models"));
    }

    #[test]
    fn default_tokenizer_lives_in_models() {
        let p = default_tokenizer_path();
        assert_eq!(p, models_dir().join("tokenizer.json"));
    }

    #[test]
    fn discovers_portable_package_root_from_bin_dir() {
        let root = std::env::temp_dir().join(format!("fish-s2pro-test-{}", uuid::Uuid::new_v4()));
        let bin = root.join("bin");
        let models = root.join("models");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(root.join("manifest.json"), "{}").unwrap();

        let discovered = discover_project_root_from(bin).unwrap();
        assert_eq!(discovered, root);

        let _ = std::fs::remove_dir_all(discovered);
    }
}
