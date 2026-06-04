use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::error::{CoreError, Result};

#[derive(Debug, Clone)]
pub struct ConvertPlan {
    pub checkpoint_dir: PathBuf,
    pub codec_path: PathBuf,
    pub output_path: PathBuf,
    pub out_dtype: String,
    pub python_exe: String,
    pub script_path: PathBuf,
}

impl ConvertPlan {
    pub fn validate(&self) -> Result<()> {
        let config = self.checkpoint_dir.join("config.json");
        if !config.exists() {
            return Err(CoreError::Message(format!(
                "missing config.json in {}",
                self.checkpoint_dir.display()
            )));
        }
        if !self.codec_path.exists() {
            return Err(CoreError::Message(format!(
                "codec not found: {}",
                self.codec_path.display()
            )));
        }
        if !self.script_path.exists() {
            return Err(CoreError::Message(format!(
                "export script not found: {} (clone rodrigomatta/s2.cpp and set quantize/unified_export_gguf.py)",
                self.script_path.display()
            )));
        }
        Ok(())
    }

    pub fn command_preview(&self) -> String {
        format!(
            "{} {} --checkpoint-path \"{}\" --codec-checkpoint-path \"{}\" --output \"{}\" --out-dtype {}",
            self.python_exe,
            self.script_path.display(),
            self.checkpoint_dir.display(),
            self.codec_path.display(),
            self.output_path.display(),
            self.out_dtype
        )
    }

    pub fn run_blocking(&self) -> Result<String> {
        self.validate()?;
        if let Some(parent) = self.output_path.parent() {
            std::fs::create_dir_all(parent).map_err(CoreError::Io)?;
        }

        let mut cmd = Command::new(&self.python_exe);
        cmd.arg(&self.script_path)
            .arg("--checkpoint-path")
            .arg(&self.checkpoint_dir)
            .arg("--codec-checkpoint-path")
            .arg(&self.codec_path)
            .arg("--output")
            .arg(&self.output_path)
            .arg("--out-dtype")
            .arg(&self.out_dtype)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = cmd.output().map_err(|e| {
            CoreError::Message(format!("failed to run python ({}): {e}", self.python_exe))
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            return Err(CoreError::Message(format!(
                "GGUF export failed (exit {:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status.code()
            )));
        }
        Ok(format!("{stdout}\n{stderr}"))
    }
}

pub fn default_export_script_paths() -> Vec<PathBuf> {
    vec![
        PathBuf::from("scripts/unified_export_gguf.py"),
        PathBuf::from("../s2.cpp/quantize/unified_export_gguf.py"),
        PathBuf::from("s2.cpp/quantize/unified_export_gguf.py"),
    ]
}

pub fn resolve_export_script(configured: &Path) -> PathBuf {
    if configured.exists() {
        return configured.to_path_buf();
    }
    for p in default_export_script_paths() {
        if p.exists() {
            return p;
        }
    }
    configured.to_path_buf()
}

pub fn checkpoint_codec_path(checkpoint_dir: &Path) -> PathBuf {
    checkpoint_dir.join("codec.pth")
}
