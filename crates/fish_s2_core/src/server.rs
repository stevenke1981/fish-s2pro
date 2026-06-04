use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::config::copy_reference_files;
use crate::error::{CoreError, Result};
use crate::models::ModelPair;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerStatus {
    Stopped,
    Starting,
    Running,
    Error,
}

#[derive(Debug, Clone)]
pub struct ServerStartOptions {
    pub binary: PathBuf,
    pub workdir: PathBuf,
    pub port: u16,
    pub model_transformer: PathBuf,
    pub model_codec: PathBuf,
    pub vulkan_device: i32,
    pub codec_vulkan_device: i32,
    pub reference_wav: Option<PathBuf>,
    pub reference_text: Option<String>,
}

pub struct ServerProcess {
    child: Option<Child>,
    pub status: ServerStatus,
    pub last_error: Option<String>,
    pub started_at: Option<Instant>,
    pub port: u16,
}

impl ServerProcess {
    pub fn new(port: u16) -> Self {
        Self {
            child: None,
            status: ServerStatus::Stopped,
            last_error: None,
            started_at: None,
            port,
        }
    }

    pub fn is_running(&mut self) -> bool {
        if let Some(child) = self.child.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    self.status = ServerStatus::Error;
                    self.last_error = Some(format!("s2 server exited: {status}"));
                    self.child = None;
                    false
                }
                Ok(None) => true,
                Err(e) => {
                    self.status = ServerStatus::Error;
                    self.last_error = Some(e.to_string());
                    false
                }
            }
        } else {
            false
        }
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.status = ServerStatus::Stopped;
        self.started_at = None;
    }

    pub fn start(&mut self, opts: &ServerStartOptions) -> Result<()> {
        self.stop();
        if !opts.binary.exists() {
            return Err(CoreError::Message(format!(
                "s2 binary not found: {} (build mach92432/s2.cpp or set path in Settings)",
                opts.binary.display()
            )));
        }
        std::fs::create_dir_all(&opts.workdir)?;

        if let (Some(wav), Some(text)) = (&opts.reference_wav, &opts.reference_text) {
            copy_reference_files(&opts.workdir, wav, text)?;
        } else {
            let _ = std::fs::remove_file(opts.workdir.join("reference.wav"));
            let _ = std::fs::remove_file(opts.workdir.join("reference.txt"));
        }

        link_or_copy_model(&opts.workdir, "model.gguf", &opts.model_transformer)?;
        link_or_copy_model(&opts.workdir, "codec.gguf", &opts.model_codec)?;

        self.status = ServerStatus::Starting;
        let mut cmd = Command::new(&opts.binary);
        cmd.current_dir(&opts.workdir)
            .arg("-v")
            .arg(opts.vulkan_device.to_string())
            .arg("--codec-vulkan")
            .arg(opts.codec_vulkan_device.to_string())
            .arg("--model")
            .arg("model.gguf")
            .arg("--model-codec")
            .arg("codec.gguf")
            .arg("--port")
            .arg(opts.port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let child = cmd.spawn().map_err(|e| {
            self.status = ServerStatus::Error;
            self.last_error = Some(e.to_string());
            CoreError::Io(e)
        })?;

        self.child = Some(child);
        self.port = opts.port;
        self.started_at = Some(Instant::now());
        self.status = ServerStatus::Running;
        self.last_error = None;
        Ok(())
    }

    pub fn uptime(&self) -> Option<Duration> {
        self.started_at.map(|t| t.elapsed())
    }
}

pub fn default_s2_binary_candidates() -> Vec<PathBuf> {
    vec![
        PathBuf::from("s2.exe"),
        PathBuf::from("s2"),
        PathBuf::from("build/s2.exe"),
        PathBuf::from("build/s2"),
        PathBuf::from("build/Release/s2.exe"),
    ]
}

pub fn resolve_s2_binary(configured: &Path) -> PathBuf {
    if configured.exists() {
        return configured.to_path_buf();
    }
    for candidate in default_s2_binary_candidates() {
        if candidate.exists() {
            return candidate;
        }
    }
    configured.to_path_buf()
}

#[allow(clippy::too_many_arguments)]
pub fn build_start_options(
    binary: PathBuf,
    workdir: PathBuf,
    port: u16,
    pair: &ModelPair,
    vulkan_device: i32,
    codec_vulkan_device: i32,
    reference_wav: Option<PathBuf>,
    reference_text: Option<String>,
) -> ServerStartOptions {
    ServerStartOptions {
        binary,
        workdir,
        port,
        model_transformer: pair.transformer.path.clone(),
        model_codec: pair.codec.path.clone(),
        vulkan_device,
        codec_vulkan_device,
        reference_wav,
        reference_text,
    }
}

#[cfg(windows)]
fn link_or_copy_model(workdir: &Path, name: &str, source: &Path) -> Result<()> {
    let dest = workdir.join(name);
    let _ = std::fs::remove_file(&dest);
    std::fs::copy(source, &dest).map_err(CoreError::Io)?;
    Ok(())
}

#[cfg(unix)]
fn link_or_copy_model(workdir: &Path, name: &str, source: &Path) -> Result<()> {
    use std::os::unix::fs::symlink;
    let dest = workdir.join(name);
    let _ = std::fs::remove_file(&dest);
    if symlink(source, &dest).is_err() {
        std::fs::copy(source, &dest).map_err(CoreError::Io)?;
    }
    Ok(())
}
