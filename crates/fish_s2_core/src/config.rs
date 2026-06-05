use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::models::ModelPair;
use crate::paths::{ensure_project_dirs, models_dir, output_dir, project_root, server_workdir};
use crate::voice::VoiceProfile;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub models_dir: PathBuf,
    pub output_dir: PathBuf,
    pub server_workdir: PathBuf,
    pub server_port: u16,
    #[serde(default = "default_server_backend")]
    pub server_backend: String,
    #[serde(default = "default_server_max_new_tokens")]
    pub server_max_new_tokens: u32,
    pub vulkan_device: i32,
    pub codec_vulkan_device: i32,
    #[serde(default)]
    pub cuda_device: i32,
    pub use_rust_engine: bool,
    pub active_model_pair_id: Option<String>,
    pub voices: Vec<VoiceProfile>,
    pub active_voice_id: Option<uuid::Uuid>,
    pub last_script: String,
    #[serde(default = "default_tts_role")]
    pub tts_role: String,
    #[serde(default = "default_tts_tone")]
    pub tts_tone: String,
    #[serde(default = "default_tts_pace")]
    pub tts_pace: String,
    #[serde(default = "default_tts_pitch")]
    pub tts_pitch: String,
    #[serde(default = "default_tts_energy")]
    pub tts_energy: String,
    pub convert_checkpoint_dir: PathBuf,
    pub convert_script: PathBuf,
    pub python_exe: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        let _ = ensure_project_dirs();
        Self {
            models_dir: models_dir(),
            output_dir: output_dir(),
            server_workdir: server_workdir(),
            server_port: 8081,
            server_backend: default_server_backend(),
            server_max_new_tokens: default_server_max_new_tokens(),
            vulkan_device: 0,
            codec_vulkan_device: 0,
            cuda_device: 0,
            use_rust_engine: true,
            active_model_pair_id: None,
            voices: Vec::new(),
            active_voice_id: None,
            last_script: "你好，這是使用 Fish Audio S2 Pro 生成的語音。".to_string(),
            tts_role: default_tts_role(),
            tts_tone: default_tts_tone(),
            tts_pace: default_tts_pace(),
            tts_pitch: default_tts_pitch(),
            tts_energy: default_tts_energy(),
            convert_checkpoint_dir: PathBuf::new(),
            convert_script: PathBuf::new(),
            python_exe: "python".to_string(),
        }
    }
}

fn default_server_backend() -> String {
    "rust-pure".to_string()
}

fn normalize_server_backend(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return default_server_backend();
    }
    #[cfg(not(feature = "legacy-s2-exe"))]
    if matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "subprocess" | "s2" | "s2.exe"
    ) {
        return default_server_backend();
    }
    trimmed.to_string()
}

fn default_server_max_new_tokens() -> u32 {
    32
}

fn default_tts_role() -> String {
    "default".to_string()
}

fn default_tts_tone() -> String {
    "natural".to_string()
}

fn default_tts_pace() -> String {
    "normal".to_string()
}

fn default_tts_pitch() -> String {
    "normal".to_string()
}

fn default_tts_energy() -> String {
    "normal".to_string()
}

impl AppConfig {
    pub fn config_path() -> PathBuf {
        project_root().join("config.json")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        if let Ok(raw) = fs::read_to_string(&path) {
            if let Ok(mut cfg) = serde_json::from_str::<Self>(&raw) {
                if cfg.models_dir.as_os_str().is_empty() {
                    cfg.models_dir = models_dir();
                }
                cfg.server_backend = normalize_server_backend(&cfg.server_backend);
                if cfg.server_max_new_tokens == 0 {
                    cfg.server_max_new_tokens = default_server_max_new_tokens();
                }
                return cfg;
            }
        }
        let cfg = Self::default();
        let _ = cfg.save();
        cfg
    }

    pub fn save(&self) -> std::io::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let raw = serde_json::to_string_pretty(self)?;
        fs::write(path, raw)
    }

    pub fn active_voice(&self) -> Option<&VoiceProfile> {
        let id = self.active_voice_id?;
        self.voices.iter().find(|v| v.id == id)
    }

    pub fn active_model_pair<'a>(&self, pairs: &'a [ModelPair]) -> Option<&'a ModelPair> {
        let id = self.active_model_pair_id.as_ref()?;
        pairs.iter().find(|p| &p.id == id)
    }

    pub fn ensure_active_model_pair<'a>(
        &mut self,
        pairs: &'a [ModelPair],
    ) -> Option<&'a ModelPair> {
        let selected_index = self
            .active_model_pair_id
            .as_deref()
            .and_then(|id| pairs.iter().position(|pair| pair.id == id))
            .or_else(|| (!pairs.is_empty()).then_some(0))?;
        let pair = &pairs[selected_index];
        if self.active_model_pair_id.as_deref() != Some(pair.id.as_str()) {
            self.active_model_pair_id = Some(pair.id.clone());
        }
        Some(pair)
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.models_dir)?;
        fs::create_dir_all(&self.output_dir)?;
        fs::create_dir_all(&self.server_workdir)?;
        Ok(())
    }
}

pub fn copy_reference_files(workdir: &Path, wav: &Path, text: &str) -> std::io::Result<()> {
    fs::create_dir_all(workdir)?;
    fs::copy(wav, workdir.join("reference.wav"))?;
    fs::write(workdir.join("reference.txt"), text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelEntry, ModelKind};

    #[test]
    fn selects_first_model_pair_when_no_active_id_exists() {
        let mut config = AppConfig {
            active_model_pair_id: None,
            ..Default::default()
        };
        let pairs = model_pairs();
        let selected = config.ensure_active_model_pair(&pairs).unwrap();
        assert_eq!(selected.id, "pair-a");
        assert_eq!(config.active_model_pair_id.as_deref(), Some("pair-a"));
    }

    #[test]
    fn replaces_stale_active_model_pair_id() {
        let mut config = AppConfig {
            active_model_pair_id: Some("missing-pair".to_string()),
            ..Default::default()
        };
        let pairs = model_pairs();
        let selected = config.ensure_active_model_pair(&pairs).unwrap();
        assert_eq!(selected.id, "pair-a");
        assert_eq!(config.active_model_pair_id.as_deref(), Some("pair-a"));
    }

    #[test]
    fn keeps_existing_active_model_pair_id() {
        let mut config = AppConfig {
            active_model_pair_id: Some("pair-b".to_string()),
            ..Default::default()
        };
        let pairs = model_pairs();
        let selected = config.ensure_active_model_pair(&pairs).unwrap();
        assert_eq!(selected.id, "pair-b");
        assert_eq!(config.active_model_pair_id.as_deref(), Some("pair-b"));
    }

    fn model_pairs() -> Vec<ModelPair> {
        vec![
            model_pair("pair-a", "a-transformer.gguf", "a-codec.gguf"),
            model_pair("pair-b", "b-transformer.gguf", "b-codec.gguf"),
        ]
    }

    fn model_pair(id: &str, transformer: &str, codec: &str) -> ModelPair {
        ModelPair {
            id: id.to_string(),
            label: id.to_string(),
            transformer: model_entry(transformer, ModelKind::TransformerOnly),
            codec: model_entry(codec, ModelKind::CodecOnly),
        }
    }

    fn model_entry(path: &str, kind: ModelKind) -> ModelEntry {
        ModelEntry {
            path: PathBuf::from(path),
            kind,
            summary: None,
        }
    }
}
