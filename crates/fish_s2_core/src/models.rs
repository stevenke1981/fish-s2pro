use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::error::{CoreError, Result};
use crate::gguf::GgufSummary;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelKind {
    Unified,
    TransformerOnly,
    CodecOnly,
    Unknown,
}

impl ModelKind {
    pub fn from_filename(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        if lower.contains("codec-only") || lower.contains("codec_only") {
            ModelKind::CodecOnly
        } else if lower.contains("transformer-only") || lower.contains("transformer_only") {
            ModelKind::TransformerOnly
        } else if lower.ends_with(".gguf") {
            ModelKind::Unified
        } else {
            ModelKind::Unknown
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    pub path: PathBuf,
    pub kind: ModelKind,
    pub summary: Option<GgufSummary>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelPair {
    pub id: String,
    pub label: String,
    pub transformer: ModelEntry,
    pub codec: ModelEntry,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ScannedModels {
    pub entries: Vec<ModelEntry>,
    pub pairs: Vec<ModelPair>,
}

impl ScannedModels {
    pub fn scan_dir(root: impl AsRef<Path>, max_depth: usize) -> Result<Self> {
        let root = root.as_ref();
        if !root.exists() {
            return Ok(Self::default());
        }

        let mut entries = Vec::new();
        for entry in WalkDir::new(root)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(std::result::Result::ok)
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if ext.to_ascii_lowercase() != "gguf" {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            let kind = ModelKind::from_filename(&file_name);
            let summary = GgufSummary::inspect(path).ok();
            entries.push(ModelEntry {
                path: path.to_path_buf(),
                kind,
                summary,
            });
        }

        entries.sort_by(|a, b| a.path.cmp(&b.path));
        let pairs = build_pairs(&entries);
        Ok(Self { entries, pairs })
    }
}

fn build_pairs(entries: &[ModelEntry]) -> Vec<ModelPair> {
    let mut pairs = Vec::new();
    let transformers: Vec<_> = entries
        .iter()
        .filter(|e| e.kind == ModelKind::TransformerOnly || e.kind == ModelKind::Unified)
        .collect();
    let codecs: Vec<_> = entries
        .iter()
        .filter(|e| e.kind == ModelKind::CodecOnly || e.kind == ModelKind::Unified)
        .collect();

    for tr in transformers {
        let tr_stem = stem_key(tr.path.file_name().and_then(|n| n.to_str()).unwrap_or(""));
        let codec = codecs.iter().find(|c| {
            if c.kind == ModelKind::Unified && tr.kind == ModelKind::Unified {
                return c.path == tr.path;
            }
            let c_stem = stem_key(c.path.file_name().and_then(|n| n.to_str()).unwrap_or(""));
            c_stem == tr_stem && c.path != tr.path
        });
        let Some(codec) = codec.cloned() else {
            if tr.kind == ModelKind::Unified {
                pairs.push(ModelPair {
                    id: tr.path.display().to_string(),
                    label: tr.path.file_name().unwrap().to_string_lossy().into_owned(),
                    transformer: tr.clone(),
                    codec: tr.clone(),
                });
            }
            continue;
        };
        pairs.push(ModelPair {
            id: format!("{}|{}", tr.path.display(), codec.path.display()),
            label: format!(
                "{} + {}",
                tr.path.file_name().unwrap().to_string_lossy(),
                codec.path.file_name().unwrap().to_string_lossy()
            ),
            transformer: tr.clone(),
            codec: codec.clone(),
        });
    }
    pairs
}

fn stem_key(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let stripped = lower
        .replace("transformer-only", "")
        .replace("transformer_only", "")
        .replace("codec-only", "")
        .replace("codec_only", "")
        .replace(".gguf", "");
    stripped
        .trim_matches(|c: char| c == '-' || c == '_')
        .to_string()
}

pub fn validate_pair(pair: &ModelPair) -> Result<()> {
    if !pair.transformer.path.exists() {
        return Err(CoreError::Message(format!(
            "transformer model missing: {}",
            pair.transformer.path.display()
        )));
    }
    if !pair.codec.path.exists() {
        return Err(CoreError::Message(format!(
            "codec model missing: {}",
            pair.codec.path.display()
        )));
    }
    Ok(())
}
