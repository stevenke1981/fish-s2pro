use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use fish_s2_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};

use crate::error::{InferError, Result};

pub const SLOW_AR_LAYERS: usize = 36;
pub const FAST_AR_LAYERS: usize = 4;
pub const HIDDEN_SIZE: u64 = 2560;
pub const QK_NORM_SIZE: u64 = 128;
pub const WQKV_OUT: u64 = 6144;
pub const ATTENTION_OUT: u64 = 4096;
pub const FFN_SIZE: u64 = 9728;
pub const TEXT_VOCAB_SIZE: u64 = 155776;
pub const FAST_VOCAB_SIZE: u64 = 4096;
pub const CODEBOOK_EMBEDDING_SIZE: u64 = 40960;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    Embedding,
    AttentionNorm,
    AttentionQNorm,
    AttentionKNorm,
    AttentionWqkv,
    AttentionOutput,
    FfnNorm,
    FeedForwardW1,
    FeedForwardW2,
    FeedForwardW3,
    OutputNorm,
    FastEmbedding,
    FastOutput,
    CodebookEmbedding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub role: TensorRole,
    pub ggml_type: GgmlType,
    pub dimensions: Vec<u64>,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, role: TensorRole, dimensions: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            role,
            ggml_type: GgmlType::F16,
            dimensions: dimensions.into(),
        }
    }

    pub fn matches(&self, tensor: &GgufTensorInfo) -> bool {
        tensor.ggml_type == self.ggml_type && tensor.dimensions == self.dimensions
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlowArLayerWeights {
    pub layer: usize,
    pub attention_q_norm: String,
    pub attention_k_norm: String,
    pub attention_wqkv: String,
    pub attention_output: String,
    pub attention_norm: String,
    pub ffn_norm: String,
    pub feed_forward_w1: String,
    pub feed_forward_w2: String,
    pub feed_forward_w3: String,
}

impl SlowArLayerWeights {
    pub fn spec(layer: usize) -> Vec<TensorSpec> {
        let p = format!("layers.{layer}");
        vec![
            TensorSpec::new(
                format!("{p}.attention.q_norm.weight"),
                TensorRole::AttentionQNorm,
                [QK_NORM_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.attention.k_norm.weight"),
                TensorRole::AttentionKNorm,
                [QK_NORM_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.attention.wqkv.weight"),
                TensorRole::AttentionWqkv,
                [HIDDEN_SIZE, WQKV_OUT],
            ),
            TensorSpec::new(
                format!("{p}.attention.wo.weight"),
                TensorRole::AttentionOutput,
                [ATTENTION_OUT, HIDDEN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.attention_norm.weight"),
                TensorRole::AttentionNorm,
                [HIDDEN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.ffn_norm.weight"),
                TensorRole::FfnNorm,
                [HIDDEN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.feed_forward.w1.weight"),
                TensorRole::FeedForwardW1,
                [HIDDEN_SIZE, FFN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.feed_forward.w2.weight"),
                TensorRole::FeedForwardW2,
                [FFN_SIZE, HIDDEN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.feed_forward.w3.weight"),
                TensorRole::FeedForwardW3,
                [HIDDEN_SIZE, FFN_SIZE],
            ),
        ]
    }

    pub fn names(layer: usize) -> Self {
        let p = format!("layers.{layer}");
        Self {
            layer,
            attention_q_norm: format!("{p}.attention.q_norm.weight"),
            attention_k_norm: format!("{p}.attention.k_norm.weight"),
            attention_wqkv: format!("{p}.attention.wqkv.weight"),
            attention_output: format!("{p}.attention.wo.weight"),
            attention_norm: format!("{p}.attention_norm.weight"),
            ffn_norm: format!("{p}.ffn_norm.weight"),
            feed_forward_w1: format!("{p}.feed_forward.w1.weight"),
            feed_forward_w2: format!("{p}.feed_forward.w2.weight"),
            feed_forward_w3: format!("{p}.feed_forward.w3.weight"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastArLayerWeights {
    pub layer: usize,
    pub attention_wqkv: String,
    pub attention_output: String,
    pub attention_norm: String,
    pub ffn_norm: String,
    pub feed_forward_w1: String,
    pub feed_forward_w2: String,
    pub feed_forward_w3: String,
}

impl FastArLayerWeights {
    pub fn spec(layer: usize) -> Vec<TensorSpec> {
        let p = format!("fast_layers.{layer}");
        vec![
            TensorSpec::new(
                format!("{p}.attention.wqkv.weight"),
                TensorRole::AttentionWqkv,
                [HIDDEN_SIZE, WQKV_OUT],
            ),
            TensorSpec::new(
                format!("{p}.attention.wo.weight"),
                TensorRole::AttentionOutput,
                [ATTENTION_OUT, HIDDEN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.attention_norm.weight"),
                TensorRole::AttentionNorm,
                [HIDDEN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.ffn_norm.weight"),
                TensorRole::FfnNorm,
                [HIDDEN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.feed_forward.w1.weight"),
                TensorRole::FeedForwardW1,
                [HIDDEN_SIZE, FFN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.feed_forward.w2.weight"),
                TensorRole::FeedForwardW2,
                [FFN_SIZE, HIDDEN_SIZE],
            ),
            TensorSpec::new(
                format!("{p}.feed_forward.w3.weight"),
                TensorRole::FeedForwardW3,
                [HIDDEN_SIZE, FFN_SIZE],
            ),
        ]
    }

    pub fn names(layer: usize) -> Self {
        let p = format!("fast_layers.{layer}");
        Self {
            layer,
            attention_wqkv: format!("{p}.attention.wqkv.weight"),
            attention_output: format!("{p}.attention.wo.weight"),
            attention_norm: format!("{p}.attention_norm.weight"),
            ffn_norm: format!("{p}.ffn_norm.weight"),
            feed_forward_w1: format!("{p}.feed_forward.w1.weight"),
            feed_forward_w2: format!("{p}.feed_forward.w2.weight"),
            feed_forward_w3: format!("{p}.feed_forward.w3.weight"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransformerTensorRegistry {
    pub architecture: String,
    pub tensor_count: usize,
    tensors: BTreeMap<String, GgufTensorInfo>,
    required: Vec<TensorSpec>,
    slow_layers: Vec<SlowArLayerWeights>,
    fast_layers: Vec<FastArLayerWeights>,
}

impl TransformerTensorRegistry {
    pub fn from_gguf_file(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufFile::open(path).map_err(|err| InferError::Message(err.to_string()))?;
        Self::from_gguf(&gguf)
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = metadata_value(gguf, "general.architecture")
            .ok_or_else(|| InferError::Message("missing general.architecture".into()))?
            .to_string();
        if architecture != "fish-speech" {
            return Err(InferError::Message(format!(
                "expected fish-speech transformer GGUF, got {architecture}"
            )));
        }

        let tensors = gguf
            .tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), tensor.clone()))
            .collect::<BTreeMap<_, _>>();

        let mut required = root_specs();
        for layer in 0..SLOW_AR_LAYERS {
            required.extend(SlowArLayerWeights::spec(layer));
        }
        for layer in 0..FAST_AR_LAYERS {
            required.extend(FastArLayerWeights::spec(layer));
        }

        let registry = Self {
            architecture,
            tensor_count: tensors.len(),
            tensors,
            required,
            slow_layers: (0..SLOW_AR_LAYERS).map(SlowArLayerWeights::names).collect(),
            fast_layers: (0..FAST_AR_LAYERS).map(FastArLayerWeights::names).collect(),
        };
        registry.validate_required()?;
        registry.validate_layer_sets()?;
        Ok(registry)
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    pub fn required_specs(&self) -> &[TensorSpec] {
        &self.required
    }

    pub fn slow_layer(&self, layer: usize) -> Option<&SlowArLayerWeights> {
        self.slow_layers.get(layer)
    }

    pub fn fast_layer(&self, layer: usize) -> Option<&FastArLayerWeights> {
        self.fast_layers.get(layer)
    }

    pub fn slow_layer_count(&self) -> usize {
        self.slow_layers.len()
    }

    pub fn fast_layer_count(&self) -> usize {
        self.fast_layers.len()
    }

    fn validate_required(&self) -> Result<()> {
        let mut failures = Vec::new();
        for spec in &self.required {
            match self.tensors.get(&spec.name) {
                Some(tensor) if spec.matches(tensor) => {}
                Some(tensor) => failures.push(format!(
                    "{} expected {:?} {:?}, got {:?} {:?}",
                    spec.name, spec.ggml_type, spec.dimensions, tensor.ggml_type, tensor.dimensions
                )),
                None => failures.push(format!("missing {}", spec.name)),
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(InferError::Message(format!(
                "transformer tensor registry validation failed:\n{}",
                failures.join("\n")
            )))
        }
    }

    fn validate_layer_sets(&self) -> Result<()> {
        let slow = indexed_layer_set(self.tensors.keys(), "layers");
        let fast = indexed_layer_set(self.tensors.keys(), "fast_layers");
        let expected_slow = (0..SLOW_AR_LAYERS).collect::<BTreeSet<_>>();
        let expected_fast = (0..FAST_AR_LAYERS).collect::<BTreeSet<_>>();
        if slow != expected_slow {
            return Err(InferError::Message(format!(
                "unexpected Slow-AR layer set: {:?}",
                slow
            )));
        }
        if fast != expected_fast {
            return Err(InferError::Message(format!(
                "unexpected Fast-AR layer set: {:?}",
                fast
            )));
        }
        Ok(())
    }
}

fn root_specs() -> Vec<TensorSpec> {
    vec![
        TensorSpec::new(
            "codebook_embeddings.weight",
            TensorRole::CodebookEmbedding,
            [HIDDEN_SIZE, CODEBOOK_EMBEDDING_SIZE],
        ),
        TensorSpec::new(
            "embeddings.weight",
            TensorRole::Embedding,
            [HIDDEN_SIZE, TEXT_VOCAB_SIZE],
        ),
        TensorSpec::new(
            "fast_embeddings.weight",
            TensorRole::FastEmbedding,
            [HIDDEN_SIZE, FAST_VOCAB_SIZE],
        ),
        TensorSpec::new(
            "fast_output.weight",
            TensorRole::FastOutput,
            [HIDDEN_SIZE, FAST_VOCAB_SIZE],
        ),
        TensorSpec::new("fast_norm.weight", TensorRole::OutputNorm, [HIDDEN_SIZE]),
        TensorSpec::new("norm.weight", TensorRole::OutputNorm, [HIDDEN_SIZE]),
    ]
}

fn metadata_value<'a>(gguf: &'a GgufFile, key: &str) -> Option<&'a str> {
    gguf.metadata
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value.as_str())
}

fn indexed_layer_set<'a>(names: impl Iterator<Item = &'a String>, prefix: &str) -> BTreeSet<usize> {
    names
        .filter_map(|name| {
            name.strip_prefix(prefix)
                .and_then(|rest| rest.strip_prefix('.'))
                .and_then(|rest| rest.split_once('.'))
                .and_then(|(layer, _)| layer.parse::<usize>().ok())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_expected_layer_weight_names() {
        let slow = SlowArLayerWeights::names(7);
        assert_eq!(slow.attention_wqkv, "layers.7.attention.wqkv.weight");
        assert_eq!(slow.feed_forward_w2, "layers.7.feed_forward.w2.weight");

        let fast = FastArLayerWeights::names(3);
        assert_eq!(fast.attention_wqkv, "fast_layers.3.attention.wqkv.weight");
        assert_eq!(fast.feed_forward_w3, "fast_layers.3.feed_forward.w3.weight");
    }

    #[test]
    fn required_specs_cover_slow_and_fast_layers() {
        let required = {
            let mut specs = root_specs();
            for layer in 0..SLOW_AR_LAYERS {
                specs.extend(SlowArLayerWeights::spec(layer));
            }
            for layer in 0..FAST_AR_LAYERS {
                specs.extend(FastArLayerWeights::spec(layer));
            }
            specs
        };
        assert_eq!(required.len(), 6 + SLOW_AR_LAYERS * 9 + FAST_AR_LAYERS * 7);
        assert!(required
            .iter()
            .any(|spec| spec.name == "layers.35.attention.wqkv.weight"));
        assert!(required
            .iter()
            .any(|spec| spec.name == "fast_layers.3.attention.wqkv.weight"));
    }

    #[test]
    #[ignore = "requires local s2-pro transformer GGUF in models/"]
    fn validates_local_transformer_registry() {
        let path = local_model_dir().join("s2-pro-f16-transformer-only.gguf");
        let registry = TransformerTensorRegistry::from_gguf_file(path).unwrap();
        assert_eq!(registry.architecture, "fish-speech");
        assert_eq!(registry.tensor_count, 358);
        assert_eq!(registry.slow_layer_count(), SLOW_AR_LAYERS);
        assert_eq!(registry.fast_layer_count(), FAST_AR_LAYERS);
        assert_eq!(
            registry
                .tensor("layers.0.attention.wqkv.weight")
                .unwrap()
                .dimensions,
            vec![HIDDEN_SIZE, WQKV_OUT]
        );
        assert_eq!(
            registry.fast_layer(0).unwrap().attention_wqkv,
            "fast_layers.0.attention.wqkv.weight"
        );
    }

    fn local_model_dir() -> std::path::PathBuf {
        std::env::var("FISH_S2_MODEL_DIR").map_or_else(
            |_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .ancestors()
                    .nth(2)
                    .expect("workspace root")
                    .join("models")
            },
            std::path::PathBuf::from,
        )
    }
}
