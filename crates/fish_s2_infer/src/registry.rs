use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use fish_s2_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};

use crate::error::{InferError, Result};

pub const SLOW_AR_LAYERS: usize = 36;
pub const FAST_AR_LAYERS: usize = 4;
pub const SLOW_CONTEXT_LENGTH: u32 = 32768;
pub const FAST_CONTEXT_LENGTH: u32 = 11;
pub const HIDDEN_SIZE: u64 = 2560;
pub const ATTENTION_HEADS: u32 = 32;
pub const KV_HEADS: u32 = 8;
pub const HEAD_DIM: u32 = 128;
pub const QK_NORM_SIZE: u64 = 128;
pub const WQKV_OUT: u64 = 6144;
pub const ATTENTION_OUT: u64 = 4096;
pub const FFN_SIZE: u64 = 9728;
pub const TEXT_VOCAB_SIZE: u64 = 155776;
pub const FAST_VOCAB_SIZE: u64 = 4096;
pub const CODEBOOK_EMBEDDING_SIZE: u64 = 40960;
pub const CODEBOOK_SIZE: u32 = 4096;
pub const NUM_CODEBOOKS: u32 = 10;
pub const SEMANTIC_BEGIN_ID: u32 = 151678;
pub const SEMANTIC_END_ID: u32 = 155773;
pub const ROPE_FREQ_BASE: f32 = 1_000_000.0;
pub const RMS_NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub struct DualArGraphSpec {
    pub slow: ArGraphSpec,
    pub fast: ArGraphSpec,
    pub kv_cache: KvCacheSpec,
    pub codebook_size: u32,
    pub num_codebooks: u32,
    pub semantic_begin_id: u32,
    pub semantic_end_id: u32,
    pub scale_codebook_embeddings: bool,
    pub tie_word_embeddings: bool,
    pub fast_has_project_in: bool,
}

impl DualArGraphSpec {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let slow_qk_norm = metadata_bool(gguf, "fish_speech.attention_qk_norm")?;
        let slow_head_dim = if slow_qk_norm {
            QK_NORM_SIZE as u32
        } else {
            metadata_u32(gguf, "fish-speech.embedding_length")?
                / metadata_u32(gguf, "fish-speech.attention.head_count")?
        };
        let slow = ArGraphSpec::new(ArGraphSpecParams {
            context_length: metadata_u32(gguf, "fish-speech.context_length")?,
            embedding_length: metadata_u32(gguf, "fish-speech.embedding_length")?,
            feed_forward_length: metadata_u32(gguf, "fish-speech.feed_forward_length")?,
            block_count: metadata_u32(gguf, "fish-speech.block_count")?,
            head_count: metadata_u32(gguf, "fish-speech.attention.head_count")?,
            head_count_kv: metadata_u32(gguf, "fish-speech.attention.head_count_kv")?,
            head_dim: slow_head_dim,
            rope_freq_base: metadata_f32(gguf, "fish-speech.rope.freq_base")?,
            rms_norm_eps: metadata_f32(gguf, "fish-speech.attention.layer_norm_rms_epsilon")?,
            attention_qk_norm: slow_qk_norm,
        })?;

        let fast = ArGraphSpec::new(ArGraphSpecParams {
            context_length: metadata_u32(gguf, "fish_speech.fast_context_length")?,
            embedding_length: metadata_u32(gguf, "fish_speech.fast_embedding_length")?,
            feed_forward_length: metadata_u32(gguf, "fish_speech.fast_feed_forward_length")?,
            block_count: metadata_u32(gguf, "fish_speech.fast_block_count")?,
            head_count: metadata_u32(gguf, "fish_speech.fast_head_count")?,
            head_count_kv: metadata_u32(gguf, "fish_speech.fast_head_count_kv")?,
            head_dim: metadata_u32(gguf, "fish_speech.fast_head_dim")?,
            rope_freq_base: metadata_f32(gguf, "fish_speech.fast_rope_freq_base")?,
            rms_norm_eps: metadata_f32(gguf, "fish_speech.fast_layer_norm_rms_eps")?,
            attention_qk_norm: metadata_bool(gguf, "fish_speech.fast_attention_qk_norm")?,
        })?;

        let codebook_size = metadata_u32(gguf, "fish_speech.codebook_size")?;
        let num_codebooks = metadata_u32(gguf, "fish_speech.num_codebooks")?;
        let spec = Self {
            kv_cache: KvCacheSpec {
                ggml_type: GgmlType::F16,
                head_dim: slow.head_dim,
                head_count_kv: slow.head_count_kv,
                block_count: slow.block_count,
            },
            slow,
            fast,
            codebook_size,
            num_codebooks,
            semantic_begin_id: metadata_u32(gguf, "fish_speech.semantic_begin_id")?,
            semantic_end_id: metadata_u32(gguf, "fish_speech.semantic_end_id")?,
            scale_codebook_embeddings: metadata_bool(
                gguf,
                "fish_speech.scale_codebook_embeddings",
            )?,
            tie_word_embeddings: metadata_bool(gguf, "fish_speech.tie_word_embeddings")?,
            fast_has_project_in: metadata_bool(gguf, "fish_speech.fast_project_in")?,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn codebook_input_dim(&self) -> u32 {
        self.num_codebooks + 1
    }

    fn validate(&self) -> Result<()> {
        let checks = [
            (
                "slow.context_length",
                self.slow.context_length,
                SLOW_CONTEXT_LENGTH,
            ),
            (
                "slow.embedding_length",
                self.slow.embedding_length,
                HIDDEN_SIZE as u32,
            ),
            (
                "slow.feed_forward_length",
                self.slow.feed_forward_length,
                FFN_SIZE as u32,
            ),
            (
                "slow.block_count",
                self.slow.block_count,
                SLOW_AR_LAYERS as u32,
            ),
            ("slow.head_count", self.slow.head_count, ATTENTION_HEADS),
            ("slow.head_count_kv", self.slow.head_count_kv, KV_HEADS),
            ("slow.head_dim", self.slow.head_dim, HEAD_DIM),
            (
                "fast.context_length",
                self.fast.context_length,
                FAST_CONTEXT_LENGTH,
            ),
            (
                "fast.embedding_length",
                self.fast.embedding_length,
                HIDDEN_SIZE as u32,
            ),
            (
                "fast.feed_forward_length",
                self.fast.feed_forward_length,
                FFN_SIZE as u32,
            ),
            (
                "fast.block_count",
                self.fast.block_count,
                FAST_AR_LAYERS as u32,
            ),
            ("fast.head_count", self.fast.head_count, ATTENTION_HEADS),
            ("fast.head_count_kv", self.fast.head_count_kv, KV_HEADS),
            ("fast.head_dim", self.fast.head_dim, HEAD_DIM),
            ("codebook_size", self.codebook_size, CODEBOOK_SIZE),
            ("num_codebooks", self.num_codebooks, NUM_CODEBOOKS),
            (
                "semantic_begin_id",
                self.semantic_begin_id,
                SEMANTIC_BEGIN_ID,
            ),
            ("semantic_end_id", self.semantic_end_id, SEMANTIC_END_ID),
        ];
        let mut failures = Vec::new();
        for (name, actual, expected) in checks {
            if actual != expected {
                failures.push(format!("{name}: expected {expected}, got {actual}"));
            }
        }
        if !approx_eq(self.slow.rope_freq_base, ROPE_FREQ_BASE) {
            failures.push(format!(
                "slow.rope_freq_base: got {}",
                self.slow.rope_freq_base
            ));
        }
        if !approx_eq(self.fast.rope_freq_base, ROPE_FREQ_BASE) {
            failures.push(format!(
                "fast.rope_freq_base: got {}",
                self.fast.rope_freq_base
            ));
        }
        if !approx_eq(self.slow.rms_norm_eps, RMS_NORM_EPS) {
            failures.push(format!("slow.rms_norm_eps: got {}", self.slow.rms_norm_eps));
        }
        if !approx_eq(self.fast.rms_norm_eps, RMS_NORM_EPS) {
            failures.push(format!("fast.rms_norm_eps: got {}", self.fast.rms_norm_eps));
        }
        if !self.slow.attention_qk_norm {
            failures.push("slow.attention_qk_norm should be true".into());
        }
        if self.fast.attention_qk_norm {
            failures.push("fast.attention_qk_norm should be false".into());
        }
        if !self.scale_codebook_embeddings {
            failures.push("scale_codebook_embeddings should be true".into());
        }
        if !self.tie_word_embeddings {
            failures.push("tie_word_embeddings should be true".into());
        }
        if self.fast_has_project_in {
            failures.push("fast_project_in should be false for s2-pro-f16".into());
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(InferError::Message(format!(
                "graph spec validation failed:\n{}",
                failures.join("\n")
            )))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArGraphSpec {
    pub context_length: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub block_count: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub q_size: u32,
    pub kv_size: u32,
    pub head_repeat: u32,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
    pub attention_qk_norm: bool,
}

impl ArGraphSpec {
    fn new(params: ArGraphSpecParams) -> Result<Self> {
        if params.head_count_kv == 0 || !params.head_count.is_multiple_of(params.head_count_kv) {
            return Err(InferError::Message(format!(
                "invalid GQA head split: heads={}, kv_heads={}",
                params.head_count, params.head_count_kv
            )));
        }
        let q_size = params.head_count * params.head_dim;
        let kv_size = params.head_count_kv * params.head_dim;
        Ok(Self {
            context_length: params.context_length,
            embedding_length: params.embedding_length,
            feed_forward_length: params.feed_forward_length,
            block_count: params.block_count,
            head_count: params.head_count,
            head_count_kv: params.head_count_kv,
            head_dim: params.head_dim,
            q_size,
            kv_size,
            head_repeat: params.head_count / params.head_count_kv,
            rope_freq_base: params.rope_freq_base,
            rms_norm_eps: params.rms_norm_eps,
            attention_qk_norm: params.attention_qk_norm,
        })
    }
}

struct ArGraphSpecParams {
    context_length: u32,
    embedding_length: u32,
    feed_forward_length: u32,
    block_count: u32,
    head_count: u32,
    head_count_kv: u32,
    head_dim: u32,
    rope_freq_base: f32,
    rms_norm_eps: f32,
    attention_qk_norm: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheSpec {
    pub ggml_type: GgmlType,
    pub head_dim: u32,
    pub head_count_kv: u32,
    pub block_count: u32,
}

impl KvCacheSpec {
    pub fn dimensions(self, max_seq_len: u32) -> [u32; 4] {
        [
            self.head_dim,
            self.head_count_kv,
            max_seq_len,
            self.block_count,
        ]
    }

    pub fn bytes_per_cache(self, max_seq_len: u32) -> u64 {
        self.dimensions(max_seq_len)
            .iter()
            .map(|dim| u64::from(*dim))
            .product::<u64>()
            * 2
    }

    pub fn bytes_for_k_and_v(self, max_seq_len: u32) -> u64 {
        self.bytes_per_cache(max_seq_len) * 2
    }
}

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
    graph_spec: DualArGraphSpec,
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
        let graph_spec = DualArGraphSpec::from_gguf(gguf)?;

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
            graph_spec,
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

    pub fn graph_spec(&self) -> &DualArGraphSpec {
        &self.graph_spec
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

fn required_metadata<'a>(gguf: &'a GgufFile, key: &str) -> Result<&'a str> {
    metadata_value(gguf, key).ok_or_else(|| InferError::Message(format!("missing {key}")))
}

fn metadata_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    required_metadata(gguf, key)?
        .parse()
        .map_err(|err| InferError::Message(format!("invalid u32 metadata {key}: {err}")))
}

fn metadata_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    required_metadata(gguf, key)?
        .parse()
        .map_err(|err| InferError::Message(format!("invalid f32 metadata {key}: {err}")))
}

fn metadata_bool(gguf: &GgufFile, key: &str) -> Result<bool> {
    match required_metadata(gguf, key)? {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(InferError::Message(format!(
            "invalid bool metadata {key}: {other}"
        ))),
    }
}

fn approx_eq(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= 1e-6
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
    fn kv_cache_dimensions_and_bytes_match_s2_layout() {
        let spec = KvCacheSpec {
            ggml_type: GgmlType::F16,
            head_dim: HEAD_DIM,
            head_count_kv: KV_HEADS,
            block_count: SLOW_AR_LAYERS as u32,
        };
        assert_eq!(
            spec.dimensions(2048),
            [HEAD_DIM, KV_HEADS, 2048, SLOW_AR_LAYERS as u32]
        );
        let one_cache_bytes =
            u64::from(HEAD_DIM) * u64::from(KV_HEADS) * 2048 * SLOW_AR_LAYERS as u64 * 2;
        assert_eq!(spec.bytes_per_cache(2048), one_cache_bytes);
        assert_eq!(spec.bytes_for_k_and_v(2048), one_cache_bytes * 2);
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
        let graph = registry.graph_spec();
        assert_eq!(graph.codebook_input_dim(), NUM_CODEBOOKS + 1);
        assert_eq!(graph.slow.head_count, ATTENTION_HEADS);
        assert_eq!(graph.slow.head_count_kv, KV_HEADS);
        assert_eq!(graph.slow.head_dim, HEAD_DIM);
        assert_eq!(graph.slow.q_size, ATTENTION_OUT as u32);
        assert_eq!(graph.slow.kv_size, KV_HEADS * HEAD_DIM);
        assert_eq!(graph.slow.head_repeat, 4);
        assert!(graph.slow.attention_qk_norm);
        assert!(approx_eq(graph.slow.rope_freq_base, ROPE_FREQ_BASE));
        assert_eq!(graph.fast.context_length, FAST_CONTEXT_LENGTH);
        assert_eq!(graph.fast.q_size, ATTENTION_OUT as u32);
        assert_eq!(graph.fast.kv_size, KV_HEADS * HEAD_DIM);
        assert_eq!(graph.fast.head_repeat, 4);
        assert!(!graph.fast.attention_qk_norm);
        assert_eq!(
            graph.kv_cache.dimensions(123),
            [HEAD_DIM, KV_HEADS, 123, SLOW_AR_LAYERS as u32]
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
