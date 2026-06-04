//! Fast-AR codebook decoder (4-layer prefix transformer, no KV persistence).

use std::path::Path;

use fish_s2_core::gguf::GgufFile;

use crate::attention::{apply_rope_normal, gqa_decode_attention, GqaAttentionShape};
use crate::error::{InferError, Result};
use crate::registry::{
    ArGraphSpec, DualArGraphSpec, TransformerTensorRegistry, FAST_AR_LAYERS, FAST_VOCAB_SIZE,
    HIDDEN_SIZE,
};
use crate::sampling::{sample_token, RandomSource, SamplerParams};
use crate::tensor::{embedding_lookup_rows, linear, rms_norm, F16TensorView};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastArLayerShape {
    pub hidden_size: usize,
    pub feed_forward_size: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub rms_norm_eps: f32,
    pub attention_qk_norm: bool,
}

impl FastArLayerShape {
    pub fn from_ar_graph_spec(spec: &ArGraphSpec) -> Result<Self> {
        Ok(Self {
            hidden_size: usize_from_u32(spec.embedding_length, "hidden_size")?,
            feed_forward_size: usize_from_u32(spec.feed_forward_length, "feed_forward_size")?,
            head_count: usize_from_u32(spec.head_count, "head_count")?,
            head_count_kv: usize_from_u32(spec.head_count_kv, "head_count_kv")?,
            head_dim: usize_from_u32(spec.head_dim, "head_dim")?,
            rope_base: spec.rope_freq_base,
            rms_norm_eps: spec.rms_norm_eps,
            attention_qk_norm: spec.attention_qk_norm,
        })
    }

    pub fn q_size(self) -> Result<usize> {
        checked_mul(self.head_count, self.head_dim, "q_size")
    }

    pub fn kv_size(self) -> Result<usize> {
        checked_mul(self.head_count_kv, self.head_dim, "kv_size")
    }

    pub fn wqkv_out(self) -> Result<usize> {
        self.q_size()?
            .checked_add(
                self.kv_size()?
                    .checked_mul(2)
                    .ok_or_else(|| InferError::Message("WQKV output size overflow".into()))?,
            )
            .ok_or_else(|| InferError::Message("WQKV output size overflow".into()))
    }

    pub fn attn_scale(self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    fn validate(self) -> Result<()> {
        if self.hidden_size == 0 || self.feed_forward_size == 0 {
            return Err(InferError::Message(
                "fast hidden/feed_forward size must be non-zero".into(),
            ));
        }
        if self.head_count_kv == 0 || !self.head_count.is_multiple_of(self.head_count_kv) {
            return Err(InferError::Message(format!(
                "invalid fast GQA split: heads={}, kv_heads={}",
                self.head_count, self.head_count_kv
            )));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(InferError::Message(format!(
                "fast head_dim must be non-zero and even, got {}",
                self.head_dim
            )));
        }
        if self.rope_base <= 0.0 {
            return Err(InferError::Message(format!(
                "fast rope_base must be positive, got {}",
                self.rope_base
            )));
        }
        self.q_size()?;
        self.kv_size()?;
        self.wqkv_out()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FastArLayerF16Weights {
    pub attention_norm: F16TensorView,
    pub wqkv: F16TensorView,
    pub output: F16TensorView,
    pub ffn_norm: F16TensorView,
    pub feed_forward_w1: F16TensorView,
    pub feed_forward_w2: F16TensorView,
    pub feed_forward_w3: F16TensorView,
}

impl FastArLayerF16Weights {
    pub fn from_gguf_layer(
        gguf: &GgufFile,
        registry: &TransformerTensorRegistry,
        layer: usize,
    ) -> Result<Self> {
        if layer >= registry.fast_layer_count() {
            return Err(InferError::Message(format!(
                "fast layer {layer} out of range (count={})",
                registry.fast_layer_count()
            )));
        }
        let names = registry.fast_layer(layer).ok_or_else(|| {
            InferError::Message(format!("missing fast layer {layer} in registry"))
        })?;
        let weights = Self {
            attention_norm: F16TensorView::from_gguf(gguf, &names.attention_norm)?,
            wqkv: F16TensorView::from_gguf(gguf, &names.attention_wqkv)?,
            output: F16TensorView::from_gguf(gguf, &names.attention_output)?,
            ffn_norm: F16TensorView::from_gguf(gguf, &names.ffn_norm)?,
            feed_forward_w1: F16TensorView::from_gguf(gguf, &names.feed_forward_w1)?,
            feed_forward_w2: F16TensorView::from_gguf(gguf, &names.feed_forward_w2)?,
            feed_forward_w3: F16TensorView::from_gguf(gguf, &names.feed_forward_w3)?,
        };
        weights.validate_shapes(layer)?;
        Ok(weights)
    }

    fn validate_shapes(&self, layer: usize) -> Result<()> {
        let hidden = usize::try_from(HIDDEN_SIZE)
            .map_err(|_| InferError::Message("HIDDEN_SIZE overflows usize".into()))?;
        let ffn = self.feed_forward_w1.dimensions();
        if ffn != [hidden, 9728] {
            return Err(InferError::Message(format!(
                "fast_layers.{layer}.feed_forward.w1.weight: expected [2560, 9728], got {ffn:?}"
            )));
        }
        Ok(())
    }

    fn skeleton<'a>(&'a self, shape: FastArLayerShape) -> FastArLayerSkeleton<'a> {
        FastArLayerSkeleton {
            shape,
            attention_norm_weight: self.attention_norm.values(),
            wqkv_weight: self.wqkv.values(),
            output_weight: self.output.values(),
            ffn_norm_weight: self.ffn_norm.values(),
            feed_forward_w1_weight: self.feed_forward_w1.values(),
            feed_forward_w2_weight: self.feed_forward_w2.values(),
            feed_forward_w3_weight: self.feed_forward_w3.values(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FastArHeadF16Weights {
    pub embeddings: F16TensorView,
    pub norm: F16TensorView,
    pub output: F16TensorView,
}

impl FastArHeadF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let weights = Self {
            embeddings: F16TensorView::from_gguf(gguf, "fast_embeddings.weight")?,
            norm: F16TensorView::from_gguf(gguf, "fast_norm.weight")?,
            output: F16TensorView::from_gguf(gguf, "fast_output.weight")?,
        };
        weights.validate_dimensions()?;
        Ok(weights)
    }

    fn validate_dimensions(&self) -> Result<()> {
        let emb = self.embeddings.dimensions();
        if emb != [HIDDEN_SIZE as usize, FAST_VOCAB_SIZE as usize] {
            return Err(InferError::Message(format!(
                "fast_embeddings.weight: expected [2560, 4096], got {emb:?}"
            )));
        }
        let out = self.output.dimensions();
        if out != [HIDDEN_SIZE as usize, FAST_VOCAB_SIZE as usize] {
            return Err(InferError::Message(format!(
                "fast_output.weight: expected [2560, 4096], got {out:?}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FastArWeights {
    pub layers: Vec<FastArLayerF16Weights>,
    pub head: FastArHeadF16Weights,
}

impl FastArWeights {
    pub fn from_gguf(gguf: &GgufFile, registry: &TransformerTensorRegistry) -> Result<Self> {
        let mut layers = Vec::with_capacity(registry.fast_layer_count());
        for layer in 0..registry.fast_layer_count() {
            layers.push(FastArLayerF16Weights::from_gguf_layer(
                gguf, registry, layer,
            )?);
        }
        Ok(Self {
            layers,
            head: FastArHeadF16Weights::from_gguf(gguf)?,
        })
    }

    pub fn open(
        transformer_path: impl AsRef<Path>,
    ) -> Result<(GgufFile, TransformerTensorRegistry, Self)> {
        let gguf = GgufFile::open(transformer_path.as_ref())
            .map_err(|err| InferError::Message(err.to_string()))?;
        let registry = TransformerTensorRegistry::from_gguf(&gguf)?;
        let weights = Self::from_gguf(&gguf, &registry)?;
        Ok((gguf, registry, weights))
    }
}

#[derive(Debug, Clone, Copy)]
struct FastArLayerSkeleton<'a> {
    shape: FastArLayerShape,
    attention_norm_weight: &'a [f32],
    wqkv_weight: &'a [f32],
    output_weight: &'a [f32],
    ffn_norm_weight: &'a [f32],
    feed_forward_w1_weight: &'a [f32],
    feed_forward_w2_weight: &'a [f32],
    feed_forward_w3_weight: &'a [f32],
}

struct FastPreparedToken {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl FastArLayerSkeleton<'_> {
    fn forward_prefill(&self, hidden_tokens: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if hidden_tokens.is_empty() {
            return Err(InferError::Message(
                "fast prefill requires at least one token".into(),
            ));
        }
        let mut prepared = Vec::with_capacity(hidden_tokens.len());
        for (position, hidden) in hidden_tokens.iter().enumerate() {
            prepared.push(self.prepare_token(hidden, position)?);
        }
        let mut outputs = Vec::with_capacity(hidden_tokens.len());
        for (offset, hidden) in hidden_tokens.iter().enumerate() {
            let token = &prepared[offset];
            let visible_count = offset + 1;
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for entry in prepared.iter().take(visible_count) {
                keys.extend_from_slice(&entry.key);
                values.extend_from_slice(&entry.value);
            }
            let attention = gqa_decode_attention(
                &token.query,
                &keys,
                &values,
                GqaAttentionShape {
                    head_count: self.shape.head_count,
                    head_count_kv: self.shape.head_count_kv,
                    head_dim: self.shape.head_dim,
                    token_count: visible_count,
                    attn_scale: self.shape.attn_scale(),
                },
            )?;
            let projected = linear(
                &attention,
                self.output_weight,
                self.shape.q_size()?,
                self.shape.hidden_size,
            )?;
            let out_hidden = hidden
                .iter()
                .zip(&projected)
                .map(|(residual, delta)| residual + delta)
                .collect::<Vec<_>>();
            let ff = self.forward_feed_forward(&out_hidden)?;
            outputs.push(ff.hidden);
        }
        Ok(outputs)
    }

    fn prepare_token(&self, hidden: &[f32], position: usize) -> Result<FastPreparedToken> {
        expect_len("hidden", hidden, self.shape.hidden_size)?;
        let normalized = rms_norm(hidden, self.attention_norm_weight, self.shape.rms_norm_eps)?;
        let qkv = linear(
            &normalized,
            self.wqkv_weight,
            self.shape.hidden_size,
            self.shape.wqkv_out()?,
        )?;
        let q_size = self.shape.q_size()?;
        let kv_size = self.shape.kv_size()?;
        let (query_raw, rest) = qkv.split_at(q_size);
        let (key_raw, value_raw) = rest.split_at(kv_size);
        let mut query = query_raw.to_vec();
        let mut key = key_raw.to_vec();
        let value = value_raw.to_vec();
        apply_rope_normal(
            &mut query,
            self.shape.head_dim,
            position,
            self.shape.rope_base,
        )?;
        apply_rope_normal(
            &mut key,
            self.shape.head_dim,
            position,
            self.shape.rope_base,
        )?;
        Ok(FastPreparedToken { query, key, value })
    }

    fn forward_feed_forward(&self, hidden: &[f32]) -> Result<FastFeedForwardOutput> {
        let normalized = rms_norm(hidden, self.ffn_norm_weight, self.shape.rms_norm_eps)?;
        let gate = linear(
            &normalized,
            self.feed_forward_w1_weight,
            self.shape.hidden_size,
            self.shape.feed_forward_size,
        )?;
        let up = linear(
            &normalized,
            self.feed_forward_w3_weight,
            self.shape.hidden_size,
            self.shape.feed_forward_size,
        )?;
        let activated = swiglu_split(&gate, &up)?;
        let projected = linear(
            &activated,
            self.feed_forward_w2_weight,
            self.shape.feed_forward_size,
            self.shape.hidden_size,
        )?;
        let hidden = hidden
            .iter()
            .zip(&projected)
            .map(|(residual, delta)| residual + delta)
            .collect();
        Ok(FastFeedForwardOutput { hidden })
    }
}

struct FastFeedForwardOutput {
    hidden: Vec<f32>,
}

/// Slow-AR last-token hidden + codebook-space prefix → logits over `codebook_size`.
pub fn forward_codebook_prefix(
    slow_hidden: &[f32],
    prefix_codes: &[u32],
    graph: &DualArGraphSpec,
    weights: &FastArWeights,
) -> Result<Vec<f32>> {
    if graph.fast_has_project_in {
        return Err(InferError::Message(
            "fast_project_in is not implemented in pure Rust yet".into(),
        ));
    }
    let num_codebooks = usize::try_from(graph.num_codebooks)
        .map_err(|_| InferError::Message("num_codebooks overflows usize".into()))?;
    if prefix_codes.len() >= num_codebooks {
        return Err(InferError::Message(format!(
            "fast prefix too long: {} codes (max {})",
            prefix_codes.len(),
            num_codebooks.saturating_sub(1)
        )));
    }
    let hidden_size = usize::try_from(graph.fast.embedding_length)
        .map_err(|_| InferError::Message("fast embedding_length overflows usize".into()))?;
    if slow_hidden.len() != hidden_size {
        return Err(InferError::Message(format!(
            "slow hidden size mismatch: expected {hidden_size}, got {}",
            slow_hidden.len()
        )));
    }
    let codebook_size = usize::try_from(graph.codebook_size)
        .map_err(|_| InferError::Message("codebook_size overflows usize".into()))?;
    let vocab_size = usize::try_from(FAST_VOCAB_SIZE)
        .map_err(|_| InferError::Message("FAST_VOCAB_SIZE overflows usize".into()))?;

    let mut sequence = Vec::with_capacity(1 + prefix_codes.len());
    sequence.push(slow_hidden.to_vec());
    if !prefix_codes.is_empty() {
        let rows = embedding_lookup_rows(
            weights.head.embeddings.values(),
            hidden_size,
            vocab_size,
            prefix_codes,
        )?;
        sequence.extend(rows);
    }

    let max_tokens = usize::try_from(graph.fast.context_length)
        .map_err(|_| InferError::Message("fast context_length overflows usize".into()))?;
    if sequence.len() > max_tokens {
        return Err(InferError::Message(format!(
            "fast sequence length {} exceeds context_length {max_tokens}",
            sequence.len()
        )));
    }

    let shape = FastArLayerShape::from_ar_graph_spec(&graph.fast)?;
    shape.validate()?;
    let mut hidden_tokens = sequence;
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        if layer_idx >= FAST_AR_LAYERS {
            break;
        }
        hidden_tokens = layer_weights
            .skeleton(shape)
            .forward_prefill(&hidden_tokens)?;
    }

    let last = hidden_tokens
        .last()
        .ok_or_else(|| InferError::Message("fast decoder produced no hidden states".into()))?;
    let normalized = rms_norm(last, weights.head.norm.values(), shape.rms_norm_eps)?;
    let logits = linear(
        &normalized,
        weights.head.output.values(),
        hidden_size,
        codebook_size,
    )?;
    Ok(logits)
}

/// Generate codebooks `1..num_codebooks` after semantic codebook index `semantic_code`.
pub fn generate_codebooks_for_semantic<R: RandomSource + ?Sized>(
    slow_hidden: &[f32],
    semantic_code: u32,
    graph: &DualArGraphSpec,
    weights: &FastArWeights,
    sampler: &SamplerParams,
    rng: &mut R,
) -> Result<Vec<u32>> {
    let codebook_size = graph.codebook_size;
    let num_codebooks = graph.num_codebooks;
    if semantic_code >= codebook_size {
        return Err(InferError::Message(format!(
            "semantic_code {semantic_code} >= codebook_size {codebook_size}"
        )));
    }
    let mut prefix = vec![semantic_code];
    let mut out = Vec::with_capacity(num_codebooks as usize);
    out.push(semantic_code);
    for _ in 1..num_codebooks {
        let logits = forward_codebook_prefix(slow_hidden, &prefix, graph, weights)?;
        let token = sample_token(&logits, sampler, None, rng)?;
        if token >= codebook_size {
            return Err(InferError::Message(format!(
                "sampled codebook token {token} >= codebook_size {codebook_size}"
            )));
        }
        prefix.push(token);
        out.push(token);
    }
    Ok(out)
}

fn swiglu_split(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(InferError::Message(format!(
            "swiglu length mismatch: gate={} up={}",
            gate.len(),
            up.len()
        )));
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect())
}

fn checked_mul(a: usize, b: usize, name: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| InferError::Message(format!("{name} overflow")))
}

fn usize_from_u32(value: u32, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| InferError::Message(format!("{name} overflows usize")))
}

fn expect_len(name: &str, actual: &[f32], expected: usize) -> Result<()> {
    if actual.len() != expected {
        return Err(InferError::Message(format!(
            "{name} length mismatch: expected {expected}, got {}",
            actual.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_transformer_path() -> Option<PathBuf> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/s2-pro-f16-transformer-only.gguf");
        path.exists().then_some(path)
    }

    #[test]
    #[ignore = "requires local s2-pro transformer GGUF in models/"]
    fn fast_layer_shape_matches_gguf_metadata() {
        let path = fixture_transformer_path().expect("transformer gguf");
        let gguf = GgufFile::open(&path).unwrap();
        let graph = DualArGraphSpec::from_gguf(&gguf).unwrap();
        let shape = FastArLayerShape::from_ar_graph_spec(&graph.fast).unwrap();
        assert_eq!(shape.head_count, 32);
        assert!(!shape.attention_qk_norm);
    }

    #[test]
    #[ignore = "requires local s2-pro transformer GGUF in models/"]
    fn forward_codebook_prefix_returns_codebook_logits() {
        let path = fixture_transformer_path().expect("transformer gguf");
        let gguf = GgufFile::open(&path).unwrap();
        let graph = DualArGraphSpec::from_gguf(&gguf).unwrap();
        let registry = TransformerTensorRegistry::from_gguf(&gguf).unwrap();
        let weights = FastArWeights::from_gguf(&gguf, &registry).unwrap();
        let hidden = vec![0.01f32; graph.fast.embedding_length as usize];
        let logits =
            forward_codebook_prefix(&hidden, &[42], &graph, &weights).expect("fast forward");
        assert_eq!(logits.len(), graph.codebook_size as usize);
        assert!(logits.iter().any(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "requires local s2-pro transformer GGUF in models/"]
    fn loads_all_fast_ar_layers_from_gguf() {
        let path = fixture_transformer_path().expect("transformer gguf");
        let gguf = GgufFile::open(&path).unwrap();
        let registry = TransformerTensorRegistry::from_gguf(&gguf).unwrap();
        assert_eq!(registry.fast_layer_count(), FAST_AR_LAYERS);
        let weights = FastArWeights::from_gguf(&gguf, &registry).unwrap();
        assert_eq!(weights.layers.len(), FAST_AR_LAYERS);
        for layer in 0..FAST_AR_LAYERS {
            let _ = &weights.layers[layer];
        }
    }
}
