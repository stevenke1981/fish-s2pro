use std::ops::AddAssign;
use std::path::Path;
use std::time::{Duration, Instant};

use fish_s2_core::gguf::GgufFile;

use crate::attention::{apply_rope_normal, SlowArKvCache};
use crate::backend::{CpuMatmulBackend, MatmulBackend};
use crate::error::{InferError, Result};
use crate::registry::{
    ArGraphSpec, DualArGraphSpec, TransformerTensorRegistry, SLOW_CONTEXT_LENGTH,
};
use crate::tensor::{
    embedding_lookup_rows, matvec_f16_streaming, rms_norm, F16TensorBytes, F16TensorView,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlowArLayerShape {
    pub hidden_size: usize,
    pub feed_forward_size: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub rms_norm_eps: f32,
}

impl SlowArLayerShape {
    pub fn from_ar_graph_spec(spec: &ArGraphSpec) -> Result<Self> {
        Ok(Self {
            hidden_size: usize_from_u32(spec.embedding_length, "hidden_size")?,
            feed_forward_size: usize_from_u32(spec.feed_forward_length, "feed_forward_size")?,
            head_count: usize_from_u32(spec.head_count, "head_count")?,
            head_count_kv: usize_from_u32(spec.head_count_kv, "head_count_kv")?,
            head_dim: usize_from_u32(spec.head_dim, "head_dim")?,
            rope_base: spec.rope_freq_base,
            rms_norm_eps: spec.rms_norm_eps,
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
                    .ok_or_else(|| InferError::Message("WQKV output size overflow".to_string()))?,
            )
            .ok_or_else(|| InferError::Message("WQKV output size overflow".to_string()))
    }

    pub fn attn_scale(self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    fn validate(self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(InferError::Message("hidden_size must be non-zero".into()));
        }
        if self.feed_forward_size == 0 {
            return Err(InferError::Message(
                "feed_forward_size must be non-zero".into(),
            ));
        }
        if self.head_count_kv == 0 || !self.head_count.is_multiple_of(self.head_count_kv) {
            return Err(InferError::Message(format!(
                "invalid GQA split: heads={}, kv_heads={}",
                self.head_count, self.head_count_kv
            )));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(InferError::Message(format!(
                "head_dim must be non-zero and even, got {}",
                self.head_dim
            )));
        }
        if self.rope_base <= 0.0 {
            return Err(InferError::Message(format!(
                "rope_base must be positive, got {}",
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
pub struct SlowArLayerF16Weights {
    pub attention_norm: F16TensorView,
    pub q_norm: F16TensorView,
    pub k_norm: F16TensorView,
    pub wqkv: F16TensorView,
    pub output: F16TensorView,
    pub ffn_norm: F16TensorView,
    pub feed_forward_w1: F16TensorView,
    pub feed_forward_w2: F16TensorView,
    pub feed_forward_w3: F16TensorView,
}

impl SlowArLayerF16Weights {
    pub fn from_gguf_layer(
        gguf: &GgufFile,
        registry: &TransformerTensorRegistry,
        layer: usize,
    ) -> Result<Self> {
        let names = registry
            .slow_layer(layer)
            .ok_or_else(|| InferError::Message(format!("slow layer not found: {layer}")))?;
        let shape = SlowArLayerShape::from_ar_graph_spec(&registry.graph_spec().slow)?;
        let weights = Self {
            attention_norm: F16TensorView::from_gguf(gguf, &names.attention_norm)?,
            q_norm: F16TensorView::from_gguf(gguf, &names.attention_q_norm)?,
            k_norm: F16TensorView::from_gguf(gguf, &names.attention_k_norm)?,
            wqkv: F16TensorView::from_gguf(gguf, &names.attention_wqkv)?,
            output: F16TensorView::from_gguf(gguf, &names.attention_output)?,
            ffn_norm: F16TensorView::from_gguf(gguf, &names.ffn_norm)?,
            feed_forward_w1: F16TensorView::from_gguf(gguf, &names.feed_forward_w1)?,
            feed_forward_w2: F16TensorView::from_gguf(gguf, &names.feed_forward_w2)?,
            feed_forward_w3: F16TensorView::from_gguf(gguf, &names.feed_forward_w3)?,
        };
        weights.validate_dimensions(shape)?;
        Ok(weights)
    }

    pub fn skeleton(&self, shape: SlowArLayerShape) -> SlowArLayerSkeleton<'_> {
        SlowArLayerSkeleton {
            shape,
            attention_norm_weight: self.attention_norm.values(),
            q_norm_weight: self.q_norm.values(),
            k_norm_weight: self.k_norm.values(),
            wqkv_weight: self.wqkv.values(),
            output_weight: self.output.values(),
            ffn_norm_weight: self.ffn_norm.values(),
            feed_forward_w1_weight: self.feed_forward_w1.values(),
            feed_forward_w2_weight: self.feed_forward_w2.values(),
            feed_forward_w3_weight: self.feed_forward_w3.values(),
        }
    }

    fn validate_dimensions(&self, shape: SlowArLayerShape) -> Result<()> {
        expect_dims(
            "attention_norm",
            self.attention_norm.dimensions(),
            &[shape.hidden_size],
        )?;
        expect_dims("q_norm", self.q_norm.dimensions(), &[shape.head_dim])?;
        expect_dims("k_norm", self.k_norm.dimensions(), &[shape.head_dim])?;
        expect_dims(
            "wqkv",
            self.wqkv.dimensions(),
            &[shape.hidden_size, shape.wqkv_out()?],
        )?;
        expect_dims(
            "attention_output",
            self.output.dimensions(),
            &[shape.q_size()?, shape.hidden_size],
        )?;
        expect_dims("ffn_norm", self.ffn_norm.dimensions(), &[shape.hidden_size])?;
        expect_dims(
            "feed_forward_w1",
            self.feed_forward_w1.dimensions(),
            &[shape.hidden_size, shape.feed_forward_size],
        )?;
        expect_dims(
            "feed_forward_w2",
            self.feed_forward_w2.dimensions(),
            &[shape.feed_forward_size, shape.hidden_size],
        )?;
        expect_dims(
            "feed_forward_w3",
            self.feed_forward_w3.dimensions(),
            &[shape.hidden_size, shape.feed_forward_size],
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SlowArLayerSkeleton<'a> {
    pub shape: SlowArLayerShape,
    pub attention_norm_weight: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub wqkv_weight: &'a [f32],
    pub output_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    pub feed_forward_w1_weight: &'a [f32],
    pub feed_forward_w2_weight: &'a [f32],
    pub feed_forward_w3_weight: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlowArLayerForwardOutput {
    pub normalized: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention: Vec<f32>,
    pub projected: Vec<f32>,
    pub hidden: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlowArLayerFeedForwardOutput {
    pub normalized: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub activated: Vec<f32>,
    pub projected: Vec<f32>,
    pub hidden: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlowArLayerBlockOutput {
    pub attention: SlowArLayerForwardOutput,
    pub feed_forward: SlowArLayerFeedForwardOutput,
    pub hidden: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlowArOutputHeadF16Weights {
    pub norm: F16TensorView,
    pub embeddings: F16TensorBytes,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlowArLogitsOutput {
    pub normalized: Vec<f32>,
    pub logits: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlowArStepResult {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlowArDecodeProfile {
    pub rms_norm: Duration,
    pub linear: Duration,
    pub gqa_attention: Duration,
    pub rope: Duration,
    pub kv_write: Duration,
    pub swiglu: Duration,
    pub output_head: Duration,
}

impl SlowArDecodeProfile {
    pub fn total(self) -> Duration {
        self.rms_norm
            + self.linear
            + self.gqa_attention
            + self.rope
            + self.kv_write
            + self.swiglu
            + self.output_head
    }

    pub fn top_ops(self, limit: usize) -> Vec<(&'static str, Duration)> {
        let mut items = vec![
            ("linear", self.linear),
            ("rms_norm", self.rms_norm),
            ("gqa_attention", self.gqa_attention),
            ("rope", self.rope),
            ("kv_write", self.kv_write),
            ("swiglu", self.swiglu),
            ("output_head", self.output_head),
        ];
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items.truncate(limit);
        items
    }

    fn add(&mut self, op: SlowArProfileOp, elapsed: Duration) {
        match op {
            SlowArProfileOp::RmsNorm => self.rms_norm += elapsed,
            SlowArProfileOp::Linear => self.linear += elapsed,
            SlowArProfileOp::GqaAttention => self.gqa_attention += elapsed,
            SlowArProfileOp::Rope => self.rope += elapsed,
            SlowArProfileOp::KvWrite => self.kv_write += elapsed,
            SlowArProfileOp::Swiglu => self.swiglu += elapsed,
            SlowArProfileOp::OutputHead => self.output_head += elapsed,
        }
    }
}

impl AddAssign for SlowArDecodeProfile {
    fn add_assign(&mut self, rhs: Self) {
        self.rms_norm += rhs.rms_norm;
        self.linear += rhs.linear;
        self.gqa_attention += rhs.gqa_attention;
        self.rope += rhs.rope;
        self.kv_write += rhs.kv_write;
        self.swiglu += rhs.swiglu;
        self.output_head += rhs.output_head;
    }
}

#[derive(Debug, Clone, Copy)]
enum SlowArProfileOp {
    RmsNorm,
    Linear,
    GqaAttention,
    Rope,
    KvWrite,
    Swiglu,
    OutputHead,
}

pub struct SlowArDecodeBackendContext<'a, B: MatmulBackend> {
    pub backend: &'a B,
    pub profile: Option<&'a mut SlowArDecodeProfile>,
}

impl<'a, B: MatmulBackend> SlowArDecodeBackendContext<'a, B> {
    pub fn new(backend: &'a B, profile: Option<&'a mut SlowArDecodeProfile>) -> Self {
        Self { backend, profile }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlowArDecodeLayerRequest {
    pub layer_start: usize,
    pub layer_count: usize,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlowArEmbeddingWeights {
    pub semantic: F16TensorView,
    pub codebook: F16TensorView,
}

impl SlowArEmbeddingWeights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let weights = Self {
            semantic: F16TensorView::from_gguf(gguf, "embeddings.weight")?,
            codebook: F16TensorView::from_gguf(gguf, "codebook_embeddings.weight")?,
        };
        weights.validate_dimensions()?;
        Ok(weights)
    }

    fn validate_dimensions(&self) -> Result<()> {
        let semantic_dims = self.semantic.dimensions();
        if semantic_dims.len() != 2 {
            return Err(InferError::Message(format!(
                "embeddings.weight expected rank 2, got {semantic_dims:?}"
            )));
        }
        let codebook_dims = self.codebook.dimensions();
        if codebook_dims.len() != 2 || codebook_dims[0] != semantic_dims[0] {
            return Err(InferError::Message(format!(
                "codebook_embeddings.weight shape mismatch: semantic={semantic_dims:?} codebook={codebook_dims:?}"
            )));
        }
        Ok(())
    }
}

pub struct SlowArState {
    gguf: GgufFile,
    registry: TransformerTensorRegistry,
    graph: DualArGraphSpec,
    shape: SlowArLayerShape,
    embeddings: SlowArEmbeddingWeights,
    output_head: SlowArOutputHeadF16Weights,
    cache: SlowArKvCache,
    backend: CpuMatmulBackend,
    decode_profile: SlowArDecodeProfile,
    n_past: usize,
    max_seq_len: usize,
}

impl SlowArState {
    pub fn open(transformer_path: impl AsRef<Path>, max_seq_len: usize) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(InferError::Message("max_seq_len must be non-zero".into()));
        }
        let gguf = GgufFile::open(transformer_path.as_ref())
            .map_err(|err| InferError::Message(err.to_string()))?;
        let registry = TransformerTensorRegistry::from_gguf(&gguf)?;
        let graph = registry.graph_spec().clone();
        let shape = SlowArLayerShape::from_ar_graph_spec(&graph.slow)?;
        let embeddings = SlowArEmbeddingWeights::from_gguf(&gguf)?;
        let output_head = SlowArOutputHeadF16Weights::from_gguf(&gguf)?;
        let cache = SlowArKvCache::new(graph.kv_cache, max_seq_len)?;
        Ok(Self {
            gguf,
            registry,
            graph,
            shape,
            embeddings,
            output_head,
            cache,
            backend: CpuMatmulBackend::new(),
            decode_profile: SlowArDecodeProfile::default(),
            n_past: 0,
            max_seq_len,
        })
    }

    pub fn open_default_max_seq_len(transformer_path: impl AsRef<Path>) -> Result<Self> {
        Self::open(transformer_path, SLOW_CONTEXT_LENGTH as usize)
    }

    pub fn graph_spec(&self) -> &DualArGraphSpec {
        &self.graph
    }

    pub fn n_past(&self) -> usize {
        self.n_past
    }

    pub fn decode_profile(&self) -> SlowArDecodeProfile {
        self.decode_profile
    }

    pub fn reset_decode_profile(&mut self) {
        self.decode_profile = SlowArDecodeProfile::default();
    }

    pub fn reset(&mut self) {
        self.n_past = 0;
        let _ = SlowArKvCache::new(self.graph.kv_cache, self.max_seq_len)
            .map(|cache| self.cache = cache);
    }

    pub fn prefill(&mut self, flat_tokens: &[i32]) -> Result<SlowArStepResult> {
        self.eval(flat_tokens)
    }

    pub fn step(&mut self, flat_token: &[i32]) -> Result<SlowArStepResult> {
        let codebook_dim = usize::try_from(self.graph.codebook_input_dim())
            .map_err(|_| InferError::Message("codebook_input_dim overflows usize".into()))?;
        if flat_token.len() != codebook_dim {
            return Err(InferError::Message(format!(
                "step token width mismatch: expected {codebook_dim}, got {}",
                flat_token.len()
            )));
        }
        self.eval(flat_token)
    }

    fn eval(&mut self, flat_tokens: &[i32]) -> Result<SlowArStepResult> {
        let codebook_dim = usize::try_from(self.graph.codebook_input_dim())
            .map_err(|_| InferError::Message("codebook_input_dim overflows usize".into()))?;
        if !flat_tokens.len().is_multiple_of(codebook_dim) {
            return Err(InferError::Message(format!(
                "flat token length {} is not a multiple of codebook_dim {codebook_dim}",
                flat_tokens.len()
            )));
        }
        let n_tokens = flat_tokens.len() / codebook_dim;
        if n_tokens == 0 {
            return Err(InferError::Message("expected at least one token".into()));
        }
        if self.n_past + n_tokens > self.max_seq_len {
            return Err(InferError::Message(format!(
                "KV cache overflow: n_past={} n_tokens={} max_seq_len={}",
                self.n_past, n_tokens, self.max_seq_len
            )));
        }

        self.reset_decode_profile();
        let hidden_tokens = embed_slow_ar_time_major(flat_tokens, &self.graph, &self.embeddings)?;
        let start_position = self.n_past;
        let outputs = if n_tokens == 1 {
            let hidden = hidden_tokens
                .first()
                .ok_or_else(|| InferError::Message("missing embedded token".into()))?;
            let mut context =
                SlowArDecodeBackendContext::new(&self.backend, Some(&mut self.decode_profile));
            let block = forward_slow_ar_block_decode_layers_with_backend(
                &self.gguf,
                &self.registry,
                SlowArDecodeLayerRequest {
                    layer_start: 0,
                    layer_count: self.registry.slow_layer_count(),
                    position: start_position,
                },
                hidden,
                &mut self.cache,
                &mut context,
            )?;
            vec![block]
        } else {
            forward_slow_ar_block_prefill_layers_cached(
                &self.gguf,
                &self.registry,
                0,
                self.registry.slow_layer_count(),
                &hidden_tokens,
                &mut self.cache,
                start_position,
            )?
        };

        let last_hidden = outputs
            .last()
            .ok_or_else(|| InferError::Message("Slow-AR produced no outputs".into()))?
            .hidden
            .clone();
        // Match s2.cpp eval_cached: StepResult.hidden is last-token output after weights_.norm.
        let output_head_start = Instant::now();
        let logits_out = self
            .output_head
            .forward_logits(&last_hidden, self.shape.rms_norm_eps)?;
        self.decode_profile
            .add(SlowArProfileOp::OutputHead, output_head_start.elapsed());
        self.n_past = checked_add(self.n_past, n_tokens, "n_past")?;
        Ok(SlowArStepResult {
            hidden: logits_out.normalized,
            logits: logits_out.logits,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SlowArLayerPreparedToken {
    normalized: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl SlowArLayerSkeleton<'_> {
    pub fn forward_decode_sequence(
        &self,
        hidden_tokens: &[Vec<f32>],
        cache: &mut SlowArKvCache,
        layer: usize,
        start_position: usize,
    ) -> Result<Vec<SlowArLayerForwardOutput>> {
        let mut outputs = Vec::with_capacity(hidden_tokens.len());
        for (offset, hidden) in hidden_tokens.iter().enumerate() {
            outputs.push(self.forward_decode_token(
                hidden,
                cache,
                layer,
                position_for_offset(start_position, offset)?,
            )?);
        }
        Ok(outputs)
    }

    pub fn forward_prefill_sequence(
        &self,
        hidden_tokens: &[Vec<f32>],
        cache: &mut SlowArKvCache,
        layer: usize,
        start_position: usize,
    ) -> Result<Vec<SlowArLayerForwardOutput>> {
        let mut prepared = Vec::with_capacity(hidden_tokens.len());
        for (offset, hidden) in hidden_tokens.iter().enumerate() {
            prepared.push(
                self.prepare_decode_token(hidden, position_for_offset(start_position, offset)?)?,
            );
        }
        for (offset, token) in prepared.iter().enumerate() {
            cache.write_token(
                layer,
                position_for_offset(start_position, offset)?,
                &token.key,
                &token.value,
            )?;
        }
        let mut outputs = Vec::with_capacity(hidden_tokens.len());
        for (offset, (hidden, token)) in hidden_tokens.iter().zip(prepared).enumerate() {
            let visible_token_count = checked_add(
                position_for_offset(start_position, offset)?,
                1,
                "visible_token_count",
            )?;
            outputs.push(self.finish_decode_token(
                hidden,
                token,
                cache,
                layer,
                visible_token_count,
            )?);
        }
        Ok(outputs)
    }

    pub fn forward_block_prefill_sequence(
        &self,
        hidden_tokens: &[Vec<f32>],
        cache: &mut SlowArKvCache,
        layer: usize,
        start_position: usize,
    ) -> Result<Vec<SlowArLayerBlockOutput>> {
        let attention_outputs =
            self.forward_prefill_sequence(hidden_tokens, cache, layer, start_position)?;
        let mut outputs = Vec::with_capacity(attention_outputs.len());
        for attention in attention_outputs {
            let feed_forward = self.forward_feed_forward(&attention.hidden)?;
            let hidden = feed_forward.hidden.clone();
            outputs.push(SlowArLayerBlockOutput {
                attention,
                feed_forward,
                hidden,
            });
        }
        Ok(outputs)
    }

    pub fn forward_decode_token(
        &self,
        hidden: &[f32],
        cache: &mut SlowArKvCache,
        layer: usize,
        position: usize,
    ) -> Result<SlowArLayerForwardOutput> {
        let backend = CpuMatmulBackend::new();
        let mut context = SlowArDecodeBackendContext::new(&backend, None);
        self.forward_decode_token_with_backend(&mut context, hidden, cache, layer, position)
    }

    pub fn forward_feed_forward(&self, hidden: &[f32]) -> Result<SlowArLayerFeedForwardOutput> {
        let backend = CpuMatmulBackend::new();
        let mut context = SlowArDecodeBackendContext::new(&backend, None);
        self.forward_feed_forward_with_backend(&mut context, hidden)
    }

    pub fn forward_decode_token_with_backend<B: MatmulBackend>(
        &self,
        context: &mut SlowArDecodeBackendContext<'_, B>,
        hidden: &[f32],
        cache: &mut SlowArKvCache,
        layer: usize,
        position: usize,
    ) -> Result<SlowArLayerForwardOutput> {
        let prepared = self.prepare_decode_token_with_backend(context, hidden, position)?;
        profile_decode_op(&mut context.profile, SlowArProfileOp::KvWrite, || {
            cache.write_token(layer, position, &prepared.key, &prepared.value)
        })?;
        self.finish_decode_token_with_backend(
            context,
            hidden,
            prepared,
            cache,
            layer,
            checked_add(position, 1, "visible_token_count")?,
        )
    }

    pub fn forward_feed_forward_with_backend<B: MatmulBackend>(
        &self,
        context: &mut SlowArDecodeBackendContext<'_, B>,
        hidden: &[f32],
    ) -> Result<SlowArLayerFeedForwardOutput> {
        self.validate(hidden)?;
        let normalized = profile_decode_op(&mut context.profile, SlowArProfileOp::RmsNorm, || {
            context
                .backend
                .rms_norm(hidden, self.ffn_norm_weight, self.shape.rms_norm_eps)
        })?;
        let gate = profile_decode_op(&mut context.profile, SlowArProfileOp::Linear, || {
            context.backend.linear(
                &normalized,
                self.feed_forward_w1_weight,
                self.shape.hidden_size,
                self.shape.feed_forward_size,
            )
        })?;
        let up = profile_decode_op(&mut context.profile, SlowArProfileOp::Linear, || {
            context.backend.linear(
                &normalized,
                self.feed_forward_w3_weight,
                self.shape.hidden_size,
                self.shape.feed_forward_size,
            )
        })?;
        let activated = profile_decode_op(&mut context.profile, SlowArProfileOp::Swiglu, || {
            swiglu_split(&gate, &up)
        })?;
        let projected = profile_decode_op(&mut context.profile, SlowArProfileOp::Linear, || {
            context.backend.linear(
                &activated,
                self.feed_forward_w2_weight,
                self.shape.feed_forward_size,
                self.shape.hidden_size,
            )
        })?;
        let hidden = hidden
            .iter()
            .zip(&projected)
            .map(|(residual, projected)| residual + projected)
            .collect();
        Ok(SlowArLayerFeedForwardOutput {
            normalized,
            gate,
            up,
            activated,
            projected,
            hidden,
        })
    }

    fn prepare_decode_token(
        &self,
        hidden: &[f32],
        position: usize,
    ) -> Result<SlowArLayerPreparedToken> {
        let backend = CpuMatmulBackend::new();
        let mut context = SlowArDecodeBackendContext::new(&backend, None);
        self.prepare_decode_token_with_backend(&mut context, hidden, position)
    }

    fn prepare_decode_token_with_backend<B: MatmulBackend>(
        &self,
        context: &mut SlowArDecodeBackendContext<'_, B>,
        hidden: &[f32],
        position: usize,
    ) -> Result<SlowArLayerPreparedToken> {
        self.validate(hidden)?;

        let normalized = profile_decode_op(&mut context.profile, SlowArProfileOp::RmsNorm, || {
            context
                .backend
                .rms_norm(hidden, self.attention_norm_weight, self.shape.rms_norm_eps)
        })?;
        let qkv = profile_decode_op(&mut context.profile, SlowArProfileOp::Linear, || {
            context.backend.linear(
                &normalized,
                self.wqkv_weight,
                self.shape.hidden_size,
                self.shape.wqkv_out()?,
            )
        })?;
        let q_size = self.shape.q_size()?;
        let kv_size = self.shape.kv_size()?;
        let (query_raw, rest) = qkv.split_at(q_size);
        let (key_raw, value_raw) = rest.split_at(kv_size);

        let mut query = profile_decode_op(&mut context.profile, SlowArProfileOp::RmsNorm, || {
            context.backend.rms_norm_heads(
                query_raw,
                self.q_norm_weight,
                self.shape.head_dim,
                self.shape.rms_norm_eps,
            )
        })?;
        let mut key = profile_decode_op(&mut context.profile, SlowArProfileOp::RmsNorm, || {
            context.backend.rms_norm_heads(
                key_raw,
                self.k_norm_weight,
                self.shape.head_dim,
                self.shape.rms_norm_eps,
            )
        })?;
        let value = value_raw.to_vec();

        profile_decode_op(&mut context.profile, SlowArProfileOp::Rope, || {
            apply_rope_normal(
                &mut query,
                self.shape.head_dim,
                position,
                self.shape.rope_base,
            )
        })?;
        profile_decode_op(&mut context.profile, SlowArProfileOp::Rope, || {
            apply_rope_normal(
                &mut key,
                self.shape.head_dim,
                position,
                self.shape.rope_base,
            )
        })?;

        Ok(SlowArLayerPreparedToken {
            normalized,
            query,
            key,
            value,
        })
    }

    fn finish_decode_token(
        &self,
        hidden: &[f32],
        prepared: SlowArLayerPreparedToken,
        cache: &SlowArKvCache,
        layer: usize,
        visible_token_count: usize,
    ) -> Result<SlowArLayerForwardOutput> {
        let backend = CpuMatmulBackend::new();
        let mut context = SlowArDecodeBackendContext::new(&backend, None);
        self.finish_decode_token_with_backend(
            &mut context,
            hidden,
            prepared,
            cache,
            layer,
            visible_token_count,
        )
    }

    fn finish_decode_token_with_backend<B: MatmulBackend>(
        &self,
        context: &mut SlowArDecodeBackendContext<'_, B>,
        hidden: &[f32],
        prepared: SlowArLayerPreparedToken,
        cache: &SlowArKvCache,
        layer: usize,
        visible_token_count: usize,
    ) -> Result<SlowArLayerForwardOutput> {
        let attention =
            profile_decode_op(&mut context.profile, SlowArProfileOp::GqaAttention, || {
                cache.decode_attention_with_backend(
                    context.backend,
                    layer,
                    visible_token_count,
                    &prepared.query,
                    self.shape.head_count,
                    self.shape.attn_scale(),
                )
            })?;
        let projected = profile_decode_op(&mut context.profile, SlowArProfileOp::Linear, || {
            context.backend.linear(
                &attention,
                self.output_weight,
                self.shape.q_size()?,
                self.shape.hidden_size,
            )
        })?;
        let hidden = hidden
            .iter()
            .zip(&projected)
            .map(|(residual, projected)| residual + projected)
            .collect();

        Ok(SlowArLayerForwardOutput {
            normalized: prepared.normalized,
            query: prepared.query,
            key: prepared.key,
            value: prepared.value,
            attention,
            projected,
            hidden,
        })
    }

    fn validate(&self, hidden: &[f32]) -> Result<()> {
        self.shape.validate()?;
        expect_len("hidden", hidden, self.shape.hidden_size)?;
        expect_len(
            "attention_norm_weight",
            self.attention_norm_weight,
            self.shape.hidden_size,
        )?;
        expect_len("q_norm_weight", self.q_norm_weight, self.shape.head_dim)?;
        expect_len("k_norm_weight", self.k_norm_weight, self.shape.head_dim)?;
        expect_len(
            "wqkv_weight",
            self.wqkv_weight,
            checked_mul(
                self.shape.hidden_size,
                self.shape.wqkv_out()?,
                "wqkv_weight",
            )?,
        )?;
        expect_len(
            "output_weight",
            self.output_weight,
            checked_mul(
                self.shape.q_size()?,
                self.shape.hidden_size,
                "output_weight",
            )?,
        )?;
        expect_len(
            "ffn_norm_weight",
            self.ffn_norm_weight,
            self.shape.hidden_size,
        )?;
        expect_len(
            "feed_forward_w1_weight",
            self.feed_forward_w1_weight,
            checked_mul(
                self.shape.hidden_size,
                self.shape.feed_forward_size,
                "feed_forward_w1_weight",
            )?,
        )?;
        expect_len(
            "feed_forward_w2_weight",
            self.feed_forward_w2_weight,
            checked_mul(
                self.shape.feed_forward_size,
                self.shape.hidden_size,
                "feed_forward_w2_weight",
            )?,
        )?;
        expect_len(
            "feed_forward_w3_weight",
            self.feed_forward_w3_weight,
            checked_mul(
                self.shape.hidden_size,
                self.shape.feed_forward_size,
                "feed_forward_w3_weight",
            )?,
        )?;
        Ok(())
    }
}

fn profile_decode_op<T>(
    profile: &mut Option<&mut SlowArDecodeProfile>,
    op: SlowArProfileOp,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let start = Instant::now();
    let result = f();
    if let Some(profile) = profile.as_deref_mut() {
        profile.add(op, start.elapsed());
    }
    result
}

impl SlowArOutputHeadF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let weights = Self {
            norm: F16TensorView::from_gguf(gguf, "norm.weight")?,
            embeddings: F16TensorBytes::from_gguf(gguf, "embeddings.weight")?,
        };
        weights.validate_dimensions()?;
        Ok(weights)
    }

    pub fn forward_logits(&self, hidden: &[f32], rms_norm_eps: f32) -> Result<SlowArLogitsOutput> {
        self.validate_dimensions()?;
        let hidden_size = self.norm.dimensions()[0];
        let vocab_size = self.embeddings.dimensions()[1];
        let normalized = rms_norm(hidden, self.norm.values(), rms_norm_eps)?;
        let logits = matvec_f16_streaming(
            &normalized,
            self.embeddings.bytes(),
            hidden_size,
            vocab_size,
        )?;
        Ok(SlowArLogitsOutput { normalized, logits })
    }

    fn validate_dimensions(&self) -> Result<()> {
        let norm_dims = self.norm.dimensions();
        if norm_dims.len() != 1 {
            return Err(InferError::Message(format!(
                "norm.weight dimensions mismatch: expected rank 1, got {norm_dims:?}"
            )));
        }
        let embedding_dims = self.embeddings.dimensions();
        if embedding_dims.len() != 2 {
            return Err(InferError::Message(format!(
                "embeddings.weight dimensions mismatch: expected rank 2, got {embedding_dims:?}"
            )));
        }
        if embedding_dims[0] != norm_dims[0] {
            return Err(InferError::Message(format!(
                "output head hidden size mismatch: norm={} embeddings={}",
                norm_dims[0], embedding_dims[0]
            )));
        }
        Ok(())
    }
}

pub fn forward_slow_ar_block_prefill_layers(
    gguf: &GgufFile,
    registry: &TransformerTensorRegistry,
    layer_start: usize,
    layer_count: usize,
    hidden_tokens: &[Vec<f32>],
    start_position: usize,
) -> Result<Vec<SlowArLayerBlockOutput>> {
    let graph = registry.graph_spec();
    let max_seq_len = checked_add(start_position, hidden_tokens.len(), "max_seq_len")?;
    let mut cache = SlowArKvCache::new(graph.kv_cache, max_seq_len)?;
    forward_slow_ar_block_prefill_layers_cached(
        gguf,
        registry,
        layer_start,
        layer_count,
        hidden_tokens,
        &mut cache,
        start_position,
    )
}

pub fn forward_slow_ar_block_prefill_layers_cached(
    gguf: &GgufFile,
    registry: &TransformerTensorRegistry,
    layer_start: usize,
    layer_count: usize,
    hidden_tokens: &[Vec<f32>],
    cache: &mut SlowArKvCache,
    start_position: usize,
) -> Result<Vec<SlowArLayerBlockOutput>> {
    if layer_count == 0 {
        return Err(InferError::Message(
            "layer_count must be greater than zero".into(),
        ));
    }
    let layer_end = checked_add(layer_start, layer_count, "layer_end")?;
    if layer_end > registry.slow_layer_count() {
        return Err(InferError::Message(format!(
            "slow layer range out of bounds: start={layer_start} count={layer_count} available={}",
            registry.slow_layer_count()
        )));
    }

    let graph = registry.graph_spec();
    let shape = SlowArLayerShape::from_ar_graph_spec(&graph.slow)?;
    let mut layer_hidden_tokens = hidden_tokens.to_vec();
    let mut outputs = Vec::new();
    for layer in layer_start..layer_end {
        let weights = SlowArLayerF16Weights::from_gguf_layer(gguf, registry, layer)?;
        outputs = weights.skeleton(shape).forward_block_prefill_sequence(
            &layer_hidden_tokens,
            cache,
            layer,
            start_position,
        )?;
        layer_hidden_tokens = outputs.iter().map(|output| output.hidden.clone()).collect();
    }
    Ok(outputs)
}

pub fn embed_slow_ar_time_major(
    flat_tokens: &[i32],
    graph: &DualArGraphSpec,
    weights: &SlowArEmbeddingWeights,
) -> Result<Vec<Vec<f32>>> {
    let codebook_dim = usize::try_from(graph.codebook_input_dim())
        .map_err(|_| InferError::Message("codebook_input_dim overflows usize".into()))?;
    if !flat_tokens.len().is_multiple_of(codebook_dim) {
        return Err(InferError::Message(format!(
            "flat token length {} is not a multiple of codebook_dim {codebook_dim}",
            flat_tokens.len()
        )));
    }
    let n_tokens = flat_tokens.len() / codebook_dim;
    let hidden_dim = weights.semantic.dimensions()[0];
    let vocab_dim = weights.semantic.dimensions()[1];
    let codebook_vocab = weights.codebook.dimensions()[1];
    let num_codebooks = usize::try_from(graph.num_codebooks)
        .map_err(|_| InferError::Message("num_codebooks overflows usize".into()))?;
    let _codebook_size = usize::try_from(graph.codebook_size)
        .map_err(|_| InferError::Message("codebook_size overflows usize".into()))?;
    let sem_scale = if graph.scale_codebook_embeddings {
        1.0f32 / (codebook_dim as f32).sqrt()
    } else {
        1.0f32
    };

    let mut hidden_tokens = Vec::with_capacity(n_tokens);
    for token_index in 0..n_tokens {
        let base = token_index * codebook_dim;
        let semantic = flat_tokens[base];
        if semantic < 0 {
            return Err(InferError::Message(format!(
                "semantic token id must be non-negative, got {semantic}"
            )));
        }
        let semantic_id = semantic as u32;
        let is_semantic =
            semantic_id >= graph.semantic_begin_id && semantic_id <= graph.semantic_end_id;

        let mut hidden = embedding_lookup_rows(
            weights.semantic.values(),
            hidden_dim,
            vocab_dim,
            &[semantic_id],
        )?
        .pop()
        .ok_or_else(|| InferError::Message("semantic embedding row missing".into()))?;

        if is_semantic {
            let mut codebook_sum = vec![0.0f32; hidden_dim];
            for codebook_index in 0..num_codebooks {
                let raw = flat_tokens[base + 1 + codebook_index];
                if raw < 0 {
                    return Err(InferError::Message(format!(
                        "codebook token must be non-negative, got {raw}"
                    )));
                }
                let codebook_offset = u32::try_from(codebook_index)
                    .map_err(|_| InferError::Message("codebook index overflows u32".into()))?
                    .checked_mul(graph.codebook_size)
                    .ok_or_else(|| InferError::Message("codebook offset overflow".into()))?;
                let codebook_id = (raw as u32).checked_add(codebook_offset).ok_or_else(|| {
                    InferError::Message("codebook embedding index overflow".into())
                })?;
                let row = embedding_lookup_rows(
                    weights.codebook.values(),
                    hidden_dim,
                    codebook_vocab,
                    &[codebook_id],
                )?
                .pop()
                .ok_or_else(|| InferError::Message("codebook embedding row missing".into()))?;
                for (slot, value) in codebook_sum.iter_mut().zip(row) {
                    *slot += value;
                }
            }
            for (slot, value) in hidden.iter_mut().zip(codebook_sum) {
                *slot += value;
            }
        }

        if graph.scale_codebook_embeddings {
            let scale = if is_semantic { sem_scale } else { 1.0 };
            for value in &mut hidden {
                *value *= scale;
            }
        }

        hidden_tokens.push(hidden);
    }
    Ok(hidden_tokens)
}

pub fn forward_slow_ar_block_decode_layers(
    gguf: &GgufFile,
    registry: &TransformerTensorRegistry,
    layer_start: usize,
    layer_count: usize,
    hidden: &[f32],
    cache: &mut SlowArKvCache,
    position: usize,
) -> Result<SlowArLayerBlockOutput> {
    let backend = CpuMatmulBackend::new();
    let mut profile = SlowArDecodeProfile::default();
    let mut context = SlowArDecodeBackendContext::new(&backend, Some(&mut profile));
    forward_slow_ar_block_decode_layers_with_backend(
        gguf,
        registry,
        SlowArDecodeLayerRequest {
            layer_start,
            layer_count,
            position,
        },
        hidden,
        cache,
        &mut context,
    )
}

pub fn forward_slow_ar_block_decode_layers_with_backend<B: MatmulBackend>(
    gguf: &GgufFile,
    registry: &TransformerTensorRegistry,
    request: SlowArDecodeLayerRequest,
    hidden: &[f32],
    cache: &mut SlowArKvCache,
    context: &mut SlowArDecodeBackendContext<'_, B>,
) -> Result<SlowArLayerBlockOutput> {
    if request.layer_count == 0 {
        return Err(InferError::Message(
            "layer_count must be greater than zero".into(),
        ));
    }
    let layer_end = checked_add(request.layer_start, request.layer_count, "layer_end")?;
    if layer_end > registry.slow_layer_count() {
        return Err(InferError::Message(format!(
            "slow layer range out of bounds: start={} count={} available={}",
            request.layer_start,
            request.layer_count,
            registry.slow_layer_count()
        )));
    }

    let graph = registry.graph_spec();
    let shape = SlowArLayerShape::from_ar_graph_spec(&graph.slow)?;
    let mut current_hidden = hidden.to_vec();
    let mut last_output = None;
    for layer in request.layer_start..layer_end {
        let weights = SlowArLayerF16Weights::from_gguf_layer(gguf, registry, layer)?;
        let attention = weights.skeleton(shape).forward_decode_token_with_backend(
            context,
            &current_hidden,
            cache,
            layer,
            request.position,
        )?;
        let feed_forward = weights
            .skeleton(shape)
            .forward_feed_forward_with_backend(context, &attention.hidden)?;
        let hidden = feed_forward.hidden.clone();
        last_output = Some(SlowArLayerBlockOutput {
            attention,
            feed_forward,
            hidden,
        });
        current_hidden = last_output
            .as_ref()
            .expect("layer output present")
            .hidden
            .clone();
    }
    last_output.ok_or_else(|| InferError::Message("decode produced no layer outputs".into()))
}

fn swiglu_split(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(InferError::Message(format!(
            "SwiGLU length mismatch: gate={} up={}",
            gate.len(),
            up.len()
        )));
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(gate, up)| gate * sigmoid(*gate) * up)
        .collect())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn expect_dims(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        Err(InferError::Message(format!(
            "{name} dimensions mismatch: expected {:?}, got {:?}",
            expected, actual
        )))
    } else {
        Ok(())
    }
}

fn expect_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        Err(InferError::Message(format!(
            "{name} length mismatch: expected {expected}, got {}",
            values.len()
        )))
    } else {
        Ok(())
    }
}

fn checked_mul(a: usize, b: usize, name: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| InferError::Message(format!("{name} overflow")))
}

fn checked_add(a: usize, b: usize, name: &str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| InferError::Message(format!("{name} overflow")))
}

fn position_for_offset(start_position: usize, offset: usize) -> Result<usize> {
    checked_add(start_position, offset, "position")
}

fn usize_from_u32(value: u32, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| InferError::Message(format!("{name} overflows usize")))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::registry::KvCacheSpec;
    use fish_s2_core::gguf::GgmlType;

    #[test]
    fn slow_ar_layer_skeleton_runs_single_token_decode_flow() {
        let shape = toy_shape();
        let hidden = [1.0, 0.0, 0.0, 0.0];
        let attention_norm = [0.5, 1.0, 1.0, 1.0];
        let head_norm = [std::f32::consts::FRAC_1_SQRT_2; 2];
        let wqkv = output_major_weight(
            shape.hidden_size,
            shape.wqkv_out().unwrap(),
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0; 4],
                vec![0.0; 4],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0; 4],
                vec![3.0, 0.0, 0.0, 0.0],
                vec![0.0; 4],
            ],
        );
        let output = output_major_weight(
            shape.q_size().unwrap(),
            shape.hidden_size,
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 2.0, 0.0],
                vec![0.0; 4],
                vec![0.0; 4],
            ],
        );
        let ffn_norm = [1.0; 4];
        let feed_forward_w1 = vec![0.0; shape.hidden_size * shape.feed_forward_size];
        let feed_forward_w2 = vec![0.0; shape.feed_forward_size * shape.hidden_size];
        let feed_forward_w3 = vec![0.0; shape.hidden_size * shape.feed_forward_size];
        let layer = SlowArLayerSkeleton {
            shape,
            attention_norm_weight: &attention_norm,
            q_norm_weight: &head_norm,
            k_norm_weight: &head_norm,
            wqkv_weight: &wqkv,
            output_weight: &output,
            ffn_norm_weight: &ffn_norm,
            feed_forward_w1_weight: &feed_forward_w1,
            feed_forward_w2_weight: &feed_forward_w2,
            feed_forward_w3_weight: &feed_forward_w3,
        };
        let spec = KvCacheSpec {
            ggml_type: GgmlType::F16,
            head_dim: shape.head_dim as u32,
            head_count_kv: shape.head_count_kv as u32,
            block_count: 1,
        };
        let mut cache = SlowArKvCache::new(spec, 1).unwrap();

        let actual = layer
            .forward_decode_token(&hidden, &mut cache, 0, 0)
            .unwrap();

        assert_close(&actual.normalized, &[1.0, 0.0, 0.0, 0.0]);
        assert_close(&actual.query, &[1.0, 0.0, 0.0, 1.0]);
        assert_close(&actual.key, &[1.0, 0.0]);
        assert_close(&actual.value, &[3.0, 0.0]);
        assert_close(cache.key_token(0, 0).unwrap(), &[1.0, 0.0]);
        assert_close(cache.value_token(0, 0).unwrap(), &[3.0, 0.0]);
        assert_close(&actual.attention, &[3.0, 0.0, 3.0, 0.0]);
        assert_close(&actual.projected, &[3.0, 6.0, 0.0, 0.0]);
        assert_close(&actual.hidden, &[4.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn slow_ar_decode_hot_ops_route_through_backend_trait() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingBackend {
            linear: AtomicUsize,
            rms_norm: AtomicUsize,
            gqa_attention: AtomicUsize,
        }

        impl MatmulBackend for CountingBackend {
            fn name(&self) -> &'static str {
                "counting"
            }

            fn linear(
                &self,
                input: &[f32],
                weight: &[f32],
                input_dim: usize,
                output_dim: usize,
            ) -> Result<Vec<f32>> {
                self.linear.fetch_add(1, Ordering::Relaxed);
                CpuMatmulBackend::new().linear(input, weight, input_dim, output_dim)
            }

            fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
                self.rms_norm.fetch_add(1, Ordering::Relaxed);
                CpuMatmulBackend::new().rms_norm(input, weight, eps)
            }

            fn rms_norm_heads(
                &self,
                input: &[f32],
                weight: &[f32],
                head_dim: usize,
                eps: f32,
            ) -> Result<Vec<f32>> {
                self.rms_norm.fetch_add(1, Ordering::Relaxed);
                CpuMatmulBackend::new().rms_norm_heads(input, weight, head_dim, eps)
            }

            fn gqa_decode_attention(
                &self,
                query_heads: &[f32],
                key_tokens: &[f32],
                value_tokens: &[f32],
                shape: crate::attention::GqaAttentionShape,
            ) -> Result<Vec<f32>> {
                self.gqa_attention.fetch_add(1, Ordering::Relaxed);
                CpuMatmulBackend::new().gqa_decode_attention(
                    query_heads,
                    key_tokens,
                    value_tokens,
                    shape,
                )
            }
        }

        let shape = toy_shape();
        let spec = KvCacheSpec {
            ggml_type: GgmlType::F16,
            head_dim: shape.head_dim as u32,
            head_count_kv: shape.head_count_kv as u32,
            block_count: 1,
        };
        let mut cache = SlowArKvCache::new(spec, 1).unwrap();
        let weights = toy_attention_weights(shape);
        let layer = weights.skeleton(shape);
        let backend = CountingBackend {
            linear: AtomicUsize::new(0),
            rms_norm: AtomicUsize::new(0),
            gqa_attention: AtomicUsize::new(0),
        };
        let mut profile = SlowArDecodeProfile::default();

        let actual = {
            let mut context = SlowArDecodeBackendContext::new(&backend, Some(&mut profile));
            layer
                .forward_decode_token_with_backend(
                    &mut context,
                    &[1.0, 0.0, 0.0, 0.0],
                    &mut cache,
                    0,
                    0,
                )
                .unwrap()
        };

        assert_close(&actual.hidden, &[4.0, 6.0, 0.0, 0.0]);
        assert_eq!(backend.linear.load(Ordering::Relaxed), 2);
        assert_eq!(backend.rms_norm.load(Ordering::Relaxed), 3);
        assert_eq!(backend.gqa_attention.load(Ordering::Relaxed), 1);
        let names = profile
            .top_ops(7)
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"linear"));
        assert!(names.contains(&"rms_norm"));
        assert!(names.contains(&"gqa_attention"));
    }

    #[test]
    fn slow_ar_layer_skeleton_runs_multi_token_decode_sequence() {
        let shape = toy_shape();
        let spec = KvCacheSpec {
            ggml_type: GgmlType::F16,
            head_dim: shape.head_dim as u32,
            head_count_kv: shape.head_count_kv as u32,
            block_count: 1,
        };
        let hidden_tokens = vec![vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 0.0, 0.0]];
        let mut cache = SlowArKvCache::new(spec, 2).unwrap();
        let weights = toy_attention_weights(shape);
        let layer = weights.skeleton(shape);

        let outputs = layer
            .forward_decode_sequence(&hidden_tokens, &mut cache, 0, 0)
            .unwrap();
        let mut prefill_cache = SlowArKvCache::new(spec, 2).unwrap();
        let prefill_outputs = layer
            .forward_prefill_sequence(&hidden_tokens, &mut prefill_cache, 0, 0)
            .unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(prefill_outputs, outputs);
        assert_close(&outputs[0].key, cache.key_token(0, 0).unwrap());
        assert_close(&outputs[0].value, cache.value_token(0, 0).unwrap());
        assert_close(&outputs[1].key, cache.key_token(0, 1).unwrap());
        assert_close(&outputs[1].value, cache.value_token(0, 1).unwrap());
        assert_close(
            prefill_cache.key_token(0, 0).unwrap(),
            cache.key_token(0, 0).unwrap(),
        );
        assert_close(
            prefill_cache.value_token(0, 1).unwrap(),
            cache.value_token(0, 1).unwrap(),
        );
        assert!(all_finite(&outputs[1].attention));
        assert!(all_finite(&outputs[1].hidden));
        assert_ne!(outputs[0].hidden, outputs[1].hidden);
    }

    #[test]
    fn slow_ar_layer_skeleton_runs_block_prefill_with_ffn_residual() {
        let shape = toy_shape();
        let spec = KvCacheSpec {
            ggml_type: GgmlType::F16,
            head_dim: shape.head_dim as u32,
            head_count_kv: shape.head_count_kv as u32,
            block_count: 1,
        };
        let mut cache = SlowArKvCache::new(spec, 1).unwrap();
        let weights = toy_attention_weights(shape);
        let layer = weights.skeleton(shape);
        let outputs = layer
            .forward_block_prefill_sequence(&[vec![1.0, 0.0, 0.0, 0.0]], &mut cache, 0, 0)
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_close(&outputs[0].attention.hidden, &[4.0, 6.0, 0.0, 0.0]);
        assert_close(&outputs[0].feed_forward.projected, &[0.0, 0.0, 0.0, 0.0]);
        assert_close(&outputs[0].hidden, &outputs[0].attention.hidden);
    }

    #[test]
    fn swiglu_split_applies_silu_gate() {
        let actual = swiglu_split(&[0.0, 1.0], &[2.0, 3.0]).unwrap();
        assert_close(&actual, &[0.0, 3.0 / (1.0 + (-1.0f32).exp())]);
    }

    #[test]
    fn slow_ar_layer_skeleton_rejects_bad_weight_shapes() {
        let shape = toy_shape();
        let ffn_norm = [1.0; 4];
        let feed_forward_w1 = vec![0.0; shape.hidden_size * shape.feed_forward_size];
        let feed_forward_w2 = vec![0.0; shape.feed_forward_size * shape.hidden_size];
        let feed_forward_w3 = vec![0.0; shape.hidden_size * shape.feed_forward_size];
        let layer = SlowArLayerSkeleton {
            shape,
            attention_norm_weight: &[1.0; 4],
            q_norm_weight: &[1.0; 2],
            k_norm_weight: &[1.0; 2],
            wqkv_weight: &[0.0; 3],
            output_weight: &[0.0; 16],
            ffn_norm_weight: &ffn_norm,
            feed_forward_w1_weight: &feed_forward_w1,
            feed_forward_w2_weight: &feed_forward_w2,
            feed_forward_w3_weight: &feed_forward_w3,
        };
        let spec = KvCacheSpec {
            ggml_type: GgmlType::F16,
            head_dim: shape.head_dim as u32,
            head_count_kv: shape.head_count_kv as u32,
            block_count: 1,
        };
        let mut cache = SlowArKvCache::new(spec, 1).unwrap();
        let err = layer
            .forward_decode_token(&[0.0; 4], &mut cache, 0, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("wqkv_weight length mismatch"));
    }

    #[test]
    #[ignore = "requires local s2-pro transformer GGUF in models/"]
    fn binds_local_layer0_f16_weights_and_runs_single_token_fixture() {
        let path = local_model_dir().join("s2-pro-f16-transformer-only.gguf");
        let gguf = GgufFile::open(path).unwrap();
        let registry = TransformerTensorRegistry::from_gguf(&gguf).unwrap();
        let graph = registry.graph_spec();
        let shape = SlowArLayerShape::from_ar_graph_spec(&graph.slow).unwrap();
        let weights = SlowArLayerF16Weights::from_gguf_layer(&gguf, &registry, 0).unwrap();

        let mut hidden = vec![0.0f32; shape.hidden_size];
        hidden[0] = 1.0;
        hidden[1] = -0.5;
        hidden[shape.hidden_size - 1] = 0.25;
        let mut cache = SlowArKvCache::new(graph.kv_cache, 1).unwrap();

        let actual = weights
            .skeleton(shape)
            .forward_decode_token(&hidden, &mut cache, 0, 0)
            .unwrap();

        assert_eq!(actual.normalized.len(), shape.hidden_size);
        assert_eq!(actual.query.len(), shape.q_size().unwrap());
        assert_eq!(actual.key.len(), shape.kv_size().unwrap());
        assert_eq!(actual.value.len(), shape.kv_size().unwrap());
        assert_eq!(actual.attention.len(), shape.q_size().unwrap());
        assert_eq!(actual.projected.len(), shape.hidden_size);
        assert_eq!(actual.hidden.len(), shape.hidden_size);
        assert!(all_finite(&actual.normalized));
        assert!(all_finite(&actual.query));
        assert!(all_finite(&actual.key));
        assert!(all_finite(&actual.value));
        assert!(all_finite(&actual.attention));
        assert!(all_finite(&actual.projected));
        assert!(all_finite(&actual.hidden));
        assert_close(cache.key_token(0, 0).unwrap(), &actual.key);
        assert_close(cache.value_token(0, 0).unwrap(), &actual.value);

        let mut block_cache = SlowArKvCache::new(graph.kv_cache, 1).unwrap();
        let block = weights
            .skeleton(shape)
            .forward_block_prefill_sequence(&[hidden.clone()], &mut block_cache, 0, 0)
            .unwrap()
            .remove(0);
        assert_eq!(block.feed_forward.normalized.len(), shape.hidden_size);
        assert_eq!(block.feed_forward.gate.len(), shape.feed_forward_size);
        assert_eq!(block.feed_forward.up.len(), shape.feed_forward_size);
        assert_eq!(block.feed_forward.activated.len(), shape.feed_forward_size);
        assert_eq!(block.feed_forward.projected.len(), shape.hidden_size);
        assert_eq!(block.hidden.len(), shape.hidden_size);
        assert!(all_finite(&block.feed_forward.normalized));
        assert!(all_finite(&block.feed_forward.gate));
        assert!(all_finite(&block.feed_forward.up));
        assert!(all_finite(&block.feed_forward.activated));
        assert!(all_finite(&block.feed_forward.projected));
        assert!(all_finite(&block.hidden));

        let multi_layer =
            forward_slow_ar_block_prefill_layers(&gguf, &registry, 0, 2, &[hidden], 0).unwrap();
        assert_eq!(multi_layer.len(), 1);
        assert_eq!(multi_layer[0].hidden.len(), shape.hidden_size);
        assert!(all_finite(&multi_layer[0].attention.hidden));
        assert!(all_finite(&multi_layer[0].feed_forward.projected));
        assert!(all_finite(&multi_layer[0].hidden));
        assert_ne!(multi_layer[0].hidden, block.hidden);
    }

    struct ToyLayerWeights {
        attention_norm: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        wqkv: Vec<f32>,
        output: Vec<f32>,
        ffn_norm: Vec<f32>,
        feed_forward_w1: Vec<f32>,
        feed_forward_w2: Vec<f32>,
        feed_forward_w3: Vec<f32>,
    }

    impl ToyLayerWeights {
        fn skeleton(&self, shape: SlowArLayerShape) -> SlowArLayerSkeleton<'_> {
            SlowArLayerSkeleton {
                shape,
                attention_norm_weight: &self.attention_norm,
                q_norm_weight: &self.q_norm,
                k_norm_weight: &self.k_norm,
                wqkv_weight: &self.wqkv,
                output_weight: &self.output,
                ffn_norm_weight: &self.ffn_norm,
                feed_forward_w1_weight: &self.feed_forward_w1,
                feed_forward_w2_weight: &self.feed_forward_w2,
                feed_forward_w3_weight: &self.feed_forward_w3,
            }
        }
    }

    fn toy_shape() -> SlowArLayerShape {
        SlowArLayerShape {
            hidden_size: 4,
            feed_forward_size: 2,
            head_count: 2,
            head_count_kv: 1,
            head_dim: 2,
            rope_base: 10_000.0,
            rms_norm_eps: 0.0,
        }
    }

    fn toy_attention_weights(shape: SlowArLayerShape) -> ToyLayerWeights {
        ToyLayerWeights {
            attention_norm: vec![0.5, 1.0, 1.0, 1.0],
            q_norm: vec![std::f32::consts::FRAC_1_SQRT_2; 2],
            k_norm: vec![std::f32::consts::FRAC_1_SQRT_2; 2],
            wqkv: output_major_weight(
                shape.hidden_size,
                shape.wqkv_out().unwrap(),
                &[
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0; 4],
                    vec![0.0; 4],
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0; 4],
                    vec![3.0, 0.0, 0.0, 0.0],
                    vec![0.0; 4],
                ],
            ),
            output: output_major_weight(
                shape.q_size().unwrap(),
                shape.hidden_size,
                &[
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 2.0, 0.0],
                    vec![0.0; 4],
                    vec![0.0; 4],
                ],
            ),
            ffn_norm: vec![1.0; shape.hidden_size],
            feed_forward_w1: vec![0.0; shape.hidden_size * shape.feed_forward_size],
            feed_forward_w2: vec![0.0; shape.feed_forward_size * shape.hidden_size],
            feed_forward_w3: vec![0.0; shape.hidden_size * shape.feed_forward_size],
        }
    }

    fn output_major_weight(input_dim: usize, output_dim: usize, rows: &[Vec<f32>]) -> Vec<f32> {
        assert_eq!(rows.len(), output_dim);
        let mut values = Vec::with_capacity(input_dim * output_dim);
        for row in rows {
            assert_eq!(row.len(), input_dim);
            values.extend_from_slice(row);
        }
        values
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected {expected}, got {actual}"
            );
        }
    }

    fn all_finite(values: &[f32]) -> bool {
        values.iter().all(|value| value.is_finite())
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
