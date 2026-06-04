//! Codec GGUF tensor registry.
//!
//! This module indexes the codec-only GGUF directory without reading tensor
//! payloads. The first RVQ slice needs stable names/shapes before decode math.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use fish_s2_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};

use crate::attention::{apply_rope_normal, gqa_decode_attention, GqaAttentionShape};
use crate::error::{InferError, Result};
use crate::tensor::{embedding_lookup_rows, linear, rms_norm, F16TensorView};

pub const CODEC_ARCHITECTURE: &str = "fish-speech-codec";
pub const CODEC_HIDDEN_SIZE: u64 = 1024;
pub const CODEC_PROJECTION_DIM: u64 = 8;
pub const CODEC_SEMANTIC_CODEBOOK_SIZE: u64 = 4096;
pub const CODEC_RESIDUAL_CODEBOOK_SIZE: u64 = 1024;
pub const CODEC_RESIDUAL_QUANTIZERS: usize = 9;
pub const CODEC_TRANSFORMER_LAYERS: usize = 8;
pub const CODEC_ATTENTION_WQKV_OUT: u64 = 3072;
pub const CODEC_FEED_FORWARD_SIZE: u64 = 3072;
pub const CODEC_CONTEXT_LENGTH: u64 = 4096;
pub const CODEC_FREQ_HEADS: u64 = 32;
pub const CODEC_RVQ_HEAD_DIM: usize = 64;
pub const CODEC_RVQ_LOCAL_HEADS: usize = 16;
pub const CODEC_RVQ_ROPE_BASE: f32 = 10_000.0;
pub const CODEC_RVQ_NORM_EPS: f32 = 1e-5;
pub const CODEC_RVQ_WINDOW_SIZE: usize = 128;

#[derive(Debug, Clone)]
pub struct CodecTensorRegistry {
    pub architecture: String,
    pub tensor_count: usize,
    pub metadata: Vec<(String, String)>,
    tensors: BTreeMap<String, GgufTensorInfo>,
    ordered_tensors: Vec<GgufTensorInfo>,
    prefix_counts: BTreeMap<String, usize>,
    semantic_quantizer: CodecQuantizerWeights,
    residual_quantizers: Vec<CodecQuantizerWeights>,
    pre_module_layers: Vec<CodecTransformerLayerWeights>,
    post_module_layers: Vec<CodecTransformerLayerWeights>,
}

impl CodecTensorRegistry {
    pub fn from_gguf_file(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufFile::open(path).map_err(|err| InferError::Message(err.to_string()))?;
        Self::from_gguf(&gguf)
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = metadata_value(gguf, "general.architecture")
            .ok_or_else(|| InferError::Message("missing general.architecture".into()))?
            .to_string();
        if architecture != CODEC_ARCHITECTURE {
            return Err(InferError::Message(format!(
                "expected {CODEC_ARCHITECTURE} codec GGUF, got {architecture}"
            )));
        }

        let tensors = gguf
            .tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), tensor.clone()))
            .collect::<BTreeMap<_, _>>();
        let ordered_tensors = gguf.tensors.clone();
        let prefix_counts = prefix_counts(&ordered_tensors);
        let semantic_quantizer = CodecQuantizerWeights::semantic();
        let residual_quantizers = (0..CODEC_RESIDUAL_QUANTIZERS)
            .map(CodecQuantizerWeights::residual)
            .collect::<Vec<_>>();
        let pre_module_layers = (0..CODEC_TRANSFORMER_LAYERS)
            .map(|layer| CodecTransformerLayerWeights::new("quantizer.pre_module", layer))
            .collect::<Vec<_>>();
        let post_module_layers = (0..CODEC_TRANSFORMER_LAYERS)
            .map(|layer| CodecTransformerLayerWeights::new("quantizer.post_module", layer))
            .collect::<Vec<_>>();

        let registry = Self {
            architecture,
            tensor_count: tensors.len(),
            metadata: gguf.metadata.clone(),
            tensors,
            ordered_tensors,
            prefix_counts,
            semantic_quantizer,
            residual_quantizers,
            pre_module_layers,
            post_module_layers,
        };
        registry.validate()?;
        Ok(registry)
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.ordered_tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
    }

    pub fn prefix_counts(&self) -> &BTreeMap<String, usize> {
        &self.prefix_counts
    }

    pub fn semantic_quantizer(&self) -> &CodecQuantizerWeights {
        &self.semantic_quantizer
    }

    pub fn residual_quantizers(&self) -> &[CodecQuantizerWeights] {
        &self.residual_quantizers
    }

    pub fn pre_module_layers(&self) -> &[CodecTransformerLayerWeights] {
        &self.pre_module_layers
    }

    pub fn post_module_layers(&self) -> &[CodecTransformerLayerWeights] {
        &self.post_module_layers
    }

    pub fn dump_rows(&self, tensor_data_start: u64) -> Result<Vec<CodecTensorDumpRow>> {
        self.ordered_tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                let role = classify_codec_tensor(&tensor.name);
                Ok(CodecTensorDumpRow {
                    index,
                    component: role.component,
                    role: role.role,
                    module: role.module,
                    layer: role.layer,
                    quantizer_index: role.quantizer_index,
                    name: tensor.name.clone(),
                    ggml_type: tensor.ggml_type,
                    dimensions: tensor.dimensions.clone(),
                    elements: tensor
                        .element_count()
                        .map_err(|err| InferError::Message(err.to_string()))?,
                    bytes: tensor
                        .byte_len()
                        .map_err(|err| InferError::Message(err.to_string()))?,
                    relative_offset: tensor.relative_offset,
                    absolute_offset: tensor.absolute_offset(tensor_data_start),
                })
            })
            .collect()
    }

    fn validate(&self) -> Result<()> {
        let mut failures = Vec::new();
        let expected_prefixes = [
            ("encoder", 128usize),
            ("quantizer", 244usize),
            ("decoder", 89usize),
        ];
        for (prefix, expected) in expected_prefixes {
            let actual = self.prefix_counts.get(prefix).copied().unwrap_or(0);
            if actual != expected {
                failures.push(format!(
                    "{prefix} tensor count: expected {expected}, got {actual}"
                ));
            }
        }

        validate_quantizer(&self.tensors, &self.semantic_quantizer, true, &mut failures);
        for quantizer in &self.residual_quantizers {
            validate_quantizer(&self.tensors, quantizer, false, &mut failures);
        }
        validate_module(
            &self.tensors,
            "quantizer.pre_module",
            &self.pre_module_layers,
            &mut failures,
        );
        validate_module(
            &self.tensors,
            "quantizer.post_module",
            &self.post_module_layers,
            &mut failures,
        );

        if self.residual_layer_set() != (0..CODEC_RESIDUAL_QUANTIZERS).collect::<BTreeSet<_>>() {
            failures.push(format!(
                "residual quantizer layer set mismatch: {:?}",
                self.residual_layer_set()
            ));
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(InferError::Message(format!(
                "codec tensor registry validation failed:\n{}",
                failures.join("\n")
            )))
        }
    }

    fn residual_layer_set(&self) -> BTreeSet<usize> {
        self.tensors
            .keys()
            .filter_map(|name| {
                name.strip_prefix("quantizer.quantizer.quantizers.")
                    .and_then(|rest| rest.split_once('.'))
                    .and_then(|(layer, _)| layer.parse::<usize>().ok())
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecQuantizerWeights {
    pub index: usize,
    pub prefix: String,
    pub in_proj_weight: String,
    pub in_proj_bias: String,
    pub out_proj_weight: String,
    pub out_proj_bias: String,
    pub codebook_weight: String,
}

impl CodecQuantizerWeights {
    pub fn semantic() -> Self {
        Self::new("quantizer.semantic_quantizer.quantizers.0", 0)
    }

    pub fn residual(index: usize) -> Self {
        Self::new(format!("quantizer.quantizer.quantizers.{index}"), index)
    }

    fn new(prefix: impl Into<String>, index: usize) -> Self {
        let prefix = prefix.into();
        Self {
            index,
            in_proj_weight: format!("{prefix}.in_proj.weight"),
            in_proj_bias: format!("{prefix}.in_proj.bias"),
            out_proj_weight: format!("{prefix}.out_proj.weight"),
            out_proj_bias: format!("{prefix}.out_proj.bias"),
            codebook_weight: format!("{prefix}.codebook.weight"),
            prefix,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecTransformerLayerWeights {
    pub module: String,
    pub layer: usize,
    pub attention_wqkv: String,
    pub attention_output: String,
    pub attention_norm: String,
    pub ffn_norm: String,
    pub feed_forward_w1: String,
    pub feed_forward_w2: String,
    pub feed_forward_w3: String,
    pub attention_layer_scale: String,
    pub ffn_layer_scale: String,
}

impl CodecTransformerLayerWeights {
    pub fn new(module: impl Into<String>, layer: usize) -> Self {
        let module = module.into();
        let prefix = format!("{module}.layers.{layer}");
        Self {
            module,
            layer,
            attention_wqkv: format!("{prefix}.attention.wqkv.weight"),
            attention_output: format!("{prefix}.attention.wo.weight"),
            attention_norm: format!("{prefix}.attention_norm.weight"),
            ffn_norm: format!("{prefix}.ffn_norm.weight"),
            feed_forward_w1: format!("{prefix}.feed_forward.w1.weight"),
            feed_forward_w2: format!("{prefix}.feed_forward.w2.weight"),
            feed_forward_w3: format!("{prefix}.feed_forward.w3.weight"),
            attention_layer_scale: format!("{prefix}.attention_layer_scale.gamma"),
            ffn_layer_scale: format!("{prefix}.ffn_layer_scale.gamma"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecPostModuleF16Weights {
    pub layers: Vec<CodecTransformerLayerF16Weights>,
    pub norm_weight: F16TensorView,
}

impl CodecPostModuleF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let registry = CodecTensorRegistry::from_gguf(gguf)?;
        Self::from_gguf_registry(gguf, &registry)
    }

    pub fn from_gguf_registry(gguf: &GgufFile, registry: &CodecTensorRegistry) -> Result<Self> {
        let layers = registry
            .post_module_layers()
            .iter()
            .map(|names| CodecTransformerLayerF16Weights::from_names(gguf, names))
            .collect::<Result<Vec<_>>>()?;
        let weights = Self {
            layers,
            norm_weight: F16TensorView::from_gguf(gguf, "quantizer.post_module.norm.weight")?,
        };
        weights.validate_dimensions()?;
        Ok(weights)
    }

    fn validate_dimensions(&self) -> Result<()> {
        if self.layers.len() != CODEC_TRANSFORMER_LAYERS {
            return Err(InferError::Message(format!(
                "codec post_module layer count mismatch: expected {}, got {}",
                CODEC_TRANSFORMER_LAYERS,
                self.layers.len()
            )));
        }
        validate_f16_dims(
            self.norm_weight.name(),
            self.norm_weight.dimensions(),
            &[CODEC_HIDDEN_SIZE as usize],
        )?;
        for layer in &self.layers {
            layer.validate_dimensions()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecTransformerLayerF16Weights {
    pub module: String,
    pub layer: usize,
    pub attention_wqkv: F16TensorView,
    pub attention_output: F16TensorView,
    pub attention_norm: F16TensorView,
    pub ffn_norm: F16TensorView,
    pub feed_forward_w1: F16TensorView,
    pub feed_forward_w2: F16TensorView,
    pub feed_forward_w3: F16TensorView,
    pub attention_layer_scale: F16TensorView,
    pub ffn_layer_scale: F16TensorView,
}

impl CodecTransformerLayerF16Weights {
    fn from_names(gguf: &GgufFile, names: &CodecTransformerLayerWeights) -> Result<Self> {
        Ok(Self {
            module: names.module.clone(),
            layer: names.layer,
            attention_wqkv: F16TensorView::from_gguf(gguf, &names.attention_wqkv)?,
            attention_output: F16TensorView::from_gguf(gguf, &names.attention_output)?,
            attention_norm: F16TensorView::from_gguf(gguf, &names.attention_norm)?,
            ffn_norm: F16TensorView::from_gguf(gguf, &names.ffn_norm)?,
            feed_forward_w1: F16TensorView::from_gguf(gguf, &names.feed_forward_w1)?,
            feed_forward_w2: F16TensorView::from_gguf(gguf, &names.feed_forward_w2)?,
            feed_forward_w3: F16TensorView::from_gguf(gguf, &names.feed_forward_w3)?,
            attention_layer_scale: F16TensorView::from_gguf(gguf, &names.attention_layer_scale)?,
            ffn_layer_scale: F16TensorView::from_gguf(gguf, &names.ffn_layer_scale)?,
        })
    }

    fn validate_dimensions(&self) -> Result<()> {
        let specs = [
            (
                self.attention_wqkv.name(),
                self.attention_wqkv.dimensions(),
                vec![
                    CODEC_HIDDEN_SIZE as usize,
                    CODEC_ATTENTION_WQKV_OUT as usize,
                ],
            ),
            (
                self.attention_output.name(),
                self.attention_output.dimensions(),
                vec![CODEC_HIDDEN_SIZE as usize, CODEC_HIDDEN_SIZE as usize],
            ),
            (
                self.feed_forward_w1.name(),
                self.feed_forward_w1.dimensions(),
                vec![CODEC_HIDDEN_SIZE as usize, CODEC_FEED_FORWARD_SIZE as usize],
            ),
            (
                self.feed_forward_w3.name(),
                self.feed_forward_w3.dimensions(),
                vec![CODEC_HIDDEN_SIZE as usize, CODEC_FEED_FORWARD_SIZE as usize],
            ),
            (
                self.feed_forward_w2.name(),
                self.feed_forward_w2.dimensions(),
                vec![CODEC_FEED_FORWARD_SIZE as usize, CODEC_HIDDEN_SIZE as usize],
            ),
            (
                self.ffn_norm.name(),
                self.ffn_norm.dimensions(),
                vec![CODEC_HIDDEN_SIZE as usize],
            ),
            (
                self.attention_norm.name(),
                self.attention_norm.dimensions(),
                vec![CODEC_HIDDEN_SIZE as usize],
            ),
            (
                self.attention_layer_scale.name(),
                self.attention_layer_scale.dimensions(),
                vec![CODEC_HIDDEN_SIZE as usize],
            ),
            (
                self.ffn_layer_scale.name(),
                self.ffn_layer_scale.dimensions(),
                vec![CODEC_HIDDEN_SIZE as usize],
            ),
        ];
        for (name, actual, expected) in specs {
            validate_f16_dims(name, actual, &expected)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecPostModuleResult {
    pub n_frames: u32,
    pub hidden_dim: usize,
    pub hidden: Vec<f32>,
}

pub fn forward_codec_post_module(
    latents: &[f32],
    n_frames: u32,
    weights: &CodecPostModuleF16Weights,
) -> Result<CodecPostModuleResult> {
    let n_frames_usize = usize::try_from(n_frames)
        .map_err(|_| InferError::Message("n_frames overflows usize".into()))?;
    let hidden_dim = CODEC_HIDDEN_SIZE as usize;
    let expected_len = n_frames_usize
        .checked_mul(hidden_dim)
        .ok_or_else(|| InferError::Message("post_module input length overflow".into()))?;
    if latents.len() != expected_len {
        return Err(InferError::Message(format!(
            "post_module input length mismatch: expected {expected_len}, got {}",
            latents.len()
        )));
    }
    let mut tokens = latents
        .chunks_exact(hidden_dim)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    for layer in &weights.layers {
        tokens = forward_codec_transformer_layer(&tokens, layer)?;
    }
    for token in &mut tokens {
        *token = rms_norm(token, weights.norm_weight.values(), CODEC_RVQ_NORM_EPS)?;
    }
    Ok(CodecPostModuleResult {
        n_frames,
        hidden_dim,
        hidden: tokens.into_iter().flatten().collect(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecF16Weights {
    pub semantic_quantizer: CodecQuantizerF16Weights,
    pub residual_quantizers: Vec<CodecQuantizerF16Weights>,
}

impl CodecF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let registry = CodecTensorRegistry::from_gguf(gguf)?;
        Self::from_gguf_registry(gguf, &registry)
    }

    pub fn from_gguf_registry(gguf: &GgufFile, registry: &CodecTensorRegistry) -> Result<Self> {
        let semantic_quantizer =
            CodecQuantizerF16Weights::from_names(gguf, registry.semantic_quantizer(), true)?;
        let residual_quantizers = registry
            .residual_quantizers()
            .iter()
            .map(|names| CodecQuantizerF16Weights::from_names(gguf, names, false))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            semantic_quantizer,
            residual_quantizers,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecQuantizerF16Weights {
    pub index: usize,
    pub in_proj_weight: F16TensorView,
    pub in_proj_bias: F16TensorView,
    pub out_proj_weight: F16TensorView,
    pub out_proj_bias: F16TensorView,
    pub codebook_weight: F16TensorView,
}

impl CodecQuantizerF16Weights {
    fn from_names(gguf: &GgufFile, names: &CodecQuantizerWeights, semantic: bool) -> Result<Self> {
        let weights = Self {
            index: names.index,
            in_proj_weight: F16TensorView::from_gguf(gguf, &names.in_proj_weight)?,
            in_proj_bias: F16TensorView::from_gguf(gguf, &names.in_proj_bias)?,
            out_proj_weight: F16TensorView::from_gguf(gguf, &names.out_proj_weight)?,
            out_proj_bias: F16TensorView::from_gguf(gguf, &names.out_proj_bias)?,
            codebook_weight: F16TensorView::from_gguf(gguf, &names.codebook_weight)?,
        };
        weights.validate_dimensions(semantic)?;
        Ok(weights)
    }

    fn validate_dimensions(&self, semantic: bool) -> Result<()> {
        let codebook_size = if semantic {
            CODEC_SEMANTIC_CODEBOOK_SIZE as usize
        } else {
            CODEC_RESIDUAL_CODEBOOK_SIZE as usize
        };
        let expected = [
            (
                self.in_proj_weight.name(),
                self.in_proj_weight.dimensions(),
                vec![1, CODEC_HIDDEN_SIZE as usize, CODEC_PROJECTION_DIM as usize],
            ),
            (
                self.in_proj_bias.name(),
                self.in_proj_bias.dimensions(),
                vec![CODEC_PROJECTION_DIM as usize],
            ),
            (
                self.out_proj_weight.name(),
                self.out_proj_weight.dimensions(),
                vec![1, CODEC_PROJECTION_DIM as usize, CODEC_HIDDEN_SIZE as usize],
            ),
            (
                self.out_proj_bias.name(),
                self.out_proj_bias.dimensions(),
                vec![CODEC_HIDDEN_SIZE as usize],
            ),
            (
                self.codebook_weight.name(),
                self.codebook_weight.dimensions(),
                vec![CODEC_PROJECTION_DIM as usize, codebook_size],
            ),
        ];
        let mut failures = Vec::new();
        for (name, actual, expected) in expected {
            if actual != expected {
                failures.push(format!("{name}: expected {expected:?}, got {actual:?}"));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(InferError::Message(format!(
                "codec F16 quantizer shape validation failed:\n{}",
                failures.join("\n")
            )))
        }
    }

    pub fn project_code(&self, code_id: u32, codebook_size: usize) -> Result<Vec<f32>> {
        let code = embedding_lookup_rows(
            self.codebook_weight.values(),
            CODEC_PROJECTION_DIM as usize,
            codebook_size,
            &[code_id],
        )?
        .pop()
        .ok_or_else(|| InferError::Message("codec codebook lookup returned no row".into()))?;
        let mut projected = linear(
            &code,
            self.out_proj_weight.values(),
            CODEC_PROJECTION_DIM as usize,
            CODEC_HIDDEN_SIZE as usize,
        )?;
        add_bias(&mut projected, self.out_proj_bias.values())?;
        Ok(projected)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecRvqLookupResult {
    pub num_codebooks: u32,
    pub n_frames: u32,
    pub latent_dim: usize,
    pub latents: Vec<f32>,
}

pub fn rvq_lookup_codes(
    codes: &[i32],
    num_codebooks: u32,
    n_frames: u32,
    weights: &CodecF16Weights,
) -> Result<CodecRvqLookupResult> {
    let expected_codebooks = (1 + weights.residual_quantizers.len()) as u32;
    if num_codebooks != expected_codebooks {
        return Err(InferError::Message(format!(
            "codec codebook count mismatch: expected {expected_codebooks}, got {num_codebooks}"
        )));
    }
    let num_codebooks_usize = usize::try_from(num_codebooks)
        .map_err(|_| InferError::Message("num_codebooks overflows usize".into()))?;
    let n_frames_usize = usize::try_from(n_frames)
        .map_err(|_| InferError::Message("n_frames overflows usize".into()))?;
    let expected_len = num_codebooks_usize
        .checked_mul(n_frames_usize)
        .ok_or_else(|| InferError::Message("codec codes length overflow".into()))?;
    if codes.len() != expected_len {
        return Err(InferError::Message(format!(
            "codec codes length mismatch: expected {expected_len}, got {}",
            codes.len()
        )));
    }

    let latent_dim = CODEC_HIDDEN_SIZE as usize;
    let mut latents = vec![0.0f32; n_frames_usize * latent_dim];
    for frame in 0..n_frames_usize {
        for codebook in 0..num_codebooks_usize {
            let code = codes[codebook * n_frames_usize + frame];
            if code < 0 {
                return Err(InferError::Message(format!(
                    "codec code must be non-negative, got {code}"
                )));
            }
            let projected = if codebook == 0 {
                weights
                    .semantic_quantizer
                    .project_code(code as u32, CODEC_SEMANTIC_CODEBOOK_SIZE as usize)?
            } else {
                weights.residual_quantizers[codebook - 1]
                    .project_code(code as u32, CODEC_RESIDUAL_CODEBOOK_SIZE as usize)?
            };
            let frame_start = frame * latent_dim;
            for (slot, value) in latents[frame_start..frame_start + latent_dim]
                .iter_mut()
                .zip(projected)
            {
                *slot += value;
            }
        }
    }

    Ok(CodecRvqLookupResult {
        num_codebooks,
        n_frames,
        latent_dim,
        latents,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecTensorDumpRow {
    pub index: usize,
    pub component: String,
    pub role: String,
    pub module: Option<String>,
    pub layer: Option<usize>,
    pub quantizer_index: Option<usize>,
    pub name: String,
    pub ggml_type: GgmlType,
    pub dimensions: Vec<u64>,
    pub elements: u64,
    pub bytes: usize,
    pub relative_offset: u64,
    pub absolute_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecTensorRoleInfo {
    pub component: String,
    pub role: String,
    pub module: Option<String>,
    pub layer: Option<usize>,
    pub quantizer_index: Option<usize>,
}

pub fn classify_codec_tensor(name: &str) -> CodecTensorRoleInfo {
    let component = name.split('.').next().unwrap_or("").to_string();

    if let Some(rest) = name.strip_prefix("quantizer.semantic_quantizer.quantizers.0.") {
        return role(
            component,
            semantic_quantizer_role(rest),
            None,
            None,
            Some(0),
        );
    }
    if let Some(rest) = name.strip_prefix("quantizer.quantizer.quantizers.") {
        if let Some((index, suffix)) = rest.split_once('.') {
            if let Ok(index) = index.parse::<usize>() {
                return role(
                    component,
                    residual_quantizer_role(suffix),
                    None,
                    None,
                    Some(index),
                );
            }
        }
    }
    for module in ["quantizer.pre_module", "quantizer.post_module"] {
        if name == format!("{module}.freqs_cis") {
            return role(component, "transformer_freqs_cis", Some(module), None, None);
        }
        if name == format!("{module}.causal_mask") {
            return role(
                component,
                "transformer_causal_mask",
                Some(module),
                None,
                None,
            );
        }
        if name == format!("{module}.norm.weight") {
            return role(
                component,
                "transformer_output_norm",
                Some(module),
                None,
                None,
            );
        }
        if let Some(rest) = name.strip_prefix(&format!("{module}.layers.")) {
            if let Some((layer, suffix)) = rest.split_once('.') {
                if let Ok(layer) = layer.parse::<usize>() {
                    return role(
                        component,
                        transformer_layer_role(suffix),
                        Some(module),
                        Some(layer),
                        None,
                    );
                }
            }
        }
    }
    if name.starts_with("quantizer.downsample.") {
        return role(component, "quantizer_downsample", None, None, None);
    }
    if name.starts_with("quantizer.upsample.") {
        return role(component, "quantizer_upsample", None, None, None);
    }
    if name.starts_with("encoder.") {
        return role(component, "encoder", None, None, None);
    }
    if name.starts_with("decoder.") {
        return role(component, "decoder", None, None, None);
    }
    role(component, "unknown", None, None, None)
}

fn role(
    component: String,
    role: impl Into<String>,
    module: Option<&str>,
    layer: Option<usize>,
    quantizer_index: Option<usize>,
) -> CodecTensorRoleInfo {
    CodecTensorRoleInfo {
        component,
        role: role.into(),
        module: module.map(str::to_string),
        layer,
        quantizer_index,
    }
}

fn semantic_quantizer_role(suffix: &str) -> &'static str {
    match suffix {
        "in_proj.weight" => "semantic_in_proj_weight",
        "in_proj.bias" => "semantic_in_proj_bias",
        "out_proj.weight" => "semantic_out_proj_weight",
        "out_proj.bias" => "semantic_out_proj_bias",
        "codebook.weight" => "semantic_codebook",
        _ => "semantic_quantizer",
    }
}

fn residual_quantizer_role(suffix: &str) -> &'static str {
    match suffix {
        "in_proj.weight" => "residual_in_proj_weight",
        "in_proj.bias" => "residual_in_proj_bias",
        "out_proj.weight" => "residual_out_proj_weight",
        "out_proj.bias" => "residual_out_proj_bias",
        "codebook.weight" => "residual_codebook",
        _ => "residual_quantizer",
    }
}

fn transformer_layer_role(suffix: &str) -> &'static str {
    match suffix {
        "attention.wqkv.weight" => "transformer_attention_wqkv",
        "attention.wo.weight" => "transformer_attention_output",
        "attention_norm.weight" => "transformer_attention_norm",
        "ffn_norm.weight" => "transformer_ffn_norm",
        "feed_forward.w1.weight" => "transformer_feed_forward_w1",
        "feed_forward.w2.weight" => "transformer_feed_forward_w2",
        "feed_forward.w3.weight" => "transformer_feed_forward_w3",
        "attention_layer_scale.gamma" => "transformer_attention_layer_scale",
        "ffn_layer_scale.gamma" => "transformer_ffn_layer_scale",
        _ => "transformer_layer",
    }
}

fn validate_quantizer(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    weights: &CodecQuantizerWeights,
    semantic: bool,
    failures: &mut Vec<String>,
) {
    let codebook_size = if semantic {
        CODEC_SEMANTIC_CODEBOOK_SIZE
    } else {
        CODEC_RESIDUAL_CODEBOOK_SIZE
    };
    let specs = [
        (
            &weights.in_proj_weight,
            vec![1, CODEC_HIDDEN_SIZE, CODEC_PROJECTION_DIM],
        ),
        (&weights.in_proj_bias, vec![CODEC_PROJECTION_DIM]),
        (
            &weights.out_proj_weight,
            vec![1, CODEC_PROJECTION_DIM, CODEC_HIDDEN_SIZE],
        ),
        (&weights.out_proj_bias, vec![CODEC_HIDDEN_SIZE]),
        (
            &weights.codebook_weight,
            vec![CODEC_PROJECTION_DIM, codebook_size],
        ),
    ];
    for (name, dimensions) in specs {
        validate_tensor(tensors, name, &dimensions, failures);
    }
}

fn validate_module(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    module: &str,
    layers: &[CodecTransformerLayerWeights],
    failures: &mut Vec<String>,
) {
    let root_specs = [
        (
            format!("{module}.freqs_cis"),
            vec![2, CODEC_FREQ_HEADS, CODEC_CONTEXT_LENGTH],
        ),
        (
            format!("{module}.causal_mask"),
            vec![CODEC_CONTEXT_LENGTH, CODEC_CONTEXT_LENGTH],
        ),
        (format!("{module}.norm.weight"), vec![CODEC_HIDDEN_SIZE]),
    ];
    for (name, dimensions) in root_specs {
        validate_tensor(tensors, &name, &dimensions, failures);
    }

    for layer in layers {
        let specs = [
            (
                &layer.attention_wqkv,
                vec![CODEC_HIDDEN_SIZE, CODEC_ATTENTION_WQKV_OUT],
            ),
            (
                &layer.attention_output,
                vec![CODEC_HIDDEN_SIZE, CODEC_HIDDEN_SIZE],
            ),
            (
                &layer.feed_forward_w1,
                vec![CODEC_HIDDEN_SIZE, CODEC_FEED_FORWARD_SIZE],
            ),
            (
                &layer.feed_forward_w3,
                vec![CODEC_HIDDEN_SIZE, CODEC_FEED_FORWARD_SIZE],
            ),
            (
                &layer.feed_forward_w2,
                vec![CODEC_FEED_FORWARD_SIZE, CODEC_HIDDEN_SIZE],
            ),
            (&layer.ffn_norm, vec![CODEC_HIDDEN_SIZE]),
            (&layer.attention_norm, vec![CODEC_HIDDEN_SIZE]),
            (&layer.attention_layer_scale, vec![CODEC_HIDDEN_SIZE]),
            (&layer.ffn_layer_scale, vec![CODEC_HIDDEN_SIZE]),
        ];
        for (name, dimensions) in specs {
            validate_tensor(tensors, name, &dimensions, failures);
        }
    }
}

fn validate_tensor(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    name: &str,
    dimensions: &[u64],
    failures: &mut Vec<String>,
) {
    match tensors.get(name) {
        Some(tensor) if tensor.ggml_type == GgmlType::F16 && tensor.dimensions == dimensions => {}
        Some(tensor) => failures.push(format!(
            "{name} expected F16 {dimensions:?}, got {:?} {:?}",
            tensor.ggml_type, tensor.dimensions
        )),
        None => failures.push(format!("missing {name}")),
    }
}

fn prefix_counts(tensors: &[GgufTensorInfo]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for tensor in tensors {
        let prefix = tensor.name.split('.').next().unwrap_or("").to_string();
        *counts.entry(prefix).or_insert(0) += 1;
    }
    counts
}

fn metadata_value<'a>(gguf: &'a GgufFile, key: &str) -> Option<&'a str> {
    gguf.metadata
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value.as_str())
}

fn forward_codec_transformer_layer(
    tokens: &[Vec<f32>],
    weights: &CodecTransformerLayerF16Weights,
) -> Result<Vec<Vec<f32>>> {
    if tokens.is_empty() {
        return Err(InferError::Message(
            "codec post_module requires at least one frame".into(),
        ));
    }
    let hidden_dim = CODEC_HIDDEN_SIZE as usize;
    for token in tokens {
        if token.len() != hidden_dim {
            return Err(InferError::Message(format!(
                "codec post_module token length mismatch: expected {hidden_dim}, got {}",
                token.len()
            )));
        }
    }

    let mut prepared = Vec::with_capacity(tokens.len());
    for (position, token) in tokens.iter().enumerate() {
        let normalized = rms_norm(token, weights.attention_norm.values(), CODEC_RVQ_NORM_EPS)?;
        let qkv = linear(
            &normalized,
            weights.attention_wqkv.values(),
            hidden_dim,
            CODEC_ATTENTION_WQKV_OUT as usize,
        )?;
        let q_size = hidden_dim;
        let kv_size = CODEC_RVQ_LOCAL_HEADS * CODEC_RVQ_HEAD_DIM;
        let (query_raw, rest) = qkv.split_at(q_size);
        let (key_raw, value_raw) = rest.split_at(kv_size);
        let mut query = query_raw.to_vec();
        let mut key = key_raw.to_vec();
        apply_rope_normal(
            &mut query,
            CODEC_RVQ_HEAD_DIM,
            position,
            CODEC_RVQ_ROPE_BASE,
        )?;
        apply_rope_normal(&mut key, CODEC_RVQ_HEAD_DIM, position, CODEC_RVQ_ROPE_BASE)?;
        prepared.push((query, key, value_raw.to_vec()));
    }

    let mut attention_outputs = Vec::with_capacity(tokens.len());
    for (offset, token) in tokens.iter().enumerate() {
        let visible_start = (offset + 1).saturating_sub(CODEC_RVQ_WINDOW_SIZE);
        let visible_count = offset + 1 - visible_start;
        let mut keys = Vec::with_capacity(visible_count * hidden_dim);
        let mut values = Vec::with_capacity(visible_count * hidden_dim);
        for (_, key, value) in &prepared[visible_start..=offset] {
            keys.extend_from_slice(key);
            values.extend_from_slice(value);
        }
        let attention = gqa_decode_attention(
            &prepared[offset].0,
            &keys,
            &values,
            GqaAttentionShape {
                head_count: hidden_dim / CODEC_RVQ_HEAD_DIM,
                head_count_kv: CODEC_RVQ_LOCAL_HEADS,
                head_dim: CODEC_RVQ_HEAD_DIM,
                token_count: visible_count,
                attn_scale: (CODEC_RVQ_HEAD_DIM as f32).sqrt().recip(),
            },
        )?;
        let projected = linear(
            &attention,
            weights.attention_output.values(),
            hidden_dim,
            hidden_dim,
        )?;
        let scaled = scale_channels(&projected, weights.attention_layer_scale.values())?;
        attention_outputs.push(add_residual(token, &scaled)?);
    }

    let mut outputs = Vec::with_capacity(tokens.len());
    for token in attention_outputs {
        let ff_in = rms_norm(&token, weights.ffn_norm.values(), CODEC_RVQ_NORM_EPS)?;
        let gate = linear(
            &ff_in,
            weights.feed_forward_w1.values(),
            hidden_dim,
            CODEC_FEED_FORWARD_SIZE as usize,
        )?;
        let up = linear(
            &ff_in,
            weights.feed_forward_w3.values(),
            hidden_dim,
            CODEC_FEED_FORWARD_SIZE as usize,
        )?;
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| silu(*gate) * up)
            .collect::<Vec<_>>();
        let ff = linear(
            &activated,
            weights.feed_forward_w2.values(),
            CODEC_FEED_FORWARD_SIZE as usize,
            hidden_dim,
        )?;
        let scaled = scale_channels(&ff, weights.ffn_layer_scale.values())?;
        outputs.push(add_residual(&token, &scaled)?);
    }
    Ok(outputs)
}

fn validate_f16_dims(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(InferError::Message(format!(
            "{name}: expected {expected:?}, got {actual:?}"
        )))
    }
}

fn scale_channels(values: &[f32], scale: &[f32]) -> Result<Vec<f32>> {
    if values.len() != scale.len() {
        return Err(InferError::Message(format!(
            "scale length mismatch: values={} scale={}",
            values.len(),
            scale.len()
        )));
    }
    Ok(values
        .iter()
        .zip(scale)
        .map(|(value, scale)| value * scale)
        .collect())
}

fn add_residual(residual: &[f32], delta: &[f32]) -> Result<Vec<f32>> {
    if residual.len() != delta.len() {
        return Err(InferError::Message(format!(
            "residual length mismatch: residual={} delta={}",
            residual.len(),
            delta.len()
        )));
    }
    Ok(residual
        .iter()
        .zip(delta)
        .map(|(residual, delta)| residual + delta)
        .collect())
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn add_bias(output: &mut [f32], bias: &[f32]) -> Result<()> {
    if output.len() != bias.len() {
        return Err(InferError::Message(format!(
            "bias length mismatch: output={} bias={}",
            output.len(),
            bias.len()
        )));
    }
    for (slot, value) in output.iter_mut().zip(bias) {
        *slot += value;
    }
    Ok(())
}

pub fn format_codec_dimensions(dimensions: &[u64]) -> String {
    dimensions
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_codec_path() -> Option<PathBuf> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/s2-pro-f16-codec-only.gguf");
        path.exists().then_some(path)
    }

    #[test]
    fn classifies_semantic_and_residual_quantizer_tensors() {
        let semantic =
            classify_codec_tensor("quantizer.semantic_quantizer.quantizers.0.codebook.weight");
        assert_eq!(semantic.role, "semantic_codebook");
        assert_eq!(semantic.quantizer_index, Some(0));

        let residual = classify_codec_tensor("quantizer.quantizer.quantizers.8.out_proj.weight");
        assert_eq!(residual.role, "residual_out_proj_weight");
        assert_eq!(residual.quantizer_index, Some(8));
    }

    #[test]
    fn classifies_pre_and_post_module_layers() {
        let pre = classify_codec_tensor("quantizer.pre_module.layers.7.attention.wqkv.weight");
        assert_eq!(pre.module.as_deref(), Some("quantizer.pre_module"));
        assert_eq!(pre.layer, Some(7));
        assert_eq!(pre.role, "transformer_attention_wqkv");

        let post = classify_codec_tensor("quantizer.post_module.norm.weight");
        assert_eq!(post.role, "transformer_output_norm");
        assert_eq!(post.layer, None);
    }

    #[test]
    fn codec_quantizer_names_match_gguf_prefixes() {
        let semantic = CodecQuantizerWeights::semantic();
        assert_eq!(
            semantic.codebook_weight,
            "quantizer.semantic_quantizer.quantizers.0.codebook.weight"
        );
        let residual = CodecQuantizerWeights::residual(3);
        assert_eq!(
            residual.in_proj_weight,
            "quantizer.quantizer.quantizers.3.in_proj.weight"
        );
    }

    #[test]
    #[ignore = "requires local s2-pro codec GGUF in models/"]
    fn loads_codec_f16_weights_and_runs_rvq_lookup_fixture() {
        let path = fixture_codec_path().expect("codec gguf");
        let gguf = GgufFile::open(&path).expect("codec gguf");
        let weights = CodecF16Weights::from_gguf(&gguf).expect("codec f16 weights");
        assert_eq!(
            weights.semantic_quantizer.codebook_weight.dimensions(),
            &[
                CODEC_PROJECTION_DIM as usize,
                CODEC_SEMANTIC_CODEBOOK_SIZE as usize
            ]
        );
        assert_eq!(weights.residual_quantizers.len(), CODEC_RESIDUAL_QUANTIZERS);

        let codes = vec![
            3988, 29, 487, 925, 184, 865, 526, 924, 37, 12, 189, 460, 854, 549, 947, 935, 339, 39,
            892, 855,
        ];
        let result = rvq_lookup_codes(&codes, 10, 2, &weights).expect("rvq lookup");
        assert_eq!(result.latent_dim, CODEC_HIDDEN_SIZE as usize);
        assert_eq!(result.latents.len(), 2 * CODEC_HIDDEN_SIZE as usize);
        assert!(result.latents.iter().all(|value| value.is_finite()));
        assert!(result.latents.iter().any(|value| value.abs() > 0.0));
    }

    #[test]
    #[ignore = "requires local s2-pro codec GGUF in models/"]
    fn loads_post_module_f16_weights_and_runs_fixture() {
        let path = fixture_codec_path().expect("codec gguf");
        let gguf = GgufFile::open(&path).expect("codec gguf");
        let rvq_weights = CodecF16Weights::from_gguf(&gguf).expect("codec f16 weights");
        let post_weights =
            CodecPostModuleF16Weights::from_gguf(&gguf).expect("post module f16 weights");
        assert_eq!(post_weights.layers.len(), CODEC_TRANSFORMER_LAYERS);
        assert_eq!(
            post_weights.norm_weight.dimensions(),
            &[CODEC_HIDDEN_SIZE as usize]
        );

        let codes = vec![
            3988, 29, 487, 925, 184, 865, 526, 924, 37, 12, 189, 460, 854, 549, 947, 935, 339, 39,
            892, 855,
        ];
        let rvq = rvq_lookup_codes(&codes, 10, 2, &rvq_weights).expect("rvq lookup");
        let result = forward_codec_post_module(&rvq.latents, rvq.n_frames, &post_weights)
            .expect("post module");
        assert_eq!(result.hidden_dim, CODEC_HIDDEN_SIZE as usize);
        assert_eq!(result.hidden.len(), 2 * CODEC_HIDDEN_SIZE as usize);
        assert!(result.hidden.iter().all(|value| value.is_finite()));
        assert!(result.hidden.iter().any(|value| value.abs() > 0.0));
    }

    #[test]
    #[ignore = "requires local s2-pro codec GGUF in models/"]
    fn loads_local_codec_registry_from_gguf() {
        let path = fixture_codec_path().expect("codec gguf");
        let registry = CodecTensorRegistry::from_gguf_file(path).expect("codec registry");
        assert_eq!(registry.architecture, CODEC_ARCHITECTURE);
        assert_eq!(registry.tensor_count, 461);
        assert_eq!(
            registry.residual_quantizers().len(),
            CODEC_RESIDUAL_QUANTIZERS
        );
        assert_eq!(registry.pre_module_layers().len(), CODEC_TRANSFORMER_LAYERS);
        assert_eq!(
            registry.post_module_layers().len(),
            CODEC_TRANSFORMER_LAYERS
        );
        assert_eq!(registry.prefix_counts().get("quantizer"), Some(&244));
    }
}
