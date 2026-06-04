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
pub const CODEC_UPSAMPLE_STAGES: usize = 2;
pub const CODEC_UPSAMPLE_FACTOR: usize = 2;
pub const CODEC_CONVNEXT_KERNEL_SIZE: usize = 7;
pub const CODEC_CONVNEXT_EXPANDED_SIZE: usize = 4096;
pub const CODEC_CONVNEXT_NORM_EPS: f32 = 1e-6;

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
    quantizer_upsample: CodecUpsampleWeights,
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
        let quantizer_upsample = CodecUpsampleWeights::new();

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
            quantizer_upsample,
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

    pub fn quantizer_upsample(&self) -> &CodecUpsampleWeights {
        &self.quantizer_upsample
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
        validate_upsample(&self.tensors, &self.quantizer_upsample, &mut failures);

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecUpsampleWeights {
    pub stages: Vec<CodecUpsampleStageWeights>,
}

impl CodecUpsampleWeights {
    pub fn new() -> Self {
        Self {
            stages: (0..CODEC_UPSAMPLE_STAGES)
                .map(CodecUpsampleStageWeights::new)
                .collect(),
        }
    }
}

impl Default for CodecUpsampleWeights {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecUpsampleStageWeights {
    pub index: usize,
    pub conv_transpose_weight: String,
    pub conv_transpose_bias: String,
    pub convnext_gamma: String,
    pub dwconv_weight: String,
    pub dwconv_bias: String,
    pub norm_weight: String,
    pub norm_bias: String,
    pub pwconv1_weight: String,
    pub pwconv1_bias: String,
    pub pwconv2_weight: String,
    pub pwconv2_bias: String,
}

impl CodecUpsampleStageWeights {
    pub fn new(index: usize) -> Self {
        let prefix = format!("quantizer.upsample.{index}");
        Self {
            index,
            conv_transpose_weight: format!("{prefix}.0.conv.weight"),
            conv_transpose_bias: format!("{prefix}.0.conv.bias"),
            convnext_gamma: format!("{prefix}.1.gamma"),
            dwconv_weight: format!("{prefix}.1.dwconv.conv.weight"),
            dwconv_bias: format!("{prefix}.1.dwconv.conv.bias"),
            norm_weight: format!("{prefix}.1.norm.weight"),
            norm_bias: format!("{prefix}.1.norm.bias"),
            pwconv1_weight: format!("{prefix}.1.pwconv1.weight"),
            pwconv1_bias: format!("{prefix}.1.pwconv1.bias"),
            pwconv2_weight: format!("{prefix}.1.pwconv2.weight"),
            pwconv2_bias: format!("{prefix}.1.pwconv2.bias"),
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
pub struct CodecUpsampleF16Weights {
    pub stages: Vec<CodecUpsampleStageF16Weights>,
}

impl CodecUpsampleF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let registry = CodecTensorRegistry::from_gguf(gguf)?;
        Self::from_gguf_registry(gguf, &registry)
    }

    pub fn from_gguf_registry(gguf: &GgufFile, registry: &CodecTensorRegistry) -> Result<Self> {
        let weights = Self {
            stages: registry
                .quantizer_upsample()
                .stages
                .iter()
                .map(|stage| CodecUpsampleStageF16Weights::from_names(gguf, stage))
                .collect::<Result<Vec<_>>>()?,
        };
        weights.validate_dimensions()?;
        Ok(weights)
    }

    fn validate_dimensions(&self) -> Result<()> {
        if self.stages.len() != CODEC_UPSAMPLE_STAGES {
            return Err(InferError::Message(format!(
                "codec upsample stage count mismatch: expected {}, got {}",
                CODEC_UPSAMPLE_STAGES,
                self.stages.len()
            )));
        }
        for stage in &self.stages {
            stage.validate_dimensions()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecUpsampleStageF16Weights {
    pub index: usize,
    pub conv_transpose_weight: F16TensorView,
    pub conv_transpose_bias: F16TensorView,
    pub convnext_gamma: F16TensorView,
    pub dwconv_weight: F16TensorView,
    pub dwconv_bias: F16TensorView,
    pub norm_weight: F16TensorView,
    pub norm_bias: F16TensorView,
    pub pwconv1_weight: F16TensorView,
    pub pwconv1_bias: F16TensorView,
    pub pwconv2_weight: F16TensorView,
    pub pwconv2_bias: F16TensorView,
}

impl CodecUpsampleStageF16Weights {
    fn from_names(gguf: &GgufFile, names: &CodecUpsampleStageWeights) -> Result<Self> {
        Ok(Self {
            index: names.index,
            conv_transpose_weight: F16TensorView::from_gguf(gguf, &names.conv_transpose_weight)?,
            conv_transpose_bias: F16TensorView::from_gguf(gguf, &names.conv_transpose_bias)?,
            convnext_gamma: F16TensorView::from_gguf(gguf, &names.convnext_gamma)?,
            dwconv_weight: F16TensorView::from_gguf(gguf, &names.dwconv_weight)?,
            dwconv_bias: F16TensorView::from_gguf(gguf, &names.dwconv_bias)?,
            norm_weight: F16TensorView::from_gguf(gguf, &names.norm_weight)?,
            norm_bias: F16TensorView::from_gguf(gguf, &names.norm_bias)?,
            pwconv1_weight: F16TensorView::from_gguf(gguf, &names.pwconv1_weight)?,
            pwconv1_bias: F16TensorView::from_gguf(gguf, &names.pwconv1_bias)?,
            pwconv2_weight: F16TensorView::from_gguf(gguf, &names.pwconv2_weight)?,
            pwconv2_bias: F16TensorView::from_gguf(gguf, &names.pwconv2_bias)?,
        })
    }

    fn validate_dimensions(&self) -> Result<()> {
        let hidden = CODEC_HIDDEN_SIZE as usize;
        let expanded = CODEC_CONVNEXT_EXPANDED_SIZE;
        let specs = [
            (
                self.conv_transpose_weight.name(),
                self.conv_transpose_weight.dimensions(),
                vec![CODEC_UPSAMPLE_FACTOR, hidden, hidden],
            ),
            (
                self.conv_transpose_bias.name(),
                self.conv_transpose_bias.dimensions(),
                vec![hidden],
            ),
            (
                self.convnext_gamma.name(),
                self.convnext_gamma.dimensions(),
                vec![hidden],
            ),
            (
                self.dwconv_weight.name(),
                self.dwconv_weight.dimensions(),
                vec![CODEC_CONVNEXT_KERNEL_SIZE, 1, hidden],
            ),
            (
                self.dwconv_bias.name(),
                self.dwconv_bias.dimensions(),
                vec![hidden],
            ),
            (
                self.norm_weight.name(),
                self.norm_weight.dimensions(),
                vec![hidden],
            ),
            (
                self.norm_bias.name(),
                self.norm_bias.dimensions(),
                vec![hidden],
            ),
            (
                self.pwconv1_weight.name(),
                self.pwconv1_weight.dimensions(),
                vec![hidden, expanded],
            ),
            (
                self.pwconv1_bias.name(),
                self.pwconv1_bias.dimensions(),
                vec![expanded],
            ),
            (
                self.pwconv2_weight.name(),
                self.pwconv2_weight.dimensions(),
                vec![expanded, hidden],
            ),
            (
                self.pwconv2_bias.name(),
                self.pwconv2_bias.dimensions(),
                vec![hidden],
            ),
        ];
        for (name, actual, expected) in specs {
            validate_f16_dims(name, actual, &expected)?;
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

#[derive(Debug, Clone, PartialEq)]
pub struct CodecUpsampleResult {
    pub input_frames: u32,
    pub output_frames: u32,
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

pub fn forward_codec_upsample(
    hidden: &[f32],
    n_frames: u32,
    weights: &CodecUpsampleF16Weights,
) -> Result<CodecUpsampleResult> {
    let mut frames = usize::try_from(n_frames)
        .map_err(|_| InferError::Message("n_frames overflows usize".into()))?;
    let hidden_dim = CODEC_HIDDEN_SIZE as usize;
    validate_frame_major_len("codec upsample input", hidden, frames, hidden_dim)?;
    let mut current = hidden.to_vec();
    for stage in &weights.stages {
        current = forward_codec_upsample_stage(&current, frames, stage)?;
        frames = frames
            .checked_mul(CODEC_UPSAMPLE_FACTOR)
            .ok_or_else(|| InferError::Message("codec upsample frame count overflow".into()))?;
    }
    let output_frames = u32::try_from(frames)
        .map_err(|_| InferError::Message("codec upsample output_frames overflows u32".into()))?;
    Ok(CodecUpsampleResult {
        input_frames: n_frames,
        output_frames,
        hidden_dim,
        hidden: current,
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

fn validate_upsample(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    weights: &CodecUpsampleWeights,
    failures: &mut Vec<String>,
) {
    if weights.stages.len() != CODEC_UPSAMPLE_STAGES {
        failures.push(format!(
            "quantizer upsample stage count: expected {}, got {}",
            CODEC_UPSAMPLE_STAGES,
            weights.stages.len()
        ));
    }
    for stage in &weights.stages {
        let specs = [
            (
                &stage.conv_transpose_weight,
                vec![
                    CODEC_UPSAMPLE_FACTOR as u64,
                    CODEC_HIDDEN_SIZE,
                    CODEC_HIDDEN_SIZE,
                ],
            ),
            (&stage.conv_transpose_bias, vec![CODEC_HIDDEN_SIZE]),
            (&stage.convnext_gamma, vec![CODEC_HIDDEN_SIZE]),
            (
                &stage.dwconv_weight,
                vec![CODEC_CONVNEXT_KERNEL_SIZE as u64, 1, CODEC_HIDDEN_SIZE],
            ),
            (&stage.dwconv_bias, vec![CODEC_HIDDEN_SIZE]),
            (&stage.norm_weight, vec![CODEC_HIDDEN_SIZE]),
            (&stage.norm_bias, vec![CODEC_HIDDEN_SIZE]),
            (
                &stage.pwconv1_weight,
                vec![CODEC_HIDDEN_SIZE, CODEC_CONVNEXT_EXPANDED_SIZE as u64],
            ),
            (
                &stage.pwconv1_bias,
                vec![CODEC_CONVNEXT_EXPANDED_SIZE as u64],
            ),
            (
                &stage.pwconv2_weight,
                vec![CODEC_CONVNEXT_EXPANDED_SIZE as u64, CODEC_HIDDEN_SIZE],
            ),
            (&stage.pwconv2_bias, vec![CODEC_HIDDEN_SIZE]),
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

fn forward_codec_upsample_stage(
    input: &[f32],
    frames: usize,
    weights: &CodecUpsampleStageF16Weights,
) -> Result<Vec<f32>> {
    let hidden_dim = CODEC_HIDDEN_SIZE as usize;
    validate_frame_major_len("codec upsample stage input", input, frames, hidden_dim)?;
    let conv = causal_conv_transpose_1d(
        input,
        frames,
        hidden_dim,
        hidden_dim,
        CODEC_UPSAMPLE_FACTOR,
        weights.conv_transpose_weight.values(),
        weights.conv_transpose_bias.values(),
    )?;
    forward_codec_convnext_block(&conv, frames * CODEC_UPSAMPLE_FACTOR, weights)
}

fn forward_codec_convnext_block(
    input: &[f32],
    frames: usize,
    weights: &CodecUpsampleStageF16Weights,
) -> Result<Vec<f32>> {
    let hidden_dim = CODEC_HIDDEN_SIZE as usize;
    validate_frame_major_len("codec convnext input", input, frames, hidden_dim)?;
    let mut x = depthwise_causal_conv1d(
        input,
        frames,
        hidden_dim,
        CODEC_CONVNEXT_KERNEL_SIZE,
        weights.dwconv_weight.values(),
        weights.dwconv_bias.values(),
    )?;
    layer_norm_affine_frame_major(
        &mut x,
        frames,
        hidden_dim,
        weights.norm_weight.values(),
        weights.norm_bias.values(),
        CODEC_CONVNEXT_NORM_EPS,
    )?;
    x = linear_bias_frame_major(
        &x,
        frames,
        hidden_dim,
        CODEC_CONVNEXT_EXPANDED_SIZE,
        weights.pwconv1_weight.values(),
        weights.pwconv1_bias.values(),
    )?;
    for value in &mut x {
        *value = gelu_erf(*value);
    }
    x = linear_bias_frame_major(
        &x,
        frames,
        CODEC_CONVNEXT_EXPANDED_SIZE,
        hidden_dim,
        weights.pwconv2_weight.values(),
        weights.pwconv2_bias.values(),
    )?;
    scale_frame_major_channels(&mut x, hidden_dim, weights.convnext_gamma.values())?;
    add_frame_major_residual(input, &x)
}

fn causal_conv_transpose_1d(
    input: &[f32],
    frames: usize,
    in_ch: usize,
    out_ch: usize,
    stride: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    validate_frame_major_len("conv_transpose input", input, frames, in_ch)?;
    let kernel = stride;
    let expected_weights = kernel
        .checked_mul(out_ch)
        .and_then(|value| value.checked_mul(in_ch))
        .ok_or_else(|| InferError::Message("conv_transpose weight length overflow".into()))?;
    if weight.len() != expected_weights {
        return Err(InferError::Message(format!(
            "conv_transpose weight length mismatch: expected {expected_weights}, got {}",
            weight.len()
        )));
    }
    if bias.len() != out_ch {
        return Err(InferError::Message(format!(
            "conv_transpose bias length mismatch: expected {out_ch}, got {}",
            bias.len()
        )));
    }
    let output_frames = frames
        .checked_mul(stride)
        .ok_or_else(|| InferError::Message("conv_transpose output frame overflow".into()))?;
    let mut output = vec![0.0f32; output_frames * out_ch];
    for frame in 0..frames {
        let input_row = &input[frame * in_ch..(frame + 1) * in_ch];
        for kernel_index in 0..kernel {
            let output_row_start = (frame * stride + kernel_index) * out_ch;
            for output_channel in 0..out_ch {
                let weight_start = (kernel_index * out_ch + output_channel) * in_ch;
                let weight_row = &weight[weight_start..weight_start + in_ch];
                output[output_row_start + output_channel] += dot(input_row, weight_row);
            }
        }
    }
    add_frame_bias(&mut output, out_ch, bias)?;
    Ok(output)
}

fn depthwise_causal_conv1d(
    input: &[f32],
    frames: usize,
    channels: usize,
    kernel: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    validate_frame_major_len("depthwise conv input", input, frames, channels)?;
    let expected_weights = kernel
        .checked_mul(channels)
        .ok_or_else(|| InferError::Message("depthwise conv weight length overflow".into()))?;
    if weight.len() != expected_weights {
        return Err(InferError::Message(format!(
            "depthwise conv weight length mismatch: expected {expected_weights}, got {}",
            weight.len()
        )));
    }
    if bias.len() != channels {
        return Err(InferError::Message(format!(
            "depthwise conv bias length mismatch: expected {channels}, got {}",
            bias.len()
        )));
    }
    let left_padding = kernel - 1;
    let mut output = vec![0.0f32; frames * channels];
    for frame in 0..frames {
        for channel in 0..channels {
            let mut sum = bias[channel];
            for kernel_index in 0..kernel {
                if let Some(source_frame) = (frame + kernel_index).checked_sub(left_padding) {
                    if source_frame < frames {
                        let input_index = source_frame * channels + channel;
                        let weight_index = kernel_index * channels + channel;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
            output[frame * channels + channel] = sum;
        }
    }
    Ok(output)
}

fn layer_norm_affine_frame_major(
    values: &mut [f32],
    frames: usize,
    channels: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<()> {
    validate_frame_major_len("layer_norm input", values, frames, channels)?;
    if weight.len() != channels || bias.len() != channels {
        return Err(InferError::Message(format!(
            "layer_norm affine length mismatch: channels={channels} weight={} bias={}",
            weight.len(),
            bias.len()
        )));
    }
    for frame in 0..frames {
        let row = &mut values[frame * channels..(frame + 1) * channels];
        let mean = row.iter().sum::<f32>() / channels as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / channels as f32;
        let scale = (variance + eps).sqrt().recip();
        for channel in 0..channels {
            row[channel] = (row[channel] - mean) * scale * weight[channel] + bias[channel];
        }
    }
    Ok(())
}

fn linear_bias_frame_major(
    input: &[f32],
    frames: usize,
    input_dim: usize,
    output_dim: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    validate_frame_major_len("linear frame-major input", input, frames, input_dim)?;
    if bias.len() != output_dim {
        return Err(InferError::Message(format!(
            "linear frame-major bias length mismatch: expected {output_dim}, got {}",
            bias.len()
        )));
    }
    let mut output = Vec::with_capacity(frames * output_dim);
    for frame in 0..frames {
        let row = &input[frame * input_dim..(frame + 1) * input_dim];
        let mut projected = linear(row, weight, input_dim, output_dim)?;
        add_bias(&mut projected, bias)?;
        output.extend(projected);
    }
    Ok(output)
}

fn scale_frame_major_channels(values: &mut [f32], channels: usize, scale: &[f32]) -> Result<()> {
    if scale.len() != channels {
        return Err(InferError::Message(format!(
            "frame-major scale length mismatch: channels={channels} scale={}",
            scale.len()
        )));
    }
    if !values.len().is_multiple_of(channels) {
        return Err(InferError::Message(format!(
            "frame-major values length {} is not divisible by channels {channels}",
            values.len()
        )));
    }
    for row in values.chunks_exact_mut(channels) {
        for (slot, factor) in row.iter_mut().zip(scale) {
            *slot *= factor;
        }
    }
    Ok(())
}

fn add_frame_major_residual(residual: &[f32], delta: &[f32]) -> Result<Vec<f32>> {
    if residual.len() != delta.len() {
        return Err(InferError::Message(format!(
            "frame-major residual length mismatch: residual={} delta={}",
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

fn add_frame_bias(output: &mut [f32], channels: usize, bias: &[f32]) -> Result<()> {
    if bias.len() != channels {
        return Err(InferError::Message(format!(
            "frame bias length mismatch: channels={channels} bias={}",
            bias.len()
        )));
    }
    for row in output.chunks_exact_mut(channels) {
        for (slot, value) in row.iter_mut().zip(bias) {
            *slot += value;
        }
    }
    Ok(())
}

fn validate_frame_major_len(
    name: &str,
    values: &[f32],
    frames: usize,
    channels: usize,
) -> Result<()> {
    let expected = frames
        .checked_mul(channels)
        .ok_or_else(|| InferError::Message(format!("{name} length overflow")))?;
    if values.len() == expected {
        Ok(())
    } else {
        Err(InferError::Message(format!(
            "{name} length mismatch: expected {expected}, got {}",
            values.len()
        )))
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
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

fn gelu_erf(value: f32) -> f32 {
    0.5 * value * (1.0 + erf_approx(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn erf_approx(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
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
    fn loads_upsample_f16_weights_and_runs_decode_stage_fixture() {
        let path = fixture_codec_path().expect("codec gguf");
        let gguf = GgufFile::open(&path).expect("codec gguf");
        let rvq_weights = CodecF16Weights::from_gguf(&gguf).expect("codec f16 weights");
        let post_weights =
            CodecPostModuleF16Weights::from_gguf(&gguf).expect("post module f16 weights");
        let upsample_weights =
            CodecUpsampleF16Weights::from_gguf(&gguf).expect("upsample f16 weights");
        assert_eq!(upsample_weights.stages.len(), CODEC_UPSAMPLE_STAGES);
        assert_eq!(
            upsample_weights.stages[0]
                .conv_transpose_weight
                .dimensions(),
            &[
                CODEC_UPSAMPLE_FACTOR,
                CODEC_HIDDEN_SIZE as usize,
                CODEC_HIDDEN_SIZE as usize
            ]
        );

        let codes = vec![
            3988, 29, 487, 925, 184, 865, 526, 924, 37, 12, 189, 460, 854, 549, 947, 935, 339, 39,
            892, 855,
        ];
        let rvq = rvq_lookup_codes(&codes, 10, 2, &rvq_weights).expect("rvq lookup");
        let post = forward_codec_post_module(&rvq.latents, rvq.n_frames, &post_weights)
            .expect("post module");
        let result = forward_codec_upsample(&post.hidden, post.n_frames, &upsample_weights)
            .expect("upsample");
        assert_eq!(result.input_frames, 2);
        assert_eq!(result.output_frames, 8);
        assert_eq!(result.hidden_dim, CODEC_HIDDEN_SIZE as usize);
        assert_eq!(result.hidden.len(), 8 * CODEC_HIDDEN_SIZE as usize);
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
