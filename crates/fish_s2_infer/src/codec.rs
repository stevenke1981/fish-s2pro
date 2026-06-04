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
use crate::wav::read_wav_mono_f32;

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
pub const CODEC_ENCODER_BLOCK_COUNT: usize = 4;
pub const CODEC_ENCODER_ENTRY_CHANNELS: usize = 64;
pub const CODEC_ENCODER_RATES: [usize; CODEC_ENCODER_BLOCK_COUNT] = [2, 4, 8, 8];
pub const CODEC_ENCODER_KERNELS: [usize; CODEC_ENCODER_BLOCK_COUNT] = [4, 8, 16, 16];
pub const CODEC_ENCODER_CHANNELS: [usize; CODEC_ENCODER_BLOCK_COUNT + 1] =
    [64, 128, 256, 512, 1024];
pub const CODEC_ENCODER_TRANSFORMER_LAYERS: usize = 4;
pub const CODEC_ENCODER_TRANSFORMER_CONTEXT: u64 = 16_384;
pub const CODEC_ENCODER_TRANSFORMER_WINDOW_SIZE: usize = 512;
pub const CODEC_FRAME_LENGTH: usize = 2048;
pub const CODEC_ENCODER_TAIL_BLOCK: usize = CODEC_ENCODER_BLOCK_COUNT + 1;
pub const CODEC_ENCODER_OUTPUT_BLOCK: usize = CODEC_ENCODER_BLOCK_COUNT + 2;
pub const CODEC_LATENT_DIM: usize = 1024;
pub const CODEC_DECODER_ENTRY_CHANNELS: usize = 1536;
pub const CODEC_DECODER_BLOCK_COUNT: usize = 4;
pub const CODEC_DECODER_RATES: [usize; CODEC_DECODER_BLOCK_COUNT] = [8, 8, 4, 2];
pub const CODEC_DECODER_TAIL_BLOCK: usize = CODEC_DECODER_BLOCK_COUNT + 1;
pub const CODEC_DECODER_OUTPUT_BLOCK: usize = CODEC_DECODER_BLOCK_COUNT + 2;
pub const CODEC_SAMPLE_RATE: u32 = 44_100;

#[derive(Debug, Clone)]
pub struct CodecTensorRegistry {
    pub architecture: String,
    pub tensor_count: usize,
    pub metadata: Vec<(String, String)>,
    tensors: BTreeMap<String, GgufTensorInfo>,
    ordered_tensors: Vec<GgufTensorInfo>,
    prefix_counts: BTreeMap<String, usize>,
    encoder: CodecEncoderWeights,
    semantic_quantizer: CodecQuantizerWeights,
    residual_quantizers: Vec<CodecQuantizerWeights>,
    pre_module_layers: Vec<CodecTransformerLayerWeights>,
    post_module_layers: Vec<CodecTransformerLayerWeights>,
    quantizer_downsample: CodecDownsampleWeights,
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
        let encoder = CodecEncoderWeights::new();
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
        let quantizer_downsample = CodecDownsampleWeights::new();
        let quantizer_upsample = CodecUpsampleWeights::new();

        let registry = Self {
            architecture,
            tensor_count: tensors.len(),
            metadata: gguf.metadata.clone(),
            tensors,
            ordered_tensors,
            prefix_counts,
            encoder,
            semantic_quantizer,
            residual_quantizers,
            pre_module_layers,
            post_module_layers,
            quantizer_downsample,
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

    pub fn encoder(&self) -> &CodecEncoderWeights {
        &self.encoder
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

    pub fn quantizer_downsample(&self) -> &CodecDownsampleWeights {
        &self.quantizer_downsample
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
        validate_encoder(&self.tensors, &self.encoder, &mut failures);
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
        validate_downsample(&self.tensors, &self.quantizer_downsample, &mut failures);
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
pub struct CodecDownsampleWeights {
    pub stages: Vec<CodecDownsampleStageWeights>,
}

impl CodecDownsampleWeights {
    pub fn new() -> Self {
        Self {
            stages: (0..CODEC_UPSAMPLE_STAGES)
                .map(CodecDownsampleStageWeights::new)
                .collect(),
        }
    }
}

impl Default for CodecDownsampleWeights {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecDownsampleStageWeights {
    pub index: usize,
    pub conv_weight: String,
    pub conv_bias: String,
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

impl CodecDownsampleStageWeights {
    pub fn new(index: usize) -> Self {
        let prefix = format!("quantizer.downsample.{index}");
        Self {
            index,
            conv_weight: format!("{prefix}.0.conv.weight"),
            conv_bias: format!("{prefix}.0.conv.bias"),
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
pub struct CodecPreModuleF16Weights {
    pub layers: Vec<CodecTransformerLayerF16Weights>,
    pub norm_weight: F16TensorView,
}

impl CodecPreModuleF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let registry = CodecTensorRegistry::from_gguf(gguf)?;
        Self::from_gguf_registry(gguf, &registry)
    }

    pub fn from_gguf_registry(gguf: &GgufFile, registry: &CodecTensorRegistry) -> Result<Self> {
        let layers = registry
            .pre_module_layers()
            .iter()
            .map(|names| CodecTransformerLayerF16Weights::from_names(gguf, names))
            .collect::<Result<Vec<_>>>()?;
        let weights = Self {
            layers,
            norm_weight: F16TensorView::from_gguf(gguf, "quantizer.pre_module.norm.weight")?,
        };
        weights.validate_dimensions()?;
        Ok(weights)
    }

    fn validate_dimensions(&self) -> Result<()> {
        if self.layers.len() != CODEC_TRANSFORMER_LAYERS {
            return Err(InferError::Message(format!(
                "codec pre_module layer count mismatch: expected {}, got {}",
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
pub struct CodecDownsampleF16Weights {
    pub stages: Vec<CodecDownsampleStageF16Weights>,
}

impl CodecDownsampleF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let registry = CodecTensorRegistry::from_gguf(gguf)?;
        Self::from_gguf_registry(gguf, &registry)
    }

    pub fn from_gguf_registry(gguf: &GgufFile, registry: &CodecTensorRegistry) -> Result<Self> {
        let weights = Self {
            stages: registry
                .quantizer_downsample()
                .stages
                .iter()
                .map(|stage| CodecDownsampleStageF16Weights::from_names(gguf, stage))
                .collect::<Result<Vec<_>>>()?,
        };
        weights.validate_dimensions()?;
        Ok(weights)
    }

    fn validate_dimensions(&self) -> Result<()> {
        if self.stages.len() != CODEC_UPSAMPLE_STAGES {
            return Err(InferError::Message(format!(
                "codec downsample stage count mismatch: expected {}, got {}",
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
pub struct CodecDownsampleStageF16Weights {
    pub index: usize,
    pub conv_weight: F16TensorView,
    pub conv_bias: F16TensorView,
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

impl CodecDownsampleStageF16Weights {
    fn from_names(gguf: &GgufFile, names: &CodecDownsampleStageWeights) -> Result<Self> {
        Ok(Self {
            index: names.index,
            conv_weight: F16TensorView::from_gguf(gguf, &names.conv_weight)?,
            conv_bias: F16TensorView::from_gguf(gguf, &names.conv_bias)?,
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
                self.conv_weight.name(),
                self.conv_weight.dimensions(),
                vec![CODEC_UPSAMPLE_FACTOR, hidden, hidden],
            ),
            (
                self.conv_bias.name(),
                self.conv_bias.dimensions(),
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

#[derive(Debug, Clone, PartialEq)]
pub struct CodecEncodeStageResult {
    pub input_frames: u32,
    pub output_frames: u32,
    pub hidden_dim: usize,
    pub hidden: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecReferenceEncoderF16Weights {
    pub encoder: CodecEncoderF16Weights,
    pub quantizer_downsample: CodecDownsampleF16Weights,
    pub quantizer_pre_module: CodecPreModuleF16Weights,
    pub rvq: CodecF16Weights,
}

impl CodecReferenceEncoderF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let registry = CodecTensorRegistry::from_gguf(gguf)?;
        Ok(Self {
            encoder: CodecEncoderF16Weights::from_gguf(gguf)?,
            quantizer_downsample: CodecDownsampleF16Weights::from_gguf_registry(gguf, &registry)?,
            quantizer_pre_module: CodecPreModuleF16Weights::from_gguf_registry(gguf, &registry)?,
            rvq: CodecF16Weights::from_gguf_registry(gguf, &registry)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecReferenceAudioResult {
    pub input_samples: u32,
    pub padded_samples: u32,
    pub encoder_frames: u32,
    pub quantizer_frames: u32,
    pub num_codebooks: u32,
    pub codes: Vec<i32>,
    pub final_residual_l2: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecEncoderFrontendResult {
    pub input_samples: u32,
    pub padded_samples: u32,
    pub output_frames: u32,
    pub hidden_dim: usize,
    pub hidden: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CodecEncoderFrontendCheckpoint {
    pub name: String,
    pub frames: usize,
    pub channels: usize,
    pub hidden_len: usize,
    pub hidden_l2: f64,
    pub hidden_mean_abs: f64,
    pub hidden_max_abs: f64,
    pub hidden_first8: Vec<f64>,
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

pub fn forward_codec_quantizer_encode_stage(
    latents: &[f32],
    n_frames: u32,
    downsample_weights: &CodecDownsampleF16Weights,
    pre_weights: &CodecPreModuleF16Weights,
) -> Result<CodecEncodeStageResult> {
    let mut frames = usize::try_from(n_frames)
        .map_err(|_| InferError::Message("n_frames overflows usize".into()))?;
    let hidden_dim = CODEC_HIDDEN_SIZE as usize;
    validate_frame_major_len("codec quantizer encode input", latents, frames, hidden_dim)?;
    let mut current = latents.to_vec();
    for stage in &downsample_weights.stages {
        current = forward_codec_downsample_stage(&current, frames, stage)?;
        frames = downsample_output_frames(frames)?;
    }
    let mut tokens = current
        .chunks_exact(hidden_dim)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    for layer in &pre_weights.layers {
        tokens = forward_codec_transformer_layer(&tokens, layer)?;
    }
    for token in &mut tokens {
        *token = rms_norm(token, pre_weights.norm_weight.values(), CODEC_RVQ_NORM_EPS)?;
    }
    let output_frames = u32::try_from(frames).map_err(|_| {
        InferError::Message("codec quantizer encode output_frames overflows u32".into())
    })?;
    Ok(CodecEncodeStageResult {
        input_frames: n_frames,
        output_frames,
        hidden_dim,
        hidden: tokens.into_iter().flatten().collect(),
    })
}

pub fn encode_reference_audio(
    audio: &[f32],
    weights: &CodecReferenceEncoderF16Weights,
) -> Result<CodecReferenceAudioResult> {
    let encoder = forward_codec_encoder_frontend(audio, &weights.encoder)?;
    let quantizer = forward_codec_quantizer_encode_stage(
        &encoder.hidden,
        encoder.output_frames,
        &weights.quantizer_downsample,
        &weights.quantizer_pre_module,
    )?;
    let vq = rvq_encode_latents_nearest(&quantizer.hidden, quantizer.output_frames, &weights.rvq)?;
    Ok(CodecReferenceAudioResult {
        input_samples: encoder.input_samples,
        padded_samples: encoder.padded_samples,
        encoder_frames: encoder.output_frames,
        quantizer_frames: quantizer.output_frames,
        num_codebooks: vq.num_codebooks,
        codes: vq.codes,
        final_residual_l2: vq.final_residual_l2,
    })
}

pub fn encode_reference_wav_file(
    path: &Path,
    weights: &CodecReferenceEncoderF16Weights,
) -> Result<CodecReferenceAudioResult> {
    let audio = read_wav_mono_f32(path, CODEC_SAMPLE_RATE)?;
    encode_reference_audio(&audio, weights)
}

pub fn forward_codec_encoder_frontend(
    audio: &[f32],
    weights: &CodecEncoderF16Weights,
) -> Result<CodecEncoderFrontendResult> {
    forward_codec_encoder_frontend_impl(audio, weights, false).map(|(result, _)| result)
}

pub fn forward_codec_encoder_frontend_with_checkpoints(
    audio: &[f32],
    weights: &CodecEncoderF16Weights,
) -> Result<(
    CodecEncoderFrontendResult,
    Vec<CodecEncoderFrontendCheckpoint>,
)> {
    forward_codec_encoder_frontend_impl(audio, weights, true)
}

fn forward_codec_encoder_frontend_impl(
    audio: &[f32],
    weights: &CodecEncoderF16Weights,
    collect_checkpoints: bool,
) -> Result<(
    CodecEncoderFrontendResult,
    Vec<CodecEncoderFrontendCheckpoint>,
)> {
    let input_samples = u32::try_from(audio.len())
        .map_err(|_| InferError::Message("encoder input sample count overflows u32".into()))?;
    let padded_samples = pad_sample_count(audio.len(), CODEC_FRAME_LENGTH)?;
    let mut current = vec![0.0f32; padded_samples];
    current[..audio.len()].copy_from_slice(audio);
    let mut checkpoints = Vec::new();

    current = causal_conv_1d_frame_major(
        &current,
        Conv1dFrameMajorSpec {
            frames: padded_samples,
            in_ch: 1,
            out_ch: CODEC_ENCODER_ENTRY_CHANNELS,
            stride: 1,
            dilation: 1,
        },
        weights.entry_conv_weight.values(),
        weights.entry_conv_bias.values(),
    )?;
    let mut frames = padded_samples;
    let mut channels = CODEC_ENCODER_ENTRY_CHANNELS;
    maybe_push_encoder_checkpoint(
        &mut checkpoints,
        collect_checkpoints,
        "entry_conv",
        frames,
        channels,
        &current,
    )?;

    for (block, &stride) in weights.blocks.iter().zip(CODEC_ENCODER_RATES.iter()) {
        current = forward_codec_encoder_block(&current, frames, channels, stride, block)?;
        frames = encoder_downsample_output_frames(frames, stride)?;
        channels = encoder_block_output_channels(block.index)?;
        maybe_push_encoder_checkpoint(
            &mut checkpoints,
            collect_checkpoints,
            format!("encoder_block_{}", block.index),
            frames,
            channels,
            &current,
        )?;
    }

    current = snake_activation_frame_major(
        &current,
        frames,
        channels,
        weights.tail_snake_alpha.values(),
    )?;
    maybe_push_encoder_checkpoint(
        &mut checkpoints,
        collect_checkpoints,
        "tail_snake",
        frames,
        channels,
        &current,
    )?;
    current = causal_conv_1d_frame_major(
        &current,
        Conv1dFrameMajorSpec {
            frames,
            in_ch: channels,
            out_ch: CODEC_LATENT_DIM,
            stride: 1,
            dilation: 1,
        },
        weights.output_conv_weight.values(),
        weights.output_conv_bias.values(),
    )?;
    maybe_push_encoder_checkpoint(
        &mut checkpoints,
        collect_checkpoints,
        "output_conv",
        frames,
        CODEC_LATENT_DIM,
        &current,
    )?;
    let output_frames = u32::try_from(frames)
        .map_err(|_| InferError::Message("encoder output_frames overflows u32".into()))?;
    let padded_samples_u32 = u32::try_from(padded_samples)
        .map_err(|_| InferError::Message("encoder padded_samples overflows u32".into()))?;
    Ok((
        CodecEncoderFrontendResult {
            input_samples,
            padded_samples: padded_samples_u32,
            output_frames,
            hidden_dim: CODEC_LATENT_DIM,
            hidden: current,
        },
        checkpoints,
    ))
}

fn maybe_push_encoder_checkpoint(
    checkpoints: &mut Vec<CodecEncoderFrontendCheckpoint>,
    enabled: bool,
    name: impl Into<String>,
    frames: usize,
    channels: usize,
    hidden: &[f32],
) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    validate_frame_major_len("encoder checkpoint", hidden, frames, channels)?;
    checkpoints.push(encoder_checkpoint(name, frames, channels, hidden));
    Ok(())
}

fn encoder_checkpoint(
    name: impl Into<String>,
    frames: usize,
    channels: usize,
    hidden: &[f32],
) -> CodecEncoderFrontendCheckpoint {
    let hidden_l2 = hidden
        .iter()
        .map(|value| {
            let v = f64::from(*value);
            v * v
        })
        .sum::<f64>()
        .sqrt();
    let hidden_mean_abs = if hidden.is_empty() {
        0.0
    } else {
        hidden
            .iter()
            .map(|value| f64::from(value.abs()))
            .sum::<f64>()
            / hidden.len() as f64
    };
    let hidden_max_abs = hidden
        .iter()
        .map(|value| f64::from(value.abs()))
        .fold(0.0, f64::max);
    CodecEncoderFrontendCheckpoint {
        name: name.into(),
        frames,
        channels,
        hidden_len: hidden.len(),
        hidden_l2,
        hidden_mean_abs,
        hidden_max_abs,
        hidden_first8: hidden
            .iter()
            .take(8)
            .map(|value| f64::from(*value))
            .collect(),
    }
}

/// RVQ latents after lookup → post-module transformer → quantizer upsample (s2.cpp
/// `build_quantizer_decode_stage`). Output is frame-major `[output_frames * hidden_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecDecodeLatentsResult {
    pub input_frames: u32,
    pub output_frames: u32,
    pub hidden_dim: usize,
    pub hidden: Vec<f32>,
}

pub fn rvq_decode_latents(
    latents: &[f32],
    n_frames: u32,
    post_weights: &CodecPostModuleF16Weights,
    upsample_weights: &CodecUpsampleF16Weights,
) -> Result<CodecDecodeLatentsResult> {
    let post = forward_codec_post_module(latents, n_frames, post_weights)?;
    let upsample = forward_codec_upsample(&post.hidden, post.n_frames, upsample_weights)?;
    Ok(CodecDecodeLatentsResult {
        input_frames: upsample.input_frames,
        output_frames: upsample.output_frames,
        hidden_dim: upsample.hidden_dim,
        hidden: upsample.hidden,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecResidualUnitWeights {
    pub snake0_alpha: String,
    pub conv0_weight: String,
    pub conv0_bias: String,
    pub snake1_alpha: String,
    pub conv1_weight: String,
    pub conv1_bias: String,
}

impl CodecResidualUnitWeights {
    fn new(prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        Self {
            snake0_alpha: format!("{prefix}.block.0.alpha"),
            conv0_weight: format!("{prefix}.block.1.conv.weight"),
            conv0_bias: format!("{prefix}.block.1.conv.bias"),
            snake1_alpha: format!("{prefix}.block.2.alpha"),
            conv1_weight: format!("{prefix}.block.3.conv.weight"),
            conv1_bias: format!("{prefix}.block.3.conv.bias"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecEncoderBlockWeights {
    pub index: usize,
    pub residual_1: CodecResidualUnitWeights,
    pub residual_2: CodecResidualUnitWeights,
    pub residual_3: CodecResidualUnitWeights,
    pub snake_alpha: String,
    pub down_conv_weight: String,
    pub down_conv_bias: String,
    pub transformer_layers: Vec<CodecTransformerLayerWeights>,
    pub transformer_norm_weight: Option<String>,
}

impl CodecEncoderBlockWeights {
    fn new(index: usize) -> Self {
        let prefix = format!("encoder.block.{index}.block");
        let transformer_layers = if index == CODEC_ENCODER_BLOCK_COUNT {
            (0..CODEC_ENCODER_TRANSFORMER_LAYERS)
                .map(|layer| CodecTransformerLayerWeights::new(format!("{prefix}.5"), layer))
                .collect()
        } else {
            Vec::new()
        };
        let transformer_norm_weight =
            (index == CODEC_ENCODER_BLOCK_COUNT).then(|| format!("{prefix}.5.norm.weight"));
        Self {
            index,
            residual_1: CodecResidualUnitWeights::new(format!("{prefix}.0")),
            residual_2: CodecResidualUnitWeights::new(format!("{prefix}.1")),
            residual_3: CodecResidualUnitWeights::new(format!("{prefix}.2")),
            snake_alpha: format!("{prefix}.3.alpha"),
            down_conv_weight: format!("{prefix}.4.conv.weight"),
            down_conv_bias: format!("{prefix}.4.conv.bias"),
            transformer_layers,
            transformer_norm_weight,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecEncoderWeights {
    pub entry_conv_weight: String,
    pub entry_conv_bias: String,
    pub blocks: Vec<CodecEncoderBlockWeights>,
    pub tail_snake_alpha: String,
    pub output_conv_weight: String,
    pub output_conv_bias: String,
}

impl CodecEncoderWeights {
    pub fn new() -> Self {
        Self {
            entry_conv_weight: "encoder.block.0.conv.weight".to_string(),
            entry_conv_bias: "encoder.block.0.conv.bias".to_string(),
            blocks: (1..=CODEC_ENCODER_BLOCK_COUNT)
                .map(CodecEncoderBlockWeights::new)
                .collect(),
            tail_snake_alpha: format!("encoder.block.{CODEC_ENCODER_TAIL_BLOCK}.alpha"),
            output_conv_weight: format!("encoder.block.{CODEC_ENCODER_OUTPUT_BLOCK}.conv.weight"),
            output_conv_bias: format!("encoder.block.{CODEC_ENCODER_OUTPUT_BLOCK}.conv.bias"),
        }
    }
}

impl Default for CodecEncoderWeights {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecDecoderBlockWeights {
    pub index: usize,
    pub snake_alpha: String,
    pub conv_transpose_weight: String,
    pub conv_transpose_bias: String,
    pub residual_1: CodecResidualUnitWeights,
    pub residual_2: CodecResidualUnitWeights,
    pub residual_3: CodecResidualUnitWeights,
}

impl CodecDecoderBlockWeights {
    fn new(index: usize) -> Self {
        let prefix = format!("decoder.model.{index}");
        Self {
            index,
            snake_alpha: format!("{prefix}.block.0.alpha"),
            conv_transpose_weight: format!("{prefix}.block.1.conv.weight"),
            conv_transpose_bias: format!("{prefix}.block.1.conv.bias"),
            residual_1: CodecResidualUnitWeights::new(format!("{prefix}.block.2")),
            residual_2: CodecResidualUnitWeights::new(format!("{prefix}.block.3")),
            residual_3: CodecResidualUnitWeights::new(format!("{prefix}.block.4")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecDecoderWeights {
    pub entry_conv_weight: String,
    pub entry_conv_bias: String,
    pub blocks: Vec<CodecDecoderBlockWeights>,
    pub tail_snake_alpha: String,
    pub output_conv_weight: String,
    pub output_conv_bias: String,
}

impl CodecDecoderWeights {
    pub fn new() -> Self {
        Self {
            entry_conv_weight: "decoder.model.0.conv.weight".to_string(),
            entry_conv_bias: "decoder.model.0.conv.bias".to_string(),
            blocks: (1..=CODEC_DECODER_BLOCK_COUNT)
                .map(CodecDecoderBlockWeights::new)
                .collect(),
            tail_snake_alpha: format!("decoder.model.{CODEC_DECODER_TAIL_BLOCK}.alpha"),
            output_conv_weight: format!("decoder.model.{CODEC_DECODER_OUTPUT_BLOCK}.conv.weight"),
            output_conv_bias: format!("decoder.model.{CODEC_DECODER_OUTPUT_BLOCK}.conv.bias"),
        }
    }
}

impl Default for CodecDecoderWeights {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecResidualUnitF16Weights {
    pub snake0_alpha: F16TensorView,
    pub conv0_weight: F16TensorView,
    pub conv0_bias: F16TensorView,
    pub snake1_alpha: F16TensorView,
    pub conv1_weight: F16TensorView,
    pub conv1_bias: F16TensorView,
}

impl CodecResidualUnitF16Weights {
    fn from_names(gguf: &GgufFile, names: &CodecResidualUnitWeights) -> Result<Self> {
        Ok(Self {
            snake0_alpha: F16TensorView::from_gguf(gguf, &names.snake0_alpha)?,
            conv0_weight: F16TensorView::from_gguf(gguf, &names.conv0_weight)?,
            conv0_bias: F16TensorView::from_gguf(gguf, &names.conv0_bias)?,
            snake1_alpha: F16TensorView::from_gguf(gguf, &names.snake1_alpha)?,
            conv1_weight: F16TensorView::from_gguf(gguf, &names.conv1_weight)?,
            conv1_bias: F16TensorView::from_gguf(gguf, &names.conv1_bias)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecEncoderBlockF16Weights {
    pub index: usize,
    pub residual_1: CodecResidualUnitF16Weights,
    pub residual_2: CodecResidualUnitF16Weights,
    pub residual_3: CodecResidualUnitF16Weights,
    pub snake_alpha: F16TensorView,
    pub down_conv_weight: F16TensorView,
    pub down_conv_bias: F16TensorView,
    pub transformer_layers: Vec<CodecTransformerLayerF16Weights>,
    pub transformer_norm_weight: Option<F16TensorView>,
}

impl CodecEncoderBlockF16Weights {
    fn from_names(gguf: &GgufFile, names: &CodecEncoderBlockWeights) -> Result<Self> {
        Ok(Self {
            index: names.index,
            residual_1: CodecResidualUnitF16Weights::from_names(gguf, &names.residual_1)?,
            residual_2: CodecResidualUnitF16Weights::from_names(gguf, &names.residual_2)?,
            residual_3: CodecResidualUnitF16Weights::from_names(gguf, &names.residual_3)?,
            snake_alpha: F16TensorView::from_gguf(gguf, &names.snake_alpha)?,
            down_conv_weight: F16TensorView::from_gguf(gguf, &names.down_conv_weight)?,
            down_conv_bias: F16TensorView::from_gguf(gguf, &names.down_conv_bias)?,
            transformer_layers: names
                .transformer_layers
                .iter()
                .map(|layer| CodecTransformerLayerF16Weights::from_names(gguf, layer))
                .collect::<Result<Vec<_>>>()?,
            transformer_norm_weight: names
                .transformer_norm_weight
                .as_deref()
                .map(|name| F16TensorView::from_gguf(gguf, name))
                .transpose()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecEncoderF16Weights {
    pub entry_conv_weight: F16TensorView,
    pub entry_conv_bias: F16TensorView,
    pub blocks: Vec<CodecEncoderBlockF16Weights>,
    pub tail_snake_alpha: F16TensorView,
    pub output_conv_weight: F16TensorView,
    pub output_conv_bias: F16TensorView,
}

impl CodecEncoderF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let names = CodecEncoderWeights::new();
        Self::from_gguf_names(gguf, &names)
    }

    pub fn from_gguf_names(gguf: &GgufFile, names: &CodecEncoderWeights) -> Result<Self> {
        let weights = Self {
            entry_conv_weight: F16TensorView::from_gguf(gguf, &names.entry_conv_weight)?,
            entry_conv_bias: F16TensorView::from_gguf(gguf, &names.entry_conv_bias)?,
            blocks: names
                .blocks
                .iter()
                .map(|block| CodecEncoderBlockF16Weights::from_names(gguf, block))
                .collect::<Result<Vec<_>>>()?,
            tail_snake_alpha: F16TensorView::from_gguf(gguf, &names.tail_snake_alpha)?,
            output_conv_weight: F16TensorView::from_gguf(gguf, &names.output_conv_weight)?,
            output_conv_bias: F16TensorView::from_gguf(gguf, &names.output_conv_bias)?,
        };
        weights.validate_dimensions()?;
        Ok(weights)
    }

    fn validate_dimensions(&self) -> Result<()> {
        validate_f16_dims(
            self.entry_conv_weight.name(),
            self.entry_conv_weight.dimensions(),
            &[7, 1, CODEC_ENCODER_ENTRY_CHANNELS],
        )?;
        validate_f16_dims(
            self.entry_conv_bias.name(),
            self.entry_conv_bias.dimensions(),
            &[CODEC_ENCODER_ENTRY_CHANNELS],
        )?;
        if self.blocks.len() != CODEC_ENCODER_BLOCK_COUNT {
            return Err(InferError::Message(format!(
                "encoder block count mismatch: expected {}, got {}",
                CODEC_ENCODER_BLOCK_COUNT,
                self.blocks.len()
            )));
        }
        for (offset, block) in self.blocks.iter().enumerate() {
            let index = offset + 1;
            let in_channels = CODEC_ENCODER_CHANNELS[offset];
            let out_channels = CODEC_ENCODER_CHANNELS[offset + 1];
            validate_encoder_block_f16_dims(block, index, in_channels, out_channels)?;
        }
        validate_f16_dims(
            self.tail_snake_alpha.name(),
            self.tail_snake_alpha.dimensions(),
            &[1, CODEC_LATENT_DIM, 1],
        )?;
        validate_f16_dims(
            self.output_conv_weight.name(),
            self.output_conv_weight.dimensions(),
            &[3, CODEC_LATENT_DIM, CODEC_LATENT_DIM],
        )?;
        validate_f16_dims(
            self.output_conv_bias.name(),
            self.output_conv_bias.dimensions(),
            &[CODEC_LATENT_DIM],
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecDecoderBlockF16Weights {
    pub index: usize,
    pub snake_alpha: F16TensorView,
    pub conv_transpose_weight: F16TensorView,
    pub conv_transpose_bias: F16TensorView,
    pub residual_1: CodecResidualUnitF16Weights,
    pub residual_2: CodecResidualUnitF16Weights,
    pub residual_3: CodecResidualUnitF16Weights,
}

impl CodecDecoderBlockF16Weights {
    fn from_names(gguf: &GgufFile, names: &CodecDecoderBlockWeights) -> Result<Self> {
        Ok(Self {
            index: names.index,
            snake_alpha: F16TensorView::from_gguf(gguf, &names.snake_alpha)?,
            conv_transpose_weight: F16TensorView::from_gguf(gguf, &names.conv_transpose_weight)?,
            conv_transpose_bias: F16TensorView::from_gguf(gguf, &names.conv_transpose_bias)?,
            residual_1: CodecResidualUnitF16Weights::from_names(gguf, &names.residual_1)?,
            residual_2: CodecResidualUnitF16Weights::from_names(gguf, &names.residual_2)?,
            residual_3: CodecResidualUnitF16Weights::from_names(gguf, &names.residual_3)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecDecoderF16Weights {
    pub entry_conv_weight: F16TensorView,
    pub entry_conv_bias: F16TensorView,
    pub blocks: Vec<CodecDecoderBlockF16Weights>,
    pub tail_snake_alpha: F16TensorView,
    pub output_conv_weight: F16TensorView,
    pub output_conv_bias: F16TensorView,
}

impl CodecDecoderF16Weights {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let names = CodecDecoderWeights::new();
        Self::from_gguf_names(gguf, &names)
    }

    pub fn from_gguf_names(gguf: &GgufFile, names: &CodecDecoderWeights) -> Result<Self> {
        let weights = Self {
            entry_conv_weight: F16TensorView::from_gguf(gguf, &names.entry_conv_weight)?,
            entry_conv_bias: F16TensorView::from_gguf(gguf, &names.entry_conv_bias)?,
            blocks: names
                .blocks
                .iter()
                .map(|block| CodecDecoderBlockF16Weights::from_names(gguf, block))
                .collect::<Result<Vec<_>>>()?,
            tail_snake_alpha: F16TensorView::from_gguf(gguf, &names.tail_snake_alpha)?,
            output_conv_weight: F16TensorView::from_gguf(gguf, &names.output_conv_weight)?,
            output_conv_bias: F16TensorView::from_gguf(gguf, &names.output_conv_bias)?,
        };
        weights.validate_dimensions()?;
        Ok(weights)
    }

    fn validate_dimensions(&self) -> Result<()> {
        validate_f16_dims(
            self.entry_conv_weight.name(),
            self.entry_conv_weight.dimensions(),
            &[7, CODEC_LATENT_DIM, CODEC_DECODER_ENTRY_CHANNELS],
        )?;
        validate_f16_dims(
            self.entry_conv_bias.name(),
            self.entry_conv_bias.dimensions(),
            &[CODEC_DECODER_ENTRY_CHANNELS],
        )?;
        if self.blocks.len() != CODEC_DECODER_BLOCK_COUNT {
            return Err(InferError::Message(format!(
                "decoder block count mismatch: expected {}, got {}",
                CODEC_DECODER_BLOCK_COUNT,
                self.blocks.len()
            )));
        }
        validate_f16_dims(
            self.tail_snake_alpha.name(),
            self.tail_snake_alpha.dimensions(),
            &[1, 96, 1],
        )?;
        validate_f16_dims(
            self.output_conv_weight.name(),
            self.output_conv_weight.dimensions(),
            &[7, 96, 1],
        )?;
        validate_f16_dims(
            self.output_conv_bias.name(),
            self.output_conv_bias.dimensions(),
            &[1],
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecWaveformResult {
    pub sample_rate: u32,
    pub input_frames: u32,
    pub latent_frames: u32,
    pub num_samples: usize,
    pub samples: Vec<f32>,
}

/// Full codec decode: RVQ codes → mono PCM in [-1, 1] after decoder `tanh`.
pub fn decode_waveform(
    codes: &[i32],
    num_codebooks: u32,
    n_frames: u32,
    rvq_weights: &CodecF16Weights,
    post_weights: &CodecPostModuleF16Weights,
    upsample_weights: &CodecUpsampleF16Weights,
    decoder_weights: &CodecDecoderF16Weights,
) -> Result<CodecWaveformResult> {
    let rvq = rvq_lookup_codes(codes, num_codebooks, n_frames, rvq_weights)?;
    let latents = rvq_decode_latents(&rvq.latents, rvq.n_frames, post_weights, upsample_weights)?;
    let samples = forward_codec_decoder(
        &latents.hidden,
        latents.output_frames,
        latents.hidden_dim,
        decoder_weights,
    )?;
    Ok(CodecWaveformResult {
        sample_rate: CODEC_SAMPLE_RATE,
        input_frames: n_frames,
        latent_frames: latents.output_frames,
        num_samples: samples.len(),
        samples,
    })
}

pub fn decode_waveform_to_wav(
    codes: &[i32],
    num_codebooks: u32,
    n_frames: u32,
    rvq_weights: &CodecF16Weights,
    post_weights: &CodecPostModuleF16Weights,
    upsample_weights: &CodecUpsampleF16Weights,
    decoder_weights: &CodecDecoderF16Weights,
) -> Result<Vec<u8>> {
    let waveform = decode_waveform(
        codes,
        num_codebooks,
        n_frames,
        rvq_weights,
        post_weights,
        upsample_weights,
        decoder_weights,
    )?;
    Ok(crate::wav::pcm_to_wav(
        &waveform.samples,
        waveform.sample_rate,
    ))
}

fn forward_codec_encoder_block(
    input: &[f32],
    frames: usize,
    channels: usize,
    stride: usize,
    weights: &CodecEncoderBlockF16Weights,
) -> Result<Vec<f32>> {
    validate_frame_major_len("encoder block input", input, frames, channels)?;
    let mut x = forward_codec_residual_unit(input, frames, channels, 1, &weights.residual_1)?;
    x = forward_codec_residual_unit(&x, frames, channels, 3, &weights.residual_2)?;
    x = forward_codec_residual_unit(&x, frames, channels, 9, &weights.residual_3)?;
    x = snake_activation_frame_major(&x, frames, channels, weights.snake_alpha.values())?;
    let out_channels = encoder_block_output_channels(weights.index)?;
    x = causal_conv_1d_frame_major(
        &x,
        Conv1dFrameMajorSpec {
            frames,
            in_ch: channels,
            out_ch: out_channels,
            stride,
            dilation: 1,
        },
        weights.down_conv_weight.values(),
        weights.down_conv_bias.values(),
    )?;

    if weights.transformer_layers.is_empty() {
        return Ok(x);
    }
    let out_frames = encoder_downsample_output_frames(frames, stride)?;
    let mut tokens = x
        .chunks_exact(out_channels)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    for layer in &weights.transformer_layers {
        tokens = forward_codec_transformer_layer_with_window(
            &tokens,
            layer,
            CODEC_ENCODER_TRANSFORMER_WINDOW_SIZE,
        )?;
    }
    let norm_weight = weights
        .transformer_norm_weight
        .as_ref()
        .ok_or_else(|| InferError::Message("missing encoder transformer norm".into()))?;
    for token in &mut tokens {
        *token = rms_norm(token, norm_weight.values(), CODEC_RVQ_NORM_EPS)?;
    }
    let hidden = tokens.into_iter().flatten().collect::<Vec<_>>();
    validate_frame_major_len(
        "encoder transformer output",
        &hidden,
        out_frames,
        out_channels,
    )?;
    Ok(hidden)
}

fn encoder_downsample_output_frames(frames: usize, stride: usize) -> Result<usize> {
    if stride == 0 {
        return Err(InferError::Message(
            "encoder downsample stride must be non-zero".into(),
        ));
    }
    frames
        .checked_add(stride - 1)
        .map(|value| value / stride)
        .ok_or_else(|| InferError::Message("encoder downsample frame count overflow".into()))
}

fn encoder_block_output_channels(index: usize) -> Result<usize> {
    CODEC_ENCODER_CHANNELS
        .get(index)
        .copied()
        .ok_or_else(|| InferError::Message(format!("invalid encoder block index {index}")))
}

fn pad_sample_count(samples: usize, frame_length: usize) -> Result<usize> {
    if frame_length == 0 {
        return Err(InferError::Message(
            "codec frame_length must be non-zero".into(),
        ));
    }
    if samples == 0 {
        return Ok(frame_length);
    }
    samples
        .checked_add(frame_length - 1)
        .map(|value| (value / frame_length) * frame_length)
        .ok_or_else(|| InferError::Message("codec padded sample count overflow".into()))
}

pub fn forward_codec_decoder(
    hidden: &[f32],
    n_frames: u32,
    hidden_dim: usize,
    weights: &CodecDecoderF16Weights,
) -> Result<Vec<f32>> {
    let frames = usize::try_from(n_frames)
        .map_err(|_| InferError::Message("decoder n_frames overflows usize".into()))?;
    validate_frame_major_len("decoder input", hidden, frames, hidden_dim)?;
    if hidden_dim != CODEC_LATENT_DIM {
        return Err(InferError::Message(format!(
            "decoder expects latent dim {CODEC_LATENT_DIM}, got {hidden_dim}"
        )));
    }

    let mut current = causal_conv_1d_frame_major(
        hidden,
        Conv1dFrameMajorSpec {
            frames,
            in_ch: CODEC_LATENT_DIM,
            out_ch: CODEC_DECODER_ENTRY_CHANNELS,
            stride: 1,
            dilation: 1,
        },
        weights.entry_conv_weight.values(),
        weights.entry_conv_bias.values(),
    )?;
    let mut frame_count = frames;
    let mut channels = CODEC_DECODER_ENTRY_CHANNELS;

    for (block, &stride) in weights.blocks.iter().zip(CODEC_DECODER_RATES.iter()) {
        current = forward_codec_decoder_block(&current, frame_count, channels, stride, block)?;
        frame_count = frame_count
            .checked_mul(stride)
            .ok_or_else(|| InferError::Message("decoder frame count overflow".into()))?;
        channels = block_output_channels(block.index)?;
    }

    current = snake_activation_frame_major(
        &current,
        frame_count,
        channels,
        weights.tail_snake_alpha.values(),
    )?;
    current = causal_conv_1d_frame_major(
        &current,
        Conv1dFrameMajorSpec {
            frames: frame_count,
            in_ch: channels,
            out_ch: 1,
            stride: 1,
            dilation: 1,
        },
        weights.output_conv_weight.values(),
        weights.output_conv_bias.values(),
    )?;
    for sample in &mut current {
        *sample = sample.tanh();
    }
    Ok(current)
}

fn block_output_channels(block_index: usize) -> Result<usize> {
    match block_index {
        1 => Ok(768),
        2 => Ok(384),
        3 => Ok(192),
        4 => Ok(96),
        other => Err(InferError::Message(format!(
            "unknown decoder block index {other}"
        ))),
    }
}

fn forward_codec_decoder_block(
    input: &[f32],
    frames: usize,
    channels: usize,
    stride: usize,
    weights: &CodecDecoderBlockF16Weights,
) -> Result<Vec<f32>> {
    validate_frame_major_len("decoder block input", input, frames, channels)?;
    let mut x =
        snake_activation_frame_major(input, frames, channels, weights.snake_alpha.values())?;
    let out_channels = block_output_channels(weights.index)?;
    x = causal_conv_transpose_1d(
        &x,
        ConvTranspose1dSpec {
            frames,
            in_ch: channels,
            out_ch: out_channels,
            stride,
            crop_right: 0,
        },
        weights.conv_transpose_weight.values(),
        weights.conv_transpose_bias.values(),
    )?;
    let out_frames = frames
        .checked_mul(stride)
        .ok_or_else(|| InferError::Message("decoder block frame count overflow".into()))?;
    x = forward_codec_residual_unit(&x, out_frames, out_channels, 1, &weights.residual_1)?;
    x = forward_codec_residual_unit(&x, out_frames, out_channels, 3, &weights.residual_2)?;
    forward_codec_residual_unit(&x, out_frames, out_channels, 9, &weights.residual_3)
}

fn forward_codec_residual_unit(
    input: &[f32],
    frames: usize,
    channels: usize,
    dilation: usize,
    weights: &CodecResidualUnitF16Weights,
) -> Result<Vec<f32>> {
    validate_frame_major_len("residual unit input", input, frames, channels)?;
    let mut branch =
        snake_activation_frame_major(input, frames, channels, weights.snake0_alpha.values())?;
    branch = causal_conv_1d_frame_major(
        &branch,
        Conv1dFrameMajorSpec {
            frames,
            in_ch: channels,
            out_ch: channels,
            stride: 1,
            dilation,
        },
        weights.conv0_weight.values(),
        weights.conv0_bias.values(),
    )?;
    branch =
        snake_activation_frame_major(&branch, frames, channels, weights.snake1_alpha.values())?;
    branch = causal_conv_1d_frame_major(
        &branch,
        Conv1dFrameMajorSpec {
            frames,
            in_ch: channels,
            out_ch: channels,
            stride: 1,
            dilation: 1,
        },
        weights.conv1_weight.values(),
        weights.conv1_bias.values(),
    )?;
    add_frame_major_residual(input, &branch)
}

fn snake_activation_frame_major(
    input: &[f32],
    frames: usize,
    channels: usize,
    alpha: &[f32],
) -> Result<Vec<f32>> {
    validate_frame_major_len("snake input", input, frames, channels)?;
    if alpha.len() != channels {
        return Err(InferError::Message(format!(
            "snake alpha length mismatch: expected {channels}, got {}",
            alpha.len()
        )));
    }
    let mut output = input.to_vec();
    for frame in 0..frames {
        let row = &mut output[frame * channels..(frame + 1) * channels];
        for (slot, &alpha) in row.iter_mut().zip(alpha) {
            let safe_alpha = if alpha.abs() < 1e-8 { 1e-8 } else { alpha };
            let ax = safe_alpha * *slot;
            let sin_ax = ax.sin();
            *slot += (sin_ax * sin_ax) / safe_alpha;
        }
    }
    Ok(output)
}

fn extra_padding_for_conv1d(
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let n_frames =
        (length as f32 - kernel_size as f32 + padding_total as f32) / stride as f32 + 1.0;
    let ideal = (n_frames.ceil() as usize)
        .saturating_sub(1)
        .saturating_mul(stride)
        .saturating_add(kernel_size.saturating_sub(padding_total));
    ideal.saturating_sub(length)
}

#[derive(Debug, Clone, Copy)]
struct Conv1dFrameMajorSpec {
    frames: usize,
    in_ch: usize,
    out_ch: usize,
    stride: usize,
    dilation: usize,
}

fn causal_conv_1d_frame_major(
    input: &[f32],
    spec: Conv1dFrameMajorSpec,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    let Conv1dFrameMajorSpec {
        frames,
        in_ch,
        out_ch,
        stride,
        dilation,
    } = spec;
    validate_frame_major_len("causal conv input", input, frames, in_ch)?;
    let weight_kernel = infer_conv_kernel_dim(weight.len(), in_ch, out_ch);
    let kernel_size = (weight_kernel - 1) * dilation + 1;
    let expected_weights = weight_kernel
        .checked_mul(in_ch)
        .and_then(|value| value.checked_mul(out_ch))
        .ok_or_else(|| InferError::Message("causal conv weight length overflow".into()))?;
    if weight.len() != expected_weights {
        return Err(InferError::Message(format!(
            "causal conv weight length mismatch: expected {expected_weights}, got {}",
            weight.len()
        )));
    }
    if bias.len() != out_ch {
        return Err(InferError::Message(format!(
            "causal conv bias length mismatch: expected {out_ch}, got {}",
            bias.len()
        )));
    }

    let padding_total = kernel_size.saturating_sub(stride);
    let left_pad = padding_total;
    let extra = extra_padding_for_conv1d(frames, kernel_size, stride, padding_total);
    let padded_frames = frames
        .checked_add(left_pad)
        .and_then(|value| value.checked_add(extra))
        .ok_or_else(|| InferError::Message("causal conv padded frame overflow".into()))?;
    let mut padded = vec![0.0f32; padded_frames * in_ch];
    for frame in 0..frames {
        let dst = (frame + left_pad) * in_ch;
        padded[dst..dst + in_ch].copy_from_slice(&input[frame * in_ch..(frame + 1) * in_ch]);
    }

    let out_frames = padded_frames
        .saturating_sub(kernel_size)
        .checked_div(stride)
        .map(|value| value + 1)
        .ok_or_else(|| InferError::Message("causal conv output frame overflow".into()))?;
    let mut output = vec![0.0f32; out_frames * out_ch];
    for out_frame in 0..out_frames {
        let start = out_frame * stride;
        for out_channel in 0..out_ch {
            let mut sum = bias[out_channel];
            for tap in 0..weight_kernel {
                let source_frame = start + tap * dilation;
                if source_frame >= padded_frames {
                    continue;
                }
                for in_channel in 0..in_ch {
                    let weight_index = ggml_tensor_index_3d(
                        weight_kernel,
                        in_ch,
                        out_ch,
                        tap,
                        in_channel,
                        out_channel,
                    );
                    let input_index = source_frame * in_ch + in_channel;
                    sum += padded[input_index] * weight[weight_index];
                }
            }
            output[out_frame * out_ch + out_channel] = sum;
        }
    }
    Ok(output)
}

fn infer_conv_kernel_dim(weight_len: usize, in_ch: usize, out_ch: usize) -> usize {
    if in_ch == 0 || out_ch == 0 {
        return 0;
    }
    weight_len / (in_ch * out_ch)
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

    fn nearest_code_for_residual(
        &self,
        residual: &[f32],
        codebook_size: usize,
    ) -> Result<CodecNearestCodeResult> {
        let hidden_dim = CODEC_HIDDEN_SIZE as usize;
        let projection_dim = CODEC_PROJECTION_DIM as usize;
        if residual.len() != hidden_dim {
            return Err(InferError::Message(format!(
                "codec VQ residual length mismatch: expected {hidden_dim}, got {}",
                residual.len()
            )));
        }
        let mut projected = linear(
            residual,
            self.in_proj_weight.values(),
            hidden_dim,
            projection_dim,
        )?;
        add_bias(&mut projected, self.in_proj_bias.values())?;

        let codebook = self.codebook_weight.values();
        let expected_codebook_len = projection_dim
            .checked_mul(codebook_size)
            .ok_or_else(|| InferError::Message("codec codebook length overflow".into()))?;
        if codebook.len() != expected_codebook_len {
            return Err(InferError::Message(format!(
                "codec codebook length mismatch: expected {expected_codebook_len}, got {}",
                codebook.len()
            )));
        }

        let projected_norm = projected
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .max(1e-12)
            .sqrt();
        let mut best_code = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for code in 0..codebook_size {
            let row = &codebook[code * projection_dim..(code + 1) * projection_dim];
            let row_norm = row
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .max(1e-12)
                .sqrt();
            let mut score = 0.0f32;
            for (actual, candidate) in projected.iter().zip(row) {
                score += (actual / projected_norm) * (candidate / row_norm);
            }
            if score > best_score {
                best_code = code;
                best_score = score;
            }
        }

        let reconstructed = self.project_code(best_code as u32, codebook_size)?;
        Ok(CodecNearestCodeResult {
            code: best_code as u32,
            reconstructed,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecRvqLookupResult {
    pub num_codebooks: u32,
    pub n_frames: u32,
    pub latent_dim: usize,
    pub latents: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecVqEncodeResult {
    pub num_codebooks: u32,
    pub n_frames: u32,
    pub latent_dim: usize,
    pub codes: Vec<i32>,
    pub final_residual_l2: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct CodecNearestCodeResult {
    code: u32,
    reconstructed: Vec<f32>,
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

pub fn rvq_encode_latents_nearest(
    latents: &[f32],
    n_frames: u32,
    weights: &CodecF16Weights,
) -> Result<CodecVqEncodeResult> {
    let num_codebooks = 1usize
        .checked_add(weights.residual_quantizers.len())
        .ok_or_else(|| InferError::Message("codec codebook count overflow".into()))?;
    let n_frames_usize = usize::try_from(n_frames)
        .map_err(|_| InferError::Message("n_frames overflows usize".into()))?;
    let latent_dim = CODEC_HIDDEN_SIZE as usize;
    validate_frame_major_len("codec VQ encode input", latents, n_frames_usize, latent_dim)?;

    let codes_len = num_codebooks
        .checked_mul(n_frames_usize)
        .ok_or_else(|| InferError::Message("codec VQ codes length overflow".into()))?;
    let mut codes = vec![0i32; codes_len];
    let mut final_residual_l2 = Vec::with_capacity(n_frames_usize);

    for frame in 0..n_frames_usize {
        let start = frame * latent_dim;
        let mut residual = latents[start..start + latent_dim].to_vec();

        for codebook in 0..num_codebooks {
            let (quantizer, codebook_size) = if codebook == 0 {
                (
                    &weights.semantic_quantizer,
                    CODEC_SEMANTIC_CODEBOOK_SIZE as usize,
                )
            } else {
                (
                    &weights.residual_quantizers[codebook - 1],
                    CODEC_RESIDUAL_CODEBOOK_SIZE as usize,
                )
            };
            let nearest = quantizer.nearest_code_for_residual(&residual, codebook_size)?;
            codes[codebook * n_frames_usize + frame] = i32::try_from(nearest.code)
                .map_err(|_| InferError::Message("codec VQ code overflows i32".into()))?;
            for (slot, reconstructed) in residual.iter_mut().zip(nearest.reconstructed) {
                *slot -= reconstructed;
            }
        }

        final_residual_l2.push(
            residual
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt(),
        );
    }

    let num_codebooks = u32::try_from(num_codebooks)
        .map_err(|_| InferError::Message("codec codebook count overflows u32".into()))?;
    Ok(CodecVqEncodeResult {
        num_codebooks,
        n_frames,
        latent_dim,
        codes,
        final_residual_l2,
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

fn validate_encoder(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    weights: &CodecEncoderWeights,
    failures: &mut Vec<String>,
) {
    validate_tensor(
        tensors,
        &weights.entry_conv_weight,
        &[7, 1, CODEC_ENCODER_ENTRY_CHANNELS as u64],
        failures,
    );
    validate_tensor(
        tensors,
        &weights.entry_conv_bias,
        &[CODEC_ENCODER_ENTRY_CHANNELS as u64],
        failures,
    );
    if weights.blocks.len() != CODEC_ENCODER_BLOCK_COUNT {
        failures.push(format!(
            "encoder block count: expected {}, got {}",
            CODEC_ENCODER_BLOCK_COUNT,
            weights.blocks.len()
        ));
    }
    for (offset, block) in weights.blocks.iter().enumerate() {
        let index = offset + 1;
        let in_channels = CODEC_ENCODER_CHANNELS[offset] as u64;
        let out_channels = CODEC_ENCODER_CHANNELS[offset + 1] as u64;
        validate_encoder_block(tensors, block, index, in_channels, out_channels, failures);
    }
    validate_tensor(
        tensors,
        &weights.tail_snake_alpha,
        &[1, CODEC_LATENT_DIM as u64, 1],
        failures,
    );
    validate_tensor(
        tensors,
        &weights.output_conv_weight,
        &[3, CODEC_LATENT_DIM as u64, CODEC_LATENT_DIM as u64],
        failures,
    );
    validate_tensor(
        tensors,
        &weights.output_conv_bias,
        &[CODEC_LATENT_DIM as u64],
        failures,
    );
}

fn validate_encoder_block(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    block: &CodecEncoderBlockWeights,
    index: usize,
    in_channels: u64,
    out_channels: u64,
    failures: &mut Vec<String>,
) {
    if block.index != index {
        failures.push(format!(
            "encoder block index: expected {index}, got {}",
            block.index
        ));
    }
    validate_residual_unit(tensors, &block.residual_1, in_channels, failures);
    validate_residual_unit(tensors, &block.residual_2, in_channels, failures);
    validate_residual_unit(tensors, &block.residual_3, in_channels, failures);
    validate_tensor(tensors, &block.snake_alpha, &[1, in_channels, 1], failures);
    validate_tensor(
        tensors,
        &block.down_conv_weight,
        &[
            CODEC_ENCODER_KERNELS[index - 1] as u64,
            in_channels,
            out_channels,
        ],
        failures,
    );
    validate_tensor(tensors, &block.down_conv_bias, &[out_channels], failures);
    if index == CODEC_ENCODER_BLOCK_COUNT {
        validate_encoder_transformer(tensors, block, failures);
    } else if !block.transformer_layers.is_empty() || block.transformer_norm_weight.is_some() {
        failures.push(format!(
            "encoder block {index} unexpectedly has transformer weights"
        ));
    }
}

fn validate_residual_unit(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    unit: &CodecResidualUnitWeights,
    channels: u64,
    failures: &mut Vec<String>,
) {
    validate_tensor(tensors, &unit.snake0_alpha, &[1, channels, 1], failures);
    validate_tensor(
        tensors,
        &unit.conv0_weight,
        &[7, channels, channels],
        failures,
    );
    validate_tensor(tensors, &unit.conv0_bias, &[channels], failures);
    validate_tensor(tensors, &unit.snake1_alpha, &[1, channels, 1], failures);
    validate_tensor(
        tensors,
        &unit.conv1_weight,
        &[1, channels, channels],
        failures,
    );
    validate_tensor(tensors, &unit.conv1_bias, &[channels], failures);
}

fn validate_encoder_transformer(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    block: &CodecEncoderBlockWeights,
    failures: &mut Vec<String>,
) {
    if block.transformer_layers.len() != CODEC_ENCODER_TRANSFORMER_LAYERS {
        failures.push(format!(
            "encoder transformer layer count: expected {}, got {}",
            CODEC_ENCODER_TRANSFORMER_LAYERS,
            block.transformer_layers.len()
        ));
    }
    let prefix = format!("encoder.block.{}.block.5", block.index);
    validate_tensor(
        tensors,
        &format!("{prefix}.freqs_cis"),
        &[2, CODEC_FREQ_HEADS, CODEC_ENCODER_TRANSFORMER_CONTEXT],
        failures,
    );
    validate_tensor(
        tensors,
        &format!("{prefix}.causal_mask"),
        &[
            CODEC_ENCODER_TRANSFORMER_CONTEXT,
            CODEC_ENCODER_TRANSFORMER_CONTEXT,
        ],
        failures,
    );
    if let Some(norm) = &block.transformer_norm_weight {
        validate_tensor(tensors, norm, &[CODEC_LATENT_DIM as u64], failures);
    } else {
        failures.push(format!("missing {prefix}.norm.weight"));
    }
    for layer in &block.transformer_layers {
        validate_codec_transformer_layer(tensors, layer, failures);
    }
}

fn validate_codec_transformer_layer(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    layer: &CodecTransformerLayerWeights,
    failures: &mut Vec<String>,
) {
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

fn validate_downsample(
    tensors: &BTreeMap<String, GgufTensorInfo>,
    weights: &CodecDownsampleWeights,
    failures: &mut Vec<String>,
) {
    if weights.stages.len() != CODEC_UPSAMPLE_STAGES {
        failures.push(format!(
            "quantizer downsample stage count: expected {}, got {}",
            CODEC_UPSAMPLE_STAGES,
            weights.stages.len()
        ));
    }
    for stage in &weights.stages {
        let specs = [
            (
                &stage.conv_weight,
                vec![
                    CODEC_UPSAMPLE_FACTOR as u64,
                    CODEC_HIDDEN_SIZE,
                    CODEC_HIDDEN_SIZE,
                ],
            ),
            (&stage.conv_bias, vec![CODEC_HIDDEN_SIZE]),
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
    forward_codec_transformer_layer_with_window(tokens, weights, CODEC_RVQ_WINDOW_SIZE)
}

fn forward_codec_transformer_layer_with_window(
    tokens: &[Vec<f32>],
    weights: &CodecTransformerLayerF16Weights,
    window_size: usize,
) -> Result<Vec<Vec<f32>>> {
    if tokens.is_empty() {
        return Err(InferError::Message(
            "codec transformer requires at least one frame".into(),
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
        let visible_start = (offset + 1).saturating_sub(window_size);
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
        ConvTranspose1dSpec {
            frames,
            in_ch: hidden_dim,
            out_ch: hidden_dim,
            stride: CODEC_UPSAMPLE_FACTOR,
            crop_right: 0,
        },
        weights.conv_transpose_weight.values(),
        weights.conv_transpose_bias.values(),
    )?;
    forward_codec_convnext_block(&conv, frames * CODEC_UPSAMPLE_FACTOR, weights)
}

fn forward_codec_downsample_stage(
    input: &[f32],
    frames: usize,
    weights: &CodecDownsampleStageF16Weights,
) -> Result<Vec<f32>> {
    let hidden_dim = CODEC_HIDDEN_SIZE as usize;
    validate_frame_major_len("codec downsample stage input", input, frames, hidden_dim)?;
    let conv = causal_conv_1d_frame_major(
        input,
        Conv1dFrameMajorSpec {
            frames,
            in_ch: hidden_dim,
            out_ch: hidden_dim,
            stride: CODEC_UPSAMPLE_FACTOR,
            dilation: 1,
        },
        weights.conv_weight.values(),
        weights.conv_bias.values(),
    )?;
    forward_codec_downsample_convnext_block(&conv, downsample_output_frames(frames)?, weights)
}

fn downsample_output_frames(frames: usize) -> Result<usize> {
    if frames < CODEC_UPSAMPLE_FACTOR {
        return Err(InferError::Message(format!(
            "codec downsample requires at least {} frames, got {frames}",
            CODEC_UPSAMPLE_FACTOR
        )));
    }
    Ok((frames - CODEC_UPSAMPLE_FACTOR) / CODEC_UPSAMPLE_FACTOR + 1)
}

fn forward_codec_downsample_convnext_block(
    input: &[f32],
    frames: usize,
    weights: &CodecDownsampleStageF16Weights,
) -> Result<Vec<f32>> {
    let hidden_dim = CODEC_HIDDEN_SIZE as usize;
    validate_frame_major_len("codec downsample convnext input", input, frames, hidden_dim)?;
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

/// Row-major index for ggml 3D tensors stored as `[ne0, ne1, ne2]`.
fn ggml_tensor_index_3d(
    ne0: usize,
    ne1: usize,
    _ne2: usize,
    i0: usize,
    i1: usize,
    i2: usize,
) -> usize {
    i0 + i1 * ne0 + i2 * ne0 * ne1
}

#[derive(Debug, Clone, Copy)]
struct ConvTranspose1dSpec {
    frames: usize,
    in_ch: usize,
    out_ch: usize,
    stride: usize,
    crop_right: usize,
}

fn causal_conv_transpose_1d(
    input: &[f32],
    spec: ConvTranspose1dSpec,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    let ConvTranspose1dSpec {
        frames,
        in_ch,
        out_ch,
        stride,
        crop_right,
    } = spec;
    validate_frame_major_len("conv_transpose input", input, frames, in_ch)?;
    let kernel = infer_conv_kernel_dim(weight.len(), in_ch, out_ch);
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
    let mut output_frames = frames
        .checked_mul(stride)
        .ok_or_else(|| InferError::Message("conv_transpose output frame overflow".into()))?;
    let mut output = vec![0.0f32; output_frames * out_ch];
    for frame in 0..frames {
        let input_row = &input[frame * in_ch..(frame + 1) * in_ch];
        for kernel_index in 0..kernel {
            let output_row_start = (frame * stride + kernel_index) * out_ch;
            if output_row_start / out_ch >= output_frames {
                continue;
            }
            for output_channel in 0..out_ch {
                for (input_channel, input_value) in input_row.iter().enumerate().take(in_ch) {
                    let weight_index = ggml_tensor_index_3d(
                        kernel,
                        out_ch,
                        in_ch,
                        kernel_index,
                        output_channel,
                        input_channel,
                    );
                    output[output_row_start + output_channel] += input_value * weight[weight_index];
                }
            }
        }
    }
    add_frame_bias(&mut output, out_ch, bias)?;
    if crop_right > 0 {
        if crop_right >= output_frames {
            return Err(InferError::Message(format!(
                "conv_transpose crop_right {crop_right} exceeds output_frames {output_frames}"
            )));
        }
        output_frames -= crop_right;
        let keep_len = output_frames * out_ch;
        output.truncate(keep_len);
    }
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
                        let weight_index =
                            ggml_tensor_index_3d(kernel, 1, channels, kernel_index, 0, channel);
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

fn validate_f16_dims(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(InferError::Message(format!(
            "{name}: expected {expected:?}, got {actual:?}"
        )))
    }
}

fn validate_encoder_block_f16_dims(
    block: &CodecEncoderBlockF16Weights,
    index: usize,
    in_channels: usize,
    out_channels: usize,
) -> Result<()> {
    if block.index != index {
        return Err(InferError::Message(format!(
            "encoder block index mismatch: expected {index}, got {}",
            block.index
        )));
    }
    validate_residual_unit_f16_dims(&block.residual_1, in_channels)?;
    validate_residual_unit_f16_dims(&block.residual_2, in_channels)?;
    validate_residual_unit_f16_dims(&block.residual_3, in_channels)?;
    validate_f16_dims(
        block.snake_alpha.name(),
        block.snake_alpha.dimensions(),
        &[1, in_channels, 1],
    )?;
    validate_f16_dims(
        block.down_conv_weight.name(),
        block.down_conv_weight.dimensions(),
        &[CODEC_ENCODER_KERNELS[index - 1], in_channels, out_channels],
    )?;
    validate_f16_dims(
        block.down_conv_bias.name(),
        block.down_conv_bias.dimensions(),
        &[out_channels],
    )?;
    if index == CODEC_ENCODER_BLOCK_COUNT {
        if block.transformer_layers.len() != CODEC_ENCODER_TRANSFORMER_LAYERS {
            return Err(InferError::Message(format!(
                "encoder transformer layer count mismatch: expected {}, got {}",
                CODEC_ENCODER_TRANSFORMER_LAYERS,
                block.transformer_layers.len()
            )));
        }
        for layer in &block.transformer_layers {
            layer.validate_dimensions()?;
        }
        let norm = block
            .transformer_norm_weight
            .as_ref()
            .ok_or_else(|| InferError::Message("missing encoder transformer norm".into()))?;
        validate_f16_dims(norm.name(), norm.dimensions(), &[CODEC_LATENT_DIM])?;
    } else if !block.transformer_layers.is_empty() || block.transformer_norm_weight.is_some() {
        return Err(InferError::Message(format!(
            "encoder block {index} unexpectedly has transformer weights"
        )));
    }
    Ok(())
}

fn validate_residual_unit_f16_dims(
    unit: &CodecResidualUnitF16Weights,
    channels: usize,
) -> Result<()> {
    validate_f16_dims(
        unit.snake0_alpha.name(),
        unit.snake0_alpha.dimensions(),
        &[1, channels, 1],
    )?;
    validate_f16_dims(
        unit.conv0_weight.name(),
        unit.conv0_weight.dimensions(),
        &[7, channels, channels],
    )?;
    validate_f16_dims(
        unit.conv0_bias.name(),
        unit.conv0_bias.dimensions(),
        &[channels],
    )?;
    validate_f16_dims(
        unit.snake1_alpha.name(),
        unit.snake1_alpha.dimensions(),
        &[1, channels, 1],
    )?;
    validate_f16_dims(
        unit.conv1_weight.name(),
        unit.conv1_weight.dimensions(),
        &[1, channels, channels],
    )?;
    validate_f16_dims(
        unit.conv1_bias.name(),
        unit.conv1_bias.dimensions(),
        &[channels],
    )?;
    Ok(())
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
        let result =
            rvq_decode_latents(&rvq.latents, rvq.n_frames, &post_weights, &upsample_weights)
                .expect("decode latents");
        assert_eq!(result.input_frames, 2);
        assert_eq!(result.output_frames, 8);
        assert_eq!(result.hidden_dim, CODEC_HIDDEN_SIZE as usize);
        assert_eq!(result.hidden.len(), 8 * CODEC_HIDDEN_SIZE as usize);
        assert!(result.hidden.iter().all(|value| value.is_finite()));
        assert!(result.hidden.iter().any(|value| value.abs() > 0.0));
    }

    #[test]
    #[ignore = "requires local s2-pro codec GGUF in models/"]
    fn loads_quantizer_encode_stage_f16_weights_fixture() {
        let path = fixture_codec_path().expect("codec gguf");
        let gguf = GgufFile::open(&path).expect("codec gguf");
        let registry = CodecTensorRegistry::from_gguf(&gguf).expect("codec registry");
        let downsample_weights = CodecDownsampleF16Weights::from_gguf_registry(&gguf, &registry)
            .expect("downsample f16 weights");
        let pre_weights = CodecPreModuleF16Weights::from_gguf_registry(&gguf, &registry)
            .expect("pre module f16 weights");

        assert_eq!(downsample_weights.stages.len(), CODEC_UPSAMPLE_STAGES);
        assert_eq!(
            downsample_weights.stages[0].conv_weight.dimensions(),
            &[
                CODEC_UPSAMPLE_FACTOR,
                CODEC_HIDDEN_SIZE as usize,
                CODEC_HIDDEN_SIZE as usize
            ]
        );
        assert_eq!(
            downsample_weights.stages[0].pwconv1_weight.dimensions(),
            &[CODEC_HIDDEN_SIZE as usize, CODEC_CONVNEXT_EXPANDED_SIZE]
        );
        assert_eq!(pre_weights.layers.len(), CODEC_TRANSFORMER_LAYERS);
        assert_eq!(
            pre_weights.norm_weight.dimensions(),
            &[CODEC_HIDDEN_SIZE as usize]
        );
    }

    #[test]
    #[ignore = "requires local s2-pro codec GGUF in models/"]
    fn runs_quantizer_encode_stage_on_synthetic_latents_fixture() {
        let path = fixture_codec_path().expect("codec gguf");
        let gguf = GgufFile::open(&path).expect("codec gguf");
        let registry = CodecTensorRegistry::from_gguf(&gguf).expect("codec registry");
        let downsample_weights = CodecDownsampleF16Weights::from_gguf_registry(&gguf, &registry)
            .expect("downsample f16 weights");
        let pre_weights = CodecPreModuleF16Weights::from_gguf_registry(&gguf, &registry)
            .expect("pre module f16 weights");

        let input_frames = 8u32;
        let hidden_dim = CODEC_HIDDEN_SIZE as usize;
        let latents = (0..input_frames as usize * hidden_dim)
            .map(|index| ((index % 97) as f32 - 48.0) / 512.0)
            .collect::<Vec<_>>();
        let result = forward_codec_quantizer_encode_stage(
            &latents,
            input_frames,
            &downsample_weights,
            &pre_weights,
        )
        .expect("quantizer encode stage");
        assert_eq!(result.input_frames, input_frames);
        assert_eq!(result.output_frames, 2);
        assert_eq!(result.hidden_dim, hidden_dim);
        assert_eq!(result.hidden.len(), 2 * hidden_dim);
        assert!(result.hidden.iter().all(|value| value.is_finite()));
        assert!(result.hidden.iter().any(|value| value.abs() > 0.0));
    }

    #[test]
    #[ignore = "requires local s2-pro codec GGUF in models/"]
    fn runs_vq_nearest_search_on_synthetic_encode_stage_fixture() {
        let path = fixture_codec_path().expect("codec gguf");
        let gguf = GgufFile::open(&path).expect("codec gguf");
        let registry = CodecTensorRegistry::from_gguf(&gguf).expect("codec registry");
        let downsample_weights = CodecDownsampleF16Weights::from_gguf_registry(&gguf, &registry)
            .expect("downsample f16 weights");
        let pre_weights = CodecPreModuleF16Weights::from_gguf_registry(&gguf, &registry)
            .expect("pre module f16 weights");
        let rvq_weights = CodecF16Weights::from_gguf(&gguf).expect("codec f16 weights");

        let input_frames = 8u32;
        let hidden_dim = CODEC_HIDDEN_SIZE as usize;
        let latents = (0..input_frames as usize * hidden_dim)
            .map(|index| ((index % 127) as f32 - 63.0) / 768.0)
            .collect::<Vec<_>>();
        let stage = forward_codec_quantizer_encode_stage(
            &latents,
            input_frames,
            &downsample_weights,
            &pre_weights,
        )
        .expect("quantizer encode stage");
        let result = rvq_encode_latents_nearest(&stage.hidden, stage.output_frames, &rvq_weights)
            .expect("VQ nearest encode");

        assert_eq!(result.num_codebooks, 1 + CODEC_RESIDUAL_QUANTIZERS as u32);
        assert_eq!(result.n_frames, stage.output_frames);
        assert_eq!(result.latent_dim, hidden_dim);
        assert_eq!(
            result.codes.len(),
            result.num_codebooks as usize * result.n_frames as usize
        );
        assert!(result
            .final_residual_l2
            .iter()
            .all(|value| value.is_finite()));

        for frame in 0..result.n_frames as usize {
            let semantic = result.codes[frame];
            assert!(semantic >= 0);
            assert!(semantic < CODEC_SEMANTIC_CODEBOOK_SIZE as i32);
            for codebook in 1..result.num_codebooks as usize {
                let code = result.codes[codebook * result.n_frames as usize + frame];
                assert!(code >= 0);
                assert!(code < CODEC_RESIDUAL_CODEBOOK_SIZE as i32);
            }
        }

        let decoded = rvq_lookup_codes(
            &result.codes,
            result.num_codebooks,
            result.n_frames,
            &rvq_weights,
        )
        .expect("encoded codes lookup");
        assert_eq!(decoded.latent_dim, hidden_dim);
        assert_eq!(decoded.latents.len(), stage.hidden.len());
        assert!(decoded.latents.iter().all(|value| value.is_finite()));
    }

    #[test]
    #[ignore = "requires local s2-pro codec GGUF in models/"]
    fn encodes_reference_audio_to_prompt_codes_fixture() {
        let path = fixture_codec_path().expect("codec gguf");
        let gguf = GgufFile::open(&path).expect("codec gguf");
        let weights =
            CodecReferenceEncoderF16Weights::from_gguf(&gguf).expect("reference encoder weights");

        let audio = (0..CODEC_FRAME_LENGTH)
            .map(|index| (((index % 97) as f32) - 48.0) / 4096.0)
            .collect::<Vec<_>>();
        let result = encode_reference_audio(&audio, &weights).expect("reference audio encode");

        assert_eq!(result.input_samples, CODEC_FRAME_LENGTH as u32);
        assert_eq!(result.padded_samples, CODEC_FRAME_LENGTH as u32);
        assert_eq!(result.encoder_frames, 4);
        assert_eq!(result.quantizer_frames, 1);
        assert_eq!(result.num_codebooks, 1 + CODEC_RESIDUAL_QUANTIZERS as u32);
        assert_eq!(result.codes.len(), result.num_codebooks as usize);
        assert!(result
            .final_residual_l2
            .iter()
            .all(|value| value.is_finite()));
        assert!(result.codes[0] >= 0);
        assert!(result.codes[0] < CODEC_SEMANTIC_CODEBOOK_SIZE as i32);
        for code in result.codes.iter().skip(1) {
            assert!(*code >= 0);
            assert!(*code < CODEC_RESIDUAL_CODEBOOK_SIZE as i32);
        }
    }

    #[test]
    #[ignore = "requires local s2-pro codec GGUF in models/"]
    fn loads_encoder_frontend_f16_weights_fixture() {
        let path = fixture_codec_path().expect("codec gguf");
        let gguf = GgufFile::open(&path).expect("codec gguf");
        let weights = CodecEncoderF16Weights::from_gguf(&gguf).expect("encoder f16 weights");

        assert_eq!(
            weights.entry_conv_weight.dimensions(),
            &[7, 1, CODEC_ENCODER_ENTRY_CHANNELS]
        );
        assert_eq!(weights.blocks.len(), CODEC_ENCODER_BLOCK_COUNT);
        assert_eq!(
            weights.blocks[0].down_conv_weight.dimensions(),
            &[4, 64, 128]
        );
        assert_eq!(
            weights.blocks[1].down_conv_weight.dimensions(),
            &[8, 128, 256]
        );
        assert_eq!(
            weights.blocks[2].down_conv_weight.dimensions(),
            &[16, 256, 512]
        );
        assert_eq!(
            weights.blocks[3].down_conv_weight.dimensions(),
            &[16, 512, 1024]
        );
        assert_eq!(
            weights.blocks[3].transformer_layers.len(),
            CODEC_ENCODER_TRANSFORMER_LAYERS
        );
        assert_eq!(
            weights.output_conv_weight.dimensions(),
            &[3, CODEC_LATENT_DIM, CODEC_LATENT_DIM]
        );
    }

    #[test]
    #[ignore = "requires local s2-pro codec GGUF in models/"]
    fn runs_encoder_frontend_on_synthetic_pcm_fixture() {
        let path = fixture_codec_path().expect("codec gguf");
        let gguf = GgufFile::open(&path).expect("codec gguf");
        let weights = CodecEncoderF16Weights::from_gguf(&gguf).expect("encoder f16 weights");

        let audio = (0..CODEC_FRAME_LENGTH)
            .map(|index| (((index % 97) as f32) - 48.0) / 4096.0)
            .collect::<Vec<_>>();
        let result = forward_codec_encoder_frontend(&audio, &weights).expect("encoder frontend");
        assert_eq!(result.input_samples, CODEC_FRAME_LENGTH as u32);
        assert_eq!(result.padded_samples, CODEC_FRAME_LENGTH as u32);
        assert_eq!(result.output_frames, 4);
        assert_eq!(result.hidden_dim, CODEC_LATENT_DIM);
        assert_eq!(result.hidden.len(), 4 * CODEC_LATENT_DIM);
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
        assert_eq!(registry.encoder().blocks.len(), CODEC_ENCODER_BLOCK_COUNT);
        assert_eq!(
            registry.residual_quantizers().len(),
            CODEC_RESIDUAL_QUANTIZERS
        );
        assert_eq!(registry.pre_module_layers().len(), CODEC_TRANSFORMER_LAYERS);
        assert_eq!(
            registry.post_module_layers().len(),
            CODEC_TRANSFORMER_LAYERS
        );
        assert_eq!(
            registry.quantizer_downsample().stages.len(),
            CODEC_UPSAMPLE_STAGES
        );
        assert_eq!(
            registry.quantizer_upsample().stages.len(),
            CODEC_UPSAMPLE_STAGES
        );
        assert_eq!(registry.prefix_counts().get("quantizer"), Some(&244));
    }
}
