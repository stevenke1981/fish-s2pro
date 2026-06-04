//! Codec GGUF tensor registry.
//!
//! This module indexes the codec-only GGUF directory without reading tensor
//! payloads. The first RVQ slice needs stable names/shapes before decode math.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use fish_s2_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};

use crate::error::{InferError, Result};

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
