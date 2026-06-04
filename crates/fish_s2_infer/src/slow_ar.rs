use fish_s2_core::gguf::GgufFile;

use crate::attention::{apply_rope_normal, SlowArKvCache};
use crate::error::{InferError, Result};
use crate::registry::{ArGraphSpec, TransformerTensorRegistry};
use crate::tensor::{linear, rms_norm, F16TensorView};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlowArLayerShape {
    pub hidden_size: usize,
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

impl SlowArLayerSkeleton<'_> {
    pub fn forward_decode_token(
        &self,
        hidden: &[f32],
        cache: &mut SlowArKvCache,
        layer: usize,
        position: usize,
    ) -> Result<SlowArLayerForwardOutput> {
        self.validate(hidden)?;

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

        let mut query = rms_norm_heads(
            query_raw,
            self.q_norm_weight,
            self.shape.head_dim,
            self.shape.rms_norm_eps,
        )?;
        let mut key = rms_norm_heads(
            key_raw,
            self.k_norm_weight,
            self.shape.head_dim,
            self.shape.rms_norm_eps,
        )?;
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

        cache.write_token(layer, position, &key, &value)?;
        let attention = cache.decode_attention(
            layer,
            position + 1,
            &query,
            self.shape.head_count,
            self.shape.attn_scale(),
        )?;
        let projected = linear(
            &attention,
            self.output_weight,
            self.shape.q_size()?,
            self.shape.hidden_size,
        )?;
        let hidden = hidden
            .iter()
            .zip(&projected)
            .map(|(residual, projected)| residual + projected)
            .collect();

        Ok(SlowArLayerForwardOutput {
            normalized,
            query,
            key,
            value,
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
        Ok(())
    }
}

fn rms_norm_heads(input: &[f32], weight: &[f32], head_dim: usize, eps: f32) -> Result<Vec<f32>> {
    if !input.len().is_multiple_of(head_dim) {
        return Err(InferError::Message(format!(
            "head RMSNorm input length {} is not a multiple of head_dim {head_dim}",
            input.len()
        )));
    }
    let mut output = Vec::with_capacity(input.len());
    for head in input.chunks_exact(head_dim) {
        output.extend(rms_norm(head, weight, eps)?);
    }
    Ok(output)
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
        let shape = SlowArLayerShape {
            hidden_size: 4,
            head_count: 2,
            head_count_kv: 1,
            head_dim: 2,
            rope_base: 10_000.0,
            rms_norm_eps: 0.0,
        };
        let hidden = [1.0, 0.0, 0.0, 0.0];
        let attention_norm = [0.5, 1.0, 1.0, 1.0];
        let head_norm = [std::f32::consts::FRAC_1_SQRT_2; 2];
        let wqkv = row_major_weight(
            shape.hidden_size,
            shape.wqkv_out().unwrap(),
            &[
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 0.0],
                vec![0.0; 8],
                vec![0.0; 8],
                vec![0.0; 8],
            ],
        );
        let output = row_major_weight(
            shape.q_size().unwrap(),
            shape.hidden_size,
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0; 4],
                vec![0.0, 2.0, 0.0, 0.0],
                vec![0.0; 4],
            ],
        );
        let layer = SlowArLayerSkeleton {
            shape,
            attention_norm_weight: &attention_norm,
            q_norm_weight: &head_norm,
            k_norm_weight: &head_norm,
            wqkv_weight: &wqkv,
            output_weight: &output,
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
    fn slow_ar_layer_skeleton_rejects_bad_weight_shapes() {
        let shape = SlowArLayerShape {
            hidden_size: 4,
            head_count: 2,
            head_count_kv: 1,
            head_dim: 2,
            rope_base: 10_000.0,
            rms_norm_eps: 0.0,
        };
        let layer = SlowArLayerSkeleton {
            shape,
            attention_norm_weight: &[1.0; 4],
            q_norm_weight: &[1.0; 2],
            k_norm_weight: &[1.0; 2],
            wqkv_weight: &[0.0; 3],
            output_weight: &[0.0; 16],
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
    }

    fn row_major_weight(input_dim: usize, output_dim: usize, rows: &[Vec<f32>]) -> Vec<f32> {
        assert_eq!(rows.len(), input_dim);
        let mut values = Vec::with_capacity(input_dim * output_dim);
        for row in rows {
            assert_eq!(row.len(), output_dim);
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
