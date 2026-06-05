use fish_s2_core::gguf::{GgmlType, GgufFile};

use crate::backend::{CpuMatmulBackend, MatmulBackend};
use crate::error::{InferError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct F16TensorView {
    name: String,
    dimensions: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct F16TensorBytes {
    name: String,
    dimensions: Vec<usize>,
    bytes: Vec<u8>,
}

impl F16TensorView {
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self> {
        let tensor = gguf
            .tensor(name)
            .ok_or_else(|| InferError::Message(format!("tensor not found: {name}")))?;
        if tensor.ggml_type != GgmlType::F16 {
            return Err(InferError::Message(format!(
                "expected F16 tensor {name}, got {:?}",
                tensor.ggml_type
            )));
        }
        let dimensions = tensor
            .dimensions
            .iter()
            .map(|dim| {
                usize::try_from(*dim).map_err(|_| {
                    InferError::Message(format!("tensor dimension overflows usize: {name}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes = gguf
            .tensor_bytes(name)
            .map_err(|err| InferError::Message(err.to_string()))?;
        Self::from_f16_le_bytes(name, dimensions, &bytes)
    }

    pub fn from_f16_le_bytes(
        name: impl Into<String>,
        dimensions: impl Into<Vec<usize>>,
        bytes: &[u8],
    ) -> Result<Self> {
        let dimensions = dimensions.into();
        let element_count = dimensions.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| InferError::Message("F16 tensor element count overflow".to_string()))
        })?;
        let expected_bytes = element_count
            .checked_mul(2)
            .ok_or_else(|| InferError::Message("F16 tensor byte length overflow".to_string()))?;
        if bytes.len() != expected_bytes {
            return Err(InferError::Message(format!(
                "F16 tensor byte length mismatch: expected {expected_bytes}, got {}",
                bytes.len()
            )));
        }
        let values = bytes
            .chunks_exact(2)
            .map(|chunk| f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect();
        Ok(Self {
            name: name.into(),
            dimensions,
            values,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

impl F16TensorBytes {
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self> {
        let tensor = gguf
            .tensor(name)
            .ok_or_else(|| InferError::Message(format!("tensor not found: {name}")))?;
        if tensor.ggml_type != GgmlType::F16 {
            return Err(InferError::Message(format!(
                "expected F16 tensor {name}, got {:?}",
                tensor.ggml_type
            )));
        }
        let dimensions = tensor
            .dimensions
            .iter()
            .map(|dim| {
                usize::try_from(*dim).map_err(|_| {
                    InferError::Message(format!("tensor dimension overflows usize: {name}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes = gguf
            .tensor_bytes(name)
            .map_err(|err| InferError::Message(err.to_string()))?;
        Self::from_f16_le_bytes(name, dimensions, bytes)
    }

    pub fn from_f16_le_bytes(
        name: impl Into<String>,
        dimensions: impl Into<Vec<usize>>,
        bytes: Vec<u8>,
    ) -> Result<Self> {
        let dimensions = dimensions.into();
        let element_count = dimensions.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| InferError::Message("F16 tensor element count overflow".to_string()))
        })?;
        let expected_bytes = element_count
            .checked_mul(2)
            .ok_or_else(|| InferError::Message("F16 tensor byte length overflow".to_string()))?;
        if bytes.len() != expected_bytes {
            return Err(InferError::Message(format!(
                "F16 tensor byte length mismatch: expected {expected_bytes}, got {}",
                bytes.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            dimensions,
            bytes,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(InferError::Message("rms_norm input is empty".into()));
    }
    if input.len() != weight.len() {
        return Err(InferError::Message(format!(
            "rms_norm length mismatch: input={}, weight={}",
            input.len(),
            weight.len()
        )));
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * weight)
        .collect())
}

/// Row gather for embedding tables stored like ggml `get_rows` on `[hidden_dim, vocab_dim]`
/// weights (linear layout: each row is one vocab/code index, length `hidden_dim`).
pub fn embedding_lookup_rows(
    weight: &[f32],
    hidden_dim: usize,
    vocab_dim: usize,
    indices: &[u32],
) -> Result<Vec<Vec<f32>>> {
    let expected_weights = hidden_dim
        .checked_mul(vocab_dim)
        .ok_or_else(|| InferError::Message("embedding weight length overflow".to_string()))?;
    if weight.len() != expected_weights {
        return Err(InferError::Message(format!(
            "embedding weight length mismatch: expected {expected_weights}, got {}",
            weight.len()
        )));
    }
    let mut outputs = Vec::with_capacity(indices.len());
    for &index in indices {
        let row = usize::try_from(index).map_err(|_| {
            InferError::Message(format!("embedding index overflows usize: {index}"))
        })?;
        if row >= vocab_dim {
            return Err(InferError::Message(format!(
                "embedding index {row} out of range for vocab_dim {vocab_dim}"
            )));
        }
        let start = row
            .checked_mul(hidden_dim)
            .ok_or_else(|| InferError::Message("embedding row offset overflow".into()))?;
        let end = start + hidden_dim;
        outputs.push(weight[start..end].to_vec());
    }
    Ok(outputs)
}

pub fn linear(
    input: &[f32],
    weight: &[f32],
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<f32>> {
    linear_with_backend(
        &CpuMatmulBackend::new(),
        input,
        weight,
        input_dim,
        output_dim,
    )
}

pub fn linear_with_backend(
    backend: &impl MatmulBackend,
    input: &[f32],
    weight: &[f32],
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<f32>> {
    backend.linear(input, weight, input_dim, output_dim)
}

pub fn matvec_f16_streaming(
    input: &[f32],
    weight_f16_le: &[u8],
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<f32>> {
    if input.len() != input_dim {
        return Err(InferError::Message(format!(
            "matvec_f16_streaming input length mismatch: expected {input_dim}, got {}",
            input.len()
        )));
    }
    let expected_weights = input_dim
        .checked_mul(output_dim)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| InferError::Message("matvec_f16_streaming weight length overflow".into()))?;
    if weight_f16_le.len() != expected_weights {
        return Err(InferError::Message(format!(
            "matvec_f16_streaming weight byte length mismatch: expected {expected_weights}, got {}",
            weight_f16_le.len()
        )));
    }

    let mut output = vec![0.0f32; output_dim];
    let row_stride = input_dim
        .checked_mul(2)
        .ok_or_else(|| InferError::Message("matvec_f16_streaming row stride overflow".into()))?;
    for (output_index, output_value) in output.iter_mut().enumerate() {
        let row_start = output_index.checked_mul(row_stride).ok_or_else(|| {
            InferError::Message("matvec_f16_streaming row offset overflow".into())
        })?;
        let row = &weight_f16_le[row_start..row_start + row_stride];
        let mut sum = 0.0f32;
        for (input_value, weight_bytes) in input.iter().zip(row.chunks_exact(2)) {
            let weight = f16_bits_to_f32(u16::from_le_bytes([weight_bytes[0], weight_bytes[1]]));
            sum += input_value * weight;
        }
        *output_value = sum;
    }
    Ok(output)
}

pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    match (exponent, fraction) {
        (0, 0) => f32::from_bits(sign),
        (0, _) => {
            let value = f32::from(fraction) * 2f32.powi(-24);
            if sign == 0 {
                value
            } else {
                -value
            }
        }
        (0x1f, 0) => f32::from_bits(sign | 0x7f80_0000),
        (0x1f, _) => f32::from_bits(sign | 0x7f80_0000 | (u32::from(fraction) << 13)),
        _ => {
            let exponent = u32::from(exponent) + 112;
            f32::from_bits(sign | (exponent << 23) | (u32::from(fraction) << 13))
        }
    }
}

pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00;
    }

    let half_exp = exponent - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let rounded = mantissa + ((1u32 << (shift - 1)) - 1) + ((mantissa >> shift) & 1);
        return sign | ((rounded >> shift) as u16);
    }

    let rounded = mantissa + 0x0000_0fff + ((mantissa >> 13) & 1);
    if rounded & 0x0080_0000 != 0 {
        let next_exp = half_exp + 1;
        if next_exp >= 0x1f {
            return sign | 0x7c00;
        }
        return sign | ((next_exp as u16) << 10);
    }
    sign | ((half_exp as u16) << 10) | ((rounded >> 13) as u16)
}

pub(crate) fn round_f32_to_f16(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::registry::HIDDEN_SIZE;

    #[test]
    fn decodes_f16_le_bytes_to_f32_values() {
        let bytes = f16_bytes(&[0x3c00, 0xc000, 0x3800, 0x4400]);
        let tensor = F16TensorView::from_f16_le_bytes("toy", [2, 2], &bytes).unwrap();
        assert_eq!(tensor.name(), "toy");
        assert_eq!(tensor.dimensions(), &[2, 2]);
        assert_eq!(tensor.values(), &[1.0, -2.0, 0.5, 4.0]);
    }

    #[test]
    fn stores_f16_tensor_bytes_without_expanding_values() {
        let bytes = f16_bytes(&[0x3c00, 0xc000, 0x3800, 0x4400]);
        let tensor = F16TensorBytes::from_f16_le_bytes("toy", [2, 2], bytes.clone()).unwrap();
        assert_eq!(tensor.name(), "toy");
        assert_eq!(tensor.dimensions(), &[2, 2]);
        assert_eq!(tensor.bytes(), bytes.as_slice());
    }

    #[test]
    fn rounds_f32_to_f16_values() {
        assert_eq!(round_f32_to_f16(1.0), 1.0);
        assert_eq!(round_f32_to_f16(-2.0), -2.0);
        assert_eq!(round_f32_to_f16(0.5), 0.5);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16_bits(f32::NAN) & 0x7c00, 0x7c00);
    }

    #[test]
    fn rms_norm_matches_manual_smoke() {
        let output = rms_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0, 0.5, 2.0, -1.0], 1e-6).unwrap();
        let scale = (7.5f32 + 1e-6).sqrt().recip();
        assert_close(&output, &[scale, scale, 6.0 * scale, -4.0 * scale]);
    }

    #[test]
    fn linear_matches_ggml_output_input_weight_layout() {
        let output = linear(&[2.0, -1.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        assert_close(&output, &[0.0, 2.0, 4.0]);
    }

    #[test]
    fn matvec_f16_streaming_matches_expanded_linear() {
        let weight_f16 = f16_bytes(&[
            f32_to_f16_bits(1.0),
            f32_to_f16_bits(2.0),
            f32_to_f16_bits(3.0),
            f32_to_f16_bits(4.0),
            f32_to_f16_bits(5.0),
            f32_to_f16_bits(6.0),
        ]);
        let output = matvec_f16_streaming(&[2.0, -1.0], &weight_f16, 2, 3).unwrap();
        assert_close(&output, &[0.0, 2.0, 4.0]);
    }

    #[test]
    #[ignore = "requires local s2-pro transformer GGUF in models/"]
    fn loads_local_norm_weight_as_f16_tensor() {
        let path = local_model_dir().join("s2-pro-f16-transformer-only.gguf");
        let gguf = GgufFile::open(path).unwrap();
        let tensor = F16TensorView::from_gguf(&gguf, "norm.weight").unwrap();
        assert_eq!(tensor.name(), "norm.weight");
        assert_eq!(tensor.dimensions(), &[HIDDEN_SIZE as usize]);
        assert_eq!(tensor.values().len(), HIDDEN_SIZE as usize);
        assert!(tensor.values().iter().all(|value| value.is_finite()));
    }

    fn f16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "expected {expected}, got {actual}"
            );
        }
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
