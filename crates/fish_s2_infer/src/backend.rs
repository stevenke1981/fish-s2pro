use crate::attention::{gqa_decode_attention, GqaAttentionShape};
use crate::error::{InferError, Result};
use rayon::prelude::*;

pub trait MatmulBackend: Send + Sync {
    fn name(&self) -> &'static str;

    fn linear(
        &self,
        input: &[f32],
        weight: &[f32],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f32>>;

    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
        rms_norm_f32(input, weight, eps)
    }

    fn rms_norm_heads(
        &self,
        input: &[f32],
        weight: &[f32],
        head_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        rms_norm_heads_f32(input, weight, head_dim, eps)
    }

    fn gqa_decode_attention(
        &self,
        query_heads: &[f32],
        key_tokens: &[f32],
        value_tokens: &[f32],
        shape: GqaAttentionShape,
    ) -> Result<Vec<f32>> {
        gqa_decode_attention(query_heads, key_tokens, value_tokens, shape)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuMatmulBackend;

impl CpuMatmulBackend {
    pub const fn new() -> Self {
        Self
    }
}

impl MatmulBackend for CpuMatmulBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn linear(
        &self,
        input: &[f32],
        weight: &[f32],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f32>> {
        validate_linear_dimensions(input, weight, input_dim, output_dim)?;

        let mut output = vec![0.0f32; output_dim];
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(output_index, output_value)| {
                let row = &weight[output_index * input_dim..(output_index + 1) * input_dim];
                *output_value = dot_f32(input, row);
            });
        Ok(output)
    }
}

fn dot_f32(input: &[f32], row: &[f32]) -> f32 {
    input
        .iter()
        .zip(row)
        .map(|(input_value, weight)| input_value * weight)
        .sum()
}

pub fn rms_norm_f32(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
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

pub fn rms_norm_heads_f32(
    input: &[f32],
    weight: &[f32],
    head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if !input.len().is_multiple_of(head_dim) {
        return Err(InferError::Message(format!(
            "head RMSNorm input length {} is not a multiple of head_dim {head_dim}",
            input.len()
        )));
    }
    let mut output = Vec::with_capacity(input.len());
    for head in input.chunks_exact(head_dim) {
        output.extend(rms_norm_f32(head, weight, eps)?);
    }
    Ok(output)
}

#[cfg(feature = "candle-backend")]
#[derive(Debug, Clone)]
pub struct CandleMatmulBackend {
    device: candle_core::Device,
}

#[cfg(feature = "candle-backend")]
impl CandleMatmulBackend {
    pub fn new_cpu() -> Self {
        Self {
            device: candle_core::Device::Cpu,
        }
    }

    #[cfg(feature = "candle-cuda")]
    pub fn new_cuda(device_id: usize) -> Result<Self> {
        let device = candle_core::Device::new_cuda(device_id).map_err(candle_error)?;
        Ok(Self { device })
    }

    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }
}

#[cfg(feature = "candle-backend")]
impl MatmulBackend for CandleMatmulBackend {
    fn name(&self) -> &'static str {
        "candle"
    }

    fn linear(
        &self,
        input: &[f32],
        weight: &[f32],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f32>> {
        validate_linear_dimensions(input, weight, input_dim, output_dim)?;
        let input = candle_core::Tensor::from_slice(input, (1, input_dim), &self.device)
            .map_err(candle_error)?;
        let weight = candle_core::Tensor::from_slice(weight, (output_dim, input_dim), &self.device)
            .map_err(candle_error)?;
        let output = input
            .matmul(&weight.t().map_err(candle_error)?)
            .map_err(candle_error)?;
        let output = output.squeeze(0).map_err(candle_error)?;
        output.to_vec1::<f32>().map_err(candle_error)
    }
}

#[cfg(feature = "candle-backend")]
fn candle_error(err: candle_core::Error) -> InferError {
    InferError::Message(format!("candle backend error: {err}"))
}

pub fn validate_linear_dimensions(
    input: &[f32],
    weight: &[f32],
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if input.len() != input_dim {
        return Err(InferError::Message(format!(
            "linear input length mismatch: expected {input_dim}, got {}",
            input.len()
        )));
    }
    let expected_weights = input_dim
        .checked_mul(output_dim)
        .ok_or_else(|| InferError::Message("linear weight length overflow".to_string()))?;
    if weight.len() != expected_weights {
        return Err(InferError::Message(format!(
            "linear weight length mismatch: expected {expected_weights}, got {}",
            weight.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_linear_matches_output_input_weight_layout() {
        let backend = CpuMatmulBackend::new();
        let output = backend
            .linear(&[2.0, -1.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
            .unwrap();
        assert_eq!(backend.name(), "cpu");
        assert_eq!(output, &[0.0, 2.0, 4.0]);
    }

    #[test]
    fn cpu_backend_rejects_bad_weight_shape() {
        let err = CpuMatmulBackend::new()
            .linear(&[1.0, 2.0], &[1.0, 2.0, 3.0], 2, 2)
            .unwrap_err();
        assert!(err.to_string().contains("linear weight length mismatch"));
    }

    #[cfg(feature = "candle-backend")]
    #[test]
    fn candle_backend_linear_matches_cpu_reference() {
        let cpu = CpuMatmulBackend::new()
            .linear(&[2.0, -1.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
            .unwrap();
        let candle = CandleMatmulBackend::new_cpu()
            .linear(&[2.0, -1.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
            .unwrap();
        assert_eq!(candle, cpu);
    }
}
