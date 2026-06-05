use crate::error::{InferError, Result};

pub trait MatmulBackend: Send + Sync {
    fn name(&self) -> &'static str;

    fn linear(
        &self,
        input: &[f32],
        weight: &[f32],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f32>>;
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
        for output_index in 0..output_dim {
            let row = &weight[output_index * input_dim..(output_index + 1) * input_dim];
            for input_index in 0..input_dim {
                output[output_index] += input[input_index] * row[input_index];
            }
        }
        Ok(output)
    }
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
}
