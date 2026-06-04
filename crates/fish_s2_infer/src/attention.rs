use crate::error::{InferError, Result};
use crate::registry::KvCacheSpec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GqaAttentionShape {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub token_count: usize,
    pub attn_scale: f32,
}

#[derive(Debug, Clone)]
pub struct SlowArKvCache {
    head_dim: usize,
    head_count_kv: usize,
    max_seq_len: usize,
    block_count: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl SlowArKvCache {
    pub fn new(spec: KvCacheSpec, max_seq_len: usize) -> Result<Self> {
        let head_dim = usize::try_from(spec.head_dim)
            .map_err(|_| InferError::Message("KV cache head_dim overflows usize".into()))?;
        let head_count_kv = usize::try_from(spec.head_count_kv)
            .map_err(|_| InferError::Message("KV cache head_count_kv overflows usize".into()))?;
        let block_count = usize::try_from(spec.block_count)
            .map_err(|_| InferError::Message("KV cache block_count overflows usize".into()))?;
        let len = head_dim
            .checked_mul(head_count_kv)
            .and_then(|len| len.checked_mul(max_seq_len))
            .and_then(|len| len.checked_mul(block_count))
            .ok_or_else(|| InferError::Message("KV cache length overflow".into()))?;
        Ok(Self {
            head_dim,
            head_count_kv,
            max_seq_len,
            block_count,
            keys: vec![0.0; len],
            values: vec![0.0; len],
        })
    }

    pub fn dimensions(&self) -> [usize; 4] {
        [
            self.head_dim,
            self.head_count_kv,
            self.max_seq_len,
            self.block_count,
        ]
    }

    pub fn write_token(
        &mut self,
        layer: usize,
        token: usize,
        key_heads: &[f32],
        value_heads: &[f32],
    ) -> Result<()> {
        let len = self.token_width();
        if key_heads.len() != len || value_heads.len() != len {
            return Err(InferError::Message(format!(
                "KV token width mismatch: expected {len}, got key={} value={}",
                key_heads.len(),
                value_heads.len()
            )));
        }
        let start = self.token_offset(layer, token, 0, 0)?;
        let end = start + len;
        self.keys[start..end].copy_from_slice(key_heads);
        self.values[start..end].copy_from_slice(value_heads);
        Ok(())
    }

    pub fn key_token(&self, layer: usize, token: usize) -> Result<&[f32]> {
        let start = self.token_offset(layer, token, 0, 0)?;
        Ok(&self.keys[start..start + self.token_width()])
    }

    pub fn value_token(&self, layer: usize, token: usize) -> Result<&[f32]> {
        let start = self.token_offset(layer, token, 0, 0)?;
        Ok(&self.values[start..start + self.token_width()])
    }

    pub fn decode_attention(
        &self,
        layer: usize,
        token_count: usize,
        query_heads: &[f32],
        head_count: usize,
        attn_scale: f32,
    ) -> Result<Vec<f32>> {
        if token_count > self.max_seq_len {
            return Err(InferError::Message(format!(
                "token_count {token_count} exceeds KV cache max_seq_len {}",
                self.max_seq_len
            )));
        }
        if layer >= self.block_count {
            return Err(InferError::Message(format!(
                "layer {layer} exceeds KV cache block_count {}",
                self.block_count
            )));
        }
        let mut keys = Vec::with_capacity(token_count * self.token_width());
        let mut values = Vec::with_capacity(token_count * self.token_width());
        for token in 0..token_count {
            keys.extend_from_slice(self.key_token(layer, token)?);
            values.extend_from_slice(self.value_token(layer, token)?);
        }
        gqa_decode_attention(
            query_heads,
            &keys,
            &values,
            GqaAttentionShape {
                head_count,
                head_count_kv: self.head_count_kv,
                head_dim: self.head_dim,
                token_count,
                attn_scale,
            },
        )
    }

    fn token_width(&self) -> usize {
        self.head_dim * self.head_count_kv
    }

    fn token_offset(
        &self,
        layer: usize,
        token: usize,
        kv_head: usize,
        dim: usize,
    ) -> Result<usize> {
        if layer >= self.block_count
            || token >= self.max_seq_len
            || kv_head >= self.head_count_kv
            || dim >= self.head_dim
        {
            return Err(InferError::Message(format!(
                "KV cache index out of bounds: layer={layer}, token={token}, kv_head={kv_head}, dim={dim}"
            )));
        }
        Ok(dim
            + self.head_dim * (kv_head + self.head_count_kv * (token + self.max_seq_len * layer)))
    }
}

pub fn apply_rope_normal(
    values: &mut [f32],
    head_dim: usize,
    position: usize,
    rope_base: f32,
) -> Result<()> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(InferError::Message(format!(
            "RoPE head_dim must be non-zero and even, got {head_dim}"
        )));
    }
    if !values.len().is_multiple_of(head_dim) {
        return Err(InferError::Message(format!(
            "RoPE values length {} is not a multiple of head_dim {head_dim}",
            values.len()
        )));
    }
    if rope_base <= 0.0 {
        return Err(InferError::Message(format!(
            "RoPE base must be positive, got {rope_base}"
        )));
    }

    for head in values.chunks_exact_mut(head_dim) {
        for pair in (0..head_dim).step_by(2) {
            let theta = position as f32 / rope_base.powf(pair as f32 / head_dim as f32);
            let (sin_theta, cos_theta) = theta.sin_cos();
            let x0 = head[pair];
            let x1 = head[pair + 1];
            head[pair] = x0 * cos_theta - x1 * sin_theta;
            head[pair + 1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
    Ok(())
}

pub fn gqa_decode_attention(
    query_heads: &[f32],
    key_tokens: &[f32],
    value_tokens: &[f32],
    shape: GqaAttentionShape,
) -> Result<Vec<f32>> {
    validate_gqa_shapes(query_heads, key_tokens, value_tokens, shape)?;
    let head_repeat = shape.head_count / shape.head_count_kv;
    let mut output = vec![0.0f32; shape.head_count * shape.head_dim];
    for query_head in 0..shape.head_count {
        let kv_head = query_head / head_repeat;
        let query = head_slice(query_heads, query_head, shape.head_dim);
        let mut scores = Vec::with_capacity(shape.token_count);
        for token in 0..shape.token_count {
            let key = token_head_slice(
                key_tokens,
                token,
                kv_head,
                shape.head_count_kv,
                shape.head_dim,
            );
            scores.push(dot(query, key) * shape.attn_scale);
        }
        softmax_in_place(&mut scores);
        for (token, probability) in scores.iter().enumerate() {
            let value = token_head_slice(
                value_tokens,
                token,
                kv_head,
                shape.head_count_kv,
                shape.head_dim,
            );
            let out = head_slice_mut(&mut output, query_head, shape.head_dim);
            for dim in 0..shape.head_dim {
                out[dim] += probability * value[dim];
            }
        }
    }
    Ok(output)
}

fn validate_gqa_shapes(
    query_heads: &[f32],
    key_tokens: &[f32],
    value_tokens: &[f32],
    shape: GqaAttentionShape,
) -> Result<()> {
    if shape.head_count_kv == 0 || !shape.head_count.is_multiple_of(shape.head_count_kv) {
        return Err(InferError::Message(format!(
            "invalid GQA split: heads={}, kv_heads={}",
            shape.head_count, shape.head_count_kv
        )));
    }
    if shape.head_dim == 0 || shape.token_count == 0 {
        return Err(InferError::Message(format!(
            "invalid attention dims: head_dim={}, token_count={}",
            shape.head_dim, shape.token_count
        )));
    }
    let expected_query = shape
        .head_count
        .checked_mul(shape.head_dim)
        .ok_or_else(|| InferError::Message("query length overflow".into()))?;
    let expected_kv = shape
        .token_count
        .checked_mul(shape.head_count_kv)
        .and_then(|len| len.checked_mul(shape.head_dim))
        .ok_or_else(|| InferError::Message("KV length overflow".into()))?;
    if query_heads.len() != expected_query {
        return Err(InferError::Message(format!(
            "query length mismatch: expected {expected_query}, got {}",
            query_heads.len()
        )));
    }
    if key_tokens.len() != expected_kv || value_tokens.len() != expected_kv {
        return Err(InferError::Message(format!(
            "KV length mismatch: expected {expected_kv}, got key={} value={}",
            key_tokens.len(),
            value_tokens.len()
        )));
    }
    Ok(())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |max, value| max.max(value));
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn head_slice(values: &[f32], head: usize, head_dim: usize) -> &[f32] {
    &values[head * head_dim..(head + 1) * head_dim]
}

fn head_slice_mut(values: &mut [f32], head: usize, head_dim: usize) -> &mut [f32] {
    &mut values[head * head_dim..(head + 1) * head_dim]
}

fn token_head_slice(
    values: &[f32],
    token: usize,
    head: usize,
    head_count: usize,
    head_dim: usize,
) -> &[f32] {
    let offset = (token * head_count + head) * head_dim;
    &values[offset..offset + head_dim]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::KvCacheSpec;
    use fish_s2_core::gguf::GgmlType;

    #[test]
    fn rope_normal_rotates_adjacent_pairs() {
        let mut values = vec![1.0, 0.0, 0.0, 1.0];
        apply_rope_normal(&mut values, 4, 1, 10_000.0).unwrap();
        let (sin_1, cos_1) = 1.0f32.sin_cos();
        let (sin_001, cos_001) = 0.01f32.sin_cos();
        assert_close(&values, &[cos_1, sin_1, -sin_001, cos_001]);
    }

    #[test]
    fn kv_cache_uses_s2_layout_for_token_writes() {
        let spec = KvCacheSpec {
            ggml_type: GgmlType::F16,
            head_dim: 2,
            head_count_kv: 2,
            block_count: 2,
        };
        let mut cache = SlowArKvCache::new(spec, 3).unwrap();
        assert_eq!(cache.dimensions(), [2, 2, 3, 2]);
        cache
            .write_token(1, 2, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(cache.key_token(1, 2).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.value_token(1, 2).unwrap(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn gqa_decode_attention_repeats_kv_heads() {
        let query = [
            1.0, 0.0, // q head 0 -> kv head 0
            0.0, 1.0, // q head 1 -> kv head 0
            1.0, 0.0, // q head 2 -> kv head 1
            0.0, 1.0, // q head 3 -> kv head 1
        ];
        let keys = [
            1.0, 0.0, 0.0, 1.0, // token 0, kv heads 0..1
            0.0, 1.0, 1.0, 0.0, // token 1, kv heads 0..1
        ];
        let values = [
            10.0, 0.0, 0.0, 10.0, // token 0
            20.0, 0.0, 0.0, 20.0, // token 1
        ];
        let output = gqa_decode_attention(
            &query,
            &keys,
            &values,
            GqaAttentionShape {
                head_count: 4,
                head_count_kv: 2,
                head_dim: 2,
                token_count: 2,
                attn_scale: 1.0,
            },
        )
        .unwrap();
        let p_hi = 1.0f32.exp() / (1.0f32.exp() + 1.0);
        let p_lo = 1.0 - p_hi;
        assert_close(
            &output,
            &[
                10.0 * p_hi + 20.0 * p_lo,
                0.0,
                10.0 * p_lo + 20.0 * p_hi,
                0.0,
                0.0,
                10.0 * p_lo + 20.0 * p_hi,
                0.0,
                10.0 * p_hi + 20.0 * p_lo,
            ],
        );
    }

    #[test]
    fn decode_attention_reads_written_kv_cache() {
        let spec = KvCacheSpec {
            ggml_type: GgmlType::F16,
            head_dim: 2,
            head_count_kv: 1,
            block_count: 1,
        };
        let mut cache = SlowArKvCache::new(spec, 2).unwrap();
        cache.write_token(0, 0, &[1.0, 0.0], &[2.0, 0.0]).unwrap();
        cache.write_token(0, 1, &[0.0, 1.0], &[4.0, 0.0]).unwrap();
        let output = cache
            .decode_attention(0, 2, &[1.0, 0.0, 0.0, 1.0], 2, 1.0)
            .unwrap();
        let p_hi = 1.0f32.exp() / (1.0f32.exp() + 1.0);
        let p_lo = 1.0 - p_hi;
        assert_close(
            &output,
            &[2.0 * p_hi + 4.0 * p_lo, 0.0, 2.0 * p_lo + 4.0 * p_hi, 0.0],
        );
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
}
