//! Top-k / top-p / temperature sampling aligned with `s2.cpp` (`s2_sampler.cpp`,
//! `s2_generate.cpp`).

use crate::error::{InferError, Result};

/// Logit value used for disallowed vocabulary entries (matches s2.cpp mask bias).
pub const LOGIT_MASKED: f32 = f32::NEG_INFINITY;

/// Finite logit threshold below which `always_include_id` is ignored (s2.cpp: `-1e30`).
const ALWAYS_INCLUDE_MIN_LOGIT: f32 = -1e30;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.7,
            top_k: 30,
        }
    }
}

/// RNG hook for deterministic tests (`sample_token` uses this instead of a global generator).
pub trait RandomSource {
    fn next_unit(&mut self) -> f64;
}

/// Simple deterministic RNG for unit tests and parity fixtures.
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RandomSource for SeededRng {
    fn next_unit(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mantissa = (self.state >> 11) & ((1u64 << 53) - 1);
        mantissa as f64 / ((1u64 << 53) as f64)
    }
}

/// Build semantic mask bias: `0` on `[sem_begin, sem_end]` and optional `im_end_id`, `-inf` elsewhere.
pub fn build_semantic_bias(
    vocab_size: usize,
    sem_begin: u32,
    sem_end: u32,
    im_end_id: Option<u32>,
) -> Vec<f32> {
    let mut bias = vec![LOGIT_MASKED; vocab_size];
    for id in sem_begin..=sem_end {
        if let Some(slot) = bias.get_mut(id as usize) {
            *slot = 0.0;
        }
    }
    if let Some(id) = im_end_id {
        if let Some(slot) = bias.get_mut(id as usize) {
            *slot = 0.0;
        }
    }
    bias
}

/// Add semantic bias and optionally block `im_end_id` (s2.cpp `apply_mask_and_sample`).
pub fn apply_semantic_bias(
    logits: &mut [f32],
    bias: &[f32],
    block_im_end: bool,
    im_end_id: Option<u32>,
) -> Result<()> {
    if logits.len() != bias.len() {
        return Err(InferError::Message(format!(
            "semantic bias length mismatch: logits={} bias={}",
            logits.len(),
            bias.len()
        )));
    }
    for (logit, mask) in logits.iter_mut().zip(bias) {
        *logit += mask;
    }
    if block_im_end {
        if let Some(id) = im_end_id {
            if let Some(logit) = logits.get_mut(id as usize) {
                *logit = LOGIT_MASKED;
            }
        }
    }
    Ok(())
}

/// In-place semantic mask: allowed ids keep logits; others become `-inf`.
pub fn semantic_mask_logits(
    logits: &mut [f32],
    sem_begin: u32,
    sem_end: u32,
    im_end_id: Option<u32>,
    block_im_end: bool,
) {
    for (index, logit) in logits.iter_mut().enumerate() {
        let id = index as u32;
        let in_semantic = id >= sem_begin && id <= sem_end;
        let im_end_allowed = im_end_id == Some(id) && !block_im_end;
        if !in_semantic && !im_end_allowed {
            *logit = LOGIT_MASKED;
        }
    }
}

/// Sample one token from logits (top-k → force-include → temperature softmax → top-p → discrete sample).
pub fn sample_token<R: RandomSource + ?Sized>(
    logits: &[f32],
    params: &SamplerParams,
    always_include_id: Option<u32>,
    rng: &mut R,
) -> Result<u32> {
    let vocab_size = logits.len();
    if vocab_size == 0 {
        return Ok(0);
    }

    let mut items: Vec<(f32, u32)> = logits
        .iter()
        .enumerate()
        .map(|(id, &value)| (value, id as u32))
        .collect();
    items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let k = if params.top_k > 0 {
        (params.top_k as usize).min(vocab_size)
    } else {
        vocab_size
    };
    items.truncate(k);

    if let Some(force_id) = always_include_id {
        if (force_id as usize) < vocab_size {
            let force_logit = logits[force_id as usize];
            if force_logit > ALWAYS_INCLUDE_MIN_LOGIT
                && !items.iter().any(|(_, id)| *id == force_id)
            {
                items.push((force_logit, force_id));
            }
        }
    }

    items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = items.len();
    if n == 0 {
        return Ok(0);
    }

    let mut probs: Vec<f32> = items.iter().map(|(logit, _)| *logit).collect();
    apply_softmax(&mut probs, params.temperature);

    let always_pos =
        always_include_id.and_then(|force_id| items.iter().position(|(_, id)| *id == force_id));

    let mut p_idx = 0usize;
    let mut cumsum = 0.0f32;
    while p_idx < n {
        cumsum += probs[p_idx];
        p_idx += 1;
        if cumsum >= params.top_p {
            break;
        }
    }
    if p_idx == 0 {
        p_idx = 1;
    }
    if let Some(pos) = always_pos {
        if pos >= p_idx {
            p_idx = pos + 1;
        }
    }

    items.truncate(p_idx);
    probs.truncate(p_idx);

    let sum_p: f32 = probs.iter().sum();
    if sum_p > 0.0 {
        for prob in &mut probs {
            *prob /= sum_p;
        }
    }

    let choice = sample_discrete(&probs, rng)?;
    Ok(items[choice].1)
}

fn apply_softmax(probs: &mut [f32], temperature: f32) {
    if probs.is_empty() {
        return;
    }
    let max_val = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for prob in probs.iter_mut() {
        if temperature > 0.0 {
            *prob = ((*prob - max_val) / temperature).exp();
        } else {
            *prob = if (*prob - max_val).abs() < 1e-6 {
                1.0
            } else {
                0.0
            };
        }
        sum += *prob;
    }
    if sum > 0.0 {
        for prob in probs.iter_mut() {
            *prob /= sum;
        }
    }
}

fn sample_discrete<R: RandomSource + ?Sized>(probs: &[f32], rng: &mut R) -> Result<usize> {
    if probs.is_empty() {
        return Err(InferError::Message("empty probability vector".into()));
    }
    let draw = rng.next_unit();
    let target = (draw as f32).clamp(0.0, 1.0 - 1e-7);
    let mut cumsum = 0.0f32;
    for (index, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if target < cumsum {
            return Ok(index);
        }
    }
    Ok(probs.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_mask_leaves_only_allowed_ids_finite() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        semantic_mask_logits(&mut logits, 1, 2, Some(4), false);
        assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], 3.0);
        assert!(logits[3].is_infinite());
        assert_eq!(logits[4], 5.0);
    }

    #[test]
    fn semantic_mask_can_block_im_end() {
        let mut logits = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        semantic_mask_logits(&mut logits, 1, 2, Some(4), true);
        assert!(logits[4].is_infinite() && logits[4].is_sign_negative());
    }

    #[test]
    fn apply_semantic_bias_zeros_disallowed_logits() {
        let logits_src = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let bias = build_semantic_bias(5, 1, 2, Some(4));
        let mut biased = logits_src.clone();
        apply_semantic_bias(&mut biased, &bias, false, Some(4)).unwrap();
        assert!(biased[0].is_infinite());
        assert_eq!(biased[1], 0.2);
        assert_eq!(biased[4], 0.5);
    }

    #[test]
    fn greedy_temp_zero_picks_top_logit() {
        let logits = vec![0.1, 5.0, 3.0, 2.0];
        let params = SamplerParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        };
        let mut rng = SeededRng::new(1);
        let token = sample_token(&logits, &params, None, &mut rng).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn top_k_limits_candidates() {
        let logits = vec![10.0, 9.0, 8.0, 0.0];
        let params = SamplerParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 2,
        };
        let mut rng = SeededRng::new(99);
        let token = sample_token(&logits, &params, None, &mut rng).unwrap();
        assert!(token <= 1);
    }

    #[test]
    fn always_include_id_survives_top_k_truncation() {
        let logits = vec![LOGIT_MASKED, LOGIT_MASKED, LOGIT_MASKED, LOGIT_MASKED, 5.0];
        let params = SamplerParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
        };
        let mut rng = SeededRng::new(1);
        let token = sample_token(&logits, &params, Some(4), &mut rng).unwrap();
        assert_eq!(token, 4);
    }

    #[test]
    fn seeded_sampling_is_repeatable() {
        let logits = vec![1.0, 3.0, 2.0, 0.5, 4.0];
        let params = SamplerParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 3,
        };
        let mut rng_a = SeededRng::new(42);
        let mut rng_b = SeededRng::new(42);
        let a = sample_token(&logits, &params, None, &mut rng_a).unwrap();
        let b = sample_token(&logits, &params, None, &mut rng_b).unwrap();
        assert_eq!(a, b);
    }
}
