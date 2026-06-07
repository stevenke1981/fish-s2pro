//! Dual-AR generation: Slow-AR semantic tokens + Fast-AR VQ codebooks per frame.

use crate::error::{InferError, Result};
use crate::fast_ar::{generate_codebooks_for_semantic, FastArWeights};
use crate::prompt::{transpose_to_time_major, PromptTensor};
use crate::registry::DualArGraphSpec;
use crate::sampling::{
    apply_semantic_bias, build_semantic_bias, sample_token, RandomSource, SamplerParams,
};
use crate::slow_ar::{SlowArDecodeProfile, SlowArState, SlowArStepResult};
use crate::tokenizer::S2TokenizerConfig;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerateParams {
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_tokens_before_end: u32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.7,
            top_p: 0.7,
            top_k: 30,
            min_tokens_before_end: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateSemanticResult {
    pub token_ids: Vec<u32>,
}

/// Flattened `(num_codebooks, n_frames)` codes in codebook-major order (matches s2.cpp `GenerateResult`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateCodesResult {
    pub codes: Vec<i32>,
    pub num_codebooks: u32,
    pub n_frames: u32,
    pub slow_ar_profile: SlowArDecodeProfile,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FastArFirstFrameResult {
    pub prompt_cols: usize,
    pub main_token_id: u32,
    pub slow_hidden: Vec<f32>,
    pub codebook_ids: Vec<u32>,
}

pub fn generate_semantic_tokens<R: RandomSource + ?Sized>(
    state: &mut SlowArState,
    tokenizer: &S2TokenizerConfig,
    graph: &DualArGraphSpec,
    prompt: &PromptTensor,
    params: &GenerateParams,
    rng: &mut R,
) -> Result<GenerateSemanticResult> {
    let flat_prompt = transpose_to_time_major(prompt)?;
    state.reset();
    let mut step = state.prefill(&flat_prompt)?;
    let bias = build_semantic_bias(
        step.logits.len(),
        graph.semantic_begin_id,
        graph.semantic_end_id,
        Some(tokenizer.im_end_id),
    );
    let sampler = SamplerParams {
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
    };

    let prompt_cols = prompt.cols;
    let mut tokens = Vec::new();
    let mut main_token = sample_semantic_token(
        &mut step,
        &bias,
        tokenizer.im_end_id,
        0,
        params.min_tokens_before_end,
        &sampler,
        rng,
    )?;
    tokens.push(main_token);

    let mut generated = 0u32;
    while main_token != tokenizer.im_end_id && generated < params.max_new_tokens {
        let step_input = build_step_input(main_token, None, graph)?;
        step = state.step(&step_input)?;
        generated += 1;
        let generated_after_prefill = state
            .n_past()
            .saturating_sub(prompt_cols)
            .saturating_add(tokens.len().saturating_sub(1));
        main_token = sample_semantic_token(
            &mut step,
            &bias,
            tokenizer.im_end_id,
            generated_after_prefill,
            params.min_tokens_before_end,
            &sampler,
            rng,
        )?;
        tokens.push(main_token);
    }

    Ok(GenerateSemanticResult { token_ids: tokens })
}

/// Autoregressive Slow-AR + Fast-AR loop; returns VQ codes for each generated frame (no RAS).
pub fn generate_codes<R: RandomSource + ?Sized>(
    state: &mut SlowArState,
    tokenizer: &S2TokenizerConfig,
    graph: &DualArGraphSpec,
    prompt: &PromptTensor,
    params: &GenerateParams,
    fast_weights: &FastArWeights,
    rng: &mut R,
) -> Result<GenerateCodesResult> {
    let flat_prompt = transpose_to_time_major(prompt)?;
    state.reset();
    let mut step = state.prefill(&flat_prompt)?;
    let mut slow_ar_profile = state.decode_profile();
    let bias = build_semantic_bias(
        step.logits.len(),
        graph.semantic_begin_id,
        graph.semantic_end_id,
        Some(tokenizer.im_end_id),
    );
    let sampler = SamplerParams {
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
    };
    let num_codebooks = graph.num_codebooks.max(1);
    let max_new = params.max_new_tokens;
    let mut codes =
        vec![
            0i32;
            usize::try_from(num_codebooks)
                .map_err(|_| InferError::Message("num_codebooks overflows usize".into()))?
                .checked_mul(usize::try_from(max_new).map_err(|_| {
                    InferError::Message("max_new_tokens overflows usize".into())
                })?)
                .ok_or_else(|| InferError::Message("codes buffer size overflow".into()))?
        ];

    let mut main_token = sample_semantic_token(
        &mut step,
        &bias,
        tokenizer.im_end_id,
        0,
        params.min_tokens_before_end,
        &sampler,
        rng,
    )?;

    let mut step_index = 0u32;
    while main_token != tokenizer.im_end_id && step_index < max_new {
        let sem_code = semantic_to_codebook_id(main_token, graph)?;
        let codebooks = generate_codebooks_for_semantic(
            &step.hidden,
            sem_code,
            graph,
            fast_weights,
            &sampler,
            rng,
        )?;

        let frame = usize::try_from(step_index)
            .map_err(|_| InferError::Message("step_index overflows usize".into()))?;
        let num_cb = usize::try_from(num_codebooks)
            .map_err(|_| InferError::Message("num_codebooks overflows usize".into()))?;
        for (cb, &id) in codebooks.iter().enumerate().take(num_cb) {
            let offset = cb
                .checked_mul(
                    usize::try_from(max_new).map_err(|_| {
                        InferError::Message("max_new_tokens overflows usize".into())
                    })?,
                )
                .and_then(|base| base.checked_add(frame))
                .ok_or_else(|| InferError::Message("codes index overflow".into()))?;
            codes[offset] = i32::try_from(id)
                .map_err(|_| InferError::Message("codebook id does not fit i32".into()))?;
        }

        let step_input = build_step_input(main_token, Some(&codebooks), graph)?;
        step = state.step(&step_input)?;
        slow_ar_profile += state.decode_profile();
        step_index += 1;

        let generated_after_prefill = state
            .n_past()
            .saturating_sub(prompt.cols)
            .saturating_add(step_index as usize);
        main_token = sample_semantic_token(
            &mut step,
            &bias,
            tokenizer.im_end_id,
            generated_after_prefill,
            params.min_tokens_before_end,
            &sampler,
            rng,
        )?;
    }

    let n_frames = step_index;
    compact_codes(&mut codes, num_codebooks, max_new, n_frames)?;
    Ok(GenerateCodesResult {
        codes,
        num_codebooks,
        n_frames,
        slow_ar_profile,
    })
}

/// After Slow-AR prefill, sample the first semantic token and run Fast-AR for all codebooks.
pub fn generate_fast_ar_first_frame<R: RandomSource + ?Sized>(
    state: &mut SlowArState,
    tokenizer: &S2TokenizerConfig,
    graph: &DualArGraphSpec,
    prompt: &PromptTensor,
    params: &GenerateParams,
    fast_weights: &FastArWeights,
    rng: &mut R,
) -> Result<FastArFirstFrameResult> {
    let flat_prompt = transpose_to_time_major(prompt)?;
    state.reset();
    let mut step = state.prefill(&flat_prompt)?;
    let bias = build_semantic_bias(
        step.logits.len(),
        graph.semantic_begin_id,
        graph.semantic_end_id,
        Some(tokenizer.im_end_id),
    );
    let sampler = SamplerParams {
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
    };
    let main_token_id = sample_semantic_token(
        &mut step,
        &bias,
        tokenizer.im_end_id,
        0,
        params.min_tokens_before_end,
        &sampler,
        rng,
    )?;
    let sem_code = semantic_to_codebook_id(main_token_id, graph)?;
    let codebook_ids = generate_codebooks_for_semantic(
        &step.hidden,
        sem_code,
        graph,
        fast_weights,
        &sampler,
        rng,
    )?;
    Ok(FastArFirstFrameResult {
        prompt_cols: prompt.cols,
        main_token_id,
        slow_hidden: step.hidden,
        codebook_ids,
    })
}

fn semantic_to_codebook_id(main_token: u32, graph: &DualArGraphSpec) -> Result<u32> {
    if main_token < graph.semantic_begin_id || main_token > graph.semantic_end_id {
        return Ok(0);
    }
    let sem_code = main_token - graph.semantic_begin_id;
    if sem_code >= graph.codebook_size {
        return Ok(graph.codebook_size.saturating_sub(1));
    }
    Ok(sem_code)
}

fn sample_semantic_token<R: RandomSource + ?Sized>(
    step: &mut SlowArStepResult,
    bias: &[f32],
    im_end_id: u32,
    generated_after_prefill: usize,
    min_tokens_before_end: u32,
    sampler: &SamplerParams,
    rng: &mut R,
) -> Result<u32> {
    let mut logits = step.logits.clone();
    let block_im_end = (generated_after_prefill as u32) < min_tokens_before_end;
    apply_semantic_bias(&mut logits, bias, block_im_end, Some(im_end_id))?;
    let force_id = if block_im_end { None } else { Some(im_end_id) };
    sample_token(&logits, sampler, force_id, rng)
}

fn build_step_input(
    main_token: u32,
    codebooks_cb: Option<&[u32]>,
    graph: &DualArGraphSpec,
) -> Result<Vec<i32>> {
    let codebook_dim = usize::try_from(graph.codebook_input_dim())
        .map_err(|_| InferError::Message("codebook_input_dim overflows usize".into()))?;
    let num_codebooks = usize::try_from(graph.num_codebooks)
        .map_err(|_| InferError::Message("num_codebooks overflows usize".into()))?;
    let mut flat = vec![0i32; codebook_dim];
    flat[0] = i32::try_from(main_token)
        .map_err(|_| InferError::Message("main token id does not fit i32".into()))?;
    if main_token >= graph.semantic_begin_id && main_token <= graph.semantic_end_id {
        if let Some(codebooks) = codebooks_cb {
            if codebooks.len() != num_codebooks {
                return Err(InferError::Message(format!(
                    "codebooks_cb length {} != num_codebooks {num_codebooks}",
                    codebooks.len()
                )));
            }
            for (slot, &cb) in flat.iter_mut().skip(1).zip(codebooks) {
                *slot = i32::try_from(cb)
                    .map_err(|_| InferError::Message("codebook id does not fit i32".into()))?;
            }
        } else {
            let sem_code = main_token - graph.semantic_begin_id;
            if sem_code >= graph.codebook_size {
                return Err(InferError::Message(format!(
                    "semantic token {main_token} maps outside codebook_size {}",
                    graph.codebook_size
                )));
            }
            flat[1] = i32::try_from(sem_code)
                .map_err(|_| InferError::Message("semantic code does not fit i32".into()))?;
            for slot in flat
                .iter_mut()
                .skip(2)
                .take(num_codebooks.saturating_sub(1))
            {
                *slot = 0;
            }
        }
    }
    Ok(flat)
}

fn compact_codes(
    codes: &mut Vec<i32>,
    num_codebooks: u32,
    max_new_tokens: u32,
    n_frames: u32,
) -> Result<()> {
    if n_frames >= max_new_tokens {
        let keep = usize::try_from(num_codebooks)
            .map_err(|_| InferError::Message("num_codebooks overflows usize".into()))?
            .checked_mul(
                usize::try_from(n_frames)
                    .map_err(|_| InferError::Message("n_frames overflows usize".into()))?,
            )
            .ok_or_else(|| InferError::Message("compacted codes size overflow".into()))?;
        codes.truncate(keep);
        return Ok(());
    }
    let num_cb = usize::try_from(num_codebooks)
        .map_err(|_| InferError::Message("num_codebooks overflows usize".into()))?;
    let frames = usize::try_from(n_frames)
        .map_err(|_| InferError::Message("n_frames overflows usize".into()))?;
    let max_new = usize::try_from(max_new_tokens)
        .map_err(|_| InferError::Message("max_new_tokens overflows usize".into()))?;
    let mut compacted = vec![
        0i32;
        num_cb.checked_mul(frames).ok_or_else(|| {
            InferError::Message("compacted codes buffer overflow".into())
        })?
    ];
    for cb in 0..num_cb {
        for t in 0..frames {
            let src = cb
                .checked_mul(max_new)
                .and_then(|b| b.checked_add(t))
                .ok_or_else(|| InferError::Message("codes src index overflow".into()))?;
            let dst = cb
                .checked_mul(frames)
                .and_then(|b| b.checked_add(t))
                .ok_or_else(|| InferError::Message("codes dst index overflow".into()))?;
            compacted[dst] = codes[src];
        }
    }
    *codes = compacted;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{build_prompt, PromptBuildOptions};
    use crate::registry::DualArGraphSpec;
    use crate::sampling::SeededRng;
    use crate::tokenizer::S2Tokenizer;
    use fish_s2_core::gguf::GgufFile;
    use std::path::PathBuf;

    fn test_graph() -> DualArGraphSpec {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/s2-pro-f16-transformer-only.gguf");
        let gguf = GgufFile::open(&path).expect("gguf");
        DualArGraphSpec::from_gguf(&gguf).expect("graph")
    }

    #[test]
    fn compact_codes_trims_stride_when_fewer_frames() {
        // codebook-major stride: codes[cb * max_new + t]
        let mut codes = vec![10, 11, 0, 20, 21, 0];
        compact_codes(&mut codes, 2, 3, 2).expect("compact");
        assert_eq!(codes, vec![10, 11, 20, 21]);
    }

    #[test]
    #[ignore = "requires local s2-pro transformer GGUF in models/"]
    fn build_step_input_accepts_full_fast_codebooks() {
        let graph = test_graph();
        let sem_main = graph.semantic_begin_id + 42;
        let cbs: Vec<u32> = (0..graph.num_codebooks).collect();
        let flat = build_step_input(sem_main, Some(&cbs), &graph).expect("step input");
        assert_eq!(flat.len() as u32, graph.codebook_input_dim());
        assert_eq!(flat[0], sem_main as i32);
        for (slot, &cb) in flat.iter().skip(1).zip(cbs.iter()) {
            assert_eq!(*slot, cb as i32);
        }
    }

    fn fixture_transformer_path() -> Option<PathBuf> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/s2-pro-f16-transformer-only.gguf");
        path.exists().then_some(path)
    }

    #[test]
    #[ignore = "requires local GGUF + tokenizer; slow (36L F16 CPU prefill)"]
    fn generate_codes_greedy_short_prompt() {
        let transformer = fixture_transformer_path().expect("transformer gguf");
        let tokenizer_path = crate::paths::default_tokenizer_path();
        let tokenizer = S2Tokenizer::from_file(&tokenizer_path).expect("tokenizer");
        let mut state = SlowArState::open_default_max_seq_len(&transformer).expect("state");
        let graph = state.graph_spec().clone();
        let gguf = GgufFile::open(&transformer).expect("gguf");
        let registry =
            crate::registry::TransformerTensorRegistry::from_gguf(&gguf).expect("registry");
        let fast_weights =
            crate::fast_ar::FastArWeights::from_gguf(&gguf, &registry).expect("fast weights");
        let prompt = build_prompt(
            &tokenizer,
            PromptBuildOptions {
                text: "hi",
                prompt_text: None,
                prompt_codes: None,
                graph: &graph,
            },
        )
        .expect("prompt");
        let mut rng = SeededRng::new(0);
        let result = generate_codes(
            &mut state,
            &tokenizer.config(),
            &graph,
            &prompt,
            &GenerateParams {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                min_tokens_before_end: 0,
            },
            &fast_weights,
            &mut rng,
        )
        .expect("generate codes");
        assert_eq!(result.num_codebooks, graph.num_codebooks);
        assert!(result.n_frames > 0);
        assert_eq!(
            result.codes.len(),
            usize::try_from(result.num_codebooks * result.n_frames).unwrap()
        );
    }

    #[test]
    #[ignore = "requires local GGUF + tokenizer; slow (36L F16 CPU prefill)"]
    fn generate_semantic_greedy_short_prompt() {
        let transformer = fixture_transformer_path().expect("transformer gguf");
        let tokenizer_path = crate::paths::default_tokenizer_path();
        let tokenizer = S2Tokenizer::from_file(&tokenizer_path).expect("tokenizer");
        let mut state = SlowArState::open_default_max_seq_len(&transformer).expect("state");
        let graph = state.graph_spec().clone();
        let prompt = build_prompt(
            &tokenizer,
            PromptBuildOptions {
                text: "hi",
                prompt_text: None,
                prompt_codes: None,
                graph: &graph,
            },
        )
        .expect("prompt");
        let mut rng = SeededRng::new(0);
        let result = generate_semantic_tokens(
            &mut state,
            &tokenizer.config(),
            &graph,
            &prompt,
            &GenerateParams {
                max_new_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                min_tokens_before_end: 0,
            },
            &mut rng,
        )
        .expect("generate");
        assert!(!result.token_ids.is_empty());
    }
}
