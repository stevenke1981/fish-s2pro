//! Pure-Rust S2 synthesis pipeline glue.
//!
//! This module deliberately stays thin: it wires the already parity-pinned
//! Slow-AR/Fast-AR code generation path to the codec waveform decoder without
//! changing the production engine backend selection yet.

use std::path::{Path, PathBuf};

use fish_s2_core::gguf::GgufFile;

use crate::codec::{
    decode_waveform, CodecDecoderF16Weights, CodecF16Weights, CodecPostModuleF16Weights,
    CodecUpsampleF16Weights, CodecWaveformResult,
};
use crate::error::{InferError, Result};
use crate::fast_ar::FastArWeights;
use crate::generate::{generate_codes, GenerateCodesResult, GenerateParams};
use crate::prompt::{build_prompt, PromptBuildOptions, PromptCodes};
use crate::registry::DualArGraphSpec;
use crate::sampling::SeededRng;
use crate::slow_ar::SlowArState;
use crate::tokenizer::S2Tokenizer;
use crate::wav::pcm_to_wav;
use crate::{default_tokenizer_path, TransformerTensorRegistry};

#[derive(Debug, Clone)]
pub struct RustPipelineConfig {
    pub transformer_gguf: PathBuf,
    pub codec_gguf: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_seq_len: Option<usize>,
}

impl RustPipelineConfig {
    pub fn new(transformer_gguf: impl Into<PathBuf>, codec_gguf: impl Into<PathBuf>) -> Self {
        Self {
            transformer_gguf: transformer_gguf.into(),
            codec_gguf: codec_gguf.into(),
            tokenizer_path: default_tokenizer_path(),
            max_seq_len: None,
        }
    }

    pub fn with_tokenizer(mut self, tokenizer_path: impl Into<PathBuf>) -> Self {
        self.tokenizer_path = tokenizer_path.into();
        self
    }

    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = Some(max_seq_len);
        self
    }
}

#[derive(Debug, Clone)]
pub struct RustSynthesisOptions {
    pub text: String,
    pub prompt_text: Option<String>,
    pub prompt_codes: Option<PromptCodes>,
    pub params: GenerateParams,
    pub seed: u64,
}

impl RustSynthesisOptions {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            prompt_text: None,
            prompt_codes: None,
            params: GenerateParams::default(),
            seed: 0,
        }
    }

    pub fn with_reference(
        mut self,
        prompt_text: impl Into<String>,
        prompt_codes: PromptCodes,
    ) -> Self {
        self.prompt_text = Some(prompt_text.into());
        self.prompt_codes = Some(prompt_codes);
        self
    }

    pub fn with_params(mut self, params: GenerateParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RustSynthesisResult {
    pub codes: GenerateCodesResult,
    pub waveform: CodecWaveformResult,
    pub wav_bytes: Vec<u8>,
}

pub struct RustPipeline {
    tokenizer: S2Tokenizer,
    graph: DualArGraphSpec,
    slow_state: SlowArState,
    fast_weights: FastArWeights,
    rvq_weights: CodecF16Weights,
    post_weights: CodecPostModuleF16Weights,
    upsample_weights: CodecUpsampleF16Weights,
    decoder_weights: CodecDecoderF16Weights,
}

impl RustPipeline {
    pub fn load(config: RustPipelineConfig) -> Result<Self> {
        require_file("transformer GGUF", &config.transformer_gguf)?;
        require_file("codec GGUF", &config.codec_gguf)?;
        require_file("tokenizer", &config.tokenizer_path)?;

        let tokenizer = S2Tokenizer::from_file(&config.tokenizer_path)?;
        let slow_state = match config.max_seq_len {
            Some(max_seq_len) => SlowArState::open(&config.transformer_gguf, max_seq_len)?,
            None => SlowArState::open_default_max_seq_len(&config.transformer_gguf)?,
        };
        let graph = slow_state.graph_spec().clone();

        let transformer = GgufFile::open(&config.transformer_gguf)
            .map_err(|err| InferError::Message(err.to_string()))?;
        let registry = TransformerTensorRegistry::from_gguf(&transformer)?;
        let fast_weights = FastArWeights::from_gguf(&transformer, &registry)?;

        let codec = GgufFile::open(&config.codec_gguf)
            .map_err(|err| InferError::Message(err.to_string()))?;
        let rvq_weights = CodecF16Weights::from_gguf(&codec)?;
        let post_weights = CodecPostModuleF16Weights::from_gguf(&codec)?;
        let upsample_weights = CodecUpsampleF16Weights::from_gguf(&codec)?;
        let decoder_weights = CodecDecoderF16Weights::from_gguf(&codec)?;

        Ok(Self {
            tokenizer,
            graph,
            slow_state,
            fast_weights,
            rvq_weights,
            post_weights,
            upsample_weights,
            decoder_weights,
        })
    }

    pub fn synthesize(&mut self, options: &RustSynthesisOptions) -> Result<RustSynthesisResult> {
        validate_prompt_pair(
            options.prompt_text.as_deref(),
            options.prompt_codes.as_ref(),
        )?;
        let prompt = build_prompt(
            &self.tokenizer,
            PromptBuildOptions {
                text: &options.text,
                prompt_text: options.prompt_text.as_deref(),
                prompt_codes: options.prompt_codes.as_ref(),
                graph: &self.graph,
            },
        )?;
        let mut rng = SeededRng::new(options.seed);
        let codes = generate_codes(
            &mut self.slow_state,
            &self.tokenizer.config(),
            &self.graph,
            &prompt,
            &options.params,
            &self.fast_weights,
            &mut rng,
        )?;
        let waveform = decode_waveform(
            &codes.codes,
            codes.num_codebooks,
            codes.n_frames,
            &self.rvq_weights,
            &self.post_weights,
            &self.upsample_weights,
            &self.decoder_weights,
        )?;
        let wav_bytes = pcm_to_wav(&waveform.samples, waveform.sample_rate);
        Ok(RustSynthesisResult {
            codes,
            waveform,
            wav_bytes,
        })
    }

    pub fn graph_spec(&self) -> &DualArGraphSpec {
        &self.graph
    }
}

fn require_file(kind: &str, path: &Path) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    Err(InferError::Message(format!(
        "{kind} not found: {}",
        path.display()
    )))
}

fn validate_prompt_pair(
    prompt_text: Option<&str>,
    prompt_codes: Option<&PromptCodes>,
) -> Result<()> {
    match (prompt_text, prompt_codes) {
        (Some(_), Some(_)) | (None, None) => Ok(()),
        _ => Err(InferError::Message(
            "prompt_text and prompt_codes must both be set or both omitted".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_pair_validation_requires_both_reference_parts() {
        let codes = PromptCodes {
            num_codebooks: 10,
            cols: 1,
            data: vec![0; 10],
        };
        assert!(validate_prompt_pair(None, None).is_ok());
        assert!(validate_prompt_pair(Some("ref"), Some(&codes)).is_ok());
        assert!(validate_prompt_pair(Some("ref"), None).is_err());
        assert!(validate_prompt_pair(None, Some(&codes)).is_err());
    }
}
