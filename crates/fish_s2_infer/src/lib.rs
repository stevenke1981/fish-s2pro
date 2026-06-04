pub mod attention;
pub mod codec;
mod engine;
mod error;
pub mod fast_ar;
mod generate;
mod paths;
pub mod prompt;
pub mod registry;
pub mod sampling;
pub mod slow_ar;
pub mod tensor;
pub mod tokenizer;
mod tokenizer_s2cpp;
mod wav;

#[cfg(feature = "server")]
pub mod server;

pub use attention::{apply_rope_normal, gqa_decode_attention, GqaAttentionShape, SlowArKvCache};
pub use codec::{
    classify_codec_tensor, decode_waveform, decode_waveform_to_wav, format_codec_dimensions,
    forward_codec_decoder, forward_codec_post_module, forward_codec_quantizer_encode_stage,
    forward_codec_upsample, rvq_decode_latents, rvq_lookup_codes, CodecDecodeLatentsResult,
    CodecDecoderF16Weights, CodecDecoderWeights, CodecDownsampleF16Weights,
    CodecDownsampleStageF16Weights, CodecDownsampleStageWeights, CodecDownsampleWeights,
    CodecEncodeStageResult, CodecF16Weights, CodecPostModuleF16Weights, CodecPostModuleResult,
    CodecPreModuleF16Weights, CodecQuantizerF16Weights, CodecQuantizerWeights,
    CodecRvqLookupResult, CodecTensorDumpRow, CodecTensorRegistry, CodecTensorRoleInfo,
    CodecTransformerLayerF16Weights, CodecTransformerLayerWeights, CodecUpsampleF16Weights,
    CodecUpsampleResult, CodecUpsampleStageF16Weights, CodecUpsampleStageWeights,
    CodecUpsampleWeights, CodecWaveformResult, CODEC_ARCHITECTURE, CODEC_DECODER_RATES,
    CODEC_RESIDUAL_QUANTIZERS, CODEC_SAMPLE_RATE, CODEC_TRANSFORMER_LAYERS,
};
pub use engine::{EngineConfig, InferenceEngine, SynthesisRequest};
pub use error::{InferError, Result};
pub use fast_ar::{
    forward_codebook_prefix, generate_codebooks_for_semantic, FastArHeadF16Weights,
    FastArLayerF16Weights, FastArLayerShape, FastArWeights,
};
pub use generate::{
    generate_codes, generate_fast_ar_first_frame, generate_semantic_tokens, FastArFirstFrameResult,
    GenerateCodesResult, GenerateParams, GenerateSemanticResult,
};
pub use paths::{default_tokenizer_path, ensure_project_dirs, models_dir, project_root};
pub use prompt::{
    build_prompt, load_prompt_codes, load_prompt_codes_file, transpose_to_time_major,
    PromptBuildOptions, PromptCodes, PromptCodesFile, PromptTensor,
};
pub use registry::{
    ArGraphSpec, DualArGraphSpec, FastArLayerWeights, KvCacheSpec, SlowArLayerWeights, TensorRole,
    TensorSpec, TransformerTensorRegistry,
};
pub use sampling::{
    apply_semantic_bias, build_semantic_bias, sample_token, semantic_mask_logits, RandomSource,
    SamplerParams, SeededRng, LOGIT_MASKED,
};
pub use slow_ar::{
    embed_slow_ar_time_major, forward_slow_ar_block_prefill_layers,
    forward_slow_ar_block_prefill_layers_cached, SlowArEmbeddingWeights, SlowArLayerBlockOutput,
    SlowArLayerF16Weights, SlowArLayerFeedForwardOutput, SlowArLayerForwardOutput,
    SlowArLayerShape, SlowArLayerSkeleton, SlowArLogitsOutput, SlowArOutputHeadF16Weights,
    SlowArState, SlowArStepResult,
};
pub use tensor::{embedding_lookup_rows, linear, rms_norm, F16TensorView};
pub use tokenizer::{bytelevel_encode_utf8, gpt2_byte_to_unicode, S2Tokenizer, TokenizedText};
pub use wav::pcm_to_wav;

#[cfg(feature = "server")]
pub use server::{spawn_server, InlineServer, ServerHandle};
