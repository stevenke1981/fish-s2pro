pub mod attention;
pub mod backend;
pub mod codec;
mod engine;
mod error;
pub mod fast_ar;
mod generate;
mod paths;
mod pipeline;
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
pub use backend::{CpuMatmulBackend, MatmulBackend};
pub use codec::{
    classify_codec_tensor, decode_waveform, decode_waveform_to_wav, encode_reference_audio,
    encode_reference_wav_file, format_codec_dimensions, forward_codec_decoder,
    forward_codec_encoder_frontend, forward_codec_encoder_frontend_with_checkpoints,
    forward_codec_post_module, forward_codec_quantizer_encode_stage, forward_codec_upsample,
    rvq_decode_latents, rvq_encode_latents_nearest, rvq_lookup_codes, CodecDecodeLatentsResult,
    CodecDecoderF16Weights, CodecDecoderWeights, CodecDownsampleF16Weights,
    CodecDownsampleStageF16Weights, CodecDownsampleStageWeights, CodecDownsampleWeights,
    CodecEncodeStageResult, CodecEncoderBlockF16Weights, CodecEncoderBlockWeights,
    CodecEncoderF16Weights, CodecEncoderFrontendCheckpoint, CodecEncoderFrontendResult,
    CodecEncoderWeights, CodecF16Weights, CodecPostModuleF16Weights, CodecPostModuleResult,
    CodecPreModuleF16Weights, CodecQuantizerF16Weights, CodecQuantizerWeights,
    CodecReferenceAudioResult, CodecReferenceEncoderF16Weights, CodecRvqLookupResult,
    CodecTensorDumpRow, CodecTensorRegistry, CodecTensorRoleInfo, CodecTransformerLayerF16Weights,
    CodecTransformerLayerWeights, CodecUpsampleF16Weights, CodecUpsampleResult,
    CodecUpsampleStageF16Weights, CodecUpsampleStageWeights, CodecUpsampleWeights,
    CodecVqEncodeResult, CodecWaveformResult, CODEC_ARCHITECTURE, CODEC_DECODER_RATES,
    CODEC_ENCODER_KERNELS, CODEC_ENCODER_RATES, CODEC_ENCODER_TRANSFORMER_WINDOW_SIZE,
    CODEC_FRAME_LENGTH, CODEC_RESIDUAL_QUANTIZERS, CODEC_SAMPLE_RATE, CODEC_TRANSFORMER_LAYERS,
};
pub use engine::{EngineBackend, EngineConfig, InferenceEngine, SynthesisRequest};
pub use error::{InferError, Result};
pub use fast_ar::{
    forward_codebook_prefix, generate_codebooks_for_semantic, FastArHeadF16Weights,
    FastArLayerF16Weights, FastArLayerShape, FastArWeights,
};
pub use generate::{
    generate_codes, generate_fast_ar_first_frame, generate_semantic_tokens, FastArFirstFrameResult,
    GenerateCodesResult, GenerateParams, GenerateSemanticResult,
};
pub use paths::{
    default_tokenizer_path, ensure_project_dirs, models_dir, output_dir, project_root,
    server_workdir,
};
pub use pipeline::{RustPipeline, RustPipelineConfig, RustSynthesisOptions, RustSynthesisResult};
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
pub use tensor::{
    embedding_lookup_rows, linear, linear_with_backend, matvec_f16_streaming, rms_norm,
    F16TensorBytes, F16TensorView,
};
pub use tokenizer::{bytelevel_encode_utf8, gpt2_byte_to_unicode, S2Tokenizer, TokenizedText};
pub use wav::{pcm_to_wav, read_wav_mono_f32, wav_mono_f32_from_bytes};

#[cfg(feature = "server")]
pub use server::{spawn_server, InlineServer, ServerHandle};
