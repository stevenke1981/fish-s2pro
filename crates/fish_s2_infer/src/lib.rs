pub mod attention;
mod engine;
mod error;
mod paths;
pub mod registry;
pub mod slow_ar;
pub mod tensor;
pub mod tokenizer;
mod wav;

#[cfg(feature = "server")]
pub mod server;

pub use attention::{apply_rope_normal, gqa_decode_attention, GqaAttentionShape, SlowArKvCache};
pub use engine::{EngineConfig, InferenceEngine, SynthesisRequest};
pub use error::{InferError, Result};
pub use paths::{default_tokenizer_path, ensure_project_dirs, models_dir, project_root};
pub use registry::{
    ArGraphSpec, DualArGraphSpec, FastArLayerWeights, KvCacheSpec, SlowArLayerWeights, TensorRole,
    TensorSpec, TransformerTensorRegistry,
};
pub use slow_ar::{
    SlowArLayerF16Weights, SlowArLayerForwardOutput, SlowArLayerShape, SlowArLayerSkeleton,
};
pub use tensor::{linear, rms_norm, F16TensorView};
pub use tokenizer::{bytelevel_encode_utf8, gpt2_byte_to_unicode, S2Tokenizer, TokenizedText};
pub use wav::pcm_to_wav;

#[cfg(feature = "server")]
pub use server::{spawn_server, InlineServer, ServerHandle};
