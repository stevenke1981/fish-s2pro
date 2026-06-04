mod engine;
mod error;
mod paths;
pub mod registry;
pub mod tensor;
pub mod tokenizer;
mod wav;

#[cfg(feature = "server")]
pub mod server;

pub use engine::{EngineConfig, InferenceEngine, SynthesisRequest};
pub use error::{InferError, Result};
pub use paths::{default_tokenizer_path, ensure_project_dirs, models_dir, project_root};
pub use registry::{
    ArGraphSpec, DualArGraphSpec, FastArLayerWeights, KvCacheSpec, SlowArLayerWeights, TensorRole,
    TensorSpec, TransformerTensorRegistry,
};
pub use tensor::{linear, rms_norm, F16TensorView};
pub use tokenizer::{bytelevel_encode_utf8, gpt2_byte_to_unicode, S2Tokenizer, TokenizedText};
pub use wav::pcm_to_wav;

#[cfg(feature = "server")]
pub use server::{spawn_server, InlineServer, ServerHandle};
