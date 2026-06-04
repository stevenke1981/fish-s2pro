mod engine;
mod error;
mod paths;
mod wav;

#[cfg(feature = "server")]
pub mod server;

pub use engine::{EngineConfig, InferenceEngine, SynthesisRequest};
pub use error::{InferError, Result};
pub use paths::{default_tokenizer_path, ensure_project_dirs, models_dir, project_root};
pub use wav::pcm_to_wav;

#[cfg(feature = "server")]
pub use server::{spawn_server, InlineServer, ServerHandle};
