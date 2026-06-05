mod error;

pub mod config;
pub mod paths;
// re-export for apps
pub use paths::{models_dir, output_dir, project_root};
pub mod convert;
pub mod gguf;
pub mod models;
#[cfg(feature = "legacy-s2-exe")]
pub mod server;
pub mod tags;
pub mod tts;
pub mod voice;

pub use config::{copy_reference_files, AppConfig};
pub use convert::{checkpoint_codec_path, resolve_export_script, ConvertPlan};
pub use gguf::GgufSummary;
pub use models::{validate_pair, ModelKind, ModelPair, ScannedModels};
#[cfg(feature = "legacy-s2-exe")]
pub use server::{
    build_start_options, resolve_s2_binary, ServerProcess, ServerStartOptions, ServerStatus,
};
pub use tags::{ControlTag, CONTROL_TAGS};
#[cfg(feature = "http-client")]
pub use tts::TtsClient;
pub use tts::{TtsRequest, TtsResponse};
pub use voice::VoiceProfile;
