use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("{0}")]
    Message(String),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("native backend not linked — run scripts/build_s2_native.ps1 and rebuild with S2_CPP_LIB + --features cpp-engine")]
    NativeNotLinked,
}

pub type Result<T> = std::result::Result<T, InferError>;
