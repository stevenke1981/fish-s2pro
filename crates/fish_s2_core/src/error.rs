use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("{0}")]
    Message(String),
    #[error("HTTP error: {0}")]
    #[cfg(feature = "http-client")]
    Http(#[from] reqwest::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, CoreError>;

impl From<anyhow::Error> for CoreError {
    fn from(value: anyhow::Error) -> Self {
        CoreError::Message(value.to_string())
    }
}
