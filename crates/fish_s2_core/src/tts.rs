use std::path::PathBuf;
#[cfg(feature = "http-client")]
use std::time::Duration;

#[cfg(feature = "http-client")]
use serde::Deserialize;
use serde::Serialize;

#[cfg(feature = "http-client")]
use crate::error::CoreError;
#[cfg(feature = "http-client")]
use crate::error::Result;

#[derive(Debug, Clone, Serialize)]
pub struct TtsRequest {
    pub text: String,
    pub format: String,
}

#[derive(Debug, Clone)]
pub struct TtsResponse {
    pub wav_bytes: Vec<u8>,
    pub saved_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[cfg(feature = "http-client")]
struct TtsApiError {
    error: Option<String>,
    message: Option<String>,
}

#[cfg(feature = "http-client")]
pub struct TtsClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

#[cfg(feature = "http-client")]
impl TtsClient {
    pub fn new(port: u16) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("http client");
        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            client,
        }
    }

    pub fn health_check(&self) -> bool {
        self.client
            .get(format!("{}/", self.base_url))
            .send()
            .is_ok()
    }

    pub fn synthesize(&self, request: &TtsRequest) -> Result<TtsResponse> {
        let url = format!("{}/v1/tts", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer dummy")
            .json(request)
            .send()
            .map_err(CoreError::Http)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            if let Ok(err) = serde_json::from_str::<TtsApiError>(&body) {
                let msg = err.error.or(err.message).unwrap_or(body);
                return Err(CoreError::Message(format!("TTS failed ({status}): {msg}")));
            }
            return Err(CoreError::Message(format!("TTS failed ({status}): {body}")));
        }

        let wav_bytes = resp.bytes().map_err(CoreError::Http)?.to_vec();
        Ok(TtsResponse {
            wav_bytes,
            saved_path: None,
        })
    }

    pub fn synthesize_to_file(
        &self,
        request: &TtsRequest,
        output_path: PathBuf,
    ) -> Result<TtsResponse> {
        let mut response = self.synthesize(request)?;
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(CoreError::Io)?;
        }
        std::fs::write(&output_path, &response.wav_bytes).map_err(CoreError::Io)?;
        response.saved_path = Some(output_path);
        Ok(response)
    }
}
