use std::net::SocketAddr;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde::Deserialize;
use std::sync::Mutex;

use crate::engine::{InferenceEngine, SynthesisRequest};
use crate::error::{InferError, Result};

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<InferenceEngine>>,
}

#[derive(Debug, Deserialize)]
struct TtsBody {
    text: String,
    #[serde(default = "default_format")]
    format: String,
}

fn default_format() -> String {
    "wav".to_string()
}

pub struct InlineServer {
    _handle: JoinHandle<()>,
    port: u16,
}

pub struct ServerHandle {
    _server: InlineServer,
}

impl ServerHandle {
    pub fn port(&self) -> u16 {
        self._server.port
    }
}

pub fn spawn_server(engine: InferenceEngine, port: u16) -> Result<ServerHandle> {
    let server = InlineServer::start(engine, port)?;
    Ok(ServerHandle { _server: server })
}

impl InlineServer {
    fn start(engine: InferenceEngine, port: u16) -> Result<Self> {
        let state = AppState {
            engine: Arc::new(Mutex::new(engine)),
        };
        let app = Router::new()
            .route("/v1/tts", post(tts_handler))
            .route("/", axum::routing::get(health))
            .route("/health", axum::routing::get(health))
            .with_state(state);

        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let handle = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            rt.block_on(async {
                let listener = tokio::net::TcpListener::bind(addr)
                    .await
                    .expect("bind inference server");
                axum::serve(listener, app).await.expect("serve");
            });
        });
        Ok(Self {
            _handle: handle,
            port,
        })
    }
}

async fn health() -> &'static str {
    "ok"
}

async fn tts_handler(
    State(state): State<AppState>,
    Json(body): Json<TtsBody>,
) -> impl IntoResponse {
    if body.format.to_lowercase() != "wav" {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "only wav format supported"})),
        )
            .into_response();
    }
    let request = SynthesisRequest {
        text: body.text,
        reference_text: None,
        reference_wav: None,
    };
    let engine = state.engine.clone();
    let result = {
        let engine = engine
            .lock()
            .map_err(|_| InferError::Message("engine lock".into()));
        match engine {
            Ok(engine) => engine.synthesize_wav(&request),
            Err(e) => Err(e),
        }
    };

    match result {
        Ok(bytes) => (StatusCode::OK, [(header::CONTENT_TYPE, "audio/wav")], bytes).into_response(),
        Err(e) => error_response(e),
    }
}

fn error_response(err: InferError) -> axum::response::Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": err.to_string()})),
    )
        .into_response()
}
