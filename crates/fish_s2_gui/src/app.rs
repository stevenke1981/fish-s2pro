use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;
use fish_s2_core::{
    checkpoint_codec_path, copy_reference_files, resolve_export_script, validate_pair, AppConfig,
    ConvertPlan, GgufSummary, ModelPair, ScannedModels, VoiceProfile, CONTROL_TAGS,
};
#[cfg(feature = "http-client")]
use fish_s2_core::{TtsClient, TtsRequest};
use fish_s2_infer::{EngineBackend, EngineConfig, InferenceEngine, SynthesisRequest};
use uuid::Uuid;

use crate::audio::AudioPlayer;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Generate,
    Clone,
    Edit,
    Models,
    Convert,
    Server,
}

enum BackgroundMsg {
    Status(String),
    SynthesisLog(String),
    TtsDone(Result<(Vec<u8>, PathBuf), String>),
    ConvertDone(Result<String, String>),
    ScanDone(ScannedModels),
    GgufInspect(Result<GgufSummary, String>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NativeRustEngineKey {
    transformer: PathBuf,
    codec: PathBuf,
    workdir: PathBuf,
    backend: String,
    max_new_tokens: u32,
    cuda_device: i32,
    codec_cuda: bool,
    vulkan_device: i32,
    codec_vulkan_device: i32,
}

struct NativeRustEngineCache {
    key: NativeRustEngineKey,
    engine: Arc<Mutex<Option<InferenceEngine>>>,
}

#[derive(Debug, Clone, Copy)]
struct WavAnalysis {
    sample_rate: u32,
    channels: u16,
    duration_secs: f64,
    rms: f64,
    peak: f64,
}

fn configure_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = egui::Color32::from_rgb(25, 28, 31);
    visuals.window_fill = egui::Color32::from_rgb(31, 35, 39);
    visuals.extreme_bg_color = egui::Color32::from_rgb(16, 18, 20);
    visuals.faint_bg_color = egui::Color32::from_rgb(36, 41, 45);
    visuals.hyperlink_color = egui::Color32::from_rgb(94, 190, 175);
    visuals.selection.bg_fill = egui::Color32::from_rgb(43, 128, 121);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(43, 48, 53);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(56, 64, 70);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(65, 89, 94);
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(9.0, 7.0);
    style.spacing.button_padding = egui::vec2(11.0, 6.0);
    style.spacing.combo_width = 136.0;
    ctx.set_style(style);
}

pub struct FishS2App {
    config: AppConfig,
    tab: Tab,
    rust_server: Option<fish_s2_infer::ServerHandle>,
    native_rust_engine: Option<NativeRustEngineCache>,
    scanned: ScannedModels,
    status_line: String,
    script: String,
    script_cursor: usize,
    selected_gguf: Option<PathBuf>,
    gguf_detail: Option<GgufSummary>,
    convert_log: String,
    convert_dtype: String,
    server_log: String,
    synthesis_log: String,
    last_wav: Option<Vec<u8>>,
    last_wav_path: Option<PathBuf>,
    audio: Option<AudioPlayer>,
    busy: bool,
    bg_tx: Sender<BackgroundMsg>,
    bg_rx: Receiver<BackgroundMsg>,
    clone_name: String,
    clone_ref_wav: PathBuf,
    clone_ref_text: String,
}

impl FishS2App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_visuals(&cc.egui_ctx);
        let mut config = AppConfig::load();
        let _ = config.ensure_dirs();
        let (bg_tx, bg_rx) = mpsc::channel();
        let auto_backend_message = maybe_promote_cuda_backend(&mut config);
        let scanned = ScannedModels::scan_dir(&config.models_dir, 4).unwrap_or_default();
        let selected_label = config
            .ensure_active_model_pair(&scanned.pairs)
            .map(|pair| pair.label.clone());
        let _ = config.save();
        let audio = AudioPlayer::new();
        let script = config.last_script.clone();
        let mut status_line =
            model_scan_status(&config.models_dir, &scanned, selected_label.as_deref());
        if let Some(message) = auto_backend_message {
            status_line = format!("{status_line}；{message}");
        }
        Self {
            tab: Tab::Generate,
            rust_server: None,
            native_rust_engine: None,
            scanned,
            status_line,
            script,
            script_cursor: 0,
            selected_gguf: None,
            gguf_detail: None,
            convert_log: String::new(),
            convert_dtype: "f16".to_string(),
            server_log: String::new(),
            synthesis_log: String::new(),
            last_wav: None,
            last_wav_path: None,
            audio,
            busy: false,
            bg_tx,
            bg_rx,
            clone_name: "我的聲音".to_string(),
            clone_ref_wav: PathBuf::new(),
            clone_ref_text: String::new(),
            config,
        }
    }

    fn persist(&mut self) {
        self.config.last_script = self.script.clone();
        let _ = self.config.save();
    }

    fn poll_background(&mut self) {
        while let Ok(msg) = self.bg_rx.try_recv() {
            match msg {
                BackgroundMsg::Status(line) => {
                    self.status_line = line;
                }
                BackgroundMsg::SynthesisLog(line) => {
                    append_log_line(&mut self.synthesis_log, &line);
                }
                BackgroundMsg::TtsDone(Ok((bytes, path))) => {
                    self.busy = false;
                    self.last_wav = Some(bytes.clone());
                    self.last_wav_path = Some(path.clone());
                    self.status_line = format!("已生成：{}", path.display());
                    if let Some(analysis) = analyze_wav_bytes(&bytes) {
                        append_log_line(
                            &mut self.synthesis_log,
                            &format!(
                                "WAV 診斷：{} Hz / {}ch / {:.3}s / RMS {:.6} / peak {:.6}",
                                analysis.sample_rate,
                                analysis.channels,
                                analysis.duration_secs,
                                analysis.rms,
                                analysis.peak
                            ),
                        );
                        if let Some(warning) =
                            wav_warning(&analysis, self.config.server_max_new_tokens)
                        {
                            self.status_line = warning.clone();
                            append_log_line(&mut self.synthesis_log, &format!("警告：{warning}"));
                        }
                    } else {
                        append_log_line(
                            &mut self.synthesis_log,
                            "WAV 診斷：無法解析 PCM16 WAV header",
                        );
                    }
                    append_log_line(
                        &mut self.synthesis_log,
                        &format!("完成：已寫出 {} bytes 到 {}", bytes.len(), path.display()),
                    );
                    if let Some(player) = &self.audio {
                        if let Err(e) = player.play_wav_bytes(&bytes) {
                            self.status_line = format!("已儲存但播放失敗：{e}");
                            append_log_line(&mut self.synthesis_log, &format!("播放失敗：{e}"));
                        } else {
                            append_log_line(&mut self.synthesis_log, "播放：已送出 WAV 到音訊裝置");
                        }
                    }
                }
                BackgroundMsg::TtsDone(Err(e)) => {
                    self.busy = false;
                    append_log_line(&mut self.synthesis_log, &format!("失敗：{e}"));
                    self.status_line = e;
                }
                BackgroundMsg::ConvertDone(Ok(log)) => {
                    self.busy = false;
                    self.convert_log = log;
                    self.status_line = "GGUF 轉換完成。正在重新掃描模型…".to_string();
                    self.rescan_models_async();
                }
                BackgroundMsg::ConvertDone(Err(e)) => {
                    self.busy = false;
                    self.convert_log = e.clone();
                    self.status_line = e;
                }
                BackgroundMsg::ScanDone(models) => {
                    self.adopt_scanned_models(models);
                }
                BackgroundMsg::GgufInspect(Ok(summary)) => {
                    self.gguf_detail = Some(summary);
                }
                BackgroundMsg::GgufInspect(Err(e)) => self.status_line = e,
            }
        }
    }

    fn server_running(&self) -> bool {
        self.rust_server.is_some()
    }

    fn rescan_models_async(&mut self) {
        let dir = self.config.models_dir.clone();
        let tx = self.bg_tx.clone();
        thread::spawn(move || {
            let result = ScannedModels::scan_dir(&dir, 4).unwrap_or_default();
            let _ = tx.send(BackgroundMsg::ScanDone(result));
        });
    }

    fn active_pair(&self) -> Option<&ModelPair> {
        self.config.active_model_pair(&self.scanned.pairs)
    }

    fn ensure_active_pair(&mut self) -> Option<ModelPair> {
        let previous_id = self.config.active_model_pair_id.clone();
        let selected = self
            .config
            .ensure_active_model_pair(&self.scanned.pairs)
            .cloned();
        if previous_id != self.config.active_model_pair_id {
            self.native_rust_engine = None;
            let _ = self.config.save();
        }
        selected
    }

    fn adopt_scanned_models(&mut self, models: ScannedModels) {
        let previous_id = self.config.active_model_pair_id.clone();
        self.scanned = models;
        let selected_label = self
            .config
            .ensure_active_model_pair(&self.scanned.pairs)
            .map(|pair| pair.label.clone());
        if previous_id != self.config.active_model_pair_id {
            self.native_rust_engine = None;
            let _ = self.config.save();
        }
        self.status_line = model_scan_status(
            &self.config.models_dir,
            &self.scanned,
            selected_label.as_deref(),
        );
    }

    fn start_server(&mut self) {
        let pair = match self.ensure_active_pair() {
            Some(p) => p,
            None => {
                self.status_line =
                    missing_model_pair_message(&self.config.models_dir, &self.scanned);
                return;
            }
        };
        if let Err(e) = validate_pair(&pair) {
            self.status_line = e.to_string();
            return;
        }

        let voice = self.config.active_voice();
        let (ref_wav, ref_text) = voice
            .map(|v| {
                (
                    Some(v.reference_wav.clone()),
                    Some(v.reference_text.clone()),
                )
            })
            .unwrap_or((None, None));

        let mut engine_cfg =
            match EngineConfig::new(pair.transformer.path.clone(), pair.codec.path.clone()) {
                Ok(c) => c,
                Err(e) => {
                    self.status_line = e.to_string();
                    return;
                }
            };
        engine_cfg.workdir = self.config.server_workdir.clone();
        match EngineBackend::parse(&self.config.server_backend) {
            Ok(backend) => engine_cfg.backend = backend,
            Err(e) => {
                self.status_line = e.to_string();
                return;
            }
        }
        engine_cfg.generate_params.max_new_tokens = self.config.server_max_new_tokens;
        engine_cfg.vulkan_device = self.config.vulkan_device;
        engine_cfg.codec_vulkan_device = self.config.codec_vulkan_device;
        engine_cfg.cuda_device = self.config.cuda_device;
        engine_cfg.codec_cuda = self.config.codec_cuda;

        if let (Some(wav), Some(text)) = (&ref_wav, &ref_text) {
            let _ = copy_reference_files(&engine_cfg.workdir, wav, text);
        }

        match InferenceEngine::load(engine_cfg) {
            Ok(engine) => {
                let backend = engine.backend();
                match fish_s2_infer::spawn_server(engine, self.config.server_port) {
                    Ok(handle) => {
                        self.rust_server = Some(handle);
                        self.server_log = format!(
                            "Rust 推理引擎：http://127.0.0.1:{}\nBackend: {}\n{}\nTransformer: {}\nCodec: {}",
                            self.config.server_port,
                            backend.as_str(),
                            backend_device_line(
                                backend,
                                self.config.cuda_device,
                                self.config.codec_cuda
                            ),
                            pair.transformer.path.display(),
                            pair.codec.path.display()
                        );
                        self.status_line =
                            "Rust 伺服器已啟動（首次載入 GGUF 可能需要數秒）".to_string();
                    }
                    Err(e) => {
                        self.status_line = e.to_string();
                        self.server_log = self.status_line.clone();
                    }
                }
            }
            Err(e) => {
                self.status_line = e.to_string();
                self.server_log = self.status_line.clone();
            }
        }
    }

    fn run_tts(&mut self) {
        if self.config.use_rust_engine {
            self.run_native_rust_tts();
        } else {
            self.run_server_tts();
        }
    }

    fn run_native_rust_tts(&mut self) {
        let pair = match self.ensure_active_pair() {
            Some(p) => p,
            None => {
                self.status_line =
                    missing_model_pair_message(&self.config.models_dir, &self.scanned);
                return;
            }
        };
        if let Err(e) = validate_pair(&pair) {
            self.status_line = e.to_string();
            return;
        }
        let raw_text = self.script.trim();
        let sanitized_text = sanitize_tts_script(raw_text);
        if sanitized_text.is_empty() {
            self.status_line = "請輸入要合成的文字".to_string();
            return;
        }
        let style_prefix = tts_style_prefix(&self.config);
        let text = apply_tts_style(&sanitized_text, &style_prefix);

        let voice = self.config.active_voice().cloned();
        self.busy = true;
        self.synthesis_log.clear();
        append_log_line(&mut self.synthesis_log, "開始：原生 Rust 直接生成");
        append_log_line(&mut self.synthesis_log, &format!("模型：{}", pair.label));
        append_log_line(
            &mut self.synthesis_log,
            &format!("Transformer：{}", pair.transformer.path.display()),
        );
        append_log_line(
            &mut self.synthesis_log,
            &format!("Codec：{}", pair.codec.path.display()),
        );
        append_log_line(
            &mut self.synthesis_log,
            &format!(
                "文字：{} bytes / {} chars",
                text.len(),
                text.chars().count()
            ),
        );
        if sanitized_text != raw_text {
            append_log_line(
                &mut self.synthesis_log,
                "文字清理：已移除 system/user/assistant 或 ChatML 標記，避免被念出來。",
            );
        }
        if !style_prefix.is_empty() {
            append_log_line(
                &mut self.synthesis_log,
                &format!("朗讀控制：{style_prefix}"),
            );
        }
        append_log_line(
            &mut self.synthesis_log,
            &format!("文字預覽：{}", text_preview(&text)),
        );
        let backend = match EngineBackend::parse(&self.config.server_backend) {
            Ok(backend) => backend,
            Err(e) => {
                self.status_line = e.to_string();
                return;
            }
        };
        let (effective_max_new_tokens, token_note) =
            effective_tts_max_new_tokens(self.config.server_max_new_tokens, &text, backend);
        append_log_line(
            &mut self.synthesis_log,
            &format!("max_new_tokens：{effective_max_new_tokens}"),
        );
        if let Some(note) = token_note {
            append_log_line(&mut self.synthesis_log, &note);
        }
        append_log_line(
            &mut self.synthesis_log,
            &format!("Backend：{}", backend.as_str()),
        );
        append_log_line(
            &mut self.synthesis_log,
            &backend_device_line(backend, self.config.cuda_device, self.config.codec_cuda),
        );
        if backend == EngineBackend::RustPure && self.config.server_max_new_tokens > 4 {
            append_log_line(
                &mut self.synthesis_log,
                "警告：目前後端是 rust-pure CPU，tokens>4 會非常慢；CUDA 版請切到 ffi-cuda。",
            );
        }
        self.status_line = "正在準備原生 Rust 引擎…".to_string();
        let output_dir = self.config.output_dir.clone();
        let key = NativeRustEngineKey {
            transformer: pair.transformer.path.clone(),
            codec: pair.codec.path.clone(),
            workdir: self.config.server_workdir.clone(),
            backend: backend.as_str().to_string(),
            max_new_tokens: effective_max_new_tokens,
            cuda_device: self.config.cuda_device,
            codec_cuda: self.config.codec_cuda,
            vulkan_device: self.config.vulkan_device,
            codec_vulkan_device: self.config.codec_vulkan_device,
        };
        let engine_slot = self.native_engine_slot(key.clone());
        let tx = self.bg_tx.clone();
        thread::spawn(move || {
            let total_start = Instant::now();
            let result = (|| {
                std::fs::create_dir_all(&output_dir)?;
                send_debug(&tx, &format!("輸出目錄：{}", output_dir.display()));
                let mut engine_guard = engine_slot.lock().map_err(|_| {
                    fish_s2_infer::InferError::Message("原生 Rust 引擎鎖定失敗".into())
                })?;
                if engine_guard.is_none() {
                    send_status(
                        &tx,
                        "首次使用原生 Rust：正在載入 GGUF 與 tokenizer，之後同模型會快很多…",
                    );
                    send_debug(&tx, "載入：開始建立 RustPure InferenceEngine");
                    let load_start = Instant::now();
                    let mut engine_cfg =
                        EngineConfig::new(pair.transformer.path.clone(), pair.codec.path.clone())?;
                    engine_cfg.backend = backend;
                    engine_cfg.workdir = key.workdir.clone();
                    engine_cfg.generate_params.max_new_tokens = key.max_new_tokens;
                    engine_cfg.cuda_device = key.cuda_device;
                    engine_cfg.codec_cuda = key.codec_cuda;
                    engine_cfg.vulkan_device = key.vulkan_device;
                    engine_cfg.codec_vulkan_device = key.codec_vulkan_device;
                    send_debug(&tx, &format!("工作目錄：{}", engine_cfg.workdir.display()));
                    send_debug(
                        &tx,
                        &format!(
                            "後端：{}；{}",
                            engine_cfg.backend.as_str(),
                            backend_device_line(
                                engine_cfg.backend,
                                engine_cfg.cuda_device,
                                engine_cfg.codec_cuda
                            )
                        ),
                    );
                    send_debug(
                        &tx,
                        &format!("tokenizer：{}", engine_cfg.tokenizer_path.display()),
                    );
                    *engine_guard = Some(InferenceEngine::load(engine_cfg)?);
                    send_debug(
                        &tx,
                        &format!("載入：完成，用時 {}", format_elapsed(load_start.elapsed())),
                    );
                    send_status(&tx, "原生 Rust 引擎已載入，正在合成語音…");
                } else {
                    send_status(&tx, "正在使用已載入的原生 Rust 引擎合成語音…");
                    send_debug(&tx, "載入：重用已快取的 RustPure InferenceEngine");
                }
                let request = SynthesisRequest {
                    text,
                    reference_wav: voice.as_ref().map(|v| v.reference_wav.clone()),
                    reference_text: voice.as_ref().map(|v| v.reference_text.clone()),
                };
                if let Some(wav) = &request.reference_wav {
                    send_debug(&tx, &format!("Reference WAV：{}", wav.display()));
                } else {
                    send_debug(&tx, "Reference：未選擇 voice profile，使用預設提示");
                }
                let engine = engine_guard.as_ref().ok_or_else(|| {
                    fish_s2_infer::InferError::Message("原生 Rust 引擎尚未載入".into())
                })?;
                send_debug(&tx, "合成：開始 Slow-AR / Fast-AR / codec decode");
                let synth_start = Instant::now();
                let bytes = engine.synthesize_wav(&request)?;
                send_debug(
                    &tx,
                    &format!(
                        "合成：完成，用時 {}，WAV {} bytes",
                        format_elapsed(synth_start.elapsed()),
                        bytes.len()
                    ),
                );
                let filename = format!("tts_{}.wav", chrono::Utc::now().format("%Y%m%d_%H%M%S"));
                let path = output_dir.join(filename);
                send_debug(&tx, &format!("寫檔：{}", path.display()));
                std::fs::write(&path, &bytes)?;
                send_debug(
                    &tx,
                    &format!("總耗時：{}", format_elapsed(total_start.elapsed())),
                );
                Ok::<_, fish_s2_infer::InferError>((bytes, path))
            })()
            .map_err(|e| e.to_string());
            let _ = tx.send(BackgroundMsg::TtsDone(result));
        });
    }

    fn native_engine_slot(
        &mut self,
        key: NativeRustEngineKey,
    ) -> Arc<Mutex<Option<InferenceEngine>>> {
        if let Some(cache) = &self.native_rust_engine {
            if cache.key == key {
                return cache.engine.clone();
            }
        }
        let engine = Arc::new(Mutex::new(None));
        self.native_rust_engine = Some(NativeRustEngineCache {
            key,
            engine: engine.clone(),
        });
        engine
    }

    fn run_server_tts(&mut self) {
        if !self.server_running() {
            self.status_line = "請先啟動 Rust 推理伺服器".to_string();
            return;
        }
        let raw_text = self.script.trim();
        let sanitized_text = sanitize_tts_script(raw_text);
        if sanitized_text.is_empty() {
            self.status_line = "請輸入要合成的文字".to_string();
            return;
        }
        let style_prefix = tts_style_prefix(&self.config);
        let text = apply_tts_style(&sanitized_text, &style_prefix);
        self.busy = true;
        self.synthesis_log.clear();
        append_log_line(&mut self.synthesis_log, "開始：HTTP server /v1/tts 生成");
        append_log_line(
            &mut self.synthesis_log,
            &format!(
                "Endpoint：http://127.0.0.1:{}/v1/tts",
                self.config.server_port
            ),
        );
        append_log_line(
            &mut self.synthesis_log,
            &format!(
                "文字：{} bytes / {} chars",
                text.len(),
                text.chars().count()
            ),
        );
        if sanitized_text != raw_text {
            append_log_line(
                &mut self.synthesis_log,
                "文字清理：已移除 system/user/assistant 或 ChatML 標記，避免被念出來。",
            );
        }
        if !style_prefix.is_empty() {
            append_log_line(
                &mut self.synthesis_log,
                &format!("朗讀控制：{style_prefix}"),
            );
        }
        self.status_line = "正在合成語音…".to_string();
        #[cfg(not(feature = "http-client"))]
        {
            self.status_line =
                "此 build 未編入 HTTP client；請使用原生 Rust 直接生成，或以 --features http-client 重新編譯。"
                    .to_string();
            append_log_line(
                &mut self.synthesis_log,
                "HTTP：未編入 http-client feature，已略過 /v1/tts 呼叫。",
            );
        }
        #[cfg(feature = "http-client")]
        {
            let port = self.config.server_port;
            let output_dir = self.config.output_dir.clone();
            let tx = self.bg_tx.clone();
            thread::spawn(move || {
                let start = Instant::now();
                let client = TtsClient::new(port);
                let req = TtsRequest {
                    text,
                    format: "wav".to_string(),
                };
                let filename = format!("tts_{}.wav", chrono::Utc::now().format("%Y%m%d_%H%M%S"));
                let path = output_dir.join(filename);
                send_debug(&tx, &format!("HTTP：POST /v1/tts，輸出 {}", path.display()));
                let result = client
                    .synthesize_to_file(&req, path)
                    .map(|r| {
                        let path = r.saved_path.unwrap_or_default();
                        send_debug(
                            &tx,
                            &format!(
                                "HTTP：完成，用時 {}，WAV {} bytes",
                                format_elapsed(start.elapsed()),
                                r.wav_bytes.len()
                            ),
                        );
                        (r.wav_bytes, path)
                    })
                    .map_err(|e| e.to_string());
                let _ = tx.send(BackgroundMsg::TtsDone(result));
            });
        }
    }

    fn run_convert(&mut self) {
        let plan = ConvertPlan {
            checkpoint_dir: self.config.convert_checkpoint_dir.clone(),
            codec_path: checkpoint_codec_path(&self.config.convert_checkpoint_dir),
            output_path: self
                .config
                .models_dir
                .join(format!("s2-pro-export-{}.gguf", self.convert_dtype)),
            out_dtype: self.convert_dtype.clone(),
            python_exe: self.config.python_exe.clone(),
            script_path: resolve_export_script(&self.config.convert_script),
        };
        self.busy = true;
        self.convert_log = plan.command_preview();
        self.status_line = "正在轉換為 GGUF（可能需要數十分鐘）…".to_string();
        let tx = self.bg_tx.clone();
        thread::spawn(move || {
            let result = plan.run_blocking().map_err(|e| e.to_string());
            let _ = tx.send(BackgroundMsg::ConvertDone(result));
        });
    }

    fn inspect_gguf_async(&mut self, path: PathBuf) {
        self.selected_gguf = Some(path.clone());
        let tx = self.bg_tx.clone();
        thread::spawn(move || {
            let result = GgufSummary::inspect(&path).map_err(|e| e.to_string());
            let _ = tx.send(BackgroundMsg::GgufInspect(result));
        });
    }
}

impl eframe::App for FishS2App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_background();
        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(200));
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(egui::RichText::new("Fish S2 Pro Studio").strong());
                ui.separator();
                ui.label(
                    egui::RichText::new("fishaudio/s2-pro · GGUF · 本地語音")
                        .color(egui::Color32::from_rgb(175, 185, 188)),
                );
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Generate, "語音生成");
                ui.selectable_value(&mut self.tab, Tab::Clone, "聲音複製");
                ui.selectable_value(&mut self.tab, Tab::Edit, "腳本編輯");
                ui.selectable_value(&mut self.tab, Tab::Models, "模型 / GGUF");
                ui.selectable_value(&mut self.tab, Tab::Convert, "轉換 GGUF");
                ui.selectable_value(&mut self.tab, Tab::Server, "伺服器");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let running = self.server_running();
                    let (label, color) = if running {
                        ("Server online", egui::Color32::from_rgb(114, 205, 155))
                    } else {
                        ("Server idle", egui::Color32::from_rgb(166, 173, 176))
                    };
                    ui.label(
                        egui::RichText::new(format!("{label} · :{}", self.config.server_port))
                            .color(color),
                    );
                });
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status_line);
                if self.busy {
                    ui.spinner();
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Generate => self.ui_generate(ui),
            Tab::Clone => self.ui_clone(ui),
            Tab::Edit => self.ui_edit(ui),
            Tab::Models => self.ui_models(ui),
            Tab::Convert => self.ui_convert(ui),
            Tab::Server => self.ui_server(ui),
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.rust_server = None;
        self.persist();
    }
}

impl FishS2App {
    fn ui_generate(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.heading("語音生成");
            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "{} · {}",
                    self.config.server_backend,
                    backend_device_line(
                        EngineBackend::parse(&self.config.server_backend)
                            .unwrap_or(EngineBackend::RustPure),
                        self.config.cuda_device,
                        self.config.codec_cuda
                    )
                ))
                .color(egui::Color32::from_rgb(170, 183, 186)),
            );
        });

        ui.group(|ui| {
            ui.horizontal_wrapped(|ui| {
                let engine_changed = ui
                    .checkbox(&mut self.config.use_rust_engine, "原生 Rust 直接生成")
                    .changed();
                ui.label("生成 token");
                let token_changed = ui
                    .add(
                        egui::DragValue::new(&mut self.config.server_max_new_tokens)
                            .range(1..=2048),
                    )
                    .changed();
                if engine_changed || token_changed {
                    self.native_rust_engine = None;
                    self.persist();
                }
                ui.separator();
                if ui
                    .checkbox(&mut self.config.codec_cuda, "Codec CUDA 診斷")
                    .changed()
                {
                    self.native_rust_engine = None;
                    self.persist();
                }
                if ui.button("儲存設定").clicked() {
                    self.persist();
                    self.status_line = "設定已儲存".to_string();
                }
            });
            ui.horizontal_wrapped(|ui| {
                let mut changed = false;
                changed |= preset_combo(
                    ui,
                    "tts_role",
                    "角色",
                    &mut self.config.tts_role,
                    TTS_ROLE_OPTIONS,
                );
                changed |= preset_combo(
                    ui,
                    "tts_tone",
                    "語調",
                    &mut self.config.tts_tone,
                    TTS_TONE_OPTIONS,
                );
                changed |= preset_combo(
                    ui,
                    "tts_pace",
                    "速度",
                    &mut self.config.tts_pace,
                    TTS_PACE_OPTIONS,
                );
                changed |= preset_combo(
                    ui,
                    "tts_pitch",
                    "音高",
                    &mut self.config.tts_pitch,
                    TTS_PITCH_OPTIONS,
                );
                changed |= preset_combo(
                    ui,
                    "tts_energy",
                    "能量",
                    &mut self.config.tts_energy,
                    TTS_ENERGY_OPTIONS,
                );
                if changed {
                    self.persist();
                }
            });
        });
        if self.config.server_max_new_tokens <= 1 {
            ui.colored_label(
                egui::Color32::YELLOW,
                "目前 token=1 只會產生極短 smoke WAV，通常聽起來像沒有聲音。",
            );
        } else if self.config.server_max_new_tokens < MIN_AUDIBLE_TTS_TOKENS {
            ui.colored_label(
                egui::Color32::YELLOW,
                format!(
                    "目前 token={} 偏短；按生成語音時，CUDA/FFI 會依文字自動提高，避免輸出太短。",
                    self.config.server_max_new_tokens
                ),
            );
        }
        if self.config.codec_cuda {
            ui.colored_label(
                egui::Color32::from_rgb(255, 202, 120),
                "Codec CUDA 診斷已請求，但一般生成會被 C++ guard 改用 CPU codec backend，避免 GGML CUDA IM2COL crash。",
            );
        }
        ui.add(
            egui::TextEdit::multiline(&mut self.script)
                .desired_width(f32::INFINITY)
                .desired_rows(12)
                .hint_text("例如：[excited] 大家好！[pause] 歡迎使用 Fish S2 Pro。"),
        );

        ui.horizontal(|ui| {
            ui.menu_button("插入常用標籤", |ui| {
                for tag in CONTROL_TAGS {
                    if ui.button(format!("{}  {}", tag.label, tag.token)).clicked() {
                        let (t, c) = fish_s2_core::tags::insert_tag_at_cursor(
                            &self.script,
                            self.script.len(),
                            tag.token,
                        );
                        self.script = t;
                        self.script_cursor = c;
                        ui.close_menu();
                    }
                }
            });
            if ui.button("生成語音").clicked() && !self.busy {
                self.persist();
                self.run_tts();
            }
            if ui
                .add_enabled(self.last_wav.is_some(), egui::Button::new("播放"))
                .clicked()
            {
                if let (Some(bytes), Some(player)) = (&self.last_wav, &self.audio) {
                    let _ = player.play_wav_bytes(bytes);
                }
            }
            if ui
                .add_enabled(
                    self.last_wav_path.is_some(),
                    egui::Button::new("開啟輸出資料夾"),
                )
                .clicked()
            {
                if let Some(path) = &self.last_wav_path {
                    open_in_explorer(path);
                }
            }
        });

        if let Some(path) = &self.last_wav_path {
            ui.label(format!("最近輸出：{}", path.display()));
        }

        ui.separator();
        ui.collapsing("合成除錯紀錄", |ui| {
            ui.horizontal(|ui| {
                if ui.button("清除紀錄").clicked() {
                    self.synthesis_log.clear();
                }
                if let Some(path) = &self.last_wav_path {
                    ui.label(format!("輸出：{}", path.display()));
                }
            });
            ui.add(
                egui::TextEdit::multiline(&mut self.synthesis_log)
                    .desired_width(f32::INFINITY)
                    .desired_rows(12)
                    .font(egui::TextStyle::Monospace),
            );
        });

        ui.separator();
        ui.collapsing("目前聲音設定", |ui| {
            if let Some(v) = self.config.active_voice() {
                ui.label(format!("名稱：{}", v.name));
                ui.label(format!("參考音：{}", v.reference_wav.display()));
                ui.label(format!("參考文本：{}", v.reference_text));
            } else {
                ui.label("未選擇克隆聲音（使用預設音色）。可在「聲音複製」分頁建立。");
            }
            if let Some(pair) = self.active_pair() {
                ui.label(format!("模型：{}", pair.label));
            } else {
                ui.colored_label(egui::Color32::YELLOW, "尚未選擇 GGUF 模型對");
            }
        });
    }

    fn ui_clone(&mut self, ui: &mut egui::Ui) {
        ui.label("聲音複製需要 3–10 秒清晰參考音訊 + 對應文本。啟動伺服器時會寫入 reference.wav / reference.txt。");
        ui.horizontal(|ui| {
            ui.label("名稱");
            ui.text_edit_singleline(&mut self.clone_name);
        });
        ui.horizontal(|ui| {
            ui.label("參考 WAV");
            ui.label(
                self.clone_ref_wav
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "（未選擇）".to_string()),
            );
            if ui.button("瀏覽…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("WAV", &["wav"])
                    .pick_file()
                {
                    self.clone_ref_wav = path;
                }
            }
        });
        ui.label("參考文本（與音訊內容一致）");
        ui.add(egui::TextEdit::multiline(&mut self.clone_ref_text).desired_rows(4));

        ui.horizontal(|ui| {
            if ui.button("儲存為聲音設定檔").clicked() {
                if self.clone_ref_wav.exists() && !self.clone_ref_text.trim().is_empty() {
                    let profile = VoiceProfile::new(
                        self.clone_name.clone(),
                        self.clone_ref_wav.clone(),
                        self.clone_ref_text.clone(),
                    );
                    let id = profile.id;
                    self.config.voices.push(profile);
                    self.config.active_voice_id = Some(id);
                    self.persist();
                    self.status_line = "已儲存聲音設定。重新啟動伺服器以套用克隆。".to_string();
                } else {
                    self.status_line = "請提供有效的 WAV 與參考文本".to_string();
                }
            }
            if ui.button("套用到伺服器工作目錄").clicked()
                && self.clone_ref_wav.exists()
                && copy_reference_files(
                    &self.config.server_workdir,
                    &self.clone_ref_wav,
                    &self.clone_ref_text,
                )
                .is_ok()
            {
                self.status_line =
                    "已寫入 reference 檔案。若伺服器正在運行，請重新啟動。".to_string();
            }
        });

        ui.separator();
        ui.heading("已儲存的聲音");
        egui::ScrollArea::vertical()
            .max_height(220.0)
            .show(ui, |ui| {
                let mut remove_id: Option<Uuid> = None;
                let mut activate: Option<Uuid> = None;
                for voice in &self.config.voices {
                    ui.horizontal(|ui| {
                        let active = self.config.active_voice_id == Some(voice.id);
                        if ui.selectable_label(active, &voice.name).clicked() {
                            activate = Some(voice.id);
                        }
                        ui.label(voice.reference_wav.file_name().unwrap().to_string_lossy());
                        if ui.small_button("刪除").clicked() {
                            remove_id = Some(voice.id);
                        }
                    });
                }
                if let Some(id) = activate {
                    self.config.active_voice_id = Some(id);
                    self.persist();
                }
                if let Some(id) = remove_id {
                    self.config.voices.retain(|v| v.id != id);
                    if self.config.active_voice_id == Some(id) {
                        self.config.active_voice_id = self.config.voices.first().map(|v| v.id);
                    }
                    self.persist();
                }
            });
    }

    fn ui_edit(&mut self, ui: &mut egui::Ui) {
        ui.label("腳本編輯器：分段管理長文本，避免短句尾端 artifact（建議每段 ≥ 90 字）。");
        ui.horizontal(|ui| {
            for tag in CONTROL_TAGS.iter().take(8) {
                if ui.small_button(tag.label).clicked() {
                    let (t, c) = fish_s2_core::tags::insert_tag_at_cursor(
                        &self.script,
                        self.script_cursor,
                        tag.token,
                    );
                    self.script = t;
                    self.script_cursor = c;
                }
            }
        });

        let response = ui.add(
            egui::TextEdit::multiline(&mut self.script)
                .desired_width(f32::INFINITY)
                .desired_rows(18)
                .cursor_at_end(false),
        );
        if response.changed() {
            self.script_cursor = self.script.len();
        }

        ui.horizontal(|ui| {
            if ui.button("依空行分段預覽").clicked() {
                let parts: Vec<_> = self
                    .script
                    .split("\n\n")
                    .filter(|p| !p.trim().is_empty())
                    .collect();
                self.status_line = format!("共 {} 段（合成時請逐段生成或合併腳本）", parts.len());
            }
            if ui.button("同步到「語音生成」").clicked() {
                self.tab = Tab::Generate;
            }
        });
    }

    fn ui_models(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("模型目錄");
            let mut dir = self.config.models_dir.display().to_string();
            ui.text_edit_singleline(&mut dir);
            if ui.button("瀏覽…").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    self.config.models_dir = path;
                    self.persist();
                    self.rescan_models_async();
                }
            }
            if ui.button("重新掃描").clicked() {
                self.rescan_models_async();
            }
        });

        ui.columns(2, |columns| {
            columns[0].heading("模型對（推理）");
            egui::ScrollArea::vertical().show(&mut columns[0], |ui| {
                if self.scanned.pairs.is_empty() {
                    ui.label("未找到配對的 transformer + codec GGUF。\n可從 Hugging Face 下載 mach9243/s2-pro-gguf。");
                }
                let mut pick_pair: Option<String> = None;
                for pair in &self.scanned.pairs {
                    let selected = self.config.active_model_pair_id.as_deref() == Some(pair.id.as_str());
                    if ui.selectable_label(selected, &pair.label).clicked() {
                        pick_pair = Some(pair.id.clone());
                    }
                }
                if let Some(id) = pick_pair {
                    if self.config.active_model_pair_id.as_deref() != Some(id.as_str()) {
                        self.native_rust_engine = None;
                    }
                    self.config.active_model_pair_id = Some(id);
                    self.persist();
                }
            });

            columns[1].heading("GGUF 檔案");
            egui::ScrollArea::vertical().show(&mut columns[1], |ui| {
                let mut inspect_path: Option<PathBuf> = None;
                for entry in &self.scanned.entries {
                    let name = entry.path.file_name().unwrap().to_string_lossy();
                    if ui.button(name.as_ref()).clicked() {
                        inspect_path = Some(entry.path.clone());
                    }
                }
                if let Some(path) = inspect_path {
                    self.inspect_gguf_async(path);
                }
            });
        });

        if let Some(summary) = &self.gguf_detail {
            ui.separator();
            ui.heading(summary.display_name());
            ui.label(format!(
                "大小 {} · tensors {} · arch {:?}",
                summary.size_human(),
                summary.tensor_count,
                summary.architecture
            ));
            egui::ScrollArea::vertical()
                .max_height(160.0)
                .show(ui, |ui| {
                    for (k, v) in summary.metadata.iter().take(40) {
                        ui.label(format!("{k} = {v}"));
                    }
                });
        }
    }

    fn ui_convert(&mut self, ui: &mut egui::Ui) {
        ui.label("將 fishaudio/s2-pro 的 Safetensors 檢查點匯出為 GGUF（需 Python + PyTorch + 官方 quantize 腳本）。");
        ui.hyperlink_to(
            "模型：fishaudio/s2-pro",
            "https://huggingface.co/fishaudio/s2-pro",
        );
        ui.hyperlink_to(
            "預量化 GGUF：mach9243/s2-pro-gguf",
            "https://huggingface.co/mach9243/s2-pro-gguf",
        );

        ui.horizontal(|ui| {
            ui.label("Checkpoint 目錄");
            let mut s = self.config.convert_checkpoint_dir.display().to_string();
            ui.text_edit_singleline(&mut s);
            if ui.button("…").clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                    self.config.convert_checkpoint_dir = p;
                    self.persist();
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("匯出腳本");
            let mut s = self.config.convert_script.display().to_string();
            ui.text_edit_singleline(&mut s);
            if ui.button("…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("Python", &["py"])
                    .pick_file()
                {
                    self.config.convert_script = p;
                    self.persist();
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Python");
            ui.text_edit_singleline(&mut self.config.python_exe);
            ui.label("精度");
            egui::ComboBox::from_id_salt("dtype")
                .selected_text(&self.convert_dtype)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.convert_dtype, "f16".to_string(), "f16");
                    ui.selectable_value(&mut self.convert_dtype, "f32".to_string(), "f32");
                });
        });

        let plan = ConvertPlan {
            checkpoint_dir: self.config.convert_checkpoint_dir.clone(),
            codec_path: checkpoint_codec_path(&self.config.convert_checkpoint_dir),
            output_path: self
                .config
                .models_dir
                .join(format!("s2-pro-export-{}.gguf", self.convert_dtype)),
            out_dtype: self.convert_dtype.clone(),
            python_exe: self.config.python_exe.clone(),
            script_path: resolve_export_script(&self.config.convert_script),
        };
        ui.monospace(plan.command_preview());

        if ui
            .add_enabled(!self.busy, egui::Button::new("開始轉換"))
            .clicked()
        {
            self.run_convert();
        }

        if !self.convert_log.is_empty() {
            ui.separator();
            ui.label("轉換日誌");
            ui.add(
                egui::TextEdit::multiline(&mut self.convert_log)
                    .desired_width(f32::INFINITY)
                    .desired_rows(10),
            );
        }
    }

    fn ui_server(&mut self, ui: &mut egui::Ui) {
        ui.label("內建 Rust 推理引擎（fish_s2_infer），API 相容 /v1/tts。");
        ui.label("GGUF 預設目錄：專案 models/（可放 transformer-only + codec-only 配對）。");
        ui.hyperlink_to(
            "建置 native ggml 後端",
            "https://github.com/mach92432/s2.cpp",
        );

        ui.horizontal(|ui| {
            ui.label("工作目錄");
            ui.label(self.config.server_workdir.display().to_string());
        });
        ui.horizontal(|ui| {
            ui.label("連接埠");
            ui.add(egui::DragValue::new(&mut self.config.server_port).range(1024..=65535));
            ui.label("後端");
            egui::ComboBox::from_id_salt("server_backend")
                .selected_text(&self.config.server_backend)
                .show_ui(ui, |ui| {
                    let backend_before = self.config.server_backend.clone();
                    ui.selectable_value(
                        &mut self.config.server_backend,
                        "rust-pure".to_string(),
                        "rust-pure",
                    );
                    ui.selectable_value(&mut self.config.server_backend, "ffi".to_string(), "ffi");
                    ui.selectable_value(
                        &mut self.config.server_backend,
                        "ffi-cuda".to_string(),
                        "ffi-cuda",
                    );
                    #[cfg(feature = "legacy-s2-exe")]
                    ui.selectable_value(
                        &mut self.config.server_backend,
                        "subprocess".to_string(),
                        "subprocess",
                    );
                    if self.config.server_backend != backend_before {
                        self.native_rust_engine = None;
                    }
                });
            ui.label("生成幀數");
            ui.add(egui::DragValue::new(&mut self.config.server_max_new_tokens).range(1..=2048));
        });
        ui.horizontal(|ui| {
            ui.label("CUDA 裝置");
            if ui
                .add(egui::DragValue::new(&mut self.config.cuda_device).range(0..=16))
                .changed()
            {
                self.native_rust_engine = None;
            }
            ui.label("Vulkan 裝置");
            if ui
                .add(egui::DragValue::new(&mut self.config.vulkan_device))
                .changed()
            {
                self.native_rust_engine = None;
            }
            ui.label("Codec Vulkan");
            if ui
                .add(egui::DragValue::new(&mut self.config.codec_vulkan_device).range(-1..=8))
                .changed()
            {
                self.native_rust_engine = None;
            }
            if ui
                .checkbox(&mut self.config.codec_cuda, "Codec CUDA 診斷")
                .changed()
            {
                self.native_rust_engine = None;
            }
        });
        ui.horizontal_wrapped(|ui| {
            let backend = EngineBackend::parse(&self.config.server_backend)
                .unwrap_or(EngineBackend::RustPure);
            ui.label(
                egui::RichText::new(backend_device_line(
                    backend,
                    self.config.cuda_device,
                    self.config.codec_cuda,
                ))
                .color(egui::Color32::from_rgb(170, 183, 186)),
            );
            if self.config.codec_cuda {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 202, 120),
                    "目前 codec CUDA 會被 guard 為 CPU fallback；只有設定 FISH_S2_CODEC_CUDA_UNSAFE=1 才會強制進入不穩定路徑。",
                );
            }
        });

        ui.horizontal(|ui| {
            if ui.button("啟動伺服器").clicked() {
                self.start_server();
                self.persist();
            }
            if ui.button("停止").clicked() {
                self.rust_server = None;
                self.status_line = "伺服器已停止".to_string();
            }
            if ui.button("測試連線").clicked() {
                #[cfg(feature = "http-client")]
                {
                    let client = TtsClient::new(self.config.server_port);
                    self.status_line = if client.health_check() {
                        "HTTP 端點可連線".to_string()
                    } else {
                        "無法連線（伺服器可能仍在載入模型）".to_string()
                    };
                }
                #[cfg(not(feature = "http-client"))]
                {
                    self.status_line =
                        "此 build 未編入 HTTP client；測試連線需要 --features http-client。"
                            .to_string();
                }
            }
        });

        ui.separator();
        ui.add(
            egui::TextEdit::multiline(&mut self.server_log)
                .desired_width(f32::INFINITY)
                .desired_rows(12),
        );
    }
}

fn open_in_explorer(path: &std::path::Path) {
    #[cfg(windows)]
    {
        let arg = if path.is_file() {
            format!("/select,{}", path.display())
        } else {
            path.display().to_string()
        };
        let _ = std::process::Command::new("explorer").arg(arg).spawn();
    }
    #[cfg(not(windows))]
    {
        let _ = std::process::Command::new("xdg-open")
            .arg(path.parent().unwrap_or(path))
            .spawn();
    }
}

fn model_scan_status(
    models_dir: &std::path::Path,
    scanned: &ScannedModels,
    selected_label: Option<&str>,
) -> String {
    if let Some(label) = selected_label {
        return format!(
            "找到 {} 個 GGUF、{} 組可用模型對；已自動選用：{}",
            scanned.entries.len(),
            scanned.pairs.len(),
            label
        );
    }
    missing_model_pair_message(models_dir, scanned)
}

fn missing_model_pair_message(models_dir: &std::path::Path, scanned: &ScannedModels) -> String {
    if scanned.entries.is_empty() {
        return format!(
            "找不到 GGUF 模型。請將 transformer-only + codec-only GGUF 放入 {}，或執行 scripts\\download_models.ps1 -IncludeGguf -Quant f16。",
            models_dir.display()
        );
    }
    format!(
        "已找到 {} 個 GGUF，但沒有可用的 transformer + codec 配對。請確認檔名包含 transformer-only 與 codec-only，或重新下載 s2-pro GGUF pair。",
        scanned.entries.len()
    )
}

fn send_status(tx: &Sender<BackgroundMsg>, line: impl Into<String>) {
    let _ = tx.send(BackgroundMsg::Status(line.into()));
}

fn send_debug(tx: &Sender<BackgroundMsg>, line: &str) {
    let _ = tx.send(BackgroundMsg::SynthesisLog(line.to_string()));
}

fn append_log_line(log: &mut String, line: &str) {
    if !log.is_empty() {
        log.push('\n');
    }
    log.push_str(&timestamped_log_line(line));
}

fn timestamped_log_line(line: &str) -> String {
    format!("[{}] {line}", chrono::Local::now().format("%H:%M:%S"))
}

fn format_elapsed(duration: Duration) -> String {
    if duration.as_secs() >= 1 {
        format!("{:.2}s", duration.as_secs_f64())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

#[derive(Clone, Copy)]
struct TtsPreset {
    id: &'static str,
    label: &'static str,
    tag: &'static str,
}

const TTS_ROLE_OPTIONS: &[TtsPreset] = &[
    TtsPreset {
        id: "default",
        label: "預設",
        tag: "",
    },
    TtsPreset {
        id: "female",
        label: "女聲",
        tag: "[female voice]",
    },
    TtsPreset {
        id: "male",
        label: "男聲",
        tag: "[male voice]",
    },
    TtsPreset {
        id: "narrator",
        label: "旁白",
        tag: "[narrator]",
    },
    TtsPreset {
        id: "young",
        label: "年輕",
        tag: "[young voice]",
    },
    TtsPreset {
        id: "mature",
        label: "成熟",
        tag: "[mature voice]",
    },
];

const TTS_TONE_OPTIONS: &[TtsPreset] = &[
    TtsPreset {
        id: "natural",
        label: "自然",
        tag: "",
    },
    TtsPreset {
        id: "warm",
        label: "溫柔",
        tag: "[warm]",
    },
    TtsPreset {
        id: "calm",
        label: "平靜",
        tag: "[calm]",
    },
    TtsPreset {
        id: "excited",
        label: "興奮",
        tag: "[excited]",
    },
    TtsPreset {
        id: "serious",
        label: "嚴肅",
        tag: "[serious]",
    },
    TtsPreset {
        id: "whisper",
        label: "耳語",
        tag: "[whisper]",
    },
];

const TTS_PACE_OPTIONS: &[TtsPreset] = &[
    TtsPreset {
        id: "normal",
        label: "正常",
        tag: "",
    },
    TtsPreset {
        id: "slow",
        label: "慢",
        tag: "[slow]",
    },
    TtsPreset {
        id: "fast",
        label: "快",
        tag: "[fast]",
    },
];

const TTS_PITCH_OPTIONS: &[TtsPreset] = &[
    TtsPreset {
        id: "normal",
        label: "正常",
        tag: "",
    },
    TtsPreset {
        id: "low",
        label: "低",
        tag: "[low voice]",
    },
    TtsPreset {
        id: "high",
        label: "高",
        tag: "[high pitch]",
    },
];

const TTS_ENERGY_OPTIONS: &[TtsPreset] = &[
    TtsPreset {
        id: "normal",
        label: "正常",
        tag: "",
    },
    TtsPreset {
        id: "soft",
        label: "柔和",
        tag: "[volume down]",
    },
    TtsPreset {
        id: "strong",
        label: "有力",
        tag: "[volume up] [emphasis]",
    },
];

fn preset_combo(
    ui: &mut egui::Ui,
    id: &'static str,
    label: &str,
    selected: &mut String,
    options: &[TtsPreset],
) -> bool {
    let before = selected.clone();
    ui.label(label);
    egui::ComboBox::from_id_salt(id)
        .selected_text(selected_preset_label(selected, options))
        .show_ui(ui, |ui| {
            for option in options {
                ui.selectable_value(selected, option.id.to_string(), option.label);
            }
        });
    *selected != before
}

fn selected_preset_label(selected: &str, options: &[TtsPreset]) -> &'static str {
    options
        .iter()
        .find(|option| option.id == selected)
        .map_or(options[0].label, |option| option.label)
}

fn tts_style_prefix(config: &AppConfig) -> String {
    [
        preset_tag(&config.tts_role, TTS_ROLE_OPTIONS),
        preset_tag(&config.tts_tone, TTS_TONE_OPTIONS),
        preset_tag(&config.tts_pace, TTS_PACE_OPTIONS),
        preset_tag(&config.tts_pitch, TTS_PITCH_OPTIONS),
        preset_tag(&config.tts_energy, TTS_ENERGY_OPTIONS),
    ]
    .into_iter()
    .filter(|tag| !tag.is_empty())
    .collect::<Vec<_>>()
    .join(" ")
}

fn preset_tag(selected: &str, options: &[TtsPreset]) -> &'static str {
    options
        .iter()
        .find(|option| option.id == selected)
        .map_or("", |option| option.tag)
}

fn sanitize_tts_script(text: &str) -> String {
    let without_specials = text
        .replace("<|im_start|>system", " ")
        .replace("<|im_start|>user", " ")
        .replace("<|im_start|>assistant", " ")
        .replace("<|im_start|>", " ")
        .replace("<|im_end|>", " ")
        .replace("<|voice|>", " ");
    let mut out = Vec::new();
    for line in without_specials.lines() {
        let trimmed = line.trim();
        if is_system_role_line(trimmed) {
            continue;
        }
        let line = strip_chat_role_prefix(trimmed);
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if matches!(lower.as_str(), "system" | "user" | "assistant") {
            continue;
        }
        out.push(line.to_string());
    }
    out.join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn strip_chat_role_prefix(line: &str) -> &str {
    let mut rest = line;
    loop {
        let trimmed = rest.trim_start();
        let lower = trimmed.to_ascii_lowercase();
        let Some(prefix_len) = chat_role_prefix_len(&lower) else {
            return trimmed;
        };
        rest = &trimmed[prefix_len..];
    }
}

fn chat_role_prefix_len(lower: &str) -> Option<usize> {
    for prefix in [
        "system:",
        "system：",
        "user:",
        "user：",
        "assistant:",
        "assistant：",
        "系統:",
        "系統：",
        "使用者:",
        "使用者：",
        "助理:",
        "助理：",
    ] {
        if lower.starts_with(prefix) {
            return Some(prefix.len());
        }
    }
    None
}

fn is_system_role_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower == "system"
        || lower.starts_with("system:")
        || lower.starts_with("system：")
        || line.starts_with("系統:")
        || line.starts_with("系統：")
}

fn apply_tts_style(text: &str, style_prefix: &str) -> String {
    if style_prefix.is_empty() {
        text.to_string()
    } else {
        format!("{style_prefix} {text}")
    }
}

fn backend_device_line(backend: EngineBackend, cuda_device: i32, codec_cuda: bool) -> String {
    if backend.uses_cuda() {
        if codec_cuda {
            format!(
                "CUDA：device {cuda_device}（Transformer 使用 GGML CUDA；codec CUDA 已 guard 為 CPU fallback）"
            )
        } else {
            format!(
                "CUDA：device {cuda_device}（Transformer 使用 GGML CUDA；codec 使用 CPU fallback，避開 CUDA IM2COL）"
            )
        }
    } else {
        "CUDA：未使用".to_string()
    }
}

const MIN_AUDIBLE_TTS_TOKENS: u32 = 128;
const MAX_AUTO_TTS_TOKENS: u32 = 1024;

fn effective_tts_max_new_tokens(
    configured: u32,
    text: &str,
    backend: EngineBackend,
) -> (u32, Option<String>) {
    if !backend.is_ffi() {
        return (configured, None);
    }
    let recommended = recommended_tts_max_new_tokens(text);
    if configured >= recommended {
        return (configured, None);
    }
    (
        recommended,
        Some(format!(
            "提示：設定的 max_new_tokens={configured} 對這段文字偏短，已自動提高到 {recommended}，避免產生幾乎無聲的短 WAV。"
        )),
    )
}

fn recommended_tts_max_new_tokens(text: &str) -> u32 {
    let content_chars = text.chars().filter(|c| !c.is_whitespace()).count() as u32;
    if content_chars == 0 {
        return MIN_AUDIBLE_TTS_TOKENS;
    }
    content_chars
        .saturating_mul(4)
        .clamp(MIN_AUDIBLE_TTS_TOKENS, MAX_AUTO_TTS_TOKENS)
}

fn text_preview(text: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 80;
    let mut preview: String = text.chars().take(MAX_PREVIEW_CHARS).collect();
    if text.chars().count() > MAX_PREVIEW_CHARS {
        preview.push_str("...");
    }
    preview.replace(['\r', '\n', '\t'], " ")
}

fn maybe_promote_cuda_backend(config: &mut AppConfig) -> Option<String> {
    if !fish_s2_infer::cpp_engine_linked() {
        return None;
    }
    if config.server_backend != "rust-pure" {
        return None;
    }
    config.server_backend = "ffi-cuda".to_string();
    Some("已偵測 CUDA/cpp-engine 版，後端自動切換為 ffi-cuda".to_string())
}

fn wav_warning(analysis: &WavAnalysis, max_new_tokens: u32) -> Option<String> {
    if analysis.duration_secs < 0.5 {
        return Some(format!(
            "生成的 WAV 只有 {:.3}s，太短所以幾乎聽不到。max_new_tokens={max_new_tokens} 只適合 smoke/debug，請在「生成 token」調高後再試。",
            analysis.duration_secs
        ));
    }
    if analysis.rms < 0.001 || analysis.peak < 0.002 {
        return Some(format!(
            "生成的 WAV 音量接近靜音：RMS {:.6}, peak {:.6}。請檢查 prompt/reference 或提高生成 token 後再試。",
            analysis.rms, analysis.peak
        ));
    }
    None
}

fn analyze_wav_bytes(bytes: &[u8]) -> Option<WavAnalysis> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }

    let mut offset = 12usize;
    let mut audio_format = None;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut data_range = None;

    while offset.checked_add(8)? <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?) as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start.checked_add(size)?;
        if chunk_end > bytes.len() {
            return None;
        }

        match id {
            b"fmt " if size >= 16 => {
                audio_format = Some(u16::from_le_bytes(
                    bytes[chunk_start..chunk_start + 2].try_into().ok()?,
                ));
                channels = Some(u16::from_le_bytes(
                    bytes[chunk_start + 2..chunk_start + 4].try_into().ok()?,
                ));
                sample_rate = Some(u32::from_le_bytes(
                    bytes[chunk_start + 4..chunk_start + 8].try_into().ok()?,
                ));
                bits_per_sample = Some(u16::from_le_bytes(
                    bytes[chunk_start + 14..chunk_start + 16].try_into().ok()?,
                ));
            }
            b"data" => {
                data_range = Some(chunk_start..chunk_end);
            }
            _ => {}
        }

        offset = chunk_end + (size % 2);
    }

    if audio_format? != 1 || bits_per_sample? != 16 {
        return None;
    }
    let channels = channels?;
    let sample_rate = sample_rate?;
    if channels == 0 || sample_rate == 0 {
        return None;
    }

    let data = &bytes[data_range?];
    let sample_width = 2usize;
    let frame_width = sample_width.checked_mul(channels as usize)?;
    let frames = data.len() / frame_width;
    if frames == 0 {
        return Some(WavAnalysis {
            sample_rate,
            channels,
            duration_secs: 0.0,
            rms: 0.0,
            peak: 0.0,
        });
    }

    let mut square_sum = 0.0f64;
    let mut peak = 0.0f64;
    let mut sample_count = 0usize;
    for chunk in data.chunks_exact(2) {
        let value = i16::from_le_bytes([chunk[0], chunk[1]]) as f64 / 32768.0;
        square_sum += value * value;
        peak = peak.max(value.abs());
        sample_count += 1;
    }

    Some(WavAnalysis {
        sample_rate,
        channels,
        duration_secs: frames as f64 / sample_rate as f64,
        rms: (square_sum / sample_count as f64).sqrt(),
        peak,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyzes_pcm16_wav_duration_and_level() {
        let wav = test_wav(&[0, 16_384, -16_384, 0], 44_100, 1);
        let analysis = analyze_wav_bytes(&wav).unwrap();
        assert_eq!(analysis.sample_rate, 44_100);
        assert_eq!(analysis.channels, 1);
        assert!((analysis.duration_secs - (4.0 / 44_100.0)).abs() < 1e-9);
        assert!(analysis.rms > 0.3);
        assert!(analysis.peak > 0.49);
    }

    #[test]
    fn warns_when_wav_is_too_short() {
        let wav = test_wav(&[0; 2048], 44_100, 1);
        let analysis = analyze_wav_bytes(&wav).unwrap();
        let warning = wav_warning(&analysis, 1).unwrap();
        assert!(warning.contains("太短"));
        assert!(warning.contains("max_new_tokens=1"));
    }

    #[test]
    fn ffi_generation_raises_too_short_token_limit_for_text() {
        let (tokens, note) = effective_tts_max_new_tokens(
            20,
            "你好，這是使用 Fish Audio S2 Pro 生成的語音。",
            EngineBackend::FfiCuda,
        );
        assert_eq!(tokens, MIN_AUDIBLE_TTS_TOKENS);
        assert!(note.unwrap().contains("自動提高"));
    }

    #[test]
    fn rust_pure_keeps_configured_debug_token_limit() {
        let (tokens, note) = effective_tts_max_new_tokens(4, "短測試", EngineBackend::RustPure);
        assert_eq!(tokens, 4);
        assert!(note.is_none());
    }

    #[test]
    fn sanitizes_chat_template_text_before_tts() {
        let cleaned = sanitize_tts_script(
            "<|im_start|>system\nsystem: do not read this\nassistant: 你好，開始朗讀。\n<|im_end|>",
        );
        assert_eq!(cleaned, "你好，開始朗讀。");
        assert!(!cleaned.to_ascii_lowercase().contains("system"));
        assert!(!cleaned.to_ascii_lowercase().contains("assistant"));
    }

    #[test]
    fn applies_selected_tts_style_tags() {
        let config = AppConfig {
            tts_role: "female".to_string(),
            tts_tone: "calm".to_string(),
            tts_pitch: "low".to_string(),
            ..AppConfig::default()
        };
        let prefix = tts_style_prefix(&config);
        assert_eq!(prefix, "[female voice] [calm] [low voice]");
        assert_eq!(
            apply_tts_style("你好", &prefix),
            "[female voice] [calm] [low voice] 你好"
        );
    }

    fn test_wav(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
        let data_len = samples.len() * 2;
        let mut bytes = Vec::with_capacity(44 + data_len);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_len as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * channels as u32 * 2;
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        let block_align = channels * 2;
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }
}
