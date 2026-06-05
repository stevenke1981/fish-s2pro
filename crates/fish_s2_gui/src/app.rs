use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;
use fish_s2_core::{
    checkpoint_codec_path, copy_reference_files, resolve_export_script, validate_pair, AppConfig,
    ConvertPlan, GgufSummary, ModelPair, ScannedModels, TtsClient, TtsRequest, VoiceProfile,
    CONTROL_TAGS,
};
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
    max_new_tokens: u32,
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
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut config = AppConfig::load();
        let _ = config.ensure_dirs();
        let (bg_tx, bg_rx) = mpsc::channel();
        let scanned = ScannedModels::scan_dir(&config.models_dir, 4).unwrap_or_default();
        let selected_label = config
            .ensure_active_model_pair(&scanned.pairs)
            .map(|pair| pair.label.clone());
        let _ = config.save();
        let audio = AudioPlayer::new();
        let script = config.last_script.clone();
        let status_line =
            model_scan_status(&config.models_dir, &scanned, selected_label.as_deref());
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

        if let (Some(wav), Some(text)) = (&ref_wav, &ref_text) {
            let _ = copy_reference_files(&engine_cfg.workdir, wav, text);
        }

        match InferenceEngine::load(engine_cfg) {
            Ok(engine) => match fish_s2_infer::spawn_server(engine, self.config.server_port) {
                Ok(handle) => {
                    self.rust_server = Some(handle);
                    self.server_log = format!(
                        "Rust 推理引擎：http://127.0.0.1:{}\nTransformer: {}\nCodec: {}",
                        self.config.server_port,
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
            },
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
        let text = self.script.trim().to_string();
        if text.is_empty() {
            self.status_line = "請輸入要合成的文字".to_string();
            return;
        }

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
        append_log_line(
            &mut self.synthesis_log,
            &format!("max_new_tokens：{}", self.config.server_max_new_tokens),
        );
        self.status_line = "正在準備原生 Rust 引擎…".to_string();
        let output_dir = self.config.output_dir.clone();
        let key = NativeRustEngineKey {
            transformer: pair.transformer.path.clone(),
            codec: pair.codec.path.clone(),
            workdir: self.config.server_workdir.clone(),
            max_new_tokens: self.config.server_max_new_tokens,
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
                    engine_cfg.backend = EngineBackend::RustPure;
                    engine_cfg.workdir = key.workdir.clone();
                    engine_cfg.generate_params.max_new_tokens = key.max_new_tokens;
                    send_debug(&tx, &format!("工作目錄：{}", engine_cfg.workdir.display()));
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
        let text = self.script.trim().to_string();
        if text.is_empty() {
            self.status_line = "請輸入要合成的文字".to_string();
            return;
        }
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
        self.status_line = "正在合成語音…".to_string();
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
                ui.heading("Fish S2 Pro Studio");
                ui.separator();
                ui.label("fishaudio/s2-pro · GGUF · 本地語音");
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
                    let dot = if running { "🟢" } else { "⚪" };
                    ui.label(format!("{dot} :{}", self.config.server_port));
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
        ui.label("輸入文字並使用 [tag] 控制語氣（S2 Pro 支援 15000+ 種自然語言標籤）。");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.config.use_rust_engine, "原生 Rust 直接生成");
            ui.label("生成 token");
            let token_changed = ui
                .add(egui::DragValue::new(&mut self.config.server_max_new_tokens).range(1..=2048))
                .changed();
            if token_changed {
                self.native_rust_engine = None;
                self.persist();
            }
            if ui.button("儲存設定").clicked() {
                self.persist();
                self.status_line = "設定已儲存".to_string();
            }
        });
        if self.config.server_max_new_tokens <= 1 {
            ui.colored_label(
                egui::Color32::YELLOW,
                "目前 token=1 只會產生極短 smoke WAV，通常聽起來像沒有聲音。",
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
                    ui.selectable_value(
                        &mut self.config.server_backend,
                        "rust-pure".to_string(),
                        "rust-pure",
                    );
                    ui.selectable_value(&mut self.config.server_backend, "ffi".to_string(), "ffi");
                    #[cfg(feature = "legacy-s2-exe")]
                    ui.selectable_value(
                        &mut self.config.server_backend,
                        "subprocess".to_string(),
                        "subprocess",
                    );
                });
            ui.label("生成幀數");
            ui.add(egui::DragValue::new(&mut self.config.server_max_new_tokens).range(1..=2048));
        });
        ui.horizontal(|ui| {
            ui.label("Vulkan 裝置");
            ui.add(egui::DragValue::new(&mut self.config.vulkan_device));
            ui.label("Codec Vulkan");
            ui.add(egui::DragValue::new(&mut self.config.codec_vulkan_device).range(-1..=8));
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
                let client = TtsClient::new(self.config.server_port);
                self.status_line = if client.health_check() {
                    "HTTP 端點可連線".to_string()
                } else {
                    "無法連線（伺服器可能仍在載入模型）".to_string()
                };
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
