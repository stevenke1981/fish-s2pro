# Fish S2 Pro Rust

Fish Audio S2 Pro 的 Rust 原生推論與桌面工具。

目前 `rust-pure` 路徑已可直接從 GGUF runtime pair 合成 WAV，不需要
`s2.exe` 或連結 C++ library。已包含 tokenizer、Slow-AR、Fast-AR、RVQ
codec decode、reference audio prompt-code encode、HTTP `/v1/tts` server、
Windows GUI，以及對 `s2.cpp` 的 parity 驗證工具。

## 工作區

```text
crates/
  fish_s2_core    GGUF 工具、模型掃描、設定、TTS client
  fish_s2_infer   Rust 原生推論、server、parity dump binaries
  fish_s2_gui     egui/eframe 桌面 GUI
  fish_s2_parity  WAV/code/tensor 比對工具
```

## 模型檔案

把 runtime assets 放在 `models/`：

```text
models/
  tokenizer.json
  s2-pro-f16-transformer-only.gguf
  s2-pro-f16-codec-only.gguf
```

server 啟動時可選用 voice clone 參考檔：

```text
runtime/s2_server/reference.wav
runtime/s2_server/reference.txt
```

下載 helper：

```powershell
.\scripts\download_models.ps1 -IncludeGguf -Quant f16
```

Rust 推論路徑使用 GGUF runtime pair，目前主要以 `mach9243/s2-pro-gguf`
驗證。`fishaudio/s2-pro` 是官方 checkpoint/tokenizer 來源。

## Server

短 smoke：

```powershell
.\scripts\smoke_rust_server.ps1 -MaxNewTokens 1
```

手動啟動：

```powershell
cargo run --release -p fish_s2_infer --bin fish_s2_server -- `
  --backend rust-pure `
  --max-new-tokens 1 `
  --port 8081
```

POST 合成 WAV：

```powershell
$body = @{ text = "hi"; format = "wav" } | ConvertTo-Json -Compress
Invoke-WebRequest `
  -Uri http://127.0.0.1:8081/v1/tts `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $body `
  -OutFile output\hi.wav
```

可選 backend：

```text
rust-pure   純 Rust 路徑，不需要 s2.exe / C++ library
ffi         使用 cpp-engine 連結 s2.cpp backend
```

舊的外部 `s2.exe` subprocess backend 預設不會出現在 build 中；若需要相容
路徑，請用 `--features legacy-s2-exe` 建置。

## GUI

```powershell
cargo run --release -p fish_s2_gui
```

Server 分頁預設可選 `rust-pure` 或 `ffi`，也可設定短測用的 `max_new_tokens`。
Voice profile 會把 `reference.wav` 與 `reference.txt` 複製到 server workdir；
RustPure 會在 server load 時編碼 reference prompt codes。

## 驗證

MVP 驗收 gate：

```powershell
.\scripts\verify_mvp.ps1
```

這會檢查模型檔、format、unit tests、GUI build check、strict clippy 與
RustPure server CLI，並寫出 `output\mvp_report.json`。若要在同一份報告中
加入較慢的真模型 HTTP 合成 smoke：

```powershell
.\scripts\verify_mvp.ps1 -RunServerSmoke -MaxNewTokens 1
```

常用本機 gate：

```powershell
cargo fmt
cargo test -p fish_s2_core
cargo test -p fish_s2_infer
cargo check -p fish_s2_gui
cargo clippy -p fish_s2_core -p fish_s2_infer -p fish_s2_gui --all-targets -- -D warnings
```

與 `s2.cpp` 做 1-token full E2E parity：

```powershell
.\scripts\dump_e2e_wav_parity.ps1 -MaxNewTokens 1
```

會驗證 generated code IDs、codec waveform stats 與 WAV envelope。

## 授權

本專案 Rust 程式碼使用 MIT license。

Fish Audio S2 Pro model weights、tokenizer assets、GGUF conversions 不屬於本
repo 的 MIT license。`fishaudio/s2-pro` 使用 Fish Audio Research License；
研究、非商業、散布與商業使用條款請以 upstream model license 為準。連結與
attribution notes 見 `docs/THIRD_PARTY_NOTICES.md`。
