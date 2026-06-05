# Fish S2 Pro Rust

Rust-native Fish Audio S2 Pro inference and desktop tooling.

The current `rust-pure` path can synthesize WAV from GGUF runtime pairs without
`s2.exe` or a linked C++ library. It includes tokenizer loading, Slow-AR,
Fast-AR, RVQ codec decode, reference-audio prompt-code encoding, an HTTP
`/v1/tts` server, a Windows GUI, and parity tools against `s2.cpp`.

## Workspace

```text
crates/
  fish_s2_core    GGUF utilities, model scanning, config, TTS client
  fish_s2_infer   Rust-native inference, server, parity dump binaries
  fish_s2_gui     Desktop GUI built with egui/eframe
  fish_s2_parity  WAV/code/tensor comparison helpers
```

## Model Files

Put runtime assets in `models/`:

```text
models/
  tokenizer.json
  s2-pro-f16-transformer-only.gguf
  s2-pro-f16-codec-only.gguf
```

Optional voice clone files for server startup:

```text
runtime/s2_server/reference.wav
runtime/s2_server/reference.txt
```

Download helper:

```powershell
.\scripts\download_models.ps1 -IncludeGguf -Quant f16
```

The Rust inference path consumes GGUF runtime pairs, currently tested with
`mach9243/s2-pro-gguf`. The official `fishaudio/s2-pro` repository remains the
source checkpoint/tokenizer repository.

## Server

Short RustPure smoke:

```powershell
.\scripts\smoke_rust_server.ps1 -MaxNewTokens 1
```

Manual server run:

```powershell
cargo run --release -p fish_s2_infer --bin fish_s2_server -- `
  --backend rust-pure `
  --max-new-tokens 1 `
  --port 8081
```

POST a WAV request:

```powershell
$body = @{ text = "hi"; format = "wav" } | ConvertTo-Json -Compress
Invoke-WebRequest `
  -Uri http://127.0.0.1:8081/v1/tts `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $body `
  -OutFile output\hi.wav
```

Available backends:

```text
rust-pure   Pure Rust path, no s2.exe / C++ library
ffi         Linked s2.cpp backend when built with cpp-engine
subprocess  Legacy s2.exe fallback
```

## GUI

```powershell
cargo run --release -p fish_s2_gui
```

The Server tab can select `rust-pure`, `ffi`, or `subprocess`, and can set a
short `max_new_tokens` value for smoke tests. Voice profiles copy
`reference.wav` and `reference.txt` into the server workdir; RustPure encodes
those reference files at server load time.

## Validation

MVP acceptance gate:

```powershell
.\scripts\verify_mvp.ps1
```

This writes `output\mvp_report.json` after checking model assets, formatting,
unit tests, GUI build checks, strict clippy, and the RustPure server CLI. To run
the slow real-model HTTP synthesis smoke as part of the same report:

```powershell
.\scripts\verify_mvp.ps1 -RunServerSmoke -MaxNewTokens 1
```

Common local gates:

```powershell
cargo fmt
cargo test -p fish_s2_core
cargo test -p fish_s2_infer
cargo check -p fish_s2_gui
cargo clippy -p fish_s2_core -p fish_s2_infer -p fish_s2_gui --all-targets -- -D warnings
```

Full one-token E2E parity against `s2.cpp`:

```powershell
.\scripts\dump_e2e_wav_parity.ps1 -MaxNewTokens 1
```

This verifies generated code IDs, codec waveform stats, and WAV envelope.

## License

Project Rust code is MIT licensed.

Fish Audio S2 Pro model weights, tokenizer assets, and GGUF conversions are not
covered by this repository's MIT license. `fishaudio/s2-pro` is licensed under
the Fish Audio Research License; follow the upstream model license for research,
non-commercial, distribution, and commercial-use terms. See
`docs/THIRD_PARTY_NOTICES.md` for links and attribution notes.
