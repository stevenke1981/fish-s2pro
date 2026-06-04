# Pure-Rust S2 Pro Dual-AR Inference — Agent TODO

**Goal:** Remove dependency on `s2.exe` / `s2.cpp` while keeping API compatibility (`/v1/tts`, `models/` layout, voice clone via `reference.wav` + `reference.txt`).

**Current state (fish-s2pro):**

| Layer | Location | Status |
|-------|----------|--------|
| App / GUI | `crates/fish_s2_gui`, `crates/fish_s2_core` | Done |
| Rust orchestration + HTTP | `crates/fish_s2_infer` (`engine.rs`, `server.rs`) | Done |
| GGUF metadata only | `crates/fish_s2_core/src/gguf.rs` | Partial (header parse, no tensor load) |
| C++ FFI shim | `crates/fish_s2_infer/ffi/s2_engine_ffi.{h,cpp}` | Scaffold (needs link to real `s2::Pipeline`) |
| Fallback CLI | `engine.rs` → `bin/s2.exe` | Works if binary present |

**Reference implementation (source of truth for math):**

- [mach92432/s2.cpp](https://github.com/mach92432/s2.cpp) — `include/s2_pipeline.h`, `src/` (Dual-AR + codec on **ggml**)
- Pipeline: text → BPE tokens → **Slow-AR** (36L Qwen3, KV cache) → **Fast-AR** (4L, 10 codebooks/step) → **RVQ codec** → WAV
- Models: [mach9243/s2-pro-gguf](https://huggingface.co/mach9243/s2-pro-gguf) (`*-transformer-only.gguf` + `*-codec-only.gguf`)

---

## Can one agent “just do it”?

**No** for a full pure-Rust port in a single session. Realistic options:

| Path | Effort | C++ free? | Notes |
|------|--------|-----------|-------|
| **A. Finish in-process FFI** | 1–3 days | No (static link) | Fastest path to “no external `s2.exe` process”; reuse ggml graphs from s2.cpp |
| **B. Rust → ggml via `cxx`/bindgen** | 1–2 weeks | No (ggml C lib) | Thin Rust API over vendored ggml + ported graph builders |
| **C. True pure Rust (Candle / Burn / custom)** | 4–12+ weeks | Yes | Must reimplement or import every op + quant formats + Vulkan |

**Recommendation:** Ship **Path A** first (production), parallel **Path C** research spike (Phase 0–2 below).

---

## Phase 0 — Baseline & golden tests (BLOCKER for everything)

- [x] **0.1** Vendor pin: document exact `s2.cpp` commit hash used for parity in `docs/S2_CPP_PIN.md`.
- [x] **0.2** Add `tests/fixtures/` (or env var paths): `crates/fish_s2_parity/tests/fixtures/` contains prompts/reference text; reference WAV is supplied locally via `-ReferenceWav` or `models/reference.wav`.
- [x] **0.3** Script `scripts/parity_run_s2cpp.ps1`: run official `s2` binary, write `output/golden.wav` + SHA256 sidecar.
- [x] **0.4** Crate `fish_s2_parity` (or `tests/parity/`): compare WAV duration, sample rate, RMS envelope — **not** bitwise (FP/Vulkan variance). Default tolerances: 0.10s duration, 0.03 RMS, 0.04 envelope MAE.
- [x] **0.5** CI gate: parity job optional (needs GPU + GGUF); mark `#[ignore]` locally without models.

**Acceptance:** With models in `models/`, C++ baseline produces stable WAV; hash/envelope logged for later Rust comparison.

---

## Phase 1 — Complete Path A (FFI, no separate process)

*Depends on: Phase 0.1*

- [ ] **1.1** Fix `scripts/build_s2_native.ps1`: link **full** `s2` + `ggml` static libs from CMake targets (not only `s2_engine_ffi.obj`).
- [ ] **1.2** Align `s2_engine_ffi.cpp` includes with vendored tree (`s2_pipeline.h` path — today `../include/` is wrong when built from `fish_s2_infer/ffi/`).
- [ ] **1.3** Expose reference-audio preload in FFI (match s2.cpp server: encode ref at init, not per request).
- [ ] **1.4** Wire `cpp-engine` feature: `CARGO_FEATURE_CPP_ENGINE` + `S2_CPP_LIB` documented in `models/README.txt`.
- [ ] **1.5** GUI default: detect linked backend; hide “download s2.exe” when `s2_cpp_linked`.
- [ ] **1.6** Integration test: `InferenceEngine::synthesize_wav` returns valid RIFF when FFI linked.

**Acceptance:** `cargo build -p fish_s2_gui --features cpp-engine` + models → TTS without `bin/s2.exe`.

---

## Phase 2 — GGUF tensor loading in Rust (shared for B/C)

*Depends on: Phase 0*

- [ ] **2.1** Choose loader strategy (document in this file):
  - Option 1: Extend `fish_s2_core::gguf` → mmap + tensor index + `f16`/`q8_0`/`q4_k` dequant read
  - Option 2: Crate `gguf` / `llama-gguf` for metadata + map tensors by name
  - Option 3: FFI only for load, Rust for orchestration (hybrid)
- [ ] **2.2** Implement `GgufTensorView` + name lookup matching s2.cpp tensor names (dump names from C++ once).
- [ ] **2.3** Unit test: load `models/*-transformer-only.gguf`, assert `general.architecture`, tensor count, key weight shapes.
- [ ] **2.4** Same for `*-codec-only.gguf`.

**Acceptance:** Rust can enumerate and read at least one weight tensor bytes without C++.

---

## Phase 3 — Tokenizer parity (Rust)

*Depends on: Phase 0.3*

- [ ] **3.1** Module `fish_s2_infer::tokenizer` using existing `tokenizers` dep — load `models/tokenizer.json`.
- [ ] **3.2** Port **ByteLevel** pre-BPE table from s2.cpp (GPT-2 byte↔unicode); add vectors of known strings from HF reference.
- [ ] **3.3** Golden test: token IDs vs Python `transformers` or s2.cpp CLI debug output for 20 strings (tags, CJK, emoji).

**Acceptance:** Token ID sequence matches reference on all fixture strings.

---

## Phase 4 — Slow-AR forward (largest block)

*Depends on: Phase 2, 3*

- [ ] **4.1** Extract graph spec from s2.cpp: layers, GQA (32h/8kv), RoPE base 1M, QK norm, hidden size, vocab.
- [ ] **4.2** Implement prefill + decode step with **KV cache** (separate Slow-AR `gallocr` equivalent).
- [ ] **4.3** Quantized matmul: `f16`, `q8_0`, `q4_k_m` — match ggml op semantics (or delegate to ggml via Path B).
- [ ] **4.4** Sampling: temperature, top_p, top_k — match `PipelineParams` defaults in `s2_engine_ffi.cpp`.
- [ ] **4.5** Golden test: single forward step hidden state L2 distance vs C++ dump (tight tolerance on CPU).

**Acceptance:** Slow-AR produces semantic hidden states matching C++ for fixed seed on CPU.

---

## Phase 5 — Fast-AR codebook decoder

*Depends on: Phase 4*

- [ ] **5.1** 4-layer AR over **10 codebooks** per semantic step (4096 entries each).
- [ ] **5.2** Persistent Fast-AR allocator / cache (separate from Slow-AR per s2.cpp notes).
- [ ] **5.3** Golden test: codebook token sequence vs C++ for short prompt.

**Acceptance:** Code indices match reference for greedy decode (temp=0).

---

## Phase 6 — Audio codec (RVQ)

*Depends on: Phase 2, 5*

- [ ] **6.1** Map codec GGUF to conv encoder/decoder + RVQ (10 × 4096).
- [ ] **6.2** Implement decode: codes → waveform (24 kHz or rate from metadata).
- [ ] **6.3** Optional: encode path for reference audio at startup (voice clone).
- [ ] **6.4** Golden test: codec-only reconstruct sin sweep or short ref WAV within SNR threshold.

**Acceptance:** WAV output from codes alone is intelligible / matches C++ envelope.

---

## Phase 7 — End-to-end pipeline in Rust

*Depends on: Phase 3–6*

- [ ] **7.1** `fish_s2_infer::pipeline::Pipeline` mirroring `s2::Pipeline` API.
- [ ] **7.2** Replace `InferenceEngine` backend selection: `Backend::RustPure` vs `Backend::Ffi` vs `Backend::Subprocess` (feature flags).
- [ ] **7.3** Reference conditioning: load `reference.wav` + `reference.txt` at `load()` time.
- [ ] **7.4** E2E parity: same text as Phase 0.3 — envelope/SNR vs `golden.wav`.

**Acceptance:** `fish_s2_server` + GUI work with **no** `s2.exe` and **no** `fish_s2_cpp.lib`.

---

## Phase 8 — GPU / Vulkan (optional but required for perf)

*Depends on: Phase 7 CPU path*

- [ ] **8.1** Evaluate `ggml` Vulkan via Path B vs pure-Rust GPU crate.
- [ ] **8.2** Match `-v` / `--codec-vulkan` device selection from `EngineConfig`.
- [ ] **8.3** Windows CI note: s2.cpp alpha — document tested matrix (Linux + RTX only).

**Acceptance:** RTF ≤ 1.5× C++ baseline on same GPU (not necessarily equal).

---

## Phase 9 — Cleanup & product

*Depends on: Phase 7*

- [ ] **9.1** Remove subprocess fallback or gate behind `legacy-s2-exe` feature.
- [ ] **9.2** Delete unused `fish_s2_core::server::ServerProcess` if fully deprecated.
- [ ] **9.3** Update `models/README.txt` + download script for quant variants.
- [ ] **9.4** License attribution: Fish Audio Research License in binary distributions.

---

## Suggested agent split

| Agent | Phases | Skills |
|-------|--------|--------|
| **build-agent** | 1, 8 | CMake, MSVC, Vulkan SDK, `build.rs` |
| **parity-agent** | 0, 3, golden tests | Python HF tokenizer, WAV analysis |
| **gguf-agent** | 2 | mmap, quant formats |
| **graph-agent** | 4, 5 | transformers, ggml source reading |
| **codec-agent** | 6 | DSP, RVQ, conv nets |
| **integrator** | 7, 9 | `fish_s2_infer`, axum, GUI wiring |

---

## Files to touch (quick index)

```
crates/fish_s2_infer/src/engine.rs      # backend selection
crates/fish_s2_infer/src/lib.rs
crates/fish_s2_infer/ffi/             # Path A
crates/fish_s2_core/src/gguf.rs       # Path C loader
scripts/build_s2_native.ps1
scripts/download_models.ps1
docs/PURE_RUST_DUAL_AR_TODO.md        # this file
```

---

## Definition of done (project-level)

1. `cargo test` passes (unit + ignored parity with `FISH_S2_PARITY=1`).
2. `cargo run -p fish_s2_infer --bin fish_s2_server` synthesizes WAV from `models/` **without** C++ binary or static lib.
3. GUI “Rust 推理引擎” works on Windows with documented GPU setup.
4. README states license + model download steps.

---

*Last updated: 2026-06-04 — fish-s2pro workspace*
