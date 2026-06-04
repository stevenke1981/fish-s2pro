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
- Models:
  - [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) — official Safetensors/BF16 source checkpoint and tokenizer.
  - [mach9243/s2-pro-gguf](https://huggingface.co/mach9243/s2-pro-gguf) — GGUF runtime pairs currently consumed by the Rust GGUF loader and s2.cpp-compatible fallback (`*-transformer-only.gguf` + `*-codec-only.gguf`).

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

## Remaining Modular Work Packages for Parallel Agents

**Rough time left from 2026-06-04 status:**

| Target | Estimate | What it means |
|--------|----------|---------------|
| Production without external `s2.exe` | 1-3 days | Finish Path A FFI/static-link backend; still uses C++/ggml in-process. |
| Pure-Rust semantic token MVP | 1-2 weeks | Text/tokenizer -> Slow-AR semantic token generation on F16 CPU, no Fast-AR/codec WAV yet. |
| Pure-Rust CPU end-to-end WAV | 4-8+ weeks | Slow-AR + Fast-AR + codec decode + integration, likely slow but C++ free. |
| Pure-Rust performant GPU end-to-end | 8-12+ weeks | Adds quantized kernels and CUDA/Vulkan/WGPU-style backend parity/perf work. |

**Current pure-Rust state:** tokenizer, GGUF tensor reads, Slow-AR F16 layer math, two-token full 36-layer hidden-state parity, and an opt-in logits/top-k dump path (`fish_s2_slow_ar_dump --logits`) exist. C++ CPU logits/top-k parity passes; CUDA logits parity is still pending.

### Package A — Slow-AR Logits and Sampling

- [x] `fish_s2_infer::slow_ar::SlowArOutputHeadF16Weights::from_gguf(gguf) -> Result<Self>`
  - Inputs: `norm.weight`, tied `embeddings.weight`.
  - Output: typed F16 views for final norm and vocab projection.
  - Acceptance: ignored local GGUF smoke loads both tensors and validates `[2560]`, `[2560,155776]`.
- [x] `fish_s2_infer::slow_ar::forward_logits(hidden, output_head, eps) -> SlowArLogitsOutput`
  - Inputs: final 36-layer `block_hidden` for last token.
  - Math: `rms_norm(hidden, norm.weight) -> linear(embeddings.weight, normalized)`.
  - Acceptance: `fish_s2_slow_ar_dump --tokens 2 --layers 36 --logits` emits `final_normalized`, `logits`, and `top_logits`.
- [x] `scripts/dump_s2cpp_slow_ar_stats.ps1 -Logits -TopK N` CPU path
  - Add matching C++ dump fields after `weights_.norm` + `weights_.embeddings`.
  - Acceptance: CPU compare passes against Rust for `slow_ar_layers0_36_*_logits.json`.
- [ ] `scripts/dump_s2cpp_slow_ar_stats.ps1 -Cuda -Logits -TopK N`
  - Validate the CUDA backend logits/top-k dump.
  - Acceptance: CUDA compare passes against Rust for `slow_ar_layers0_36_*_logits_cuda.json`.
- [x] `fish_s2_parity::compare_slow_ar_dumps` logits gate
  - Compare optional `final_normalized`, `logits`, and exact-rank `top_logits.token_id`.
  - Keep short-chain tolerances strict; allow full-stack accumulated drift only when `layer_count >= 36`.
- [ ] `fish_s2_infer::sampling::semantic_mask_logits(logits, sem_begin, sem_end, im_end_id, block_end)`
  - Match `s2_generate.cpp` semantic mask: only `[semantic_begin_id, semantic_end_id]` plus optional `im_end_id`.
  - Acceptance: unit test verifies masked logits are `-inf` outside allowed IDs.
- [ ] `fish_s2_infer::sampling::sample_token(logits, SamplerParams, always_include_id, rng)`
  - Match s2.cpp top-k -> force-include EOS -> temperature softmax -> top-p -> discrete sample order.
  - Acceptance: deterministic `temp=0`/greedy and seeded top-k/top-p unit tests.
- [ ] `fish_s2_infer::slow_ar::generate_semantic_tokens(...)`
  - Wires tokenizer prompt state -> Slow-AR prefill -> logits -> semantic sampling loop.
  - Acceptance: greedy or fixed-seed semantic token sequence parity vs s2.cpp on a short prompt.

### Package B — Prompt Embeddings and Slow-AR Stateful Decode

- [ ] `fish_s2_infer::prompt::build_prompt_tensor(text_tokens, prompt_codes) -> PromptTensor`
  - Match s2.cpp time-major layout: `(num_codebooks + 1) * n_tokens`.
  - Acceptance: fixture transposes codebook-major prompt data exactly like `s2_generate.cpp`.
- [ ] `fish_s2_infer::slow_ar::embed_slow_ar_tokens(flat_tokens, graph_spec, weights)`
  - Math: semantic `embeddings.weight` + masked/summed `codebook_embeddings.weight`, optional `1/sqrt(num_codebooks+1)` scale.
  - Acceptance: Rust embedding stats parity against s2.cpp prefill dump for one prompt token.
- [ ] `fish_s2_infer::slow_ar::SlowArState`
  - Owns KV cache, `n_past`, graph specs, layer/output-head weights.
  - Functions: `prefill(flat_tokens)`, `step(flat_token)`, `reset()`.
  - Acceptance: `prefill + step` hidden/logits parity vs C++ `SlowARModel::prefill/step`.

### Package C — Fast-AR Codebook Decoder

- [ ] `fish_s2_infer::fast_ar::FastArLayerF16Weights::from_gguf_layer(...)`
  - Bind `fast_layers.N.*`, `fast_norm.weight`, `fast_output.weight`, `fast_embeddings.weight`.
  - Acceptance: registry shape smoke for all 4 layers.
- [ ] `fish_s2_infer::fast_ar::forward_codebook_prefix(hidden, prefix_codes)`
  - Match `SlowARModel::fast_decode`: semantic hidden + prefix codebook embeddings -> 4-layer Fast-AR -> codebook logits.
  - Acceptance: C++ dump hook for `fast_logits` and Rust parity for one prefix length.
- [ ] `fish_s2_infer::fast_ar::generate_codebooks_for_semantic(hidden, semantic_code, sampler)`
  - Generate remaining codebooks 1..9.
  - Acceptance: fixed-seed or greedy codebook sequence parity vs s2.cpp for one semantic step.

### Package D — Codec/RVQ Decode

- [ ] `fish_s2_infer::codec::CodecTensorRegistry::from_gguf(codec_gguf)`
  - Map codec tensors and metadata from `s2-pro-f16-codec-only.gguf`.
  - Acceptance: tensor name/shape dump parity against `output/s2-pro-f16-codec-tensors.tsv`.
- [ ] `fish_s2_infer::codec::rvq_decode_codes(codes) -> acoustic_features`
  - Port semantic/residual quantizer dequantization.
  - Acceptance: code fixture dequant stats parity vs s2.cpp codec hook.
- [ ] `fish_s2_infer::codec::decode_waveform(codes) -> Pcm/Wav`
  - Port post-module transformer/ConvNeXt/upsample path.
  - Acceptance: codec-only WAV envelope/SNR parity on a tiny code fixture.
- [ ] `fish_s2_infer::codec::encode_reference_audio(wav) -> prompt_codes`
  - Needed for voice clone/reference conditioning.
  - Acceptance: reference WAV prompt codes match s2.cpp within exact code sequence or documented tolerance.

### Package E — Quantization and Memory Efficiency

- [ ] `fish_s2_core::gguf::MappedTensorView`
  - Avoid eagerly expanding huge tensors such as `embeddings.weight` to f32.
  - Acceptance: logits path can run with streaming/mmap rows under bounded memory.
- [ ] `fish_s2_infer::tensor::matvec_f16_streaming(input, f16_bytes, dims)`
  - Compute output-head logits without allocating the full f32 embedding matrix.
  - Acceptance: matches current F32-expanded output head stats.
- [ ] `fish_s2_infer::tensor::{q8_0,q4_k_m}::matvec`
  - Match ggml quantized matmul semantics for non-F16 GGUF variants.
  - Acceptance: tiny synthetic ggml parity plus selected real tensor parity.

### Package F — Integration Backend

- [ ] `fish_s2_infer::pipeline::RustPipeline::load(model_dir, config)`
  - Own tokenizer, transformer GGUF, codec GGUF, prompt/reference state.
  - Acceptance: can load `models/` without `s2.exe` or C++ lib.
- [ ] `fish_s2_infer::pipeline::RustPipeline::synthesize(request) -> WAV`
  - Calls tokenizer -> Slow-AR -> Fast-AR -> codec.
  - Acceptance: returns valid RIFF and passes Phase 0 envelope parity.
- [ ] `fish_s2_infer::engine::Backend::{RustPure,Ffi,Subprocess}`
  - Feature-gated backend selection with explicit logs.
  - Acceptance: GUI/server can choose RustPure when complete, FFI/subprocess fallback otherwise.

### Package G — GPU Acceleration

- [ ] `fish_s2_infer::backend::MatmulBackend` trait
  - CPU reference backend first; GPU backend later.
  - Acceptance: Slow/Fast/codec ops call through backend abstraction where practical.
- [ ] CUDA/Vulkan/WGPU spike for `matvec`, RMSNorm, RoPE, attention
  - Start with output-head matvec and layer FFN matvec as highest-cost ops.
  - Acceptance: one Slow-AR layer parity and measured speedup vs CPU.

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

- [x] **2.1** Choose loader strategy (document in this file):
  - **Selected:** Option 1, extend `fish_s2_core::gguf` with a tensor index and raw tensor byte access first. Dequant / typed views will be layered on top when Slow-AR and codec op coverage is known.
  - Option 1: Extend `fish_s2_core::gguf` → mmap + tensor index + `f16`/`q8_0`/`q4_k` dequant read
  - Option 2: Crate `gguf` / `llama-gguf` for metadata + map tensors by name
  - Option 3: FFI only for load, Rust for orchestration (hybrid)
- [x] **2.2** Implement `GgufTensorView` + name lookup matching s2.cpp tensor names (dump names from C++ once). Current API: `GgufFile::open`, `GgufFile::tensor`, `GgufFile::tensor_names`, `GgufFile::tensor_bytes`, and `GgufTensorInfo`.
- [x] **2.3** Unit test: load `models/*-transformer-only.gguf`, assert `general.architecture`, tensor count, key weight shapes. Verified with `s2-pro-f16-transformer-only.gguf`; full dump at `output/s2-pro-f16-transformer-tensors.tsv`.
- [x] **2.4** Same for `*-codec-only.gguf`. Verified with `s2-pro-f16-codec-only.gguf`; full dump at `output/s2-pro-f16-codec-tensors.tsv`.

**Acceptance:** Rust can enumerate and read at least one weight tensor bytes without C++.

---

## Phase 3 — Tokenizer parity (Rust)

*Depends on: Phase 0.3*

- [x] **3.1** Module `fish_s2_infer::tokenizer` using existing `tokenizers` dep — load `models/tokenizer.json`.
- [x] **3.2** Port **ByteLevel** pre-BPE table from s2.cpp (GPT-2 byte↔unicode); add vectors of known strings from HF reference.
- [x] **3.3** Golden test: token IDs vs Python `transformers` or s2.cpp CLI debug output for 20 strings (tags, CJK, emoji). Fixture: `crates/fish_s2_infer/tests/fixtures/tokenizer_golden.tsv`; verified against Python `transformers` on 2026-06-04.

**Acceptance:** Token ID sequence matches reference on all fixture strings.

---

## Phase 4 — Slow-AR forward (largest block)

*Depends on: Phase 2, 3*

- [x] **4.1** Extract graph spec from s2.cpp: layers, GQA (32h/8kv), RoPE base 1M, QK norm, hidden size, vocab. Rust `TransformerTensorRegistry` now validates Slow-AR/Fast-AR weight names, key shapes, metadata hparams, attention/GQA split, RoPE base, and Slow-AR KV cache layout from the transformer GGUF. Details are in `docs/S2_PRO_TRANSFORMER_REGISTRY.md`; forward math starts in 4.2+.
- [ ] **4.2** Implement prefill + decode step with **KV cache** (separate Slow-AR `gallocr` equivalent). **Prep slices complete:** `fish_s2_infer::tensor` now provides F16 typed tensor views plus scalar RMSNorm and ggml-compatible `[out,in]` linear smoke tests; `fish_s2_infer::attention` now covers ggml normal RoPE, Slow-AR KV token writes, and single-step GQA decode attention smoke tests; `fish_s2_infer::slow_ar` now has a toy single-token attention skeleton that runs `rms_norm -> WQKV split -> QK norm -> RoPE -> KV write -> GQA attention -> output projection -> residual`, plus a layer-local prefill-style multi-token sequence path: all token Q/K/V are prepared first, the full K/V span is written into cache, then each token runs causal attention over its visible prefix. `SlowArLayerSkeleton::forward_block_prefill_sequence` now adds the transformer FFN sublayer (`ffn_norm -> w1/w3 -> SwiGLU -> w2 -> residual`) with toy and real GGUF finite smoke coverage, and `forward_slow_ar_block_prefill_layers` now chains layer outputs so each layer's `block_hidden` becomes the next layer's hidden input. `fish_s2_slow_ar_dump --tokens N --layers M` emits Rust full-block JSON stats with per-token sequence entries, `scripts\dump_s2cpp_slow_ar_stats.ps1 -Tokens N -Layers M` emits the matching s2.cpp full-block JSON stats on CPU or CUDA, and `fish_s2_parity compare-slow-ar` now compares attention plus FFN/block tensors with per-token names, `layer_count` metadata, tight 1/2-layer tolerances, and wider full-stack tolerances for accumulated F16/ggml drift. Single-token layer 0, two-token 2-layer, and two-token full 36-layer CPU/CUDA s2.cpp vs Rust full-block parity pass; Slow-AR sampling/token generation is still pending.
- [ ] **4.3** Quantized matmul: `f16`, `q8_0`, `q4_k_m` — match ggml op semantics (or delegate to ggml via Path B).
- [ ] **4.4** Sampling: temperature, top_p, top_k — match `PipelineParams` defaults in `s2_engine_ffi.cpp`.
- [x] **4.5** Golden test: single forward step hidden state L2 distance vs C++ dump (tight tolerance on CPU). Rust JSON dump generation, local s2.cpp JSON dump hook, and Rust/C++ comparator exist for the layer 0 single-token fixture, a two-token two-layer fixture, and a two-token full 36-layer full-block prefill-style sequence fixture.

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
- [x] **9.3** Update `models/README.txt` + download script for model sources. `scripts/download_models.ps1` now supports `fishaudio/s2-pro` official checkpoint downloads via `-IncludeOfficialCheckpoint` and GGUF runtime pairs via `-IncludeGguf -Quant ...`; direct Rust inference still uses GGUF while official Safetensors are tokenizer/source/conversion inputs.
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
