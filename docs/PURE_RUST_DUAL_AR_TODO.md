# Pure-Rust S2 Pro Dual-AR Inference — Agent TODO

**Goal:** Remove dependency on `s2.exe` / `s2.cpp` while keeping API compatibility (`/v1/tts`, `models/` layout, voice clone via `reference.wav` + `reference.txt`).

**Current state (fish-s2pro):**

| Layer | Location | Status |
|-------|----------|--------|
| App / GUI | `crates/fish_s2_gui`, `crates/fish_s2_core` | Done |
| Rust orchestration + HTTP | `crates/fish_s2_infer` (`engine.rs`, `server.rs`) | Done |
| GGUF load + tensor bytes | `crates/fish_s2_core/src/gguf.rs` | Done (mmap index, F16 views in `fish_s2_infer::tensor`) |
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
| Pure-Rust semantic/codebook MVP | 3-7 days | Rust semantic token generation and Fast-AR codebook IDs now run; remaining work is broader parity, reference prompt fixtures, and cleanup. |
| Pure-Rust CPU end-to-end WAV | 4-8+ weeks | Slow-AR + Fast-AR + codec decode + integration, likely slow but C++ free. |
| Pure-Rust performant GPU end-to-end | 8-12+ weeks | Adds quantized kernels and CUDA/Vulkan/WGPU-style backend parity/perf work. |

**Current pure-Rust state:** tokenizer, GGUF tensor reads, Slow-AR F16 layer math (prefill + decode with persistent KV), output-head logits, `fish_s2_infer::sampling`, `prompt::build_prompt`, `embed_slow_ar_time_major`, `SlowArState` (prefill/step/reset; `StepResult.hidden` = post-`norm.weight` like s2.cpp), `generate::{generate_semantic_tokens, generate_codes}`, and `fast_ar::{forward_codebook_prefix, generate_codebooks_for_semantic}` (4-layer causal prefix decode). CPU and CUDA Slow-AR logits/top-k parity pass. Greedy semantic token parity vs s2.cpp CPU passes (`dump_semantic_parity.ps1`). Greedy first-frame Fast-AR codebooks (all 10) pass (`dump_fast_ar_parity.ps1`). Full C++ `s2::generate` codebook-major output matches Rust `fish_s2_codes_dump` for greedy `hi`, 2 frames (`dump_generated_codes_parity.ps1`). **Codec decoder waveform** parity passes on greedy `hi` (`scripts/dump_waveform_parity.ps1`, `compare-waveform` + WAV envelope). **Not yet verified:** full E2E WAV (Slow-AR + Fast-AR + codec in one Rust path). Quantizer decode stage (RVQ → post-module → upsample) parity passes on greedy `hi`.

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
- [x] `scripts/dump_s2cpp_slow_ar_stats.ps1 -Cuda -Logits -TopK N`
  - Validate the CUDA backend logits/top-k dump.
  - Acceptance: CUDA compare passes against Rust for `slow_ar_layers0_36_*_logits_cuda.json`.
- [x] `fish_s2_parity::compare_slow_ar_dumps` logits gate
  - Compare optional `final_normalized`, `logits`, and exact-rank `top_logits.token_id`.
  - Keep short-chain tolerances strict; allow full-stack accumulated drift only when `layer_count >= 36`.
- [x] `fish_s2_infer::sampling::semantic_mask_logits(logits, sem_begin, sem_end, im_end_id, block_end)`
  - Match `s2_generate.cpp` semantic mask: only `[semantic_begin_id, semantic_end_id]` plus optional `im_end_id`.
  - Acceptance: unit test verifies masked logits are `-inf` outside allowed IDs.
- [x] `fish_s2_infer::sampling::sample_token(logits, SamplerParams, always_include_id, rng)`
  - Match s2.cpp top-k -> force-include EOS -> temperature softmax -> top-p -> discrete sample order.
  - Acceptance: deterministic `temp=0`/greedy and seeded top-k/top-p unit tests.
- [x] `fish_s2_infer::generate::generate_semantic_tokens(...)`
  - Wires `transpose_to_time_major(prompt)` -> `SlowArState::prefill` -> semantic sampling -> `step` loop.
  - Acceptance: greedy or fixed-seed semantic token sequence parity vs s2.cpp on a short prompt (`dump_semantic_parity.ps1`).
- [x] `fish_s2_infer::generate::generate_codes(...)`
  - Full Slow-AR + Fast-AR loop; codebook-major `GenerateCodesResult` (matches s2.cpp `generate()` layout; RAS not ported).
  - Slow `step` uses all Fast-AR codebooks in `build_step_input` (not only semantic cb0).
- [x] `fish_s2_codes_dump --max-new-tokens N`
  - Dumps Rust-generated codebook-major codes as `{ num_codebooks, n_frames, codes }`.
  - Acceptance: release run wrote `output/generated_codes_hi_rust.json` for greedy `hi` (`10 codebooks x 2 frames`); `fish_s2_parity compare-generated-codes` self-check passes.
- [x] C++ full generated-codes dump parity
  - Add `s2_generate_codes_dump` helper around `s2.cpp` full generate path and compare against `fish_s2_codes_dump`.
  - Acceptance: exact `num_codebooks`, `n_frames`, and `codes` match for greedy `hi`, `max_new_tokens=2` via `scripts/dump_generated_codes_parity.ps1`.
- [ ] Reference-prompt generated-codes parity
  - Extend `fish_s2_codes_dump` and `s2_generate_codes_dump` to accept `prompt_text` + prompt code fixture.
  - Acceptance: exact generated `codes` match on one short reference-prompt fixture.
- [x] Parity gate: `scripts/dump_semantic_parity.ps1` + `fish_s2_parity compare-semantic-tokens` (UTF-8 JSON); greedy `hi` short prompt (`main_token_ids` exact match vs s2.cpp CPU dump).

### Package B — Prompt Embeddings and Slow-AR Stateful Decode

- [x] `fish_s2_infer::prompt::build_prompt` + `transpose_to_time_major` -> time-major `Vec<i32>`
  - Match s2.cpp `build_prompt()` layout: `(num_codebooks + 1) * n_tokens`.
  - Acceptance: unit test for transpose; reference-prompt path matches `s2_generate.cpp` structure (parity dump pending).
- [x] `fish_s2_infer::slow_ar::embed_slow_ar_time_major(flat_tokens, graph, weights)`
  - Math: semantic `embeddings.weight` + summed `codebook_embeddings.weight`, optional `1/sqrt(num_codebooks+1)` scale.
  - Acceptance: Rust embedding stats parity against s2.cpp prefill dump for one prompt token (pending).
- [x] `fish_s2_infer::slow_ar::SlowArState`
  - Owns KV cache, `n_past`, graph specs, embeddings, output-head weights.
  - Functions: `prefill(flat_tokens)`, `step(flat_token)`, `reset()`; multi-token prefill uses `forward_slow_ar_block_prefill_layers_cached` on persistent cache.
  - Acceptance: greedy semantic token sequence parity vs s2.cpp CPU passes; lower-level prefill/step hidden-stat dump parity remains useful follow-up.

### Package C — Fast-AR Codebook Decoder

- [x] `fish_s2_infer::fast_ar::FastArLayerF16Weights::from_gguf_layer(...)`
  - Bind `fast_layers.N.*`, `fast_norm.weight`, `fast_output.weight`, `fast_embeddings.weight`.
  - Acceptance: registry shape smoke for all 4 layers (`loads_all_fast_ar_layers_from_gguf`).
- [x] `fish_s2_infer::fast_ar::forward_codebook_prefix(hidden, prefix_codes)`
  - Match `SlowARModel::fast_decode`: semantic hidden + prefix codebook embeddings -> 4-layer Fast-AR -> codebook logits.
  - Acceptance: `scripts/dump_fast_ar_parity.ps1` greedy `hi` first frame (`compare-fast-ar-frame` exact `codebook_ids`).
- [x] `fish_s2_infer::fast_ar::generate_codebooks_for_semantic(hidden, semantic_code, sampler)`
  - Generate remaining codebooks 1..9.
  - Acceptance: same as `dump_fast_ar_parity.ps1` (greedy `hi`, all codebooks match C++ CPU).

### Package D — Codec/RVQ Decode

- [x] `fish_s2_infer::codec::CodecTensorRegistry::from_gguf(codec_gguf)`
  - Map codec tensors and metadata from `s2-pro-f16-codec-only.gguf`.
  - Acceptance: `fish_s2_codec_dump` writes `output/s2-pro-f16-codec-registry.tsv` plus metadata TSV; ignored local GGUF smoke validates 461 tensors, `encoder=128`, `quantizer=244`, `decoder=89`, semantic codebook `8x4096`, and residual quantizers `{0..8}` `8x1024`.
- [x] `fish_s2_infer::codec::CodecF16Weights::from_gguf(codec_gguf)`
  - Bind typed F16 views for semantic/residual codebooks and projection weights without expanding the full decoder yet.
  - Acceptance: ignored local GGUF smoke loads semantic/residual codebooks plus in/out projections and validates shapes.
- [x] `fish_s2_infer::codec::rvq_lookup_codes(codes, weights) -> CodecRvqLookupResult`
  - First RVQ slice: generated codebook-major codes -> per-frame 1024-d latent via codebook lookup, `out_proj`, bias, and residual sum.
  - Acceptance: `fish_s2_rvq_lookup_dump` reads `output/generated_codes_hi_rust.json` and writes finite stats to `output/rvq_lookup_hi_rust.json` (`2 frames x 1024 latent`).
- [x] C++ RVQ lookup parity hook
  - Add s2.cpp dump for the same codebook lookup/projection/sum stage before porting pre/post module math.
  - Acceptance: `scripts/dump_rvq_lookup_parity.ps1` builds `s2_rvq_lookup_dump`, compares `decode_codes_stage(...)` vs Rust `rvq_lookup_codes(...)`, and passes on greedy `hi` (`latent_l2_delta=0.00000013`, `latent_first8_mae=0.00000005`).
- [x] `fish_s2_infer::codec::forward_codec_post_module(latents) -> CodecPostModuleResult`
  - Port the RVQ post-module transformer block: RMSNorm -> WQKV -> RoPE -> causal/windowed attention -> output projection -> layer scale -> FFN/SwiGLU -> final norm.
  - Acceptance: ignored local GGUF smoke runs `rvq_lookup_codes(...)` then 8-layer post-module on greedy `hi`; `fish_s2_post_module_dump` writes finite stats to `output/post_module_hi_rust.json` (`2 frames x 1024 hidden`).
- [x] C++ post-module transformer parity hook
  - Add an s2.cpp dump for `build_transformer(quantizer.post_module)` before upsample so Rust post-module math can be pinned independently.
  - Acceptance: `scripts/dump_post_module_parity.ps1` builds `s2_post_module_dump`, compares s2.cpp `build_transformer(quantizer.post_module)` vs Rust `forward_codec_post_module(...)`, and passes on greedy `hi` (`hidden_l2_delta=0.00007342`, `hidden_first8_mae=0.00003685`).
- [x] `fish_s2_infer::codec::forward_codec_upsample(post_hidden) -> CodecUpsampleResult`
  - Port quantizer upsample ConvTranspose + ConvNeXt stages after post-module parity is pinned.
  - Acceptance: typed F16 registry binds `quantizer.upsample.{0,1}` weights; ignored local GGUF smoke runs RVQ lookup -> post-module -> 2-stage upsample on greedy `hi`; `fish_s2_decode_stage_dump` writes finite stats to `output/decode_stage_hi_rust.json` (`2 -> 8 frames x 1024 hidden`).
- [x] C++ quantizer decode-stage parity hook
  - Compare s2.cpp `build_quantizer_decode_stage(...)` against Rust RVQ lookup -> post-module -> upsample before expanding decoder waveform path.
  - Acceptance: `scripts/dump_decode_stage_parity.ps1` builds `s2_decode_stage_dump`, compares `output/decode_stage_hi_cpp.json` vs `output/decode_stage_hi_rust.json` via `fish_s2_parity compare-decode-stage`.
- [x] `fish_s2_infer::codec::rvq_decode_latents(latents) -> acoustic_features`
  - Wrap post-module + upsample into `CodecDecodeLatentsResult`; `fish_s2_decode_stage_dump` uses the same path.
  - Acceptance: code fixture dequant stats parity vs s2.cpp codec hook (`dump_decode_stage_parity.ps1`).
- [x] `fish_s2_infer::codec::decode_waveform(codes) -> Pcm/Wav`
  - RVQ lookup → `rvq_decode_latents` → pure-Rust decoder (entry conv, 4 upsample blocks, Snake, output conv, tanh).
  - Acceptance: `scripts/dump_waveform_parity.ps1` + `fish_s2_parity compare-waveform` on greedy `hi` (`samples_l2_delta≈3.2e-4`, WAV `passed=true`). C++ `s2_codec.cpp` `causal_conv_1d` keeps F16 kernels (ggml `im2col` requirement).
- [ ] `fish_s2_infer::codec::encode_reference_audio(wav) -> prompt_codes`
  - Needed for voice clone/reference conditioning.
  - [x] Pre-slice: bind/validate `quantizer.downsample.{0,1}` ConvNeXt weights and `quantizer.pre_module` transformer F16 weights for the encode path.
  - [x] Port quantizer downsample + pre-module forward from encoder latents to VQ input (`forward_codec_quantizer_encode_stage` synthetic-latent GGUF smoke).
  - [ ] Port encoder frontend from mono PCM to 1024-d latent frames.
    - [x] Bind/validate encoder frontend F16 weights: entry conv, 4 residual/downsample blocks, block4 transformer, tail/output conv.
    - [x] Port encoder frontend forward math shape path (`forward_codec_encoder_frontend` synthetic PCM GGUF smoke).
    - [ ] Add C++ encoder-latent parity hook for reference WAV.
  - [ ] Port VQ nearest-code search for semantic + 9 residual codebooks.
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
- [x] **4.2** Stateful prefill + decode step with **KV cache** (`SlowArState`). Layer-local prefill path, 36-layer chain, KV layout, real token embeddings, output-head logits, `SlowArState::{prefill,step,reset}`, CPU/CUDA block/logits parity, and greedy semantic token parity vs s2.cpp CPU are in place. Lower-level prompt prefill/step hidden-stat parity can still be added as an extra diagnostic.
- [ ] **4.3** Quantized matmul: `f16`, `q8_0`, `q4_k_m` — match ggml op semantics (or delegate to ggml via Path B).
- [x] **4.4** Sampling: temperature, top_p, top_k — match `PipelineParams` defaults in `s2_engine_ffi.cpp` (`fish_s2_infer::sampling`).
- [x] **4.5** Golden test: single forward step hidden state L2 distance vs C++ dump (tight tolerance on CPU). Rust JSON dump generation, local s2.cpp JSON dump hook, and Rust/C++ comparator exist for the layer 0 single-token fixture, a two-token two-layer fixture, and a two-token full 36-layer full-block prefill-style sequence fixture.

**Acceptance:** Slow-AR produces semantic tokens matching C++ for greedy CPU fixtures; CUDA logits/top-k parity is also verified.

---

## Phase 5 — Fast-AR codebook decoder

*Depends on: Phase 4*

- [x] **5.1** 4-layer AR over **10 codebooks** per semantic step (4096 entries each). Rust first-frame Fast-AR codebook parity vs s2.cpp CPU passes; `generate_codes` loops Slow-AR + Fast-AR over multiple generated frames.
- [x] **5.2** C++ full generated-code parity over multiple frames.
  - `scripts/dump_generated_codes_parity.ps1` builds `s2_generate_codes_dump`, writes C++ and Rust JSON, and compares exact `codes`.
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
crates/fish_s2_infer/src/sampling.rs  # Phase 4.4
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

*Last updated: 2026-06-05 — Codec/RVQ: encoder frontend forward shape smoke passes; next: C++ encoder-latent parity hook and VQ search*
