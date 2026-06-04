# S2 Pro Transformer Registry

Recorded: 2026-06-04

This is the Phase 4.1 slice: Rust knows the transformer GGUF weight names,
shape contracts, attention/GQA split, RoPE base, and Slow-AR KV cache layout.
It does not run Slow-AR math yet.

Source model:

- `models/s2-pro-f16-transformer-only.gguf`
- Architecture: `fish-speech`
- Tensor count: `358`

Rust entry point:

- `fish_s2_infer::registry::TransformerTensorRegistry`
- `fish_s2_infer::registry::DualArGraphSpec`

s2.cpp source references:

- Pinned commit: `2b9c6d2984f92b186419420e9126c47919b81d3a`
- HParams metadata load: `output/s2.cpp-src/src/s2_model.cpp:166`
- Slow-AR KV cache allocation: `output/s2.cpp-src/src/s2_model.cpp:343`
- Slow-AR `eval_cached` graph: `output/s2.cpp-src/src/s2_model.cpp:423`
- Fast-AR `fast_decode` graph: `output/s2.cpp-src/src/s2_model.cpp:672`
- Pipeline KV cache sizing/reuse: `output/s2.cpp-src/src/s2_pipeline.cpp:155`
- Semantic/Fast-AR generation loop: `output/s2.cpp-src/src/s2_generate.cpp:1`

Constants asserted by the registry:

| Name | Value |
|------|-------|
| `SLOW_AR_LAYERS` | `36` |
| `FAST_AR_LAYERS` | `4` |
| `SLOW_CONTEXT_LENGTH` | `32768` |
| `FAST_CONTEXT_LENGTH` | `11` |
| `HIDDEN_SIZE` | `2560` |
| `ATTENTION_HEADS` | `32` |
| `KV_HEADS` | `8` |
| `HEAD_DIM` | `128` |
| `QK_NORM_SIZE` | `128` |
| `WQKV_OUT` | `6144` |
| `ATTENTION_OUT` | `4096` |
| `FFN_SIZE` | `9728` |
| `TEXT_VOCAB_SIZE` | `155776` |
| `FAST_VOCAB_SIZE` | `4096` |
| `CODEBOOK_EMBEDDING_SIZE` | `40960` |
| `CODEBOOK_SIZE` | `4096` |
| `NUM_CODEBOOKS` | `10` |
| `SEMANTIC_BEGIN_ID` | `151678` |
| `SEMANTIC_END_ID` | `155773` |
| `ROPE_FREQ_BASE` | `1000000` |
| `RMS_NORM_EPS` | `0.000001` |

Dual-AR graph math contract:

| Area | Slow-AR | Fast-AR |
|------|---------|---------|
| Blocks | `36` | `4` |
| Context length | `32768` | `11` |
| Hidden size | `2560` | `2560` |
| FFN size | `9728` | `9728` |
| Query heads | `32` | `32` |
| KV heads | `8` | `8` |
| Head dim | `128` | `128` |
| Q size | `4096` | `4096` |
| K size | `1024` | `1024` |
| V size | `1024` | `1024` |
| GQA repeat | `4` | `4` |
| RoPE base | `1000000` | `1000000` |
| RMS norm eps | `0.000001` | `0.000001` |
| QK norm | `true` | `false` |

Slow-AR KV cache layout:

- Type: `GGML_TYPE_F16`
- K cache dimensions: `[head_dim, kv_heads, max_seq_len, slow_layers]`
- V cache dimensions: `[head_dim, kv_heads, max_seq_len, slow_layers]`
- For S2 Pro: `[128, 8, max_seq_len, 36]`
- One cache byte size: `128 * 8 * max_seq_len * 36 * 2`
- K+V byte size: `2 * 128 * 8 * max_seq_len * 36 * 2`

Token/codebook contract:

- Semantic token range: `[151678, 155773]`
- Codebook count: `10`
- Codebook size: `4096`
- Slow-AR input timestep width: semantic row plus 10 codebook rows, so
  `num_codebooks + 1 = 11`
- `scale_codebook_embeddings = true`
- `tie_word_embeddings = true`
- `fast_project_in = false` for `s2-pro-f16-transformer-only.gguf`

Phase 4.2 prep slice:

- `fish_s2_infer::tensor::F16TensorView` validates GGUF tensor type/shape/byte
  length and decodes little-endian F16 payloads to `f32`.
- `fish_s2_infer::tensor::rms_norm` implements scalar RMSNorm over one vector.
- `fish_s2_infer::tensor::linear` implements scalar linear projection for ggml
  weights stored as `[output_dim, input_dim]` rows while GGUF shape metadata is
  reported as `[input_dim, output_dim]`.
- Ignored smoke test loads `norm.weight` from the local transformer GGUF as an
  F16 typed tensor.
- `fish_s2_infer::attention::apply_rope_normal` implements the `ggml_rope_ext`
  mode used by s2.cpp (`GGML_ROPE_TYPE_NORMAL`, adjacent-pair rotation).
- `fish_s2_infer::attention::SlowArKvCache` provides a CPU smoke cache with the
  same logical dimensions as s2.cpp: `[head_dim, kv_heads, max_seq_len, layers]`.
- `fish_s2_infer::attention::gqa_decode_attention` validates the 32h/8kv GQA
  repeat and single-step decode softmax path over cached K/V tokens.
- `fish_s2_infer::slow_ar::SlowArLayerSkeleton` wires a toy single-token
  attention path through RMSNorm, WQKV split, QK norm, RoPE, KV write, GQA
  attention, output projection, and residual add.
- `SlowArLayerSkeleton::forward_prefill_sequence` runs multiple hidden tokens
  through a layer-local prefill-style path: prepare all token Q/K/V, write the
  full K/V span into cache, then compute causal attention per token over the
  visible prefix. `forward_decode_sequence` remains the looped decode reference
  path for equivalence tests.
- `SlowArLayerSkeleton::forward_block_prefill_sequence` adds the transformer FFN
  sublayer on top of the attention output: `ffn_norm -> w1/w3 -> SwiGLU -> w2
  -> residual`. This is covered by toy block smoke tests and the ignored local
  GGUF layer 0 finite smoke.
- `fish_s2_infer::slow_ar::SlowArLayerF16Weights` binds a registry layer to
  real local GGUF F16 tensors (`attention_norm`, `q_norm`, `k_norm`, `wqkv`,
  `wo`, `ffn_norm`, `w1`, `w2`, `w3`) and feeds them into the layer skeleton.
  The ignored fixture loads layer 0 and checks shape consistency plus finite
  attention and FFN outputs.
- `fish_s2_slow_ar_dump` writes JSON stats for the same layer 0 Rust fixture,
  including len/L2/mean_abs/max_abs/first8 for normalized, Q, K, V, attention,
  projection, attention residual hidden, FFN normalized/gate/up/SwiGLU/projected,
  and final block hidden state. `--tokens N` uses the prefill-style sequence path
  and emits a `sequence` array with per-token positions while keeping token 0
  stats at the top level for backward compatibility.
- `scripts\dump_s2cpp_slow_ar_stats.ps1` patches a local ignored s2.cpp clone,
  builds a standalone `s2_slow_ar_dump` helper without the Crow/server target,
  and writes the matching layer-local full-block C++ JSON stats dump. `-Tokens N`
  mirrors the Rust sequence fixture. It defaults to CPU and also supports CUDA via
  `-Cuda -CudaDevice 0`; on Visual Studio 18/2026 with CUDA 12.6, add
  `-AllowUnsupportedCudaCompiler`.
- `fish_s2_parity compare-slow-ar` compares Rust and C++ Slow-AR JSON stats
  dumps with per-token tensor names such as `token1.key` and
  `token1.ffn_projected`. Tolerances are tuned for ggml F16/RoPE vs Rust scalar
  `f32` drift across a two-token decode.

Root tensor specs:

| Tensor | Dimensions |
|--------|------------|
| `codebook_embeddings.weight` | `2560x40960` |
| `embeddings.weight` | `2560x155776` |
| `fast_embeddings.weight` | `2560x4096` |
| `fast_output.weight` | `2560x4096` |
| `fast_norm.weight` | `2560` |
| `norm.weight` | `2560` |

Slow-AR layer `N` tensor specs:

| Pattern | Dimensions |
|---------|------------|
| `layers.N.attention.q_norm.weight` | `128` |
| `layers.N.attention.k_norm.weight` | `128` |
| `layers.N.attention.wqkv.weight` | `2560x6144` |
| `layers.N.attention.wo.weight` | `4096x2560` |
| `layers.N.attention_norm.weight` | `2560` |
| `layers.N.ffn_norm.weight` | `2560` |
| `layers.N.feed_forward.w1.weight` | `2560x9728` |
| `layers.N.feed_forward.w2.weight` | `9728x2560` |
| `layers.N.feed_forward.w3.weight` | `2560x9728` |

Fast-AR layer `N` tensor specs:

| Pattern | Dimensions |
|---------|------------|
| `fast_layers.N.attention.wqkv.weight` | `2560x6144` |
| `fast_layers.N.attention.wo.weight` | `4096x2560` |
| `fast_layers.N.attention_norm.weight` | `2560` |
| `fast_layers.N.ffn_norm.weight` | `2560` |
| `fast_layers.N.feed_forward.w1.weight` | `2560x9728` |
| `fast_layers.N.feed_forward.w2.weight` | `9728x2560` |
| `fast_layers.N.feed_forward.w3.weight` | `2560x9728` |

Validation:

```powershell
cargo test --workspace
cargo test -p fish_s2_infer attention::tests
cargo test -p fish_s2_infer slow_ar::tests
cargo test -p fish_s2_infer slow_ar::tests::binds_local_layer0_f16_weights_and_runs_single_token_fixture -- --ignored
cargo test -p fish_s2_infer registry::tests::validates_local_transformer_registry -- --ignored
cargo test -p fish_s2_infer tensor::tests::loads_local_norm_weight_as_f16_tensor -- --ignored
cargo run -p fish_s2_infer --bin fish_s2_slow_ar_dump -- --transformer .\models\s2-pro-f16-transformer-only.gguf --output .\output\slow_ar_layer0_rust_stats.json
cargo run -p fish_s2_infer --bin fish_s2_slow_ar_dump -- --transformer .\models\s2-pro-f16-transformer-only.gguf --output .\output\slow_ar_layer0_rust_seq2_stats.json --tokens 2
.\scripts\dump_s2cpp_slow_ar_stats.ps1
.\scripts\dump_s2cpp_slow_ar_stats.ps1 -Output .\output\slow_ar_layer0_cpp_seq2_stats.json -Tokens 2
.\scripts\dump_s2cpp_slow_ar_stats.ps1 -Cuda -CudaDevice 0 -AllowUnsupportedCudaCompiler -Output .\output\slow_ar_layer0_cpp_stats_cuda.json
cargo run -p fish_s2_parity --bin fish_s2_parity -- compare-slow-ar .\output\slow_ar_layer0_cpp_stats.json .\output\slow_ar_layer0_rust_stats.json
cargo run -p fish_s2_parity --bin fish_s2_parity -- compare-slow-ar .\output\slow_ar_layer0_cpp_seq2_stats.json .\output\slow_ar_layer0_rust_seq2_stats.json
cargo run -p fish_s2_parity --bin fish_s2_parity -- compare-slow-ar .\output\slow_ar_layer0_cpp_stats_cuda.json .\output\slow_ar_layer0_rust_stats.json
cargo clippy --all-targets -- -D warnings
```

Next Phase 4 work:

- Add typed views for quantized weights needed by non-F16 model variants.
- Broaden the Slow-AR parity fixture from looped multi-token decode to batched
  prefill over multiple tokens.
