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
cargo test -p fish_s2_infer registry::tests::validates_local_transformer_registry -- --ignored
cargo clippy --all-targets -- -D warnings
```

Next Phase 4 work:

- Add typed tensor views for F16 weights.
- Implement a CPU-only single-layer or single-token forward smoke before trying
  full prefill/decode.
