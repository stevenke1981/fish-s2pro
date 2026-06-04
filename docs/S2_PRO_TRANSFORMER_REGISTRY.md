# S2 Pro Transformer Registry

Recorded: 2026-06-04

This is the first Phase 4 slice: Rust knows the transformer GGUF weight names
and shape contracts, but it does not run Slow-AR math yet.

Source model:

- `models/s2-pro-f16-transformer-only.gguf`
- Architecture: `fish-speech`
- Tensor count: `358`

Rust entry point:

- `fish_s2_infer::registry::TransformerTensorRegistry`

Constants asserted by the registry:

| Name | Value |
|------|-------|
| `SLOW_AR_LAYERS` | `36` |
| `FAST_AR_LAYERS` | `4` |
| `HIDDEN_SIZE` | `2560` |
| `QK_NORM_SIZE` | `128` |
| `WQKV_OUT` | `6144` |
| `ATTENTION_OUT` | `4096` |
| `FFN_SIZE` | `9728` |
| `TEXT_VOCAB_SIZE` | `155776` |
| `FAST_VOCAB_SIZE` | `4096` |
| `CODEBOOK_EMBEDDING_SIZE` | `40960` |

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
cargo test -p fish_s2_infer registry::tests::validates_local_transformer_registry -- --ignored
```

Next Phase 4 work:

- Confirm attention head layout, GQA split, RoPE base, and cache layout from
  `s2.cpp`.
- Add typed tensor views for F16 weights.
- Implement a CPU-only single-layer or single-token forward smoke before trying
  full prefill/decode.
