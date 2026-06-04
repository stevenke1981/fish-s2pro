# S2 Pro Codec GGUF Registry

Status: 2026-06-04

This document records the first codec/RVQ slice for the pure-Rust S2 Pro path:
`CodecTensorRegistry::from_gguf(codec_gguf)` plus a codec tensor name/shape dump.

## Commands

```powershell
cargo run -q -p fish_s2_infer --bin fish_s2_codec_dump -- `
  --codec .\models\s2-pro-f16-codec-only.gguf `
  --output .\output\s2-pro-f16-codec-registry.tsv `
  --metadata-output .\output\s2-pro-f16-codec-metadata.tsv

cargo test -p fish_s2_infer codec::tests::loads_local_codec_registry_from_gguf -- --ignored --nocapture
cargo test -p fish_s2_infer codec::tests::loads_codec_f16_weights_and_runs_rvq_lookup_fixture -- --ignored --nocapture

cargo run -q -p fish_s2_infer --bin fish_s2_rvq_lookup_dump -- `
  --codec .\models\s2-pro-f16-codec-only.gguf `
  --codes .\output\generated_codes_hi_rust.json `
  --output .\output\rvq_lookup_hi_rust.json

cargo run -q -p fish_s2_infer --bin fish_s2_post_module_dump -- `
  --codec .\models\s2-pro-f16-codec-only.gguf `
  --codes .\output\generated_codes_hi_rust.json `
  --output .\output\post_module_hi_rust.json

.\scripts\dump_rvq_lookup_parity.ps1
.\scripts\dump_post_module_parity.ps1
```

## Codec Metadata

Observed from `models/s2-pro-f16-codec-only.gguf`:

| Key | Value |
|-----|-------|
| `general.architecture` | `fish-speech-codec` |
| `fish_speech.codec.sample_rate` | `44100` |
| `fish_speech.codec.hop_length` | `512` |
| `fish_speech.codec.frame_length` | `2048` |
| `fish_speech.codec.latent_dim` | `1024` |
| `fish_speech.codec.quantizer_type` | `downsample_residual_vector_quantize` |
| `fish_speech.codec.quantizer_codebook_dim` | `8` |
| `fish_speech.codec.quantizer_residual_codebooks` | `9` |
| `fish_speech.codec.quantizer_residual_codebook_size` | `1024` |
| `fish_speech.codec.quantizer_semantic_codebook_size` | `4096` |
| `fish_speech.codec.rvq_transformer.n_layer` | `8` |
| `fish_speech.codec.rvq_transformer.dim` | `1024` |
| `fish_speech.codec.rvq_transformer.feed_forward_length` | `3072` |

## Tensor Groups

`fish_s2_codec_dump` validates the codec GGUF directory without reading tensor payloads.

| Component | Count |
|-----------|------:|
| `encoder` | 128 |
| `quantizer` | 244 |
| `decoder` | 89 |
| Total | 461 |

## RVQ Entry Points

Semantic codebook:

| Tensor | Shape |
|--------|-------|
| `quantizer.semantic_quantizer.quantizers.0.in_proj.weight` | `1x1024x8` |
| `quantizer.semantic_quantizer.quantizers.0.in_proj.bias` | `8` |
| `quantizer.semantic_quantizer.quantizers.0.out_proj.weight` | `1x8x1024` |
| `quantizer.semantic_quantizer.quantizers.0.out_proj.bias` | `1024` |
| `quantizer.semantic_quantizer.quantizers.0.codebook.weight` | `8x4096` |

Residual codebooks:

| Tensor Pattern | Shape | Count |
|----------------|-------|------:|
| `quantizer.quantizer.quantizers.{0..8}.in_proj.weight` | `1x1024x8` | 9 |
| `quantizer.quantizer.quantizers.{0..8}.in_proj.bias` | `8` | 9 |
| `quantizer.quantizer.quantizers.{0..8}.out_proj.weight` | `1x8x1024` | 9 |
| `quantizer.quantizer.quantizers.{0..8}.out_proj.bias` | `1024` | 9 |
| `quantizer.quantizer.quantizers.{0..8}.codebook.weight` | `8x1024` | 9 |

## RVQ Transformer Blocks

Both `quantizer.pre_module` and `quantizer.post_module` expose:

| Tensor Pattern | Shape |
|----------------|-------|
| `{module}.freqs_cis` | `2x32x4096` |
| `{module}.causal_mask` | `4096x4096` |
| `{module}.layers.{0..7}.attention.wqkv.weight` | `1024x3072` |
| `{module}.layers.{0..7}.attention.wo.weight` | `1024x1024` |
| `{module}.layers.{0..7}.feed_forward.w1.weight` | `1024x3072` |
| `{module}.layers.{0..7}.feed_forward.w3.weight` | `1024x3072` |
| `{module}.layers.{0..7}.feed_forward.w2.weight` | `3072x1024` |
| `{module}.layers.{0..7}.ffn_norm.weight` | `1024` |
| `{module}.layers.{0..7}.attention_norm.weight` | `1024` |
| `{module}.layers.{0..7}.attention_layer_scale.gamma` | `1024` |
| `{module}.layers.{0..7}.ffn_layer_scale.gamma` | `1024` |
| `{module}.norm.weight` | `1024` |

## RVQ Lookup Smoke

Completed in Rust:

- `CodecF16Weights::from_gguf(...)` binds semantic and residual codebook/projection tensors.
- `rvq_lookup_codes(...)` maps codebook-major generated codes to per-frame 1024-d latents.
- `fish_s2_rvq_lookup_dump` wrote `output/rvq_lookup_hi_rust.json` from `generated_codes_hi_rust.json`.
- `scripts/dump_rvq_lookup_parity.ps1` builds an s2.cpp `s2_rvq_lookup_dump` helper that includes `s2_codec.cpp` in the helper translation unit and calls the internal `decode_codes_stage(...)` slice directly.

Observed smoke stats for greedy `hi`, 2 frames:

| Field | Value |
|-------|------:|
| `num_codebooks` | 10 |
| `n_frames` | 2 |
| `latent_dim` | 1024 |
| `latent_len` | 2048 |
| `latent_l2` | 84.9925936284213 |
| `latent_mean_abs` | 1.4799805433885922 |
| `latent_max_abs` | 7.035152435302734 |

Observed C++ vs Rust RVQ lookup parity for the same fixture:

| Field | Value |
|-------|------:|
| `latent_l2_delta` | 0.00000013 |
| `latent_mean_abs_delta` | 0.00000000 |
| `latent_max_abs_delta` | 0.00000000 |
| `latent_first8_mae` | 0.00000005 |

## RVQ Post-Module Smoke

Completed in Rust:

- `CodecPostModuleF16Weights::from_gguf(...)` binds the 8-layer `quantizer.post_module` transformer plus final norm.
- `forward_codec_post_module(...)` runs short-sequence causal/windowed attention with RoPE, layer scale, FFN/SwiGLU, residuals, and final RMSNorm.
- `fish_s2_post_module_dump` wrote `output/post_module_hi_rust.json` from the same generated-codes fixture.
- `scripts/dump_post_module_parity.ps1` builds an s2.cpp `s2_post_module_dump` helper and stops exactly after `build_transformer(quantizer.post_module)`.

Observed smoke stats for greedy `hi`, 2 frames:

| Field | Value |
|-------|------:|
| `n_frames` | 2 |
| `hidden_dim` | 1024 |
| `hidden_len` | 2048 |
| `hidden_l2` | 11.466411357576677 |
| `hidden_mean_abs` | 0.1456731767643955 |
| `hidden_max_abs` | 5.619481086730957 |

Observed C++ vs Rust post-module parity for the same fixture:

| Field | Value |
|-------|------:|
| `hidden_l2_delta` | 0.00007342 |
| `hidden_mean_abs_delta` | 0.00000143 |
| `hidden_max_abs_delta` | 0.00004101 |
| `hidden_first8_mae` | 0.00003685 |

## Quantizer Decode Stage (RVQ → post-module → upsample)

Completed in Rust:

- `rvq_decode_latents(...)` chains `forward_codec_post_module` + `forward_codec_upsample`.
- `fish_s2_decode_stage_dump` writes `output/decode_stage_hi_rust.json` (`2 -> 8 frames x 1024 hidden`).
- `scripts/dump_decode_stage_parity.ps1` builds `s2_decode_stage_dump` and compares against Rust via `fish_s2_parity compare-decode-stage`.

Observed C++ vs Rust decode-stage parity for greedy `hi` generated codes (`2 -> 8 frames x 1024`):

| Field | Value |
|-------|------:|
| `hidden_l2_delta` | 0.00250405 |
| `hidden_mean_abs_delta` | 0.00003651 |
| `hidden_max_abs_delta` | 0.00211334 |
| `hidden_first8_mae` | 0.00151788 |

Upsample/ConvNeXt weights use ggml `[ne0, ne1, ne2]` indexing (`i0 + i1*ne0 + i2*ne0*ne1`).

## Next Slice

- `decode_waveform(codes)` parity: `scripts/dump_waveform_parity.ps1` (greedy `hi`).
- Add reference-prompt generated-codes parity fixtures.
- `encode_reference_audio(wav)` for voice clone.
