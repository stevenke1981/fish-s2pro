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

## Next Slice

- Add typed F16 views for the semantic and residual codebook/projection tensors.
- Implement `rvq_lookup_codes(codes) -> quantized_latents` for a tiny generated-codes fixture.
- Dump matching C++ RVQ/codebook lookup stats before porting decoder convolution/transformer math.
