# S2 Pro GGUF Tensor Dump

Recorded: 2026-06-04

Models:

- `models/s2-pro-f16-transformer-only.gguf`
- `models/s2-pro-f16-codec-only.gguf`

SHA256:

- `s2-pro-f16-transformer-only.gguf`: `E04E5194DAFAD1F99E3988B0F365C7DFFA679673F4E2227AFE6100B2E5825A70`
- `s2-pro-f16-codec-only.gguf`: `3258F3654F41E143FABEB6CE83ADE6A97970B5EE95FFB16BE962289D7D27B5C9`

Full local TSV dumps:

- `output/s2-pro-f16-transformer-tensors.tsv`
- `output/s2-pro-f16-codec-tensors.tsv`

The `output/` directory is intentionally ignored by git because the dumps are
generated artifacts.

## Transformer

- Architecture: `fish-speech`
- Version: `3`
- Tensor count: `358`
- Tensor data start: `27872`
- Slow-AR layer count from `layers.N.*`: `36`
- Fast-AR layer count from `fast_layers.N.*`: `4`

Key tensor shapes:

| Tensor | Type | Dimensions |
|--------|------|------------|
| `codebook_embeddings.weight` | F16 | `2560x40960` |
| `embeddings.weight` | F16 | `2560x155776` |
| `fast_embeddings.weight` | F16 | `2560x4096` |
| `fast_layers.0.attention.wqkv.weight` | F16 | `2560x6144` |
| `fast_layers.0.attention.wo.weight` | F16 | `4096x2560` |
| `fast_output.weight` | F16 | `2560x4096` |
| `layers.0.attention.q_norm.weight` | F16 | `128` |
| `layers.0.attention.wqkv.weight` | F16 | `2560x6144` |
| `layers.0.attention.wo.weight` | F16 | `4096x2560` |
| `layers.0.feed_forward.w1.weight` | F16 | `2560x9728` |
| `norm.weight` | F16 | `2560` |

## Codec

- Architecture: `fish-speech-codec`
- Version: `3`
- Tensor count: `461`
- Tensor data start: `40608`

Key tensor shapes:

| Tensor | Type | Dimensions |
|--------|------|------------|
| `encoder.block.0.conv.weight` | F16 | `7x1x64` |
| `encoder.block.4.block.5.causal_mask` | F16 | `16384x16384` |
| `encoder.block.4.block.5.layers.0.attention.wqkv.weight` | F16 | `1024x3072` |
| `quantizer.semantic_quantizer.quantizers.0.codebook.weight` | F16 | `8x4096` |
| `quantizer.quantizer.quantizers.0.codebook.weight` | F16 | `8x1024` |
| `quantizer.pre_module.layers.0.attention.wqkv.weight` | F16 | `1024x3072` |
| `quantizer.post_module.layers.0.attention.wqkv.weight` | F16 | `1024x3072` |
| `decoder.model.0.conv.weight` | F16 | `7x1024x1536` |
| `decoder.model.6.conv.weight` | F16 | `7x96x1` |

Regenerate dumps:

```powershell
cargo run -q -p fish_s2_core --bin fish_s2_gguf_dump -- .\models\s2-pro-f16-transformer-only.gguf --output .\output\s2-pro-f16-transformer-tensors.tsv
cargo run -q -p fish_s2_core --bin fish_s2_gguf_dump -- .\models\s2-pro-f16-codec-only.gguf --output .\output\s2-pro-f16-codec-tensors.tsv
```
