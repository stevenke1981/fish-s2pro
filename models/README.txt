Place Fish Audio S2 Pro assets here (project default models directory):

  models/
    tokenizer.json                    <- from fishaudio/s2-pro
    s2-pro-*-transformer-only.gguf    <- e.g. mach9243/s2-pro-gguf
    s2-pro-*-codec-only.gguf
    reference.wav / reference.txt     <- optional voice clone (server startup)

Or a unified s2-pro-f16.gguf (rodrigomt/s2-pro-gguf).

Official checkpoint (Safetensors/BF16) for GGUF export:
  models/s2-pro/config.json + model*.safetensors + codec.pth

Important:
  fishaudio/s2-pro is the official source model and tokenizer repository.
  mach9243/s2-pro-gguf is the GGUF runtime pair currently consumed by this
  workspace's Rust GGUF loader and s2.cpp-compatible fallback. Direct Rust
  inference from official Safetensors is not implemented yet; download it when
  you need tokenizer/source weights or a conversion input.

Download helper:
  .\scripts\download_models.ps1
  .\scripts\download_models.ps1 -IncludeOfficialCheckpoint -DryRun
  .\scripts\download_models.ps1 -IncludeGguf -Quant f16
  .\scripts\download_models.ps1 -IncludeGguf -Quant q4_k_m -DryRun
