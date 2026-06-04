Place Fish Audio S2 Pro assets here (project default models directory):

  models/
    tokenizer.json                    <- from fishaudio/s2-pro
    s2-pro-*-transformer-only.gguf    <- e.g. mach9243/s2-pro-gguf
    s2-pro-*-codec-only.gguf
    reference.wav / reference.txt     <- optional voice clone (server startup)

Or a unified s2-pro-f16.gguf (rodrigomt/s2-pro-gguf).

Checkpoint (Safetensors) for GGUF export:
  models/s2-pro/config.json + model*.safetensors + codec.pth