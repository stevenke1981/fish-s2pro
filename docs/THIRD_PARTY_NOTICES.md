# Third-Party Notices

This repository contains Rust source code and local tooling for Fish Audio S2
Pro inference. The repository code is MIT licensed unless a file says
otherwise.

Model files are separate assets. Do not redistribute model weights, tokenizer
files, checkpoints, or GGUF conversions under this repository's MIT license.

## Fish Audio S2 Pro

- Source model and tokenizer: https://huggingface.co/fishaudio/s2-pro
- Upstream license file: https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md
- License name shown by Hugging Face: Fish Audio Research License

Use of the official model weights and tokenizer is governed by Fish Audio's
model license, including any research, non-commercial, redistribution, and
commercial-use restrictions.

## GGUF Runtime Pairs

- Split runtime GGUF pair used by this project: https://huggingface.co/mach9243/s2-pro-gguf
- Original/community GGUF repository referenced by the scripts: https://huggingface.co/rodrigomt/s2-pro-gguf

GGUF files are conversions/derivatives of the Fish Audio S2 Pro model assets.
Treat them as model assets governed by the upstream model license and the
metadata/license files provided by the model repositories.

## s2.cpp Reference

- Reference implementation used for parity: https://github.com/mach92432/s2.cpp

The RustPure backend no longer requires `s2.exe` or a linked C++ library for
the verified one-token server smoke path. The C++ reference remains useful for
parity scripts, optional FFI experiments, and regression checks.

## Packaging Checklist

Before distributing a binary package:

1. Include this notice file and the repository license.
2. Include or link the upstream Fish Audio Research License for model assets.
3. Keep model assets outside the MIT-licensed source package unless the
   distribution terms are separately satisfied.
4. Document which model repository and quantization were used.
