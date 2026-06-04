# s2.cpp Reference Pin

The parity baseline for this workspace is pinned to:

- Repository: https://github.com/mach92432/s2.cpp
- Commit: `2b9c6d2984f92b186419420e9126c47919b81d3a`
- Recorded: 2026-06-04

Verify or refresh the pin with:

```powershell
git ls-remote https://github.com/mach92432/s2.cpp.git HEAD
```

When updating this pin, regenerate `output/golden.wav` with:

```powershell
.\scripts\parity_run_s2cpp.ps1
```

Do not compare generated WAV files bit-for-bit. The project parity gate compares
duration, sample rate, RMS, and RMS envelope because floating point and GPU
backend differences can change bytes while preserving equivalent audio.
