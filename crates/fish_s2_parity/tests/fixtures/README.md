# Parity Fixtures

These fixtures are intentionally text-only in git. Provide the reference audio
with `-ReferenceWav`, or place `models/reference.wav` locally before running:

```powershell
.\scripts\parity_run_s2cpp.ps1
```

The generated baseline is written to `output/golden.wav`, which is ignored by
git because it depends on local models, backend, and hardware.
