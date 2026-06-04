# Export fishaudio/s2-pro Safetensors checkpoint to GGUF using rodrigomatta/s2.cpp quantize script.
# Prerequisites: Python 3, pip install numpy torch gguf safetensors
# Usage:
#   .\scripts\convert_s2_pro_gguf.ps1 -CheckpointDir "D:\models\s2-pro" -Output "D:\models\s2-pro-f16.gguf"

param(
    [Parameter(Mandatory = $true)]
    [string] $CheckpointDir,
    [Parameter(Mandatory = $true)]
    [string] $Output,
    [string] $Script = "",
    [string] $OutDtype = "f16",
    [string] $Python = "python"
)

$codec = Join-Path $CheckpointDir "codec.pth"
if (-not (Test-Path $codec)) {
    Write-Error "codec.pth not found in $CheckpointDir"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($Script)) {
    $candidates = @(
        (Join-Path $PSScriptRoot "unified_export_gguf.py"),
        (Join-Path (Split-Path $PSScriptRoot -Parent) "s2.cpp\quantize\unified_export_gguf.py")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $Script = $c; break }
    }
}

if (-not (Test-Path $Script)) {
    Write-Error "unified_export_gguf.py not found. Clone https://github.com/rodrigomatta/s2.cpp and pass -Script."
    exit 1
}

& $Python $Script `
    --checkpoint-path $CheckpointDir `
    --codec-checkpoint-path $codec `
    --output $Output `
    --out-dtype $OutDtype

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "GGUF written to $Output"