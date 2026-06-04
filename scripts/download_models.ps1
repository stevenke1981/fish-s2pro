# Download Fish Audio S2 Pro assets into project models/ (default model directory).
#
# Usage:
#   .\scripts\download_models.ps1
#   .\scripts\download_models.ps1 -IncludeGguf -Quant f16
#
# Requires: huggingface-cli (pip install huggingface_hub) or git lfs

param(
    [switch] $IncludeGguf,
    [ValidateSet("f16", "f32")]
    [string] $Quant = "f16"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$models = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $models | Out-Null

function Require-HfCli {
    if (-not (Get-Command huggingface-cli -ErrorAction SilentlyContinue)) {
        Write-Error "huggingface-cli not found. Install: pip install -U huggingface_hub"
    }
}

Require-HfCli

Write-Host "Downloading tokenizer.json -> models/"
huggingface-cli download fishaudio/s2-pro tokenizer.json --local-dir $models

if ($IncludeGguf) {
    $repo = "mach9243/s2-pro-gguf"
    Write-Host "Downloading GGUF pair ($Quant) from $repo ..."
    huggingface-cli download $repo --include "*$Quant*" --local-dir $models
} else {
    Write-Host "Skipped GGUF (pass -IncludeGguf to download mach9243/s2-pro-gguf)."
}

Write-Host ""
Write-Host "Done. Place or verify in:"
Write-Host "  $models"
Write-Host "  tokenizer.json + *-transformer-only.gguf + *-codec-only.gguf"
Write-Host ""
Write-Host "Start GUI: cargo run -p fish_s2_gui"