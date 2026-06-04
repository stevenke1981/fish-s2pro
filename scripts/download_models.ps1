# Download Fish Audio S2 Pro assets into project models/ (default model directory).
#
# Usage:
#   .\scripts\download_models.ps1
#   .\scripts\download_models.ps1 -IncludeGguf -Quant f16
#   .\scripts\download_models.ps1 -IncludeGguf -Quant q4_k_m -DryRun
#
# Requires: hf (pip install huggingface_hub) or git lfs

param(
    [switch] $SkipTokenizer,
    [switch] $IncludeGguf,
    [ValidateSet("f16", "f32", "q8_0", "q6_k", "q5_k_m", "q4_k_m")]
    [string] $Quant = "f16",
    [string] $GgufRepo = "mach9243/s2-pro-gguf",
    [switch] $DryRun,
    [switch] $Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$models = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $models | Out-Null

function Require-HfCli {
    if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
        Write-Error "hf not found. Install: pip install -U huggingface_hub"
    }
}

Require-HfCli

function Invoke-HfDownload {
    param(
        [string[]] $DownloadArgs
    )
    $cmd = @("download") + $DownloadArgs + @("--local-dir", $models)
    if ($DryRun) { $cmd += "--dry-run" }
    if ($Force) { $cmd += "--force-download" }
    & hf @cmd
    if ($LASTEXITCODE -ne 0) {
        throw "hf download failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipTokenizer) {
    Write-Host "Downloading tokenizer.json -> models/"
    Invoke-HfDownload -DownloadArgs @("fishaudio/s2-pro", "tokenizer.json")
}

if ($IncludeGguf) {
    Write-Host "Downloading GGUF pair ($Quant) from $GgufRepo ..."
    Invoke-HfDownload -DownloadArgs @(
        $GgufRepo,
        "--include", "*$Quant*-transformer-only.gguf",
        "--include", "*$Quant*-codec-only.gguf"
    )
} else {
    Write-Host "Skipped GGUF (pass -IncludeGguf to download $GgufRepo)."
}

Write-Host ""
Write-Host "Done. Place or verify in:"
Write-Host "  $models"
Write-Host "  tokenizer.json + *-transformer-only.gguf + *-codec-only.gguf"
Write-Host ""
Write-Host "Start GUI: cargo run -p fish_s2_gui"
