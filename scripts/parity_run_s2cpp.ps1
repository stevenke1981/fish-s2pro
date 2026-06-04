param(
    [string] $S2Bin,
    [string] $ModelsDir,
    [string] $OutputDir,
    [string] $WorkDir,
    [string] $Text,
    [string] $ReferenceWav,
    [string] $ReferenceText,
    [int] $VulkanDevice = 0,
    [int] $CodecVulkanDevice = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $ModelsDir) { $ModelsDir = Join-Path $root "models" }
if (-not $OutputDir) { $OutputDir = Join-Path $root "output" }
if (-not $WorkDir) { $WorkDir = Join-Path $OutputDir "s2cpp_work" }
if (-not $Text) {
    $promptFile = Join-Path $root "crates\fish_s2_parity\tests\fixtures\prompts.txt"
    $Text = (Get-Content -LiteralPath $promptFile -TotalCount 1)
}

function Resolve-Executable {
    param([string] $Explicit)
    if ($Explicit) {
        if (-not (Test-Path -LiteralPath $Explicit)) {
            throw "s2 binary not found: $Explicit"
        }
        return (Resolve-Path -LiteralPath $Explicit).Path
    }
    $candidates = @(
        (Join-Path $root "bin\s2.exe"),
        (Join-Path $root "bin\s2"),
        "s2.exe",
        "s2"
    )
    foreach ($candidate in $candidates) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "s2 executable not found. Pass -S2Bin or place it in bin\."
}

function Resolve-Model {
    param([string] $Pattern)
    $matches = Get-ChildItem -LiteralPath $ModelsDir -Filter $Pattern -File |
        Sort-Object Name
    if (-not $matches) {
        throw "model file not found in $ModelsDir matching $Pattern"
    }
    return $matches[0].FullName
}

$s2 = Resolve-Executable $S2Bin
$transformer = Resolve-Model "*-transformer-only.gguf"
$codec = Resolve-Model "*-codec-only.gguf"
$tokenizer = Join-Path $ModelsDir "tokenizer.json"
if (-not (Test-Path -LiteralPath $tokenizer)) {
    throw "tokenizer not found: $tokenizer"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Copy-Item -LiteralPath $tokenizer -Destination (Join-Path $WorkDir "tokenizer.json") -Force

if (-not $ReferenceWav) {
    $candidate = Join-Path $ModelsDir "reference.wav"
    if (Test-Path -LiteralPath $candidate) { $ReferenceWav = $candidate }
}
if (-not $ReferenceText) {
    $candidate = Join-Path $ModelsDir "reference.txt"
    if (Test-Path -LiteralPath $candidate) { $ReferenceText = $candidate }
}
if ($ReferenceWav) {
    Copy-Item -LiteralPath $ReferenceWav -Destination (Join-Path $WorkDir "reference.wav") -Force
}
if ($ReferenceText) {
    Copy-Item -LiteralPath $ReferenceText -Destination (Join-Path $WorkDir "reference.txt") -Force
}

$golden = Join-Path $OutputDir "golden.wav"
Write-Host "Running s2.cpp baseline:"
Write-Host "  s2:          $s2"
Write-Host "  transformer: $transformer"
Write-Host "  codec:       $codec"
Write-Host "  output:      $golden"

Push-Location $WorkDir
try {
    & $s2 `
        -v $VulkanDevice `
        --codec-vulkan $CodecVulkanDevice `
        --model $transformer `
        --model-codec $codec `
        --text $Text `
        --output $golden
    if ($LASTEXITCODE -ne 0) {
        throw "s2 exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$hash = Get-FileHash -LiteralPath $golden -Algorithm SHA256
$sidecar = "$golden.sha256"
"$($hash.Hash)  golden.wav" | Set-Content -LiteralPath $sidecar -Encoding ascii

$metricsPath = "$golden.metrics.txt"
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    cargo run -q -p fish_s2_parity -- metrics $golden |
        Set-Content -LiteralPath $metricsPath -Encoding ascii
}

Write-Host "Wrote:"
Write-Host "  $golden"
Write-Host "  $sidecar"
if (Test-Path -LiteralPath $metricsPath) {
    Write-Host "  $metricsPath"
}
