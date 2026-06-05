param(
    [string] $DistDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "dist\fish-s2pro-mvp"),
    [switch] $SkipBuild,
    [switch] $RunVerify,
    [switch] $Archive
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = [System.IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$distRoot = [System.IO.Path]::GetFullPath((Join-Path $root "dist"))
$distDirFull = [System.IO.Path]::GetFullPath($DistDir)
$exeSuffix = if ($IsWindows -or $env:OS -match "Windows") { ".exe" } else { "" }

function Invoke-Checked {
    param(
        [scriptblock] $Command,
        [string] $Label
    )
    Write-Host "==> $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Assert-InsideDistRoot {
    param([string] $Path)

    $full = [System.IO.Path]::GetFullPath($Path)
    $rootWithSep = $distRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
        [System.IO.Path]::DirectorySeparatorChar
    if (-not ($full.StartsWith($rootWithSep, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Refusing to modify path outside dist root: $full"
    }
}

function Copy-RequiredFile {
    param(
        [string] $Source,
        [string] $Destination
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Required file not found: $Source"
    }
    $parent = Split-Path -Parent $Destination
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Get-GitValue {
    param([string[]] $GitArgs)

    try {
        $value = & git @GitArgs 2>$null
        if ($LASTEXITCODE -eq 0) {
            return ($value -join "`n").Trim()
        }
    } catch {
    }
    return $null
}

Assert-InsideDistRoot $distDirFull

if ($RunVerify) {
    Invoke-Checked -Label "running MVP fast gate" -Command {
        & (Join-Path $PSScriptRoot "verify_mvp.ps1")
    }
}

if (-not $SkipBuild) {
    Invoke-Checked -Label "building release GUI" -Command {
        cargo build --release -p fish_s2_gui
    }
    Invoke-Checked -Label "building release server" -Command {
        cargo build --release -p fish_s2_infer --bin fish_s2_server
    }
}

if (Test-Path -LiteralPath $distDirFull) {
    Remove-Item -LiteralPath $distDirFull -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $distDirFull | Out-Null

$binDir = Join-Path $distDirFull "bin"
$scriptsDir = Join-Path $distDirFull "scripts"
$docsDir = Join-Path $distDirFull "docs"
$modelsDir = Join-Path $distDirFull "models"
New-Item -ItemType Directory -Force -Path $binDir, $scriptsDir, $docsDir, $modelsDir | Out-Null

$releaseDir = Join-Path $root "target\release"
Copy-RequiredFile (Join-Path $releaseDir "fish-s2pro$exeSuffix") (Join-Path $binDir "fish-s2pro$exeSuffix")
Copy-RequiredFile (Join-Path $releaseDir "fish_s2_server$exeSuffix") (Join-Path $binDir "fish_s2_server$exeSuffix")

Copy-RequiredFile (Join-Path $root "README.md") (Join-Path $distDirFull "README.md")
Copy-RequiredFile (Join-Path $root "README.zh-TW.md") (Join-Path $distDirFull "README.zh-TW.md")
Copy-RequiredFile (Join-Path $root "docs\THIRD_PARTY_NOTICES.md") (Join-Path $docsDir "THIRD_PARTY_NOTICES.md")
Copy-RequiredFile (Join-Path $root "models\README.txt") (Join-Path $modelsDir "README.txt")
Copy-RequiredFile (Join-Path $root "scripts\download_models.ps1") (Join-Path $scriptsDir "download_models.ps1")
Copy-RequiredFile (Join-Path $root "scripts\Use-UnicodeEncoding.ps1") (Join-Path $scriptsDir "Use-UnicodeEncoding.ps1")

$runServerScript = @'
param(
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Codec = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-codec-only.gguf"),
    [int] $Port = 8081,
    [int] $MaxNewTokens = 1,
    [string] $Backend = "rust-pure"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root "bin\fish_s2_server.exe"
if (-not (Test-Path -LiteralPath $server)) { throw "server binary not found: $server" }
if (-not (Test-Path -LiteralPath $Transformer)) { throw "transformer GGUF not found: $Transformer" }
if (-not (Test-Path -LiteralPath $Codec)) { throw "codec GGUF not found: $Codec" }

& $server `
    --transformer $Transformer `
    --codec $Codec `
    --backend $Backend `
    --max-new-tokens $MaxNewTokens `
    --port $Port
'@
Write-Utf8NoBom (Join-Path $scriptsDir "run_server.ps1") ($runServerScript + "`n")

$smokeScript = @'
param(
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Codec = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-codec-only.gguf"),
    [string] $Text = "hi",
    [int] $Port = 18081,
    [int] $MaxNewTokens = 1,
    [string] $OutputWav = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\server_smoke_hi.wav"),
    [int] $StartupTimeoutSeconds = 180,
    [int] $RequestTimeoutSeconds = 1200
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = Split-Path $PSScriptRoot -Parent
$serverExe = Join-Path $root "bin\fish_s2_server.exe"
$outDir = Split-Path -Parent $OutputWav
$logDir = Join-Path $root "output"
$stdout = Join-Path $logDir "server_smoke_stdout.txt"
$stderr = Join-Path $logDir "server_smoke_stderr.txt"
$server = $null

function Test-RequiredFile {
    param([string] $Label, [string] $Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label not found: $Path" }
}

function Stop-SmokeServer {
    if ($script:server -and -not $script:server.HasExited) {
        Stop-Process -Id $script:server.Id -Force -ErrorAction SilentlyContinue
    }
}

function Assert-WavFile {
    param([string] $Path)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 44) { throw "WAV too small: $($bytes.Length) bytes" }
    $riff = [System.Text.Encoding]::ASCII.GetString($bytes, 0, 4)
    $wave = [System.Text.Encoding]::ASCII.GetString($bytes, 8, 4)
    if ($riff -ne "RIFF" -or $wave -ne "WAVE") {
        throw "invalid WAV header: riff=$riff wave=$wave"
    }
    Write-Host "wav_bytes=$($bytes.Length)"
    Write-Host "wav_header=RIFF/WAVE"
}

try {
    Test-RequiredFile "server binary" $serverExe
    Test-RequiredFile "transformer GGUF" $Transformer
    Test-RequiredFile "codec GGUF" $Codec
    New-Item -ItemType Directory -Force -Path $outDir, $logDir | Out-Null
    Remove-Item -LiteralPath $stdout, $stderr, $OutputWav -ErrorAction SilentlyContinue

    $script:server = Start-Process `
        -FilePath $serverExe `
        -ArgumentList @(
            "--transformer", $Transformer,
            "--codec", $Codec,
            "--backend", "rust-pure",
            "--max-new-tokens", $MaxNewTokens.ToString(),
            "--port", $Port.ToString()
        ) `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden

    $healthUrl = "http://127.0.0.1:$Port/health"
    $ready = $false
    for ($i = 0; $i -lt $StartupTimeoutSeconds; $i++) {
        Start-Sleep -Seconds 1
        if ($script:server.HasExited) {
            if (Test-Path -LiteralPath $stderr) { Get-Content -Path $stderr -Tail 80 }
            throw "fish_s2_server exited with code $($script:server.ExitCode)"
        }
        try {
            $health = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
            if ($health.StatusCode -eq 200) {
                Write-Host "server_ready_seconds=$($i + 1)"
                $ready = $true
                break
            }
        } catch {
        }
    }
    if (-not $ready) {
        if (Test-Path -LiteralPath $stderr) { Get-Content -Path $stderr -Tail 80 }
        throw "server did not become ready within $StartupTimeoutSeconds seconds"
    }

    $body = @{ text = $Text; format = "wav" } | ConvertTo-Json -Compress
    Invoke-WebRequest `
        -Uri "http://127.0.0.1:$Port/v1/tts" `
        -Method POST `
        -ContentType "application/json; charset=utf-8" `
        -Body $body `
        -OutFile $OutputWav `
        -TimeoutSec $RequestTimeoutSeconds `
        -UseBasicParsing

    Assert-WavFile $OutputWav
    Write-Host "output_wav=$OutputWav"
} finally {
    Stop-SmokeServer
}
'@
Write-Utf8NoBom (Join-Path $scriptsDir "smoke_server.ps1") ($smokeScript + "`n")

$packageReadme = @"
# Fish S2 Pro Rust MVP Package

This package contains the RustPure MVP binaries and support scripts.

## Layout

- bin/fish-s2pro${exeSuffix}: Windows desktop GUI.
- bin/fish_s2_server${exeSuffix}: RustPure HTTP server for /v1/tts.
- models/: put tokenizer.json and the transformer-only + codec-only GGUF pair here.
- scripts/: model download, packaged server launch, and smoke helpers.
- docs/THIRD_PARTY_NOTICES.md: upstream model/license notes.

## Quick Start

1. Download model assets with scripts/download_models.ps1 or copy them into models/.
2. Run bin/fish-s2pro$exeSuffix for the GUI.
3. Or run scripts/run_server.ps1 -MaxNewTokens 1 -Port 8081.
4. For a short HTTP smoke, run scripts/smoke_server.ps1 -MaxNewTokens 1.

The package intentionally does not include model weights or tokenizer assets.
"@
Write-Utf8NoBom (Join-Path $distDirFull "PACKAGE_README.md") ($packageReadme + "`n")

$manifest = [ordered]@{
    schema = "fish-s2pro.mvp-package.v1"
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    git_commit = Get-GitValue -GitArgs @("rev-parse", "HEAD")
    git_branch = Get-GitValue -GitArgs @("branch", "--show-current")
    package_dir = $distDirFull
    binaries = @(
        "bin/fish-s2pro$exeSuffix",
        "bin/fish_s2_server$exeSuffix"
    )
    scripts = @(
        "scripts/download_models.ps1",
        "scripts/run_server.ps1",
        "scripts/smoke_server.ps1"
    )
    model_assets_included = $false
    notes = @(
        "RustPure is the default backend.",
        "Legacy external s2.exe subprocess backend requires the legacy-s2-exe feature and is not included in this package."
    )
}
Write-Utf8NoBom (Join-Path $distDirFull "manifest.json") (($manifest | ConvertTo-Json -Depth 6) + "`n")

if ($Archive) {
    $zipPath = "$distDirFull.zip"
    Assert-InsideDistRoot $zipPath
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $distDirFull "*") -DestinationPath $zipPath -Force
    Write-Host "archive=$zipPath"
}

Write-Host "package=$distDirFull"
