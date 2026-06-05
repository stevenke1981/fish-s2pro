param(
    [string] $DistDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "dist\fish-s2pro-mvp"),
    [switch] $SkipBuild,
    [switch] $RunVerify,
    [switch] $Archive,
    [string] $Features = ""
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

function Copy-OptionalGgmlRuntimeDlls {
    param([string] $DestinationDir)

    $candidates = @()
    $candidates += Join-Path $root "target\release"
    if ($env:S2_CPP_DLL_DIR) {
        $candidates += $env:S2_CPP_DLL_DIR
    }
    if ($env:S2_CPP_LIB) {
        $libDir = [System.IO.Path]::GetFullPath($env:S2_CPP_LIB)
        $nativeRoot = Split-Path -Parent $libDir
        $candidates += Join-Path $nativeRoot "ggml\bin\Release"
        $candidates += Join-Path $nativeRoot "ggml\bin"
    }

    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        $dlls = @(Get-ChildItem -LiteralPath $candidate -Filter "ggml*.dll" -File -ErrorAction SilentlyContinue)
        if ($dlls.Count -eq 0) { continue }
        foreach ($dll in $dlls) {
            Copy-RequiredFile $dll.FullName (Join-Path $DestinationDir $dll.Name)
        }
        Write-Host "copied_ggml_runtime_dlls=$($dlls.Count)"
        Write-Host "ggml_runtime_dir=$candidate"
        return
    }
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

function ConvertTo-PackageRelativePath {
    param([string] $Path)

    $full = [System.IO.Path]::GetFullPath($Path)
    $prefix = $distDirFull.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
        [System.IO.Path]::DirectorySeparatorChar
    if (-not ($full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Path is outside package: $full"
    }
    return $full.Substring($prefix.Length).Replace("\", "/")
}

function Get-PackageFileEntries {
    Get-ChildItem -LiteralPath $distDirFull -Recurse -File |
        Where-Object {
            $_.Name -ne "manifest.json" -and
            $_.Name -ne "SHA256SUMS.txt"
        } |
        Sort-Object FullName |
        ForEach-Object {
            [ordered]@{
                path = ConvertTo-PackageRelativePath $_.FullName
                bytes = $_.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
}

Assert-InsideDistRoot $distDirFull

if ($RunVerify) {
    Invoke-Checked -Label "running MVP fast gate" -Command {
        & (Join-Path $PSScriptRoot "verify_mvp.ps1")
    }
}

if (-not $SkipBuild) {
    Invoke-Checked -Label "building release GUI" -Command {
        if ($Features.Trim()) {
            cargo build --release -p fish_s2_gui --features $Features
        } else {
            cargo build --release -p fish_s2_gui
        }
    }
    Invoke-Checked -Label "building release server" -Command {
        if ($Features.Trim()) {
            cargo build --release -p fish_s2_infer --bin fish_s2_server --features $Features
        } else {
            cargo build --release -p fish_s2_infer --bin fish_s2_server
        }
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
Copy-OptionalGgmlRuntimeDlls $binDir

Copy-RequiredFile (Join-Path $root "README.md") (Join-Path $distDirFull "README.md")
Copy-RequiredFile (Join-Path $root "README.zh-TW.md") (Join-Path $distDirFull "README.zh-TW.md")
Copy-RequiredFile (Join-Path $root "docs\THIRD_PARTY_NOTICES.md") (Join-Path $docsDir "THIRD_PARTY_NOTICES.md")
Copy-RequiredFile (Join-Path $root "models\README.txt") (Join-Path $modelsDir "README.txt")
Copy-RequiredFile (Join-Path $root "scripts\check_cuda_compat.ps1") (Join-Path $scriptsDir "check_cuda_compat.ps1")
Copy-RequiredFile (Join-Path $root "scripts\download_models.ps1") (Join-Path $scriptsDir "download_models.ps1")
Copy-RequiredFile (Join-Path $root "scripts\Use-UnicodeEncoding.ps1") (Join-Path $scriptsDir "Use-UnicodeEncoding.ps1")

$runServerScript = @'
param(
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Codec = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-codec-only.gguf"),
    [int] $Port = 8081,
    [int] $MaxNewTokens = 1,
    [string] $Backend = "rust-pure",
    [int] $CudaDevice = 0,
    [switch] $CodecCuda
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root "bin\fish_s2_server.exe"
if (-not (Test-Path -LiteralPath $server)) { throw "server binary not found: $server" }
if (-not (Test-Path -LiteralPath $Transformer)) { throw "transformer GGUF not found: $Transformer" }
if (-not (Test-Path -LiteralPath $Codec)) { throw "codec GGUF not found: $Codec" }

$args = @(
    "--transformer", $Transformer,
    "--codec", $Codec,
    "--backend", $Backend,
    "--cuda-device", $CudaDevice.ToString(),
    "--max-new-tokens", $MaxNewTokens.ToString(),
    "--port", $Port.ToString()
)
if ($CodecCuda) {
    $args += "--codec-cuda"
}

& $server @args
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

$verifyPackageScript = @'
param(
    [switch] $SkipServerHelp
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = Split-Path $PSScriptRoot -Parent
$manifestPath = Join-Path $root "manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "manifest not found: $manifestPath"
}

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if ($manifest.schema -ne "fish-s2pro.mvp-package.v1") {
    throw "unexpected manifest schema: $($manifest.schema)"
}

$checksumPath = Join-Path $root $manifest.checksum_file
if (-not (Test-Path -LiteralPath $checksumPath)) {
    throw "checksum file not found: $checksumPath"
}
$checksumLines = @(Get-Content -LiteralPath $checksumPath | Where-Object { $_.Trim() })
$manifestChecksumLines = @($manifest.files | ForEach-Object { "$($_.sha256)  $($_.path)" })
if ($checksumLines.Count -ne $manifestChecksumLines.Count) {
    throw "checksum count mismatch: expected $($manifestChecksumLines.Count), got $($checksumLines.Count)"
}
for ($i = 0; $i -lt $manifestChecksumLines.Count; $i++) {
    if ($checksumLines[$i] -ne $manifestChecksumLines[$i]) {
        throw "checksum line mismatch at $i"
    }
}

$checked = 0
foreach ($file in @($manifest.files)) {
    $path = Join-Path $root ($file.path -replace '/', [System.IO.Path]::DirectorySeparatorChar)
    if (-not (Test-Path -LiteralPath $path)) {
        throw "missing packaged file: $($file.path)"
    }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -ne [int64] $file.bytes) {
        throw "size mismatch for $($file.path): expected $($file.bytes), got $($item.Length)"
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($hash -ne $file.sha256) {
        throw "sha256 mismatch for $($file.path)"
    }
    $checked += 1
}

$forbidden = Get-ChildItem -LiteralPath (Join-Path $root "models") -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -eq "tokenizer.json" -or
        $_.Extension -in @(".gguf", ".safetensors", ".pth", ".bin")
    }
if ($forbidden) {
    $names = ($forbidden | Select-Object -ExpandProperty FullName) -join "; "
    throw "package unexpectedly includes model assets: $names"
}

foreach ($script in @("check_cuda_compat.ps1", "run_server.ps1", "smoke_server.ps1", "verify_package.ps1")) {
    $path = Join-Path $PSScriptRoot $script
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($path, [ref] $tokens, [ref] $errors) | Out-Null
    if ($errors.Count -gt 0) {
        $messages = ($errors | ForEach-Object { $_.Message }) -join "; "
        throw "PowerShell parse errors in ${script}: $messages"
    }
}

if (-not $SkipServerHelp) {
    $server = Join-Path $root "bin\fish_s2_server.exe"
    if (-not (Test-Path -LiteralPath $server)) {
        throw "server binary not found: $server"
    }
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $help = & $server --help 2>&1
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($LASTEXITCODE -ne 0) {
        throw "server --help failed with exit code $LASTEXITCODE"
    }
    $helpText = $help -join "`n"
    if ($helpText -notmatch "rust-pure\|ffi") {
        throw "server help does not advertise rust-pure|ffi"
    }

    $oldFishRoot = $env:FISH_S2PRO_ROOT
    Remove-Item Env:FISH_S2PRO_ROOT -ErrorAction SilentlyContinue
    try {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $paths = & $server --print-paths 2>&1
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
        if ($LASTEXITCODE -ne 0) {
            throw "server --print-paths failed with exit code $LASTEXITCODE"
        }
        $pathText = $paths -join "`n"
        $expectedRoot = [regex]::Escape($root)
        $expectedModels = [regex]::Escape((Join-Path $root "models"))
        if ($pathText -notmatch "project_root=$expectedRoot" -or $pathText -notmatch "models_dir=$expectedModels") {
            throw "server --print-paths did not use package root"
        }
    } finally {
        if ($null -ne $oldFishRoot) {
            $env:FISH_S2PRO_ROOT = $oldFishRoot
        }
    }
}

Write-Host "package_verified=true"
Write-Host "checked_files=$checked"
'@
Write-Utf8NoBom (Join-Path $scriptsDir "verify_package.ps1") ($verifyPackageScript + "`n")

$packageReadme = @"
# Fish S2 Pro Rust MVP Package

This package contains the RustPure MVP binaries and optional cpp-engine/ffi-cuda runtime files when built with -Features cpp-engine.

## Layout

- bin/fish-s2pro${exeSuffix}: Windows desktop GUI.
- bin/fish_s2_server${exeSuffix}: HTTP server for /v1/tts; supports rust-pure and, when linked, ffi/ffi-cuda.
- bin/ggml*.dll: optional GGML runtime DLLs copied from S2_CPP_LIB for cpp-engine/CUDA packages.
- models/: put tokenizer.json and the transformer-only + codec-only GGUF pair here.
- scripts/: model download, packaged server launch, and smoke helpers.
- scripts/check_cuda_compat.ps1: CUDA/NVIDIA toolkit compatibility report.
- docs/THIRD_PARTY_NOTICES.md: upstream model/license notes.
- manifest.json and SHA256SUMS.txt: package inventory and checksums.

## Quick Start

1. Download model assets with scripts/download_models.ps1 or copy them into models/.
2. Run bin/fish-s2pro$exeSuffix for the GUI.
3. Or run scripts/run_server.ps1 -MaxNewTokens 1 -Port 8081.
4. For CUDA FFI builds, run scripts/run_server.ps1 -Backend ffi-cuda -CudaDevice 0.
5. Experimental codec CUDA is opt-in: scripts/run_server.ps1 -Backend ffi-cuda -CudaDevice 0 -CodecCuda.
6. For a short HTTP smoke, run scripts/smoke_server.ps1 -MaxNewTokens 1.
7. Validate the package files with scripts/verify_package.ps1.
8. Diagnose package paths with bin/fish_s2_server$exeSuffix --print-paths.
9. Check CUDA compatibility with scripts/check_cuda_compat.ps1.

The package intentionally does not include model weights or tokenizer assets.
"@
Write-Utf8NoBom (Join-Path $distDirFull "PACKAGE_README.md") ($packageReadme + "`n")

$fileEntries = @(Get-PackageFileEntries)
$checksumLines = $fileEntries | ForEach-Object { "$($_.sha256)  $($_.path)" }
Write-Utf8NoBom (Join-Path $distDirFull "SHA256SUMS.txt") (($checksumLines -join "`n") + "`n")

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
        "scripts/check_cuda_compat.ps1",
        "scripts/download_models.ps1",
        "scripts/run_server.ps1",
        "scripts/smoke_server.ps1",
        "scripts/verify_package.ps1"
    )
    model_assets_included = $false
    checksum_file = "SHA256SUMS.txt"
    files = $fileEntries
    notes = @(
        "RustPure is the default backend.",
        "Legacy external s2.exe subprocess backend requires the legacy-s2-exe feature and is not included in this package."
    )
}
Write-Utf8NoBom (Join-Path $distDirFull "manifest.json") (($manifest | ConvertTo-Json -Depth 6) + "`n")

Invoke-Checked -Label "verifying packaged files" -Command {
    & (Join-Path $scriptsDir "verify_package.ps1")
}

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
