param(
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Codec = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-codec-only.gguf"),
    [string] $Text = "hi",
    [int] $Port = 18081,
    [int] $MaxNewTokens = 1,
    [string] $Backend = "rust-pure",
    [string] $ReferenceWav,
    [string] $ReferenceText,
    [string] $OutputWav = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\server_smoke_hi.wav"),
    [int] $StartupTimeoutSeconds = 180,
    [int] $RequestTimeoutSeconds = 1200
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $root "output"
$stdout = Join-Path $outDir "server_smoke_stdout.txt"
$stderr = Join-Path $outDir "server_smoke_stderr.txt"
$pidPath = Join-Path $outDir "server_smoke_pid.txt"
$workdir = Join-Path $root "runtime\s2_server"
$server = $null

function Test-RequiredFile {
    param(
        [string] $Label,
        [string] $Path
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

function Stop-SmokeServer {
    if ($script:server -and -not $script:server.HasExited) {
        Stop-Process -Id $script:server.Id -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $pidPath) {
        Remove-Item -LiteralPath $pidPath -ErrorAction SilentlyContinue
    }
}

try {
    Test-RequiredFile "transformer GGUF" $Transformer
    Test-RequiredFile "codec GGUF" $Codec
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    New-Item -ItemType Directory -Force -Path $workdir | Out-Null
    Remove-Item -LiteralPath $stdout, $stderr, $OutputWav, $pidPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $workdir "reference.wav"), (Join-Path $workdir "reference.txt") -ErrorAction SilentlyContinue

    if ($ReferenceWav -or $ReferenceText) {
        if (-not ($ReferenceWav -and $ReferenceText)) {
            throw "-ReferenceWav and -ReferenceText must both be set"
        }
        Test-RequiredFile "reference wav" $ReferenceWav
        Copy-Item -LiteralPath $ReferenceWav -Destination (Join-Path $workdir "reference.wav") -Force
        Write-Utf8NoBom (Join-Path $workdir "reference.txt") $ReferenceText
        Write-Host "using reference conditioning from $ReferenceWav"
    }

    cargo build --release -p fish_s2_infer --bin fish_s2_server
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $exe = Join-Path $root "target\release\fish_s2_server.exe"
    Test-RequiredFile "release fish_s2_server.exe" $exe

    $script:server = Start-Process `
        -FilePath $exe `
        -ArgumentList @(
            "--transformer", $Transformer,
            "--codec", $Codec,
            "--backend", $Backend,
            "--max-new-tokens", $MaxNewTokens.ToString(),
            "--port", $Port.ToString()
        ) `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden
    Write-Utf8NoBom $pidPath $script:server.Id.ToString()

    $healthUrl = "http://127.0.0.1:$Port/health"
    for ($i = 0; $i -lt $StartupTimeoutSeconds; $i++) {
        Start-Sleep -Seconds 1
        if ($script:server.HasExited) {
            if (Test-Path -LiteralPath $stderr) {
                Get-Content -Path $stderr -Tail 80
            }
            throw "fish_s2_server exited with code $($script:server.ExitCode)"
        }
        try {
            $health = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
            if ($health.StatusCode -eq 200) {
                Write-Host "server ready after $($i + 1)s"
                break
            }
        } catch {
            if ($i -eq ($StartupTimeoutSeconds - 1)) {
                if (Test-Path -LiteralPath $stderr) {
                    Get-Content -Path $stderr -Tail 80
                }
                throw "server did not become ready within $StartupTimeoutSeconds seconds"
            }
        }
    }

    $body = @{ text = $Text; format = "wav" } | ConvertTo-Json -Compress
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Invoke-WebRequest `
        -Uri "http://127.0.0.1:$Port/v1/tts" `
        -Method POST `
        -ContentType "application/json; charset=utf-8" `
        -Body $body `
        -OutFile $OutputWav `
        -TimeoutSec $RequestTimeoutSeconds `
        -UseBasicParsing
    $sw.Stop()

    $wav = Get-Item -LiteralPath $OutputWav
    Write-Host "wrote $($wav.FullName)"
    Write-Host "elapsed_seconds=$([math]::Round($sw.Elapsed.TotalSeconds, 2))"
    Write-Host "bytes=$($wav.Length)"
    cargo run -q -p fish_s2_parity -- metrics $OutputWav
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Stop-SmokeServer
}
