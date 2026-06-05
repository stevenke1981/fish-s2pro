param(
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Codec = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-codec-only.gguf"),
    [string] $Tokenizer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\tokenizer.json"),
    [string] $Text = "hi",
    [int] $MaxNewTokens = 1,
    [int] $Port = 18081,
    [switch] $RunServerSmoke,
    [switch] $CheckCuda,
    [switch] $RequireCuda,
    [string] $OutputWav = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\mvp_server_smoke.wav"),
    [string] $Report = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\mvp_report.json")
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $root "output"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$checks = [System.Collections.Generic.List[object]]::new()
$serverSmoke = [ordered]@{
    status = $(if ($RunServerSmoke) { "pending" } else { "skipped" })
    output_wav = $OutputWav
    metrics = $null
}
$cudaCompat = [ordered]@{
    status = $(if ($CheckCuda) { "pending" } else { "skipped" })
    report = (Join-Path $outDir "cuda_compat_report.json")
}

function Test-RequiredFile {
    param(
        [string] $Label,
        [string] $Path
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

function Convert-KeyValueLines {
    param([string[]] $Lines)

    $values = [ordered]@{}
    foreach ($line in $Lines) {
        if ($line -match '^([^=]+)=(.+)$') {
            $key = $matches[1]
            $raw = $matches[2]
            $number = 0.0
            if ([double]::TryParse(
                    $raw,
                    [System.Globalization.NumberStyles]::Float,
                    [System.Globalization.CultureInfo]::InvariantCulture,
                    [ref] $number
                )) {
                $values[$key] = $number
            } else {
                $values[$key] = $raw
            }
        }
    }
    return $values
}

function Save-MvpReport {
    param([string] $Status)

    $reportObject = [ordered]@{
        schema = "fish-s2pro.mvp-report.v1"
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        status = $Status
        root = $root
        model_paths = [ordered]@{
            transformer = $Transformer
            codec = $Codec
            tokenizer = $Tokenizer
        }
        smoke = [ordered]@{
            run_server_smoke = [bool] $RunServerSmoke
            text = $Text
            max_new_tokens = $MaxNewTokens
            port = $Port
        }
        checks = @($checks)
        server_smoke = $serverSmoke
        cuda_compat = $cudaCompat
    }

    $json = $reportObject | ConvertTo-Json -Depth 8
    Write-Utf8NoBom $Report ($json + "`n")
}

function Invoke-MvpStep {
    param(
        [string] $Name,
        [scriptblock] $Command
    )

    Write-Host "==> $Name"
    $step = [ordered]@{
        name = $Name
        status = "running"
        started_at = (Get-Date).ToUniversalTime().ToString("o")
        duration_seconds = 0
        exit_code = $null
        output_tail = @()
        error = $null
    }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $captured = [System.Collections.Generic.List[string]]::new()

    try {
        $global:LASTEXITCODE = 0
        & $Command 2>&1 | ForEach-Object {
            $line = $_.ToString()
            $captured.Add($line)
            Write-Host $line
        }
        $exitCode = if ($null -ne $LASTEXITCODE) { [int] $LASTEXITCODE } else { 0 }
        $step.exit_code = $exitCode
        if ($exitCode -ne 0) {
            throw "$Name failed with exit code $exitCode"
        }
        $step.status = "passed"
    } catch {
        $step.status = "failed"
        $step.error = $_.Exception.Message
        throw
    } finally {
        $sw.Stop()
        $step.duration_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        $tailStart = [Math]::Max(0, $captured.Count - 80)
        $tail = @()
        for ($i = $tailStart; $i -lt $captured.Count; $i++) {
            $tail += $captured[$i]
        }
        $step.output_tail = $tail
        $checks.Add([pscustomobject] $step) | Out-Null
    }
}

function Test-ScriptParses {
    param([string] $Path)

    $tokens = $null
    $parseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref] $tokens, [ref] $parseErrors) | Out-Null
    if ($parseErrors.Count -gt 0) {
        $messages = $parseErrors | ForEach-Object { $_.Message }
        throw "PowerShell parse errors in ${Path}: $($messages -join '; ')"
    }
}

try {
    Invoke-MvpStep "model assets present" {
        Test-RequiredFile "transformer GGUF" $Transformer
        Test-RequiredFile "codec GGUF" $Codec
        Test-RequiredFile "tokenizer" $Tokenizer
    }
    Invoke-MvpStep "PowerShell scripts parse" {
        Test-ScriptParses (Join-Path $PSScriptRoot "verify_mvp.ps1")
        Test-ScriptParses (Join-Path $PSScriptRoot "smoke_rust_server.ps1")
    }
    Invoke-MvpStep "cargo fmt --check" {
        cargo fmt --check
    }
    Invoke-MvpStep "cargo test -p fish_s2_core" {
        cargo test -p fish_s2_core
    }
    Invoke-MvpStep "cargo test -p fish_s2_infer" {
        cargo test -p fish_s2_infer
    }
    Invoke-MvpStep "cargo check -p fish_s2_gui" {
        cargo check -p fish_s2_gui
    }
    Invoke-MvpStep "cargo clippy strict workspace crates" {
        cargo clippy -p fish_s2_core -p fish_s2_infer -p fish_s2_gui --all-targets -- -D warnings
    }
    Invoke-MvpStep "fish_s2_server help" {
        cargo run -q -p fish_s2_infer --bin fish_s2_server -- --help
    }

    if ($CheckCuda) {
        $cudaCompat.status = "running"
        Invoke-MvpStep "CUDA compatibility check" {
            $cudaArgs = @{
                Report = $cudaCompat.report
            }
            if ($RequireCuda) {
                $cudaArgs["RequireCuda"] = $true
            }
            & (Join-Path $PSScriptRoot "check_cuda_compat.ps1") @cudaArgs
        }
        $cudaCompat.status = "passed"
        if (Test-Path -LiteralPath $cudaCompat.report) {
            $cudaCompat.summary = Get-Content -Raw -LiteralPath $cudaCompat.report | ConvertFrom-Json
        }
    }

    if ($RunServerSmoke) {
        $serverSmoke.status = "running"
        Invoke-MvpStep "RustPure server smoke" {
            & (Join-Path $PSScriptRoot "smoke_rust_server.ps1") `
                -Transformer $Transformer `
                -Codec $Codec `
                -Text $Text `
                -Port $Port `
                -MaxNewTokens $MaxNewTokens `
                -Backend "rust-pure" `
                -OutputWav $OutputWav
        }

        $wav = Get-Item -LiteralPath $OutputWav
        $metricLines = cargo run -q -p fish_s2_parity -- metrics $OutputWav
        if ($LASTEXITCODE -ne 0) {
            throw "fish_s2_parity metrics failed for $OutputWav"
        }
        $serverSmoke.status = "passed"
        $serverSmoke.output_wav = $wav.FullName
        $serverSmoke.bytes = $wav.Length
        $serverSmoke.metrics = Convert-KeyValueLines $metricLines
    }

    $status = if ($RunServerSmoke) { "passed" } else { "fast-passed" }
    Save-MvpReport $status
    Write-Host "MVP verification $status"
    Write-Host "report=$Report"
} catch {
    if ($RunServerSmoke -and $serverSmoke.status -eq "running") {
        $serverSmoke.status = "failed"
    }
    if ($CheckCuda -and $cudaCompat.status -eq "running") {
        $cudaCompat.status = "failed"
    }
    Save-MvpReport "failed"
    Write-Error $_
    exit 1
}
