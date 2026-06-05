param(
    [string] $CodesDumpDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2cpp_generated_codes_dump"),
    [string] $WaveformDumpDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2cpp_waveform_dump"),
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Codec = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-codec-only.gguf"),
    [string] $Tokenizer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\tokenizer.json"),
    [string] $Text = "hi",
    [int] $MaxNewTokens = 1,
    [float] $Temperature = 0,
    [float] $TopP = 1,
    [int] $TopK = 0,
    [int] $MinTokensBeforeEnd = 0,
    [int] $Seed = 0,
    [int] $Threads = 4,
    [string] $BuildType = "Release",
    [switch] $Cuda,
    [switch] $BuildOnly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $root "output"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Test-RequiredFile {
    param(
        [string] $Label,
        [string] $Path
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

function Invoke-Checked {
    param(
        [scriptblock] $Command,
        [string] $Label
    )
    Write-Host $Label
    & $Command
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Test-RequiredFile "transformer GGUF" $Transformer
Test-RequiredFile "codec GGUF" $Codec
Test-RequiredFile "tokenizer" $Tokenizer

$tag = $Text -replace '[^a-zA-Z0-9]+', '_'
if (-not $tag) {
    $tag = "text"
}

$generatedCpp = Join-Path $outDir "generated_codes_${tag}_cpp.json"
$e2eRustJson = Join-Path $outDir "e2e_wav_${tag}_rust.json"
$e2eRustCodes = Join-Path $outDir "e2e_codes_${tag}_rust.json"
$e2eRustWav = Join-Path $outDir "e2e_${tag}_rust.wav"

$waveformTag = "${tag}_cpp"
$waveformCppJson = Join-Path $outDir "waveform_${waveformTag}_cpp.json"
$waveformCppWav = Join-Path $outDir "waveform_${waveformTag}_cpp.wav"

$generatedArgs = @{
    DumpDir = $CodesDumpDir
    Transformer = $Transformer
    Tokenizer = $Tokenizer
    Text = $Text
    MaxNewTokens = $MaxNewTokens
    Temperature = $Temperature
    TopP = $TopP
    TopK = $TopK
    MinTokensBeforeEnd = $MinTokensBeforeEnd
    Seed = $Seed
    Threads = $Threads
    BuildType = $BuildType
}
if ($Cuda) {
    $generatedArgs["Cuda"] = $true
}
if ($BuildOnly) {
    $generatedArgs["BuildOnly"] = $true
}

& (Join-Path $PSScriptRoot "dump_generated_codes_parity.ps1") @generatedArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($BuildOnly) {
    exit 0
}

$waveformArgs = @{
    DumpDir = $WaveformDumpDir
    Codec = $Codec
    Codes = $generatedCpp
    Threads = $Threads
    BuildType = $BuildType
}
& (Join-Path $PSScriptRoot "dump_waveform_parity.ps1") @waveformArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Invoke-Checked -Label "running RustPipeline E2E WAV dump" -Command {
    cargo run --release -p fish_s2_infer --bin fish_s2_e2e_wav_dump -- `
        --transformer $Transformer `
        --codec $Codec `
        --tokenizer $Tokenizer `
        --text $Text `
        --output $e2eRustJson `
        --codes $e2eRustCodes `
        --wav $e2eRustWav `
        --max-new-tokens $MaxNewTokens `
        --temperature $Temperature `
        --top-p $TopP `
        --top-k $TopK `
        --min-tokens-before-end $MinTokensBeforeEnd `
        --seed $Seed
}

Invoke-Checked -Label "comparing C++ generated codes vs RustPipeline codes" -Command {
    cargo run -q -p fish_s2_parity -- compare-generated-codes $generatedCpp $e2eRustCodes
}

Invoke-Checked -Label "comparing C++ waveform stats vs RustPipeline stats" -Command {
    cargo run -q -p fish_s2_parity -- compare-waveform $waveformCppJson $e2eRustJson
}

Invoke-Checked -Label "comparing C++ waveform WAV vs RustPipeline WAV" -Command {
    cargo run -q -p fish_s2_parity -- compare $waveformCppWav $e2eRustWav
}

Write-Host "E2E WAV parity OK:"
Write-Host "  codes:    $generatedCpp vs $e2eRustCodes"
Write-Host "  waveform: $waveformCppJson vs $e2eRustJson"
Write-Host "  wav:      $waveformCppWav vs $e2eRustWav"
