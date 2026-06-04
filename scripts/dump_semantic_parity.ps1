param(
    [string] $S2CppDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2.cpp-src"),
    [string] $DumpDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2cpp_slow_ar_dump"),
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Tokenizer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\tokenizer.json"),
    [string] $Text = "hi",
    [int] $MaxNewTokens = 4,
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

$buildDir = if ($Cuda) {
    Join-Path $DumpDir "build-cuda"
} else {
    Join-Path $DumpDir "build-cpu"
}

$cudaFlag = if ($Cuda) { "-DGGML_CUDA=ON" } else { "-DGGML_CUDA=OFF" }
if ($Cuda) {
    $cudaArchFlag = "-DCMAKE_CUDA_ARCHITECTURES=86"
} else {
    $cudaArchFlag = $null
}
$cmakeArgs = @(
    "-S", $DumpDir,
    "-B", $buildDir,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    $cudaFlag
)
if ($cudaArchFlag) { $cmakeArgs += $cudaArchFlag }
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $buildDir --config $BuildType --target s2_semantic_dump --parallel

$cppExe = Join-Path $buildDir "$BuildType\s2_semantic_dump.exe"
if (-not (Test-Path -LiteralPath $cppExe)) {
    $cppExe = Join-Path $buildDir "s2_semantic_dump.exe"
}
if (-not (Test-Path -LiteralPath $cppExe)) {
    throw "s2_semantic_dump not found under $buildDir"
}

$tag = $Text -replace '[^a-zA-Z0-9]+', '_'
$cppJson = Join-Path $outDir "semantic_tokens_${tag}_cpp.json"
$rustJson = Join-Path $outDir "semantic_tokens_${tag}_rust.json"

& $cppExe `
    --transformer $Transformer `
    --tokenizer $Tokenizer `
    --output $cppJson `
    --text $Text `
    --max-new-tokens $MaxNewTokens `
    --temperature $Temperature `
    --top-p $TopP `
    --top-k $TopK `
    --min-tokens-before-end $MinTokensBeforeEnd `
    --threads $Threads

if ($BuildOnly) { exit 0 }

cargo run --release -p fish_s2_infer --bin fish_s2_semantic_dump -- `
    --transformer $Transformer `
    --tokenizer $Tokenizer `
    --output $rustJson `
    --text $Text `
    --max-new-tokens $MaxNewTokens `
    --temperature $Temperature `
    --top-p $TopP `
    --top-k $TopK `
    --min-tokens-before-end $MinTokensBeforeEnd `
    --seed $Seed

cargo run -p fish_s2_parity -- compare-semantic-tokens $cppJson $rustJson
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "semantic token parity OK: $cppJson vs $rustJson"