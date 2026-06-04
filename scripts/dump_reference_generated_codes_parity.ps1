param(
    [string] $DumpDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2cpp_reference_codes_dump"),
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Codec = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-codec-only.gguf"),
    [string] $Tokenizer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\tokenizer.json"),
    [string] $Fixture = (Join-Path (Split-Path $PSScriptRoot -Parent) "crates\fish_s2_parity\tests\fixtures\reference_prompt_codes.json"),
    [string] $PromptTextFile = (Join-Path (Split-Path $PSScriptRoot -Parent) "crates\fish_s2_parity\tests\fixtures\reference.txt"),
    [string] $Text = "hi",
    [int] $MaxNewTokens = 1,
    [float] $Temperature = 0,
    [float] $TopP = 1,
    [int] $TopK = 0,
    [int] $MinTokensBeforeEnd = 0,
    [int] $Seed = 0,
    [int] $Threads = 4,
    [string] $BuildType = "Release",
    [switch] $BuildOnly,
    [switch] $SkipFixtureBuild,
    [switch] $PromptCodesOnly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $root "output"
$s2CppDir = Join-Path $outDir "s2.cpp-src"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $DumpDir | Out-Null

if (-not (Test-Path -LiteralPath $s2CppDir)) {
    throw "missing s2.cpp source tree: $s2CppDir; run scripts\dump_generated_codes_parity.ps1 once"
}

function Write-TinyReferenceWav {
    param(
        [string] $Path,
        [int] $SampleRate = 44100,
        [int] $NumSamples = 5120
    )
    $numChannels = 1
    $bitsPerSample = 16
    $blockAlign = $numChannels * ($bitsPerSample / 8)
    $byteRate = $SampleRate * $blockAlign
    $dataSize = $NumSamples * $blockAlign
    $stream = [System.IO.File]::Create($Path)
    try {
        $writer = New-Object System.IO.BinaryWriter($stream)
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("RIFF"))
        $writer.Write([int](36 + $dataSize))
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("WAVE"))
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("fmt "))
        $writer.Write([int]16)
        $writer.Write([Int16]1)
        $writer.Write([Int16]$numChannels)
        $writer.Write([int]$SampleRate)
        $writer.Write([int]$byteRate)
        $writer.Write([Int16]$blockAlign)
        $writer.Write([Int16]$bitsPerSample)
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("data"))
        $writer.Write([int]$dataSize)
        for ($i = 0; $i -lt $NumSamples; $i++) {
            $t = [double]$i / [double]$SampleRate
            $sample = [int]([math]::Sin(2.0 * [math]::PI * 220.0 * $t) * 8000.0)
            if ($sample -gt 32767) { $sample = 32767 }
            if ($sample -lt -32768) { $sample = -32768 }
            $writer.Write([Int16]$sample)
        }
    } finally {
        $stream.Dispose()
    }
}

function Write-EncodePromptCodesMain {
    param([string] $Path)
    $main = @'
#include "json.hpp"

#include "s2_audio.h"
#include "s2_codec.cpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

namespace {

struct Args {
    std::string codec;
    std::string wav;
    std::string prompt_text;
    std::string prompt_text_file;
    std::string output;
    int threads = 4;
};

void print_help() {
    std::cerr
        << "Usage: s2_encode_prompt_codes_dump --codec <codec.gguf> --wav <ref.wav> "
           "(--prompt-text <text> | --prompt-text-file <text.txt>) "
           "--output <prompt_codes.json> [--threads 4]\n";
}

std::string read_trimmed_text_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open prompt text file: " + path);
    }
    std::string text(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
    while (!text.empty()) {
        const char tail = text.back();
        if (tail != '\r' && tail != '\n' && tail != '\t' && tail != ' ') break;
        text.pop_back();
    }
    return text;
}

bool parse_args(int argc, char ** argv, Args & args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--codec") {
            const char * v = need_value("--codec");
            if (!v) return false;
            args.codec = v;
        } else if (arg == "--wav") {
            const char * v = need_value("--wav");
            if (!v) return false;
            args.wav = v;
        } else if (arg == "--prompt-text") {
            const char * v = need_value("--prompt-text");
            if (!v) return false;
            args.prompt_text = v;
        } else if (arg == "--prompt-text-file") {
            const char * v = need_value("--prompt-text-file");
            if (!v) return false;
            args.prompt_text_file = v;
        } else if (arg == "--output") {
            const char * v = need_value("--output");
            if (!v) return false;
            args.output = v;
        } else if (arg == "--threads") {
            const char * v = need_value("--threads");
            if (!v) return false;
            args.threads = std::stoi(v);
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (!args.prompt_text.empty() && !args.prompt_text_file.empty()) {
        std::cerr << "--prompt-text and --prompt-text-file are mutually exclusive\n";
        return false;
    }
    return !args.codec.empty() && !args.wav.empty()
        && (!args.prompt_text.empty() || !args.prompt_text_file.empty())
        && !args.output.empty();
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_help();
        return 2;
    }

    s2::AudioData audio;
    if (!s2::load_audio(args.wav, audio, 0)) {
        std::cerr << "failed to load wav: " << args.wav << "\n";
        return 1;
    }

    s2::AudioCodec codec;
    if (!codec.load(args.codec, -1)) {
        std::cerr << "failed to load codec: " << args.codec << "\n";
        return 1;
    }

    std::vector<int32_t> codes;
    int32_t n_frames = 0;
    if (!codec.encode(audio.samples.data(), static_cast<int32_t>(audio.samples.size()), args.threads,
                      codes, n_frames)) {
        std::cerr << "codec.encode failed\n";
        return 1;
    }

    json doc;
    const std::string prompt_text =
        args.prompt_text_file.empty() ? args.prompt_text : read_trimmed_text_file(args.prompt_text_file);
    doc["prompt_text"] = prompt_text;
    doc["num_codebooks"] = codec.num_codebooks();
    doc["cols"] = n_frames;
    doc["codes"] = codes;

    const std::string json_utf8 = doc.dump(2) + "\n";
    std::ofstream out(args.output, std::ios::binary);
    if (!out) {
        std::cerr << "failed to open output: " << args.output << "\n";
        return 1;
    }
    out.write(json_utf8.data(), static_cast<std::streamsize>(json_utf8.size()));
    std::cout << "wrote " << args.output << " (" << codec.num_codebooks() << " codebooks x "
              << n_frames << " prompt frames)\n";
    return 0;
}
'@
    Write-Utf8NoBom $Path $main
}

function Write-ReferenceGenerateCodesDumpMain {
    param([string] $Path)
    $main = @'
#include "s2_generate.h"
#include "s2_prompt.h"
#include "s2_tokenizer.h"

#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace {

struct Args {
    std::string transformer;
    std::string tokenizer;
    std::string text = "hi";
    std::string prompt_codes_path;
    std::string output;
    int max_new_tokens = 2;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int min_tokens_before_end = 0;
    int threads = 4;
};

void print_help() {
    std::cerr
        << "Usage: s2_generate_codes_dump --transformer <gguf> --tokenizer <tokenizer.json> "
           "--prompt-codes <prompt_codes.json> --output <codes.json> [--text hi] "
           "[--max-new-tokens 2] [--temperature 0] [--top-p 1] [--top-k 0] "
           "[--min-tokens-before-end 0] [--threads 4]\n";
}

bool parse_int(const char * label, const std::string & value, int & out) {
    try {
        out = std::stoi(value);
        return true;
    } catch (const std::exception & err) {
        std::cerr << "invalid " << label << ": " << err.what() << "\n";
        return false;
    }
}

bool parse_float(const char * label, const std::string & value, float & out) {
    try {
        out = std::stof(value);
        return true;
    } catch (const std::exception & err) {
        std::cerr << "invalid " << label << ": " << err.what() << "\n";
        return false;
    }
}

bool parse_args(int argc, char ** argv, Args & args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--transformer") {
            const char * v = need_value("--transformer");
            if (!v) return false;
            args.transformer = v;
        } else if (arg == "--tokenizer") {
            const char * v = need_value("--tokenizer");
            if (!v) return false;
            args.tokenizer = v;
        } else if (arg == "--output") {
            const char * v = need_value("--output");
            if (!v) return false;
            args.output = v;
        } else if (arg == "--text") {
            const char * v = need_value("--text");
            if (!v) return false;
            args.text = v;
        } else if (arg == "--prompt-codes") {
            const char * v = need_value("--prompt-codes");
            if (!v) return false;
            args.prompt_codes_path = v;
        } else if (arg == "--max-new-tokens") {
            const char * v = need_value("--max-new-tokens");
            if (!v || !parse_int("--max-new-tokens", v, args.max_new_tokens)) return false;
        } else if (arg == "--temperature") {
            const char * v = need_value("--temperature");
            if (!v || !parse_float("--temperature", v, args.temperature)) return false;
        } else if (arg == "--top-p") {
            const char * v = need_value("--top-p");
            if (!v || !parse_float("--top-p", v, args.top_p)) return false;
        } else if (arg == "--top-k") {
            const char * v = need_value("--top-k");
            if (!v || !parse_int("--top-k", v, args.top_k)) return false;
        } else if (arg == "--min-tokens-before-end") {
            const char * v = need_value("--min-tokens-before-end");
            if (!v || !parse_int("--min-tokens-before-end", v, args.min_tokens_before_end)) return false;
        } else if (arg == "--threads") {
            const char * v = need_value("--threads");
            if (!v || !parse_int("--threads", v, args.threads)) return false;
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    return !args.transformer.empty() && !args.tokenizer.empty() && !args.output.empty()
           && !args.prompt_codes_path.empty();
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_help();
        return 2;
    }

    std::ifstream codes_in(args.prompt_codes_path, std::ios::binary);
    if (!codes_in) {
        std::cerr << "failed to open prompt codes: " << args.prompt_codes_path << "\n";
        return 1;
    }
    json prompt_doc = json::parse(codes_in);
    const std::string prompt_text = prompt_doc.at("prompt_text").get<std::string>();
    const int32_t num_codebooks = prompt_doc.at("num_codebooks").get<int32_t>();
    const int32_t t_prompt = prompt_doc.at("cols").get<int32_t>();
    std::vector<int32_t> prompt_codes = prompt_doc.at("codes").get<std::vector<int32_t>>();
    const size_t expected_codes = static_cast<size_t>(num_codebooks) * t_prompt;
    if (prompt_codes.size() != expected_codes) {
        std::cerr << "prompt codes length mismatch: expected " << expected_codes
                  << ", got " << prompt_codes.size() << "\n";
        return 1;
    }

    s2::Tokenizer tokenizer;
    if (!tokenizer.load(args.tokenizer)) {
        std::cerr << "failed to load tokenizer: " << args.tokenizer << "\n";
        return 1;
    }

    s2::SlowARModel model;
    if (!model.load(args.transformer, -1)) {
        std::cerr << "failed to load transformer: " << args.transformer << "\n";
        return 1;
    }

    {
        const s2::ModelHParams & hp = model.hparams();
        s2::TokenizerConfig & tc = tokenizer.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id > 0) tc.semantic_end_id = hp.semantic_end_id;
        if (hp.num_codebooks > 0) tc.num_codebooks = hp.num_codebooks;
        if (hp.codebook_size > 0) tc.codebook_size = hp.codebook_size;
        if (hp.vocab_size > 0) tc.vocab_size = hp.vocab_size;
    }

    const int32_t max_seq_len = model.hparams().context_length > 0
        ? model.hparams().context_length
        : 32768;
    if (!model.init_kv_cache(max_seq_len)) {
        std::cerr << "failed to init KV cache (max_seq_len=" << max_seq_len << ")\n";
        return 1;
    }

    s2::PromptTensor prompt = s2::build_prompt(
        tokenizer, args.text, prompt_text, prompt_codes.data(), num_codebooks, t_prompt);

    s2::GenerateParams gen;
    gen.max_new_tokens        = args.max_new_tokens;
    gen.temperature           = args.temperature;
    gen.top_p                 = args.top_p;
    gen.top_k                 = args.top_k;
    gen.min_tokens_before_end = args.min_tokens_before_end;
    gen.n_threads             = args.threads;
    gen.verbose               = false;

    s2::GenerateResult result = s2::generate(model, tokenizer.config(), prompt, gen);

    json doc;
    doc["backend"] = "s2.cpp";
    doc["prompt_text"] = prompt_text;
    doc["prompt_code_cols"] = t_prompt;
    doc["text"] = args.text;
    doc["temperature"] = args.temperature;
    doc["top_p"] = args.top_p;
    doc["top_k"] = args.top_k;
    doc["max_new_tokens"] = args.max_new_tokens;
    doc["min_tokens_before_end"] = args.min_tokens_before_end;
    doc["prompt_cols"] = prompt.cols;
    doc["num_codebooks"] = result.num_codebooks;
    doc["n_frames"] = result.n_frames;
    doc["codes"] = result.codes;

    const std::string json_utf8 = doc.dump(2) + "\n";
    std::ofstream out(args.output, std::ios::binary);
    if (!out) {
        std::cerr << "failed to open output: " << args.output << "\n";
        return 1;
    }
    out.write(json_utf8.data(), static_cast<std::streamsize>(json_utf8.size()));
    std::cout << "wrote " << args.output << " (" << result.num_codebooks
              << " codebooks x " << result.n_frames << " frames)\n";
    return 0;
}
'@
    Write-Utf8NoBom $Path $main
}

function Ensure-ReferenceCodesCMake {
    param(
        [string] $Path,
        [string] $SourceDir,
        [string] $BuildSourceDir,
        [string] $TargetName,
        [string] $MainCpp,
        [string[]] $ExtraSources = @()
    )
    $sourceUnix = $SourceDir.Replace('\', '/')
    $buildUnix = $BuildSourceDir.Replace('\', '/')
    $extra = ($ExtraSources | ForEach-Object { "`"`${S2_CPP_SRC}/$_`"" }) -join "`n    "
    $cmake = @"
cmake_minimum_required(VERSION 3.14)
project($TargetName LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)

if(MSVC)
    add_compile_options(
        "`$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "`$<$<COMPILE_LANGUAGE:C>:/utf-8>"
    )
endif()

add_subdirectory("$sourceUnix/ggml" ggml-build)

set(S2_CPP_SRC "$sourceUnix")

add_executable($TargetName
    $extra
    "$buildUnix/$MainCpp"
)

target_include_directories($TargetName PRIVATE
    "`${S2_CPP_SRC}/include"
    "`${S2_CPP_SRC}/src"
    "`${S2_CPP_SRC}/third_party"
    "`${S2_CPP_SRC}/ggml/include"
    "`${S2_CPP_SRC}/ggml/src"
    "$buildUnix"
)

target_link_libraries($TargetName PRIVATE ggml)

if(MSVC)
    target_compile_options($TargetName PRIVATE /EHsc /utf-8)
endif()
"@
    Write-Utf8NoBom $Path $cmake
}

function Build-S2CppTool {
    param(
        [string] $DumpDir,
        [string] $TargetName,
        [string] $MainCpp,
        [string[]] $ExtraSources,
        [string] $BuildType
    )
    $buildDir = Join-Path $DumpDir "build-$TargetName"
    Ensure-ReferenceCodesCMake `
        -Path (Join-Path $DumpDir "CMakeLists.$TargetName.txt") `
        -SourceDir $s2CppDir `
        -BuildSourceDir $DumpDir `
        -TargetName $TargetName `
        -MainCpp $MainCpp `
        -ExtraSources $ExtraSources
    Copy-Item -Force (Join-Path $DumpDir "CMakeLists.$TargetName.txt") (Join-Path $DumpDir "CMakeLists.txt")
    cmake -S $DumpDir -B $buildDir -DCMAKE_BUILD_TYPE=$BuildType | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    cmake --build $buildDir --config $BuildType --target $TargetName --parallel | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $exe = Join-Path $buildDir "$BuildType\$TargetName.exe"
    if (-not (Test-Path -LiteralPath $exe)) {
        $exe = Join-Path $buildDir "$TargetName.exe"
    }
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "$TargetName not found under $buildDir"
    }
    return $exe
}

if (-not $SkipFixtureBuild) {
    if (-not (Test-Path -LiteralPath $Codec)) {
        throw "missing codec gguf for fixture build: $Codec"
    }
    Write-EncodePromptCodesMain (Join-Path $DumpDir "encode_prompt_codes_dump_main.cpp")
    $encodeExe = Build-S2CppTool `
        -DumpDir $DumpDir `
        -TargetName "s2_encode_prompt_codes_dump" `
        -MainCpp "encode_prompt_codes_dump_main.cpp" `
        -ExtraSources @("src/s2_audio.cpp") `
        -BuildType $BuildType
    $tinyWav = Join-Path $DumpDir "reference_tiny.wav"
    Write-TinyReferenceWav -Path $tinyWav
    $cppPromptCodes = Join-Path $outDir "reference_prompt_codes_cpp.json"
    $rustPromptCodes = Join-Path $outDir "reference_prompt_codes_rust.json"
    & $encodeExe `
        --codec $Codec `
        --wav $tinyWav `
        --prompt-text-file $PromptTextFile `
        --output $cppPromptCodes `
        --threads $Threads
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not $BuildOnly) {
        cargo run -q -p fish_s2_infer --bin fish_s2_reference_codes_dump -- `
            --codec $Codec `
            --wav-input $tinyWav `
            --prompt-text-file $PromptTextFile `
            --prompt-codes-format `
            --output $rustPromptCodes
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        cargo run -q -p fish_s2_parity -- compare-prompt-codes $cppPromptCodes $rustPromptCodes
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "reference prompt-code parity OK: $cppPromptCodes vs $rustPromptCodes"
    }
    if (-not (Test-Path -LiteralPath $Fixture)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $Fixture -Parent) | Out-Null
        Copy-Item -Force -LiteralPath $cppPromptCodes -Destination $Fixture
        Write-Host "wrote reference prompt codes fixture: $Fixture"
    }
    if ($PromptCodesOnly) {
        exit 0
    }
}

if (-not (Test-Path -LiteralPath $Fixture)) {
    throw "missing prompt codes fixture: $Fixture (run without -SkipFixtureBuild)"
}

Write-ReferenceGenerateCodesDumpMain (Join-Path $DumpDir "reference_generate_codes_dump_main.cpp")
$genExe = Build-S2CppTool `
    -DumpDir $DumpDir `
    -TargetName "s2_generate_codes_dump" `
    -MainCpp "reference_generate_codes_dump_main.cpp" `
    -ExtraSources @(
        "src/s2_model.cpp",
        "src/s2_tokenizer.cpp",
        "src/s2_prompt.cpp",
        "src/s2_sampler.cpp",
        "src/s2_generate.cpp"
    ) `
    -BuildType $BuildType

$tag = "reference"
$cppJson = Join-Path $outDir "generated_codes_${tag}_cpp.json"
$rustJson = Join-Path $outDir "generated_codes_${tag}_rust.json"

& $genExe `
    --transformer $Transformer `
    --tokenizer $Tokenizer `
    --prompt-codes $Fixture `
    --output $cppJson `
    --text $Text `
    --max-new-tokens $MaxNewTokens `
    --temperature $Temperature `
    --top-p $TopP `
    --top-k $TopK `
    --min-tokens-before-end $MinTokensBeforeEnd `
    --threads $Threads
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($BuildOnly) { exit 0 }

cargo run --release -p fish_s2_infer --bin fish_s2_codes_dump -- `
    --transformer $Transformer `
    --tokenizer $Tokenizer `
    --prompt-codes $Fixture `
    --output $rustJson `
    --text $Text `
    --max-new-tokens $MaxNewTokens `
    --temperature $Temperature `
    --top-p $TopP `
    --top-k $TopK `
    --min-tokens-before-end $MinTokensBeforeEnd `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run -p fish_s2_parity -- compare-generated-codes $cppJson $rustJson
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "reference generated-codes parity OK: $cppJson vs $rustJson"
