param(
    [string] $DumpDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2cpp_generated_codes_dump"),
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Tokenizer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\tokenizer.json"),
    [string] $Text = "hi",
    [int] $MaxNewTokens = 2,
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
$s2CppDir = Join-Path $outDir "s2.cpp-src"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $DumpDir | Out-Null

function Write-GenerateCodesDumpMain {
    param([string] $Path)
    $main = @'
#include "s2_generate.h"
#include "s2_prompt.h"
#include "s2_tokenizer.h"

#include "json.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace {

struct Args {
    std::string transformer;
    std::string tokenizer;
    std::string text = "hi";
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
           "--output <codes.json> [--text hi] [--max-new-tokens 2] [--temperature 0] "
           "[--top-p 1] [--top-k 0] [--min-tokens-before-end 0] [--threads 4]\n";
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
    return !args.transformer.empty() && !args.tokenizer.empty() && !args.output.empty();
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_help();
        return 2;
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

    const int32_t max_seq_len = model.hparams().context_length > 0
        ? model.hparams().context_length
        : 32768;
    if (!model.init_kv_cache(max_seq_len)) {
        std::cerr << "failed to init KV cache (max_seq_len=" << max_seq_len << ")\n";
        return 1;
    }

    const int32_t num_codebooks =
        model.hparams().num_codebooks > 0 ? model.hparams().num_codebooks : 10;
    s2::PromptTensor prompt =
        s2::build_prompt(tokenizer, args.text, "", nullptr, num_codebooks, 0);

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

function Ensure-GenerateCodesCMake {
    param(
        [string] $Path,
        [string] $SourceDir,
        [string] $BuildSourceDir,
        [bool] $UseCuda,
        [string] $BuildType
    )

    $sourceUnix = $SourceDir.Replace('\', '/')
    $buildUnix = $BuildSourceDir.Replace('\', '/')
    $cudaValue = if ($UseCuda) { "ON" } else { "OFF" }
    $cmake = @"
cmake_minimum_required(VERSION 3.14)
project(s2_generate_codes_dump LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_CUDA $cudaValue CACHE BOOL "" FORCE)
if(GGML_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)
    string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
endif()

if(MSVC)
    add_compile_options(
        "`$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "`$<$<COMPILE_LANGUAGE:C>:/utf-8>"
    )
    if(GGML_CUDA)
        string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/utf-8")
    endif()
endif()

add_subdirectory("$sourceUnix/ggml" ggml-build)

set(S2_CPP_SRC "$sourceUnix")

add_executable(s2_generate_codes_dump
    "`${S2_CPP_SRC}/src/s2_model.cpp"
    "`${S2_CPP_SRC}/src/s2_tokenizer.cpp"
    "`${S2_CPP_SRC}/src/s2_prompt.cpp"
    "`${S2_CPP_SRC}/src/s2_sampler.cpp"
    "`${S2_CPP_SRC}/src/s2_generate.cpp"
    "$buildUnix/generate_codes_dump_main.cpp"
)

target_include_directories(s2_generate_codes_dump PRIVATE
    "`${S2_CPP_SRC}/include"
    "`${S2_CPP_SRC}/third_party"
    "`${S2_CPP_SRC}/ggml/include"
    "`${S2_CPP_SRC}/ggml/src"
    "$buildUnix"
)

target_link_libraries(s2_generate_codes_dump PRIVATE ggml)

if(MSVC)
    target_compile_options(s2_generate_codes_dump PRIVATE /EHsc /utf-8)
endif()
"@
    Write-Utf8NoBom $Path $cmake
}

$buildDir = if ($Cuda) {
    Join-Path $DumpDir "build-cuda-generated-codes"
} else {
    Join-Path $DumpDir "build-cpu-generated-codes"
}

Write-GenerateCodesDumpMain (Join-Path $DumpDir "generate_codes_dump_main.cpp")
Ensure-GenerateCodesCMake `
    -Path (Join-Path $DumpDir "CMakeLists.txt") `
    -SourceDir $s2CppDir `
    -BuildSourceDir $DumpDir `
    -UseCuda ([bool]$Cuda) `
    -BuildType $BuildType

cmake -S $DumpDir -B $buildDir -DCMAKE_BUILD_TYPE=$BuildType
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $buildDir --config $BuildType --target s2_generate_codes_dump --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$cppExe = Join-Path $buildDir "$BuildType\s2_generate_codes_dump.exe"
if (-not (Test-Path -LiteralPath $cppExe)) {
    $cppExe = Join-Path $buildDir "s2_generate_codes_dump.exe"
}
if (-not (Test-Path -LiteralPath $cppExe)) {
    throw "s2_generate_codes_dump not found under $buildDir"
}

$tag = $Text -replace '[^a-zA-Z0-9]+', '_'
$cppJson = Join-Path $outDir "generated_codes_${tag}_cpp.json"
$rustJson = Join-Path $outDir "generated_codes_${tag}_rust.json"

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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($BuildOnly) { exit 0 }

cargo run --release -p fish_s2_infer --bin fish_s2_codes_dump -- `
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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run -p fish_s2_parity -- compare-generated-codes $cppJson $rustJson
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "generated codes parity OK: $cppJson vs $rustJson"
