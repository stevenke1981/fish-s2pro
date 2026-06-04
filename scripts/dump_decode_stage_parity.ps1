param(
    [string] $DumpDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2cpp_decode_stage_dump"),
    [string] $Codec = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-codec-only.gguf"),
    [string] $Codes = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\generated_codes_hi_rust.json"),
    [int] $Threads = 4,
    [string] $BuildType = "Release",
    [switch] $BuildOnly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

$root = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $root "output"
$s2CppDir = Join-Path $outDir "s2.cpp-src"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $DumpDir | Out-Null

if (-not (Test-Path -LiteralPath $s2CppDir)) {
    throw "missing s2.cpp source tree: $s2CppDir; run scripts\dump_generated_codes_parity.ps1 once to fetch/build it"
}

function Write-DecodeStageDumpMain {
    param([string] $Path)
    $main = @'
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Reuse s2_codec.cpp static helpers in this TU (decode_codes_stage,
// build_quantizer_decode_stage) for full RVQ lookup -> post-module -> upsample.
#include "s2_codec.cpp"

using json = nlohmann::json;

namespace {

struct Args {
    std::string codec;
    std::string codes;
    std::string output;
    int threads = 4;
};

void print_help() {
    std::cerr
        << "Usage: s2_decode_stage_dump --codec <codec.gguf> --codes <generated_codes.json> "
           "--output <decode_stage.json> [--threads 4]\n";
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
        } else if (arg == "--codes") {
            const char * v = need_value("--codes");
            if (!v) return false;
            args.codes = v;
        } else if (arg == "--output") {
            const char * v = need_value("--output");
            if (!v) return false;
            args.output = v;
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
    return !args.codec.empty() && !args.codes.empty() && !args.output.empty();
}

double l2(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) {
        const double v = static_cast<double>(value);
        sum += v * v;
    }
    return std::sqrt(sum);
}

double mean_abs(const std::vector<float> & values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (float value : values) sum += std::fabs(static_cast<double>(value));
    return sum / static_cast<double>(values.size());
}

double max_abs(const std::vector<float> & values) {
    double max_value = 0.0;
    for (float value : values) {
        max_value = std::max(max_value, std::fabs(static_cast<double>(value)));
    }
    return max_value;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_help();
        return 2;
    }

    std::ifstream codes_in(args.codes, std::ios::binary);
    if (!codes_in) {
        std::cerr << "failed to open codes: " << args.codes << "\n";
        return 1;
    }
    json input = json::parse(codes_in);
    const int32_t num_codebooks = input.at("num_codebooks").get<int32_t>();
    const int32_t n_frames = input.at("n_frames").get<int32_t>();
    std::vector<int32_t> codes = input.at("codes").get<std::vector<int32_t>>();
    const size_t expected_codes = static_cast<size_t>(num_codebooks) * n_frames;
    if (codes.size() != expected_codes) {
        std::cerr << "codes length mismatch: expected " << expected_codes
                  << ", got " << codes.size() << "\n";
        return 1;
    }

    s2::AudioCodec codec;
    if (!codec.load(args.codec, -1)) {
        std::cerr << "failed to load codec: " << args.codec << "\n";
        return 1;
    }
    if (codec.num_codebooks() != num_codebooks) {
        std::cerr << "num_codebooks mismatch: codec=" << codec.num_codebooks()
                  << ", codes=" << num_codebooks << "\n";
        return 1;
    }

    std::vector<float> stage;
    if (!s2::decode_codes_stage(*codec.impl_, codes.data(), n_frames, stage)) {
        std::cerr << "decode_codes_stage failed\n";
        return 1;
    }

    std::vector<float> hidden_out;
    int32_t hidden_dim = 0;
    int32_t output_frames = 0;
    {
        const size_t ctx_size = 96u * 1024u * 1024u;
        std::vector<uint8_t> ctx_buf(ctx_size);
        ggml_init_params p = { ctx_size, ctx_buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        if (!ctx) {
            std::cerr << "failed to init ggml context\n";
            return 1;
        }

        s2::transformer_inputs inp;
        ggml_tensor * stage_in =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, codec.impl_->quantizer_input_dim, n_frames);
        ggml_tensor * hidden = nullptr;
        try {
            hidden = s2::build_quantizer_decode_stage(ctx, *codec.impl_, stage_in, inp);
            hidden = ggml_cpy(
                ctx,
                hidden,
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden->ne[0], hidden->ne[1]));
        } catch (const std::exception & e) {
            std::cerr << "decode stage build failed: " << e.what() << "\n";
            ggml_free(ctx);
            return 1;
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 131072, false);
        ggml_build_forward_expand(gf, hidden);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(codec.impl_->backend));
        if (!allocr || !ggml_gallocr_alloc_graph(allocr, gf)) {
            std::cerr << "decode stage graph allocation failed\n";
            if (allocr) ggml_gallocr_free(allocr);
            ggml_free(ctx);
            return 1;
        }

        ggml_backend_tensor_set(stage_in, stage.data(), 0, stage.size() * sizeof(float));
        if (inp.positions) {
            ggml_backend_tensor_set(
                inp.positions,
                inp.position_values.data(),
                0,
                inp.position_values.size() * sizeof(int32_t));
        }
        if (inp.mask) {
            ggml_backend_tensor_set(
                inp.mask,
                inp.mask_values.data(),
                0,
                inp.mask_values.size() * sizeof(float));
        }

        if (ggml_backend_is_cpu(codec.impl_->backend)) {
            ggml_backend_cpu_set_n_threads(codec.impl_->backend, args.threads);
        }
        if (ggml_backend_graph_compute(codec.impl_->backend, gf) != GGML_STATUS_SUCCESS) {
            std::cerr << "decode stage compute failed\n";
            ggml_gallocr_free(allocr);
            ggml_free(ctx);
            return 1;
        }

        hidden_dim = static_cast<int32_t>(hidden->ne[0]);
        output_frames = static_cast<int32_t>(hidden->ne[1]);
        hidden_out.resize(static_cast<size_t>(hidden->ne[0]) * hidden->ne[1]);
        ggml_backend_tensor_get(hidden, hidden_out.data(), 0, hidden_out.size() * sizeof(float));
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
    }

    json doc;
    doc["backend"] = "s2.cpp";
    if (input.contains("text") && !input.at("text").is_null()) {
        doc["text"] = input.at("text");
    } else {
        doc["text"] = nullptr;
    }
    doc["num_codebooks"] = num_codebooks;
    doc["input_frames"] = n_frames;
    doc["output_frames"] = output_frames;
    doc["hidden_dim"] = hidden_dim;
    doc["hidden_len"] = hidden_out.size();
    doc["hidden_l2"] = l2(hidden_out);
    doc["hidden_mean_abs"] = mean_abs(hidden_out);
    doc["hidden_max_abs"] = max_abs(hidden_out);
    doc["hidden_first8"] = json::array();
    for (size_t i = 0; i < hidden_out.size() && i < 8; ++i) {
        doc["hidden_first8"].push_back(static_cast<double>(hidden_out[i]));
    }

    const std::string json_utf8 = doc.dump(2) + "\n";
    std::ofstream out(args.output, std::ios::binary);
    if (!out) {
        std::cerr << "failed to open output: " << args.output << "\n";
        return 1;
    }
    out.write(json_utf8.data(), static_cast<std::streamsize>(json_utf8.size()));
    std::cout << "wrote " << args.output << " (" << n_frames << " -> " << output_frames
              << " frames x " << hidden_dim << " hidden)\n";
    return 0;
}
'@
    Write-Utf8NoBom $Path $main
}

function Write-DecodeStageCMake {
    param(
        [string] $Path,
        [string] $SourceDir,
        [string] $BuildSourceDir
    )

    $sourceUnix = $SourceDir.Replace('\', '/')
    $buildUnix = $BuildSourceDir.Replace('\', '/')
    $cmake = @"
cmake_minimum_required(VERSION 3.14)
project(s2_decode_stage_dump LANGUAGES C CXX)

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

add_executable(s2_decode_stage_dump
    "$buildUnix/decode_stage_dump_main.cpp"
)

target_include_directories(s2_decode_stage_dump PRIVATE
    "`${S2_CPP_SRC}/include"
    "`${S2_CPP_SRC}/src"
    "`${S2_CPP_SRC}/third_party"
    "`${S2_CPP_SRC}/ggml/include"
    "`${S2_CPP_SRC}/ggml/src"
    "$buildUnix"
)

target_link_libraries(s2_decode_stage_dump PRIVATE ggml)

if(MSVC)
    target_compile_options(s2_decode_stage_dump PRIVATE /EHsc /utf-8)
endif()
"@
    Write-Utf8NoBom $Path $cmake
}

$buildDir = Join-Path $DumpDir "build-cpu-decode-stage"

Write-DecodeStageDumpMain (Join-Path $DumpDir "decode_stage_dump_main.cpp")
Write-DecodeStageCMake `
    -Path (Join-Path $DumpDir "CMakeLists.txt") `
    -SourceDir $s2CppDir `
    -BuildSourceDir $DumpDir

cmake -S $DumpDir -B $buildDir -DCMAKE_BUILD_TYPE=$BuildType
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $buildDir --config $BuildType --target s2_decode_stage_dump --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$cppExe = Join-Path $buildDir "$BuildType\s2_decode_stage_dump.exe"
if (-not (Test-Path -LiteralPath $cppExe)) {
    $cppExe = Join-Path $buildDir "s2_decode_stage_dump.exe"
}
if (-not (Test-Path -LiteralPath $cppExe)) {
    throw "s2_decode_stage_dump not found under $buildDir"
}

$codeStem = [System.IO.Path]::GetFileNameWithoutExtension($Codes)
$tag = $codeStem -replace '^generated_codes_', ''
$cppJson = Join-Path $outDir "decode_stage_${tag}_cpp.json"
$rustJson = Join-Path $outDir "decode_stage_${tag}_rust.json"

& $cppExe `
    --codec $Codec `
    --codes $Codes `
    --output $cppJson `
    --threads $Threads
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($BuildOnly) { exit 0 }

cargo run -q -p fish_s2_infer --bin fish_s2_decode_stage_dump -- `
    --codec $Codec `
    --codes $Codes `
    --output $rustJson
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run -q -p fish_s2_parity -- compare-decode-stage $cppJson $rustJson
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "decode-stage parity OK: $cppJson vs $rustJson"