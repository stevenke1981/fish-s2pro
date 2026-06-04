param(
    [string] $DumpDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2cpp_waveform_dump"),
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

function Write-WaveformDumpMain {
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

#include "s2_codec.cpp"

using json = nlohmann::json;

namespace {

struct Args {
    std::string codec;
    std::string codes;
    std::string output;
    std::string wav;
    int threads = 4;
};

void print_help() {
    std::cerr
        << "Usage: s2_waveform_dump --codec <codec.gguf> --codes <generated_codes.json> "
           "--output <waveform.json> [--wav <out.wav>] [--threads 4]\n";
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
        } else if (arg == "--wav") {
            const char * v = need_value("--wav");
            if (!v) return false;
            args.wav = v;
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

    std::vector<float> audio;
    if (!codec.decode(codes.data(), n_frames, args.threads, audio)) {
        std::cerr << "codec.decode failed\n";
        return 1;
    }

    if (!args.wav.empty()) {
        std::ofstream wav_out(args.wav, std::ios::binary);
        if (!wav_out) {
            std::cerr << "failed to open wav: " << args.wav << "\n";
            return 1;
        }
        const uint32_t sample_rate = 44100;
        const uint16_t channels = 1;
        const uint16_t bits = 16;
        const uint32_t data_bytes = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
        const uint32_t riff_size = 36 + data_bytes;
        wav_out.write("RIFF", 4);
        wav_out.write(reinterpret_cast<const char *>(&riff_size), 4);
        wav_out.write("WAVE", 4);
        wav_out.write("fmt ", 4);
        const uint32_t fmt_size = 16;
        const uint16_t audio_format = 1;
        const uint32_t byte_rate = sample_rate * channels * (bits / 8);
        const uint16_t block_align = channels * (bits / 8);
        wav_out.write(reinterpret_cast<const char *>(&fmt_size), 4);
        wav_out.write(reinterpret_cast<const char *>(&audio_format), 2);
        wav_out.write(reinterpret_cast<const char *>(&channels), 2);
        wav_out.write(reinterpret_cast<const char *>(&sample_rate), 4);
        wav_out.write(reinterpret_cast<const char *>(&byte_rate), 4);
        wav_out.write(reinterpret_cast<const char *>(&block_align), 2);
        wav_out.write(reinterpret_cast<const char *>(&bits), 2);
        wav_out.write("data", 4);
        wav_out.write(reinterpret_cast<const char *>(&data_bytes), 4);
        for (float sample : audio) {
            const float clamped = std::max(-1.0f, std::min(1.0f, sample));
            const int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
            wav_out.write(reinterpret_cast<const char *>(&pcm), sizeof(pcm));
        }
        std::cout << "wrote " << args.wav << "\n";
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
    doc["sample_rate"] = 44100;
    doc["num_samples"] = audio.size();
    doc["samples_l2"] = l2(audio);
    doc["samples_mean_abs"] = mean_abs(audio);
    doc["samples_max_abs"] = max_abs(audio);
    doc["samples_first8"] = json::array();
    for (size_t i = 0; i < audio.size() && i < 8; ++i) {
        doc["samples_first8"].push_back(static_cast<double>(audio[i]));
    }

    const std::string json_utf8 = doc.dump(2) + "\n";
    std::ofstream out(args.output, std::ios::binary);
    if (!out) {
        std::cerr << "failed to open output: " << args.output << "\n";
        return 1;
    }
    out.write(json_utf8.data(), static_cast<std::streamsize>(json_utf8.size()));
    std::cout << "wrote " << args.output << " (" << audio.size() << " samples)\n";
    return 0;
}
'@
    Write-Utf8NoBom $Path $main
}

function Write-WaveformCMake {
    param(
        [string] $Path,
        [string] $SourceDir,
        [string] $BuildSourceDir
    )

    $sourceUnix = $SourceDir.Replace('\', '/')
    $buildUnix = $BuildSourceDir.Replace('\', '/')
    $cmake = @"
cmake_minimum_required(VERSION 3.14)
project(s2_waveform_dump LANGUAGES C CXX)

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

add_executable(s2_waveform_dump
    "$buildUnix/waveform_dump_main.cpp"
)

target_include_directories(s2_waveform_dump PRIVATE
    "`${S2_CPP_SRC}/include"
    "`${S2_CPP_SRC}/src"
    "`${S2_CPP_SRC}/third_party"
    "`${S2_CPP_SRC}/ggml/include"
    "`${S2_CPP_SRC}/ggml/src"
    "$buildUnix"
)

target_link_libraries(s2_waveform_dump PRIVATE ggml)

if(MSVC)
    target_compile_options(s2_waveform_dump PRIVATE /EHsc /utf-8)
endif()
"@
    Write-Utf8NoBom $Path $cmake
}

$buildDir = Join-Path $DumpDir "build-cpu-waveform"

Write-WaveformDumpMain (Join-Path $DumpDir "waveform_dump_main.cpp")
Write-WaveformCMake `
    -Path (Join-Path $DumpDir "CMakeLists.txt") `
    -SourceDir $s2CppDir `
    -BuildSourceDir $DumpDir

cmake -S $DumpDir -B $buildDir -DCMAKE_BUILD_TYPE=$BuildType
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $buildDir --config $BuildType --target s2_waveform_dump --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$cppExe = Join-Path $buildDir "$BuildType\s2_waveform_dump.exe"
if (-not (Test-Path -LiteralPath $cppExe)) {
    $cppExe = Join-Path $buildDir "s2_waveform_dump.exe"
}
if (-not (Test-Path -LiteralPath $cppExe)) {
    throw "s2_waveform_dump not found under $buildDir"
}

$codeStem = [System.IO.Path]::GetFileNameWithoutExtension($Codes)
$tag = $codeStem -replace '^generated_codes_', ''
$cppJson = Join-Path $outDir "waveform_${tag}_cpp.json"
$rustJson = Join-Path $outDir "waveform_${tag}_rust.json"
$cppWav = Join-Path $outDir "waveform_${tag}_cpp.wav"
$rustWav = Join-Path $outDir "waveform_${tag}_rust.wav"

& $cppExe `
    --codec $Codec `
    --codes $Codes `
    --output $cppJson `
    --wav $cppWav `
    --threads $Threads
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($BuildOnly) { exit 0 }

cargo run -q -p fish_s2_infer --bin fish_s2_waveform_dump -- `
    --codec $Codec `
    --codes $Codes `
    --output $rustJson `
    --wav $rustWav
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run -q -p fish_s2_parity -- compare-waveform $cppJson $rustJson
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run -q -p fish_s2_parity -- compare $cppWav $rustWav
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "waveform parity OK: $cppJson vs $rustJson and $cppWav vs $rustWav"