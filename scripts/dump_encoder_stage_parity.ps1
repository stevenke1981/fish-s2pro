param(
    [string] $DumpDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2cpp_encoder_stage_dump"),
    [string] $Codec = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-codec-only.gguf"),
    [string] $WavInput = "",
    [int] $Samples = 2048,
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

function Write-EncoderStageDumpMain {
    param([string] $Path)
    $main = @'
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "s2_codec.cpp"

using json = nlohmann::json;

namespace {

struct Args {
    std::string codec;
    std::string output;
    std::string wav_input;
    int samples = 2048;
    int threads = 4;
};

struct TensorCheckpoint {
    std::string name;
    ggml_tensor * tensor = nullptr;
};

struct StageSummary {
    std::string name;
    int32_t frames = 0;
    int32_t channels = 0;
    size_t hidden_len = 0;
    double hidden_l2 = 0.0;
    double hidden_mean_abs = 0.0;
    double hidden_max_abs = 0.0;
    std::vector<double> hidden_first8;
};

void print_help() {
    std::cerr
        << "Usage: s2_encoder_stage_dump --codec <codec.gguf> --output <encoder_stage.json> "
           "[--samples 2048] [--wav-input reference.wav] [--threads 4]\n";
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
        } else if (arg == "--output") {
            const char * v = need_value("--output");
            if (!v) return false;
            args.output = v;
        } else if (arg == "--wav-input") {
            const char * v = need_value("--wav-input");
            if (!v) return false;
            args.wav_input = v;
        } else if (arg == "--samples") {
            const char * v = need_value("--samples");
            if (!v || !parse_int("--samples", v, args.samples)) return false;
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
    return !args.codec.empty() && !args.output.empty() && args.samples >= 0;
}

std::vector<float> synthetic_pcm(int samples) {
    std::vector<float> audio(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        audio[static_cast<size_t>(i)] = (static_cast<float>(i % 97) - 48.0f) / 4096.0f;
    }
    return audio;
}

uint16_t read_u16(const std::vector<uint8_t> & bytes, size_t offset) {
    if (offset + 2 > bytes.size()) throw std::runtime_error("unexpected EOF reading WAV u16");
    return static_cast<uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
}

uint32_t read_u32(const std::vector<uint8_t> & bytes, size_t offset) {
    if (offset + 4 > bytes.size()) throw std::runtime_error("unexpected EOF reading WAV u32");
    return static_cast<uint32_t>(bytes[offset])
        | (static_cast<uint32_t>(bytes[offset + 1]) << 8)
        | (static_cast<uint32_t>(bytes[offset + 2]) << 16)
        | (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

std::vector<float> read_wav_mono(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open WAV: " + path);
    std::vector<uint8_t> bytes(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
    if (bytes.size() < 12
        || std::memcmp(bytes.data(), "RIFF", 4) != 0
        || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("not a RIFF/WAVE file: " + path);
    }

    bool have_fmt = false;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    const uint8_t * data = nullptr;
    size_t data_len = 0;
    size_t offset = 12;
    while (offset + 8 <= bytes.size()) {
        const uint8_t * id = bytes.data() + offset;
        const uint32_t len = read_u32(bytes, offset + 4);
        offset += 8;
        const size_t end = offset + len;
        if (end > bytes.size()) throw std::runtime_error("WAV chunk extends past file end");
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (len < 16) throw std::runtime_error("WAV fmt chunk too short");
            audio_format = read_u16(bytes, offset);
            channels = read_u16(bytes, offset + 2);
            sample_rate = read_u32(bytes, offset + 4);
            bits_per_sample = read_u16(bytes, offset + 14);
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            data = bytes.data() + offset;
            data_len = len;
        }
        offset = end + (len % 2);
    }
    if (!have_fmt) throw std::runtime_error("missing WAV fmt chunk");
    if (!data) throw std::runtime_error("missing WAV data chunk");
    if (channels == 0) throw std::runtime_error("WAV has zero channels");
    if (sample_rate != 44100) throw std::runtime_error("expected 44100 Hz WAV");

    std::vector<float> interleaved;
    if (audio_format == 1 && bits_per_sample == 16) {
        if (data_len % 2 != 0) throw std::runtime_error("PCM16 WAV data length is odd");
        interleaved.reserve(data_len / 2);
        for (size_t i = 0; i < data_len; i += 2) {
            const int16_t sample =
                static_cast<int16_t>(data[i] | (static_cast<uint16_t>(data[i + 1]) << 8));
            interleaved.push_back(static_cast<float>(sample) / 32768.0f);
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        if (data_len % 4 != 0) throw std::runtime_error("float32 WAV data length is invalid");
        interleaved.reserve(data_len / 4);
        for (size_t i = 0; i < data_len; i += 4) {
            uint32_t raw = static_cast<uint32_t>(data[i])
                | (static_cast<uint32_t>(data[i + 1]) << 8)
                | (static_cast<uint32_t>(data[i + 2]) << 16)
                | (static_cast<uint32_t>(data[i + 3]) << 24);
            float sample = 0.0f;
            std::memcpy(&sample, &raw, sizeof(sample));
            interleaved.push_back(std::max(-1.0f, std::min(1.0f, sample)));
        }
    } else {
        throw std::runtime_error("unsupported WAV format: expected PCM16 or float32");
    }

    if (interleaved.empty()) throw std::runtime_error("WAV contains no samples");
    if (channels == 1) return interleaved;
    if (interleaved.size() % channels != 0) {
        throw std::runtime_error("WAV sample count is not divisible by channel count");
    }
    std::vector<float> mono;
    mono.reserve(interleaved.size() / channels);
    for (size_t i = 0; i < interleaved.size(); i += channels) {
        float sum = 0.0f;
        for (uint16_t ch = 0; ch < channels; ++ch) sum += interleaved[i + ch];
        mono.push_back(sum / static_cast<float>(channels));
    }
    return mono;
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

StageSummary summarize_stage(const std::string & name, ggml_tensor * tensor) {
    StageSummary summary;
    summary.name = name;
    summary.channels = static_cast<int32_t>(tensor->ne[0]);
    summary.frames = static_cast<int32_t>(tensor->ne[1]);
    summary.hidden_len = static_cast<size_t>(tensor->ne[0]) * static_cast<size_t>(tensor->ne[1]);
    std::vector<float> values(summary.hidden_len);
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    summary.hidden_l2 = l2(values);
    summary.hidden_mean_abs = mean_abs(values);
    summary.hidden_max_abs = max_abs(values);
    for (size_t i = 0; i < values.size() && i < 8; ++i) {
        summary.hidden_first8.push_back(static_cast<double>(values[i]));
    }
    return summary;
}

json stage_summary_to_json(const StageSummary & summary) {
    json doc;
    doc["name"] = summary.name;
    doc["frames"] = summary.frames;
    doc["channels"] = summary.channels;
    doc["hidden_len"] = summary.hidden_len;
    doc["hidden_l2"] = summary.hidden_l2;
    doc["hidden_mean_abs"] = summary.hidden_mean_abs;
    doc["hidden_max_abs"] = summary.hidden_max_abs;
    doc["hidden_first8"] = summary.hidden_first8;
    return doc;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_help();
        return 2;
    }

    s2::AudioCodec codec;
    if (!codec.load(args.codec, -1)) {
        std::cerr << "failed to load codec: " << args.codec << "\n";
        return 1;
    }

    std::vector<float> audio;
    try {
        audio = args.wav_input.empty() ? synthetic_pcm(args.samples) : read_wav_mono(args.wav_input);
    } catch (const std::exception & e) {
        std::cerr << "audio input failed: " << e.what() << "\n";
        return 1;
    }
    const int32_t input_samples = static_cast<int32_t>(audio.size());
    const int32_t frame_length = (codec.impl_->frame_length > 0) ? codec.impl_->frame_length : 512;
    const int32_t padded = ((input_samples + frame_length - 1) / frame_length) * frame_length;
    std::vector<float> audio_padded(static_cast<size_t>(padded), 0.0f);
    std::copy(audio.begin(), audio.end(), audio_padded.begin());

    std::vector<float> latent_out;
    std::vector<StageSummary> checkpoint_summaries;
    int32_t hidden_dim = 0;
    int32_t output_frames = 0;
    {
        const size_t ctx_size = 128u * 1024u * 1024u;
        std::vector<uint8_t> ctx_buf(ctx_size);
        ggml_init_params p = { ctx_size, ctx_buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        if (!ctx) {
            std::cerr << "failed to init ggml context\n";
            return 1;
        }

        s2::transformer_inputs enc_inp;
        std::vector<TensorCheckpoint> checkpoints;
        auto checkpoint = [&](const std::string & name, ggml_tensor * x) -> ggml_tensor * {
            ggml_tensor * copy = ggml_cpy(
                ctx,
                x,
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1]));
            checkpoints.push_back({name, copy});
            return x;
        };
        ggml_tensor * audio_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, padded);
        ggml_tensor * latent = nullptr;
        try {
            ggml_tensor * x = s2::causal_conv_1d(
                ctx,
                ggml_get_tensor(codec.impl_->ctx_w, (codec.impl_->tprefix + "encoder.block.0.conv.weight").c_str()),
                ggml_get_tensor(codec.impl_->ctx_w, (codec.impl_->tprefix + "encoder.block.0.conv.bias").c_str()),
                audio_in,
                1,
                1);
            checkpoint("entry_conv", x);

            for (size_t i = 0; i < codec.impl_->encoder_rates.size(); ++i) {
                const std::string prefix =
                    codec.impl_->tprefix + "encoder.block." + std::to_string(i + 1) + ".block";
                const int32_t n_layers =
                    (i < codec.impl_->encoder_transformer_layers.size())
                        ? codec.impl_->encoder_transformer_layers[i]
                        : 0;
                x = s2::build_encoder_block(
                    ctx,
                    *codec.impl_,
                    prefix,
                    x,
                    codec.impl_->encoder_rates[i],
                    n_layers,
                    enc_inp);
                checkpoint("encoder_block_" + std::to_string(i + 1), x);
            }

            const int last = static_cast<int>(codec.impl_->encoder_rates.size()) + 1;
            auto req = [&](const std::string & name) -> ggml_tensor * {
                ggml_tensor * t = ggml_get_tensor(codec.impl_->ctx_w, name.c_str());
                if (!t) throw std::runtime_error("missing tensor: " + name);
                return t;
            };
            x = s2::snake_activation(
                ctx,
                x,
                req(codec.impl_->tprefix + "encoder.block." + std::to_string(last) + ".alpha"));
            checkpoint("tail_snake", x);
            x = s2::causal_conv_1d(
                ctx,
                req(codec.impl_->tprefix + "encoder.block." + std::to_string(last + 1) + ".conv.weight"),
                req(codec.impl_->tprefix + "encoder.block." + std::to_string(last + 1) + ".conv.bias"),
                x,
                1,
                1);
            checkpoint("output_conv", x);
            latent = ggml_cpy(ctx, x, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1]));
        } catch (const std::exception & e) {
            std::cerr << "encoder stage build failed: " << e.what() << "\n";
            ggml_free(ctx);
            return 1;
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 131072, false);
        for (const TensorCheckpoint & item : checkpoints) {
            ggml_build_forward_expand(gf, item.tensor);
        }
        ggml_build_forward_expand(gf, latent);

        ggml_gallocr_t allocr =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(codec.impl_->backend));
        if (!allocr || !ggml_gallocr_alloc_graph(allocr, gf)) {
            std::cerr << "encoder stage graph allocation failed\n";
            if (allocr) ggml_gallocr_free(allocr);
            ggml_free(ctx);
            return 1;
        }

        ggml_backend_tensor_set(
            audio_in,
            audio_padded.data(),
            0,
            audio_padded.size() * sizeof(float));
        if (enc_inp.positions) {
            ggml_backend_tensor_set(
                enc_inp.positions,
                enc_inp.position_values.data(),
                0,
                enc_inp.position_values.size() * sizeof(int32_t));
        }
        if (enc_inp.mask) {
            ggml_backend_tensor_set(
                enc_inp.mask,
                enc_inp.mask_values.data(),
                0,
                enc_inp.mask_values.size() * sizeof(float));
        }

        if (ggml_backend_is_cpu(codec.impl_->backend)) {
            ggml_backend_cpu_set_n_threads(codec.impl_->backend, args.threads);
        }
        if (ggml_backend_graph_compute(codec.impl_->backend, gf) != GGML_STATUS_SUCCESS) {
            std::cerr << "encoder stage compute failed\n";
            ggml_gallocr_free(allocr);
            ggml_free(ctx);
            return 1;
        }

        hidden_dim = static_cast<int32_t>(latent->ne[0]);
        output_frames = static_cast<int32_t>(latent->ne[1]);
        latent_out.resize(static_cast<size_t>(latent->ne[0]) * latent->ne[1]);
        ggml_backend_tensor_get(latent, latent_out.data(), 0, latent_out.size() * sizeof(float));
        for (const TensorCheckpoint & item : checkpoints) {
            checkpoint_summaries.push_back(summarize_stage(item.name, item.tensor));
        }
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
    }

    json doc;
    doc["backend"] = "s2.cpp";
    doc["input_samples"] = input_samples;
    doc["padded_samples"] = padded;
    doc["output_frames"] = output_frames;
    doc["hidden_dim"] = hidden_dim;
    doc["hidden_len"] = latent_out.size();
    doc["hidden_l2"] = l2(latent_out);
    doc["hidden_mean_abs"] = mean_abs(latent_out);
    doc["hidden_max_abs"] = max_abs(latent_out);
    doc["hidden_first8"] = json::array();
    for (size_t i = 0; i < latent_out.size() && i < 8; ++i) {
        doc["hidden_first8"].push_back(static_cast<double>(latent_out[i]));
    }
    doc["checkpoints"] = json::array();
    for (const StageSummary & summary : checkpoint_summaries) {
        doc["checkpoints"].push_back(stage_summary_to_json(summary));
    }

    const std::string json_utf8 = doc.dump(2) + "\n";
    std::ofstream out(args.output, std::ios::binary);
    if (!out) {
        std::cerr << "failed to open output: " << args.output << "\n";
        return 1;
    }
    out.write(json_utf8.data(), static_cast<std::streamsize>(json_utf8.size()));
    std::cout << "wrote " << args.output << " (" << input_samples << " -> " << padded
              << " samples, " << output_frames << " frames x " << hidden_dim << " hidden)\n";
    return 0;
}
'@
    Write-Utf8NoBom $Path $main
}

function Write-EncoderStageCMake {
    param(
        [string] $Path,
        [string] $SourceDir,
        [string] $BuildSourceDir
    )

    $sourceUnix = $SourceDir.Replace('\', '/')
    $buildUnix = $BuildSourceDir.Replace('\', '/')
    $cmake = @"
cmake_minimum_required(VERSION 3.14)
project(s2_encoder_stage_dump LANGUAGES C CXX)

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

add_executable(s2_encoder_stage_dump
    "$buildUnix/encoder_stage_dump_main.cpp"
)

target_include_directories(s2_encoder_stage_dump PRIVATE
    "`${S2_CPP_SRC}/include"
    "`${S2_CPP_SRC}/src"
    "`${S2_CPP_SRC}/third_party"
    "`${S2_CPP_SRC}/ggml/include"
    "`${S2_CPP_SRC}/ggml/src"
    "$buildUnix"
)

target_link_libraries(s2_encoder_stage_dump PRIVATE ggml)

if(MSVC)
    target_compile_options(s2_encoder_stage_dump PRIVATE /EHsc /utf-8)
endif()
"@
    Write-Utf8NoBom $Path $cmake
}

$buildDir = Join-Path $DumpDir "build-cpu-encoder-stage"

Write-EncoderStageDumpMain (Join-Path $DumpDir "encoder_stage_dump_main.cpp")
Write-EncoderStageCMake `
    -Path (Join-Path $DumpDir "CMakeLists.txt") `
    -SourceDir $s2CppDir `
    -BuildSourceDir $DumpDir

cmake -S $DumpDir -B $buildDir -DCMAKE_BUILD_TYPE=$BuildType
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $buildDir --config $BuildType --target s2_encoder_stage_dump --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$cppExe = Join-Path $buildDir "$BuildType\s2_encoder_stage_dump.exe"
if (-not (Test-Path -LiteralPath $cppExe)) {
    $cppExe = Join-Path $buildDir "s2_encoder_stage_dump.exe"
}
if (-not (Test-Path -LiteralPath $cppExe)) {
    throw "s2_encoder_stage_dump not found under $buildDir"
}

if ([string]::IsNullOrWhiteSpace($WavInput)) {
    $tag = "synthetic_${Samples}"
} else {
    $wavStem = [System.IO.Path]::GetFileNameWithoutExtension($WavInput)
    $tag = "wav_" + ($wavStem -replace '[^A-Za-z0-9_.-]', '_')
}
$cppJson = Join-Path $outDir "encoder_stage_${tag}_cpp.json"
$rustJson = Join-Path $outDir "encoder_stage_${tag}_rust.json"

$cppArgs = @("--codec", $Codec, "--output", $cppJson, "--threads", "$Threads")
if ([string]::IsNullOrWhiteSpace($WavInput)) {
    $cppArgs += @("--samples", "$Samples")
} else {
    $cppArgs += @("--wav-input", $WavInput)
}
& $cppExe @cppArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($BuildOnly) { exit 0 }

$rustArgs = @("--codec", $Codec, "--output", $rustJson)
if ([string]::IsNullOrWhiteSpace($WavInput)) {
    $rustArgs += @("--samples", "$Samples")
} else {
    $rustArgs += @("--wav-input", $WavInput)
}
cargo run -q -p fish_s2_infer --bin fish_s2_encoder_stage_dump -- @rustArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run -q -p fish_s2_parity -- compare-encoder-stage $cppJson $rustJson
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "encoder-stage parity OK: $cppJson vs $rustJson"
