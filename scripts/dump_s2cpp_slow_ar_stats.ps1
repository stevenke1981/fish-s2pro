param(
    [string] $S2CppDir,
    [string] $Transformer,
    [string] $Output,
    [int] $Layer = 0,
    [int] $Position = 0,
    [int] $Tokens = 1,
    [int] $Threads = 4,
    [switch] $Cuda,
    [int] $CudaDevice = 0,
    [string] $CudaArchitectures = "86",
    [switch] $AllowUnsupportedCudaCompiler,
    [string] $BuildType = "Release",
    [switch] $BuildOnly,
    [switch] $ConfigureOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $S2CppDir) { $S2CppDir = Join-Path $root "output\s2.cpp-src" }
if (-not $Transformer) { $Transformer = Join-Path $root "models\s2-pro-f16-transformer-only.gguf" }
if (-not $Output) { $Output = Join-Path $root "output\slow_ar_layer0_cpp_stats.json" }

function Read-Utf8 {
    param([string] $Path)
    return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

function Write-Utf8NoBom {
    param([string] $Path, [string] $Text)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Add-IncludeOnce {
    param([string] $Text, [string] $Include)
    if ($Text.Contains($Include)) { return $Text }
    return $Text.Replace("#include <stdexcept>`r`n", "#include <stdexcept>`r`n$Include`r`n")
}

function Patch-S2CppSource {
    param([string] $SourceDir)

    $headerPath = Join-Path $SourceDir "include\s2_model.h"
    $sourcePath = Join-Path $SourceDir "src\s2_model.cpp"
    if (-not (Test-Path -LiteralPath $headerPath)) { throw "missing header: $headerPath" }
    if (-not (Test-Path -LiteralPath $sourcePath)) { throw "missing source: $sourcePath" }

    $header = Read-Utf8 $headerPath
    $oldHeaderMethod = @"
    // Dump a layer-local Slow-AR attention slice for Rust parity.
    bool dump_slow_ar_layer_stats(const std::string & transformer_path_for_json,
                                  const std::string & output_path,
                                  int32_t layer,
                                  int32_t position,
                                  int32_t n_threads);

"@
    $newHeaderMethod = @"
    // Dump a layer-local Slow-AR attention slice for Rust parity.
    bool dump_slow_ar_layer_stats(const std::string & transformer_path_for_json,
                                  const std::string & output_path,
                                  int32_t layer,
                                  int32_t position,
                                  int32_t token_count,
                                  int32_t n_threads);

"@
    $header = $header.Replace($oldHeaderMethod, $newHeaderMethod)
    if (-not $header.Contains("dump_slow_ar_layer_stats")) {
        $needle = "    const ModelHParams & hparams() const { return hparams_; }`r`n"
        $header = $header.Replace($needle, $newHeaderMethod + $needle)
    }
    Write-Utf8NoBom $headerPath $header

    $source = Read-Utf8 $sourcePath
    $source = Add-IncludeOnce $source "#include <cstdlib>"
    $source = Add-IncludeOnce $source "#include <fstream>"
    $source = Add-IncludeOnce $source "#include <iomanip>"
    $source = Add-IncludeOnce $source "#include <sstream>"
    if (-not $source.Contains("#include `"ggml-cuda.h`"")) {
        $source = $source.Replace("#include `"../include/s2_model.h`"`r`n", @"
#include "../include/s2_model.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
"@)
    }
    $source = $source.Replace("#endif#include <iostream>", "#endif`r`n#include <iostream>")

    if (-not $source.Contains("FISH_S2_CUDA_DEVICE")) {
        $source = $source.Replace(@"
    if (!backend_) {
        backend_ = ggml_backend_cpu_init();
    }
"@, @"
#ifdef GGML_USE_CUDA
    if (!backend_) {
        if (const char * cuda_device_env = std::getenv("FISH_S2_CUDA_DEVICE")) {
            const int cuda_device = std::atoi(cuda_device_env);
            backend_ = ggml_backend_cuda_init(cuda_device);
            if (backend_) {
                std::cout << "[Model] CUDA backend initialized on device " << cuda_device << "." << std::endl;
            } else {
                std::cerr << "[Model] CUDA init failed for device " << cuda_device << ", falling back to CPU." << std::endl;
            }
        }
    }
#endif
    if (!backend_) {
        backend_ = ggml_backend_cpu_init();
    }
"@)
    }

    $source = $source.Replace(@"
    ggml_tensor * projected = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:dump_wo");
    ggml_tensor * hidden_out = ggml_add(ctx0, hidden_input, projected);
    ggml_build_forward_expand(gf, hidden_out);
"@, @"
    ggml_tensor * projected = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:dump_wo");
    ggml_tensor * hidden_out = ggml_add(ctx0, hidden_input, projected);

    // gallocr may reuse intermediate buffers. Copy every tensor we want to dump
    // into dedicated graph outputs so host reads are stable after compute.
    ggml_tensor * dump_normalized = ggml_cpy(ctx0, attn_in,
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, 1));
    ggml_tensor * dump_query = ggml_cpy(ctx0, q,
        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head, n_tokens));
    ggml_tensor * dump_key = ggml_cpy(ctx0, k,
        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head_kv, n_tokens));
    ggml_tensor * dump_value = ggml_cpy(ctx0, v,
        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head_kv, n_tokens));
    ggml_tensor * dump_attention = ggml_cpy(ctx0, attn_cur,
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tokens));
    ggml_tensor * dump_projected = ggml_cpy(ctx0, projected,
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, 1));
    ggml_tensor * dump_hidden = ggml_cpy(ctx0, hidden_out,
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, 1));

    ggml_build_forward_expand(gf, dump_normalized);
    ggml_build_forward_expand(gf, dump_query);
    ggml_build_forward_expand(gf, dump_key);
    ggml_build_forward_expand(gf, dump_value);
    ggml_build_forward_expand(gf, dump_attention);
    ggml_build_forward_expand(gf, dump_projected);
    ggml_build_forward_expand(gf, dump_hidden);
"@)

    $source = $source.Replace(@"
    write_tensor_stats_s2_dump(out, "normalized", attn_in, true);
    write_tensor_stats_s2_dump(out, "query", q, true);
    write_tensor_stats_s2_dump(out, "key", k, true);
    write_tensor_stats_s2_dump(out, "value", v, true);
    write_tensor_stats_s2_dump(out, "attention", attn_cur, true);
    write_tensor_stats_s2_dump(out, "projected", projected, true);
    write_tensor_stats_s2_dump(out, "hidden", hidden_out, false);
"@, @"
    write_tensor_stats_s2_dump(out, "normalized", dump_normalized, true);
    write_tensor_stats_s2_dump(out, "query", dump_query, true);
    write_tensor_stats_s2_dump(out, "key", dump_key, true);
    write_tensor_stats_s2_dump(out, "value", dump_value, true);
    write_tensor_stats_s2_dump(out, "attention", dump_attention, true);
    write_tensor_stats_s2_dump(out, "projected", dump_projected, true);
    write_tensor_stats_s2_dump(out, "hidden", dump_hidden, false);
"@)

    $source = $source.Replace(@"
    if (!memory_k_ || !memory_v_ || max_seq_len_ <= position) {
        free_kv_cache();
        if (!init_kv_cache(position + 1)) {
            return false;
        }
    } else {
"@, @"
    if (!memory_k_ || !memory_v_) {
        if (!init_kv_cache(position + 1)) {
            return false;
        }
    } else if (max_seq_len_ <= position) {
        std::fprintf(stderr, "[dump] existing KV cache too small for position %d\n", position);
        return false;
    } else {
"@)

    $source = [regex]::Replace(
        $source,
        "(?s)`r?`n// ---------------------------------------------------------------------------`r?`n// dump_slow_ar_layer_stats\(\).*?bool SlowARModel::dump_slow_ar_layer_stats.*?`r?`n}`r?`n(?=// ---------------------------------------------------------------------------`r?`n// fast_decode\(\))",
        ""
    )

    if (-not $source.Contains("SlowARModel::dump_slow_ar_layer_stats")) {
        $method = @"

// ---------------------------------------------------------------------------
// dump_slow_ar_layer_stats() - layer-local Slow-AR parity fixture
// ---------------------------------------------------------------------------

static std::string json_escape_s2_dump(const std::string & value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

static std::vector<float> tensor_to_f32_s2_dump(ggml_tensor * tensor) {
    const int64_t n = ggml_nelements(tensor);
    std::vector<float> out(static_cast<size_t>(n));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
        return out;
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(static_cast<size_t>(n));
        ggml_backend_tensor_get(tensor, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < tmp.size(); ++i) {
            out[i] = ggml_fp16_to_fp32(tmp[i]);
        }
        return out;
    }
    throw std::runtime_error("unsupported dump tensor type: " + std::to_string(static_cast<int>(tensor->type)));
}

static void write_tensor_stats_values_s2_dump(std::ostream & out,
                                              const char * indent,
                                              const char * name,
                                              const std::vector<float> & values,
                                              bool comma) {
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double max_abs = 0.0;
    for (float value : values) {
        const double abs_value = std::fabs(static_cast<double>(value));
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
        sum_abs += abs_value;
        max_abs = std::max(max_abs, abs_value);
    }
    const double mean_abs = values.empty() ? 0.0 : sum_abs / static_cast<double>(values.size());
    out << indent << "\"" << name << "\": {"
        << "\"len\": " << values.size()
        << ", \"l2\": " << std::sqrt(sum_sq)
        << ", \"mean_abs\": " << mean_abs
        << ", \"max_abs\": " << max_abs
        << ", \"first8\": [";
    const size_t first_n = std::min<size_t>(8, values.size());
    for (size_t i = 0; i < first_n; ++i) {
        if (i > 0) out << ", ";
        out << values[i];
    }
    out << "]}";
    if (comma) out << ",";
    out << "\n";
}

static void write_tensor_stats_s2_dump(std::ostream & out,
                                       const char * name,
                                       ggml_tensor * tensor,
                                       bool comma) {
    write_tensor_stats_values_s2_dump(out, "  ", name, tensor_to_f32_s2_dump(tensor), comma);
}

static std::vector<float> token_slice_s2_dump(const std::vector<float> & values,
                                              int32_t token,
                                              int32_t width) {
    const size_t begin = static_cast<size_t>(token) * static_cast<size_t>(width);
    const size_t end = begin + static_cast<size_t>(width);
    if (end > values.size()) {
        throw std::runtime_error("dump tensor slice out of range");
    }
    return std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(begin),
                              values.begin() + static_cast<std::ptrdiff_t>(end));
}

bool SlowARModel::dump_slow_ar_layer_stats(const std::string & transformer_path_for_json,
                                           const std::string & output_path,
                                           int32_t layer_index,
                                           int32_t position,
                                           int32_t token_count,
                                           int32_t n_threads) {
    if (layer_index < 0 || layer_index >= static_cast<int32_t>(weights_.layers.size())) {
        std::fprintf(stderr, "[dump] layer out of range: %d\n", layer_index);
        return false;
    }
    if (position < 0) {
        std::fprintf(stderr, "[dump] position must be non-negative\n");
        return false;
    }
    if (token_count <= 0) {
        std::fprintf(stderr, "[dump] token_count must be positive\n");
        return false;
    }
    const int32_t required_seq_len = position + token_count;
    if (!memory_k_ || !memory_v_) {
        if (!init_kv_cache(required_seq_len)) {
            return false;
        }
    } else if (max_seq_len_ < required_seq_len) {
        std::fprintf(stderr, "[dump] existing KV cache too small for %d tokens\n", required_seq_len);
        return false;
    } else {
        ggml_backend_tensor_memset(memory_k_, 0, 0, ggml_nbytes(memory_k_));
        ggml_backend_tensor_memset(memory_v_, 0, 0, ggml_nbytes(memory_v_));
    }
    n_past_ = position;

    const int32_t dim       = hparams_.embedding_length;
    const int32_t n_head    = hparams_.head_count;
    const int32_t n_head_kv = hparams_.head_count_kv;
    const auto & layer = weights_.layers[static_cast<size_t>(layer_index)];

    int32_t head_dim = 0;
    if (hparams_.attention_qk_norm && layer.q_norm) {
        head_dim = static_cast<int32_t>(layer.q_norm->ne[0]);
    } else {
        head_dim = static_cast<int32_t>(layer.wo->ne[0] / n_head);
    }

    const int32_t q_size   = n_head * head_dim;
    const int32_t kv_size  = n_head_kv * head_dim;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int32_t n_tokens = token_count;

    const size_t ctx_size = 16u * 1024u * 1024u;
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_init_params p = { ctx_size, ctx_buf.data(), true };
    ggml_context * ctx0 = ggml_init(p);
    if (!ctx0) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_tensor * hidden_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, n_tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

    ggml_tensor * attn_in = rms_norm_weighted(ctx0, hidden_input, layer.attention_norm, hparams_.rms_norm_eps);
    ggml_tensor * qkv     = mul_mat_checked(ctx0, layer.wqkv, attn_in, "mul_mat:dump_wqkv");
    const size_t elem_size = ggml_element_size(qkv);

    ggml_tensor * q2d = ggml_view_2d(ctx0, qkv, q_size, n_tokens, qkv->nb[1], 0);
    ggml_tensor * k2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], q_size * elem_size);
    ggml_tensor * v2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], (q_size + kv_size) * elem_size);

    ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q2d), head_dim, n_head, n_tokens);
    ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k2d), head_dim, n_head_kv, n_tokens);
    ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_cont(ctx0, v2d), head_dim, n_head_kv, n_tokens);

    if (hparams_.attention_qk_norm) {
        q = rms_norm_weighted(ctx0, q, layer.q_norm, hparams_.rms_norm_eps);
        k = rms_norm_weighted(ctx0, k, layer.k_norm, hparams_.rms_norm_eps);
    }

    q = ggml_rope_ext(ctx0, q, positions, nullptr, head_dim, 0,
                      hparams_.context_length, hparams_.rope_freq_base,
                      1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    k = ggml_rope_ext(ctx0, k, positions, nullptr, head_dim, 0,
                      hparams_.context_length, hparams_.rope_freq_base,
                      1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

    const size_t layer_off_k = static_cast<size_t>(layer_index) * memory_k_->nb[3];
    const size_t layer_off_v = static_cast<size_t>(layer_index) * memory_v_->nb[3];
    const size_t token_off_k = static_cast<size_t>(position) * memory_k_->nb[2];
    const size_t token_off_v = static_cast<size_t>(position) * memory_v_->nb[2];

    ggml_tensor * k_slot = ggml_view_3d(ctx0, memory_k_,
        head_dim, n_head_kv, n_tokens,
        memory_k_->nb[1], memory_k_->nb[2],
        layer_off_k + token_off_k);
    ggml_tensor * v_slot = ggml_view_3d(ctx0, memory_v_,
        head_dim, n_head_kv, n_tokens,
        memory_v_->nb[1], memory_v_->nb[2],
        layer_off_v + token_off_v);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, k, k_slot));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, v, v_slot));

    ggml_tensor * k_mem = k;
    ggml_tensor * v_mem = v;
    if (position > 0) {
        ggml_tensor * k_past = ggml_reshape_3d(ctx0,
            ggml_view_1d(ctx0, memory_k_, static_cast<int64_t>(position) * kv_size, layer_off_k),
            head_dim, n_head_kv, position);
        ggml_tensor * v_past = ggml_reshape_3d(ctx0,
            ggml_view_1d(ctx0, memory_v_, static_cast<int64_t>(position) * kv_size, layer_off_v),
            head_dim, n_head_kv, position);
        if (k_past->type != k->type) k_past = ggml_cast(ctx0, k_past, k->type);
        if (v_past->type != v->type) v_past = ggml_cast(ctx0, v_past, v->type);
        k_mem = ggml_concat(ctx0, k_past, k, 2);
        v_mem = ggml_concat(ctx0, v_past, v, 2);
    }

    if (n_head != n_head_kv && q->type != GGML_TYPE_F32) {
        q = ggml_cast(ctx0, q, GGML_TYPE_F32);
    }
    ggml_tensor * k_rep = repeat_interleave_heads(ctx0, k_mem, n_head / n_head_kv);
    ggml_tensor * v_rep = repeat_interleave_heads(ctx0, v_mem, n_head / n_head_kv);

    ggml_tensor * Q   = ggml_permute(ctx0, q,     0, 2, 1, 3);
    ggml_tensor * K   = ggml_permute(ctx0, k_rep, 0, 2, 1, 3);
    ggml_tensor * KQ  = mul_mat_checked(ctx0, K, Q, "mul_mat:dump_kq");
    ggml_tensor * KQs = ggml_scale(ctx0, KQ, attn_scale);
    ggml_tensor * KQm = ggml_diag_mask_inf(ctx0, KQs, position);
    ggml_tensor * KQf = ggml_soft_max(ctx0, KQm);

    ggml_tensor * V       = ggml_cont(ctx0, ggml_permute(ctx0, v_rep, 1, 2, 0, 3));
    ggml_tensor * KQV     = mul_mat_checked(ctx0, V, KQf, "mul_mat:dump_kqv");
    ggml_tensor * KQVm    = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
    ggml_tensor * attn_cur = ggml_cpy(ctx0, KQVm,
                                      ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tokens));
    ggml_tensor * projected = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:dump_wo");
    ggml_tensor * hidden_out = ggml_add(ctx0, hidden_input, projected);

    // gallocr may reuse intermediate buffers. Copy every tensor we want to dump
    // into dedicated graph outputs so host reads are stable after compute.
    ggml_tensor * dump_normalized = ggml_cpy(ctx0, attn_in,
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, n_tokens));
    ggml_tensor * dump_query = ggml_cpy(ctx0, q,
        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head, n_tokens));
    ggml_tensor * dump_key = ggml_cpy(ctx0, k,
        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head_kv, n_tokens));
    ggml_tensor * dump_value = ggml_cpy(ctx0, v,
        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head_kv, n_tokens));
    ggml_tensor * dump_attention = ggml_cpy(ctx0, attn_cur,
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tokens));
    ggml_tensor * dump_projected = ggml_cpy(ctx0, projected,
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, n_tokens));
    ggml_tensor * dump_hidden = ggml_cpy(ctx0, hidden_out,
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, n_tokens));

    ggml_build_forward_expand(gf, dump_normalized);
    ggml_build_forward_expand(gf, dump_query);
    ggml_build_forward_expand(gf, dump_key);
    ggml_build_forward_expand(gf, dump_value);
    ggml_build_forward_expand(gf, dump_attention);
    ggml_build_forward_expand(gf, dump_projected);
    ggml_build_forward_expand(gf, dump_hidden);

    if (!ggml_gallocr_alloc_graph(allocr_, gf)) {
        std::fprintf(stderr, "[dump] gallocr alloc failed\n");
        ggml_free(ctx0);
        return false;
    }

    std::vector<float> hidden_values(static_cast<size_t>(dim) * static_cast<size_t>(n_tokens), 0.0f);
    std::vector<int32_t> position_values(static_cast<size_t>(n_tokens));
    for (int32_t token = 0; token < n_tokens; ++token) {
        const size_t base = static_cast<size_t>(token) * static_cast<size_t>(dim);
        hidden_values[base] = 1.0f;
        hidden_values[base + 1] = -0.5f + static_cast<float>(token);
        hidden_values[base + static_cast<size_t>(dim - 1)] = 0.25f + static_cast<float>(token) * 0.125f;
        position_values[static_cast<size_t>(token)] = position + token;
    }
    ggml_backend_tensor_set(hidden_input, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));

    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[dump] compute failed\n");
        ggml_free(ctx0);
        return false;
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "[dump] cannot open output: %s\n", output_path.c_str());
        ggml_free(ctx0);
        return false;
    }
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"transformer\": \"" << json_escape_s2_dump(transformer_path_for_json) << "\",\n";
    out << "  \"layer\": " << layer_index << ",\n";
    out << "  \"position\": " << position << ",\n";
    out << "  \"token_count\": " << n_tokens << ",\n";
    out << "  \"hidden_size\": " << dim << ",\n";
    out << "  \"head_count\": " << n_head << ",\n";
    out << "  \"head_count_kv\": " << n_head_kv << ",\n";
    out << "  \"head_dim\": " << head_dim << ",\n";
    const std::vector<float> normalized_values = tensor_to_f32_s2_dump(dump_normalized);
    const std::vector<float> query_values = tensor_to_f32_s2_dump(dump_query);
    const std::vector<float> key_values = tensor_to_f32_s2_dump(dump_key);
    const std::vector<float> value_values = tensor_to_f32_s2_dump(dump_value);
    const std::vector<float> attention_values = tensor_to_f32_s2_dump(dump_attention);
    const std::vector<float> projected_values = tensor_to_f32_s2_dump(dump_projected);
    const std::vector<float> hidden_out_values = tensor_to_f32_s2_dump(dump_hidden);
    write_tensor_stats_values_s2_dump(out, "  ", "normalized", token_slice_s2_dump(normalized_values, 0, dim), true);
    write_tensor_stats_values_s2_dump(out, "  ", "query", token_slice_s2_dump(query_values, 0, q_size), true);
    write_tensor_stats_values_s2_dump(out, "  ", "key", token_slice_s2_dump(key_values, 0, kv_size), true);
    write_tensor_stats_values_s2_dump(out, "  ", "value", token_slice_s2_dump(value_values, 0, kv_size), true);
    write_tensor_stats_values_s2_dump(out, "  ", "attention", token_slice_s2_dump(attention_values, 0, q_size), true);
    write_tensor_stats_values_s2_dump(out, "  ", "projected", token_slice_s2_dump(projected_values, 0, dim), true);
    write_tensor_stats_values_s2_dump(out, "  ", "hidden", token_slice_s2_dump(hidden_out_values, 0, dim), n_tokens > 1);
    if (n_tokens > 1) {
        out << "  \"sequence\": [\n";
        for (int32_t token = 0; token < n_tokens; ++token) {
            out << "    {\n";
            out << "      \"position\": " << (position + token) << ",\n";
            write_tensor_stats_values_s2_dump(out, "      ", "normalized", token_slice_s2_dump(normalized_values, token, dim), true);
            write_tensor_stats_values_s2_dump(out, "      ", "query", token_slice_s2_dump(query_values, token, q_size), true);
            write_tensor_stats_values_s2_dump(out, "      ", "key", token_slice_s2_dump(key_values, token, kv_size), true);
            write_tensor_stats_values_s2_dump(out, "      ", "value", token_slice_s2_dump(value_values, token, kv_size), true);
            write_tensor_stats_values_s2_dump(out, "      ", "attention", token_slice_s2_dump(attention_values, token, q_size), true);
            write_tensor_stats_values_s2_dump(out, "      ", "projected", token_slice_s2_dump(projected_values, token, dim), true);
            write_tensor_stats_values_s2_dump(out, "      ", "hidden", token_slice_s2_dump(hidden_out_values, token, dim), false);
            out << "    }";
            if (token + 1 < n_tokens) out << ",";
            out << "\n";
        }
        out << "  ]\n";
    }
    out << "}\n";

    ggml_free(ctx0);
    n_past_ = position + n_tokens;
    return true;
}
"@
        $needle = "// ---------------------------------------------------------------------------`r`n// fast_decode()"
        $source = $source.Replace($needle, $method + "`r`n" + $needle)
    }
    Write-Utf8NoBom $sourcePath $source
}

function Write-DumpBuildFiles {
    param(
        [string] $SourceDir,
        [string] $BuildSourceDir,
        [bool] $UseCuda,
        [string] $CudaArch,
        [bool] $AllowUnsupportedCompiler
    )
    New-Item -ItemType Directory -Force -Path $BuildSourceDir | Out-Null
    $cudaOption = if ($UseCuda) { "ON" } else { "OFF" }
    $cudaUnsupportedCompilerLine = if ($AllowUnsupportedCompiler) {
        '    string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")'
    } else {
        ''
    }

    $main = @"
#include "s2_model.h"

#include <iostream>
#include <string>

struct Args {
    std::string transformer;
    std::string output;
    int layer = 0;
    int position = 0;
    int tokens = 1;
    int threads = 4;
};

static void print_help() {
    std::cerr
        << "Usage: s2_slow_ar_dump --transformer <transformer.gguf> --output <stats.json> [--layer 0] [--position 0] [--tokens 1] [--threads 4]\n";
}

static bool parse_int(const char * label, const std::string & value, int & out) {
    try {
        out = std::stoi(value);
        return true;
    } catch (const std::exception & err) {
        std::cerr << "invalid " << label << " value '" << value << "': " << err.what() << "\n";
        return false;
    }
}

static bool parse_args(int argc, char ** argv, Args & args) {
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
            const char * value = need_value("--transformer");
            if (!value) return false;
            args.transformer = value;
        } else if (arg == "--output") {
            const char * value = need_value("--output");
            if (!value) return false;
            args.output = value;
        } else if (arg == "--layer") {
            const char * value = need_value("--layer");
            if (!value || !parse_int("--layer", value, args.layer)) return false;
        } else if (arg == "--position") {
            const char * value = need_value("--position");
            if (!value || !parse_int("--position", value, args.position)) return false;
        } else if (arg == "--tokens") {
            const char * value = need_value("--tokens");
            if (!value || !parse_int("--tokens", value, args.tokens)) return false;
        } else if (arg == "--threads") {
            const char * value = need_value("--threads");
            if (!value || !parse_int("--threads", value, args.threads)) return false;
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (args.transformer.empty()) {
        std::cerr << "missing --transformer\n";
        return false;
    }
    if (args.output.empty()) {
        std::cerr << "missing --output\n";
        return false;
    }
    if (args.tokens <= 0) {
        std::cerr << "--tokens must be greater than zero\n";
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_help();
        return 2;
    }

    s2::SlowARModel model;
    if (!model.load(args.transformer, -1)) {
        std::cerr << "failed to load transformer: " << args.transformer << "\n";
        return 1;
    }
    if (!model.dump_slow_ar_layer_stats(args.transformer, args.output, args.layer, args.position, args.tokens, args.threads)) {
        std::cerr << "failed to dump Slow-AR stats\n";
        return 1;
    }
    std::cout << "wrote " << args.output << "\n";
    return 0;
}
"@

    $cmake = @"
cmake_minimum_required(VERSION 3.14)
project(s2_slow_ar_dump LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_CUDA $cudaOption CACHE BOOL "" FORCE)
if(GGML_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "$CudaArch" CACHE STRING "" FORCE)
$cudaUnsupportedCompilerLine
endif()

add_subdirectory("$($SourceDir.Replace('\', '/'))/ggml" ggml-build)

add_executable(s2_slow_ar_dump
    "$($SourceDir.Replace('\', '/'))/src/s2_model.cpp"
    "$($BuildSourceDir.Replace('\', '/'))/slow_ar_dump_main.cpp"
)

target_include_directories(s2_slow_ar_dump PRIVATE
    "$($SourceDir.Replace('\', '/'))/include"
    "$($SourceDir.Replace('\', '/'))/third_party"
    "$($SourceDir.Replace('\', '/'))/ggml/include"
    "$($SourceDir.Replace('\', '/'))/ggml/src"
)

target_link_libraries(s2_slow_ar_dump PRIVATE ggml)

if(MSVC)
    target_compile_options(s2_slow_ar_dump PRIVATE /EHsc /utf-8)
endif()
"@

    Write-Utf8NoBom (Join-Path $BuildSourceDir "slow_ar_dump_main.cpp") $main
    Write-Utf8NoBom (Join-Path $BuildSourceDir "CMakeLists.txt") $cmake
}

if (-not (Test-Path -LiteralPath $S2CppDir)) { throw "S2CppDir not found: $S2CppDir" }
if (-not (Test-Path -LiteralPath $Transformer)) { throw "Transformer not found: $Transformer" }
New-Item -ItemType Directory -Force -Path (Split-Path $Output -Parent) | Out-Null

$resolvedSource = (Resolve-Path -LiteralPath $S2CppDir).Path
$resolvedTransformer = (Resolve-Path -LiteralPath $Transformer).Path
$resolvedOutput = $Output
if (-not [System.IO.Path]::IsPathRooted($resolvedOutput)) {
    $resolvedOutput = Join-Path (Get-Location) $resolvedOutput
}

Patch-S2CppSource $resolvedSource

$dumpSourceDir = Join-Path $root "output\s2cpp_slow_ar_dump"
$dumpBuildDirName = if ($Cuda) { "build-cuda" } else { "build" }
$dumpBuildDir = Join-Path $dumpSourceDir $dumpBuildDirName
Write-DumpBuildFiles $resolvedSource $dumpSourceDir $Cuda.IsPresent $CudaArchitectures $AllowUnsupportedCudaCompiler.IsPresent

cmake -S $dumpSourceDir -B $dumpBuildDir -DCMAKE_BUILD_TYPE=$BuildType
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($ConfigureOnly) { exit 0 }

cmake --build $dumpBuildDir --config $BuildType --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($BuildOnly) { exit 0 }

$exe = Get-ChildItem -LiteralPath $dumpBuildDir -Recurse -Filter "s2_slow_ar_dump.exe" -File |
    Sort-Object FullName |
    Select-Object -First 1
if (-not $exe) {
    $exe = Get-ChildItem -LiteralPath $dumpBuildDir -Recurse -Filter "s2_slow_ar_dump" -File |
        Sort-Object FullName |
        Select-Object -First 1
}
if (-not $exe) { throw "built executable not found under $dumpBuildDir" }

if ($Cuda) {
    $env:FISH_S2_CUDA_DEVICE = "$CudaDevice"
} else {
    Remove-Item Env:\FISH_S2_CUDA_DEVICE -ErrorAction SilentlyContinue
}

& $exe.FullName `
    --transformer $resolvedTransformer `
    --output $resolvedOutput `
    --layer $Layer `
    --position $Position `
    --tokens $Tokens `
    --threads $Threads
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Wrote $resolvedOutput"
