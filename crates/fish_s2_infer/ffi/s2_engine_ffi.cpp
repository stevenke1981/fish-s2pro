#include "s2_engine_ffi.h"

#include "s2_pipeline.h"

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

struct S2EngineHandle {
    s2::Pipeline pipeline;
    s2::PipelineParams base;
    bool ready = false;
};

static void copy_err(char *err, size_t err_cap, const std::string &msg) {
    if (!err || err_cap == 0) {
        return;
    }
    std::strncpy(err, msg.c_str(), err_cap - 1);
    err[err_cap - 1] = '\0';
}

static void set_cuda_device_env(int32_t device) {
    const std::string value = std::to_string(device);
#ifdef _WIN32
    _putenv_s("FISH_S2_CUDA_DEVICE", value.c_str());
#else
    setenv("FISH_S2_CUDA_DEVICE", value.c_str(), 1);
#endif
}

extern "C" S2EngineHandle *s2_engine_create(const S2EngineConfig *cfg, char *err, size_t err_cap) {
    if (!cfg || !cfg->model_path || !cfg->tokenizer_path) {
        copy_err(err, err_cap, "invalid engine config");
        return nullptr;
    }
    auto *handle = new S2EngineHandle();
    const bool use_cuda = cfg->use_cuda != 0;
    if (use_cuda) {
        set_cuda_device_env(cfg->cuda_device);
    }
    handle->base.model_path = cfg->model_path;
    handle->base.codec_model_path = cfg->codec_path ? cfg->codec_path : "";
    handle->base.tokenizer_path = cfg->tokenizer_path;
    handle->base.vulkan_device = use_cuda ? -1 : cfg->vulkan_device;
    handle->base.codec_vulkan_device = use_cuda ? -1 : cfg->codec_vulkan_device;
    handle->base.gen.max_new_tokens = 2048;
    handle->base.gen.temperature = 0.7f;
    handle->base.gen.top_p = 0.8f;
    handle->base.gen.top_k = 30;
    handle->base.gen.n_threads = 4;

    if (!handle->pipeline.init(handle->base)) {
        copy_err(err, err_cap, "pipeline init failed");
        delete handle;
        return nullptr;
    }
    handle->ready = true;
    return handle;
}

extern "C" void s2_engine_destroy(S2EngineHandle *handle) {
    delete handle;
}

extern "C" int32_t s2_engine_synthesize_wav(
    S2EngineHandle *handle,
    const char *text,
    const char *reference_text,
    uint8_t **out_data,
    size_t *out_len,
    char *err,
    size_t err_cap) {
    if (!handle || !handle->ready || !text || !out_data || !out_len) {
        copy_err(err, err_cap, "engine not ready");
        return 0;
    }
    s2::PipelineParams params = handle->base;
    params.text = text;
    if (reference_text) {
        params.prompt_text = reference_text;
    }
    std::vector<char> buffer;
    if (!handle->pipeline.synthesize_to_buffer(params, buffer)) {
        copy_err(err, err_cap, "synthesis failed");
        return 0;
    }
    auto *raw = static_cast<uint8_t *>(std::malloc(buffer.size()));
    if (!raw) {
        copy_err(err, err_cap, "out of memory");
        return 0;
    }
    std::memcpy(raw, buffer.data(), buffer.size());
    *out_data = raw;
    *out_len = buffer.size();
    return 1;
}

extern "C" void s2_engine_free_buffer(uint8_t *ptr) {
    std::free(ptr);
}
