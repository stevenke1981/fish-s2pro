#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct S2EngineHandle S2EngineHandle;

typedef struct S2EngineConfig {
    const char *model_path;
    const char *codec_path;
    const char *tokenizer_path;
    const char *workdir;
    int32_t vulkan_device;
    int32_t codec_vulkan_device;
    int32_t use_cuda;
    int32_t cuda_device;
} S2EngineConfig;

S2EngineHandle *s2_engine_create(const S2EngineConfig *cfg, char *err, size_t err_cap);
void s2_engine_destroy(S2EngineHandle *handle);

int32_t s2_engine_synthesize_wav(
    S2EngineHandle *handle,
    const char *text,
    const char *reference_text,
    uint8_t **out_data,
    size_t *out_len,
    char *err,
    size_t err_cap);

void s2_engine_free_buffer(uint8_t *ptr);

#ifdef __cplusplus
}
#endif
