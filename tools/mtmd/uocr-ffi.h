#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(UOCR_FFI_BUILD)
#    define UOCR_FFI_API __declspec(dllexport)
#  else
#    define UOCR_FFI_API __declspec(dllimport)
#  endif
#else
#  define UOCR_FFI_API __attribute__((visibility("default")))
#endif

#define UOCR_FFI_ABI_VERSION 1u

typedef struct uocr_ffi_session uocr_ffi_session;

typedef enum uocr_ffi_status {
    UOCR_FFI_STATUS_OK = 0,
    UOCR_FFI_STATUS_ERROR = 1,
    UOCR_FFI_STATUS_INVALID_ARGUMENT = 2,
    UOCR_FFI_STATUS_UNSUPPORTED_ABI = 3,
    UOCR_FFI_STATUS_LOAD_FAILED = 4,
    UOCR_FFI_STATUS_RUN_FAILED = 5,
    UOCR_FFI_STATUS_CANCELLED = 6
} uocr_ffi_status;

typedef enum uocr_ffi_event_type {
    UOCR_FFI_EVENT_TOKEN = 1,
    UOCR_FFI_EVENT_LOG = 2,
    UOCR_FFI_EVENT_PROGRESS = 3,
    UOCR_FFI_EVENT_DONE = 4
} uocr_ffi_event_type;

typedef struct uocr_ffi_event {
    uint32_t struct_size;
    uint32_t type;
    const char * text_utf8;
    uint64_t text_len;
    const char * json_utf8;
    uint64_t json_len;
    int32_t code;
    uint32_t reserved_u32;
    uint64_t index;
    void * reserved_ptr0;
    void * reserved_ptr1;
    void * reserved_ptr2;
    void * reserved_ptr3;
} uocr_ffi_event;

typedef int32_t (*uocr_ffi_event_callback)(const uocr_ffi_event * event, void * user_data);

typedef struct uocr_ffi_params {
    uint32_t struct_size;
    uint32_t flags;
    const char * model_path;
    const char * mmproj_path;
    const char * chat_template;
    int32_t ctx_size;
    int32_t n_batch;
    int32_t n_gpu_layers;
    int32_t log_verbosity;
    int32_t force_prompt_eos;
    int32_t no_image_end;
    int32_t gundam_mode;
    int32_t no_repeat_ngram;
    int32_t ngram_size;
    int32_t ngram_window;
    const char * ngram_whitelist;
    int32_t prefill_aware_swa;
    int32_t legacy_kv_prune;
    int32_t decode_window;
    int32_t min_new_tokens;
    void * reserved_ptr0;
    void * reserved_ptr1;
    void * reserved_ptr2;
    void * reserved_ptr3;
} uocr_ffi_params;

typedef struct uocr_ffi_request {
    uint32_t struct_size;
    uint32_t flags;
    const char * image_path;
    const char * prompt;
    int32_t max_tokens;
    int32_t reserved_i32;
    uocr_ffi_event_callback event_callback;
    void * user_data;
    void * reserved_ptr0;
    void * reserved_ptr1;
    void * reserved_ptr2;
    void * reserved_ptr3;
} uocr_ffi_request;

UOCR_FFI_API uint32_t uocr_ffi_abi_version(void);
UOCR_FFI_API const char * uocr_ffi_build_info(void);
UOCR_FFI_API const char * uocr_ffi_media_marker(void);
UOCR_FFI_API uocr_ffi_session * uocr_ffi_create(const uocr_ffi_params * params);
UOCR_FFI_API void uocr_ffi_destroy(uocr_ffi_session * session);
UOCR_FFI_API uocr_ffi_status uocr_ffi_run_image(uocr_ffi_session * session, const uocr_ffi_request * request);
UOCR_FFI_API const char * uocr_ffi_last_error(uocr_ffi_session * session);
UOCR_FFI_API uocr_ffi_status uocr_ffi_last_status(uocr_ffi_session * session);
UOCR_FFI_API uint64_t uocr_ffi_run_count(uocr_ffi_session * session);

#ifdef __cplusplus
}
#endif
