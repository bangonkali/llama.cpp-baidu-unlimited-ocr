#include "arg.h"
#include "debug.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#ifdef UOCR_FFI_LIBRARY
#include "uocr-ffi.h"
#endif

#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits.h>
#include <cinttypes>
#include <clocale>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

// volatile, because of signal being an interrupt
static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

/**
 * Please note that this is NOT a production-ready binary.
 * It is a playground for trying multimodal support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand. Do not add unnecessary complexity. The goal is to have a simple CLI for testing multimodal support.
 */

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for multimodal\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  -hf user/repo can replace both -m and --mmproj in most cases\n"
        "  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n"
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

// this is only used by tests.sh to capture the response ; it's not meant to be used in production
static void inject_test_response_marker() {
    const char * env = std::getenv("MTMD_TEST_RESPONSE_MARKER");
    if (env) {
        LOG("%s\n", env);
    }
}

static int env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return (int) parsed;
}

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

struct mtmd_no_repeat_ngram {
    bool enabled = false;
    int ngram_size = 30;
    int window_size = 90;
    llama_tokens whitelist_tokens = {128821, 128822};
};

struct mtmd_prefill_aware_swa {
    bool enabled = false;
    bool legacy_kv_prune = false;
    int decode_window = 128;
};

struct mtmd_min_new_tokens {
    int n_tokens = 0;
};

struct uocr_trace_top_logit {
    llama_token token_id = LLAMA_TOKEN_NULL;
    float logit = 0.0f;
    std::string piece;
};

struct uocr_trace_decoder_pos {
    size_t index = 0;
    mtmd_decoder_pos pos = {};
};

struct uocr_trace_chunk {
    size_t index = 0;
    std::string type;
    size_t n_tokens = 0;
    llama_pos n_pos = 0;
    std::string id;
    llama_tokens text_tokens;
    std::vector<uocr_trace_decoder_pos> decoder_pos_sample;
};

struct uocr_trace_embedding {
    size_t chunk_index = 0;
    size_t n_tokens = 0;
    int n_embd = 0;
    double sum = 0.0;
    double abs_sum = 0.0;
    float min = 0.0f;
    float max = 0.0f;
    std::vector<float> first_values;
};

struct uocr_trace_output_embedding {
    std::string phase;
    int index = 0;
    llama_token token_id = LLAMA_TOKEN_NULL;
    int n_embd = 0;
    double sum = 0.0;
    double abs_sum = 0.0;
    float min = 0.0f;
    float max = 0.0f;
    std::vector<float> first_values;
};

struct uocr_trace_generation_step {
    int index = 0;
    llama_token token_id = LLAMA_TOKEN_NULL;
    std::string piece;
    bool is_eog = false;
    bool is_antiprompt = false;
    int raw_top_rank = -1;
    llama_tokens banned_tokens;
    std::vector<uocr_trace_top_logit> top_logits;
};

struct uocr_trace {
    bool enabled = false;
    std::string path;
    int top_k = 8;
    std::string tool;
    std::string model_path;
    std::string mmproj_path;
    std::string chat_template;
    std::string formatted_prompt;
    bool add_special = false;
    bool parse_special = true;
    int n_predict = 0;
    int n_vocab = 0;
    int n_embd = 0;
    int n_embd_inp = 0;
    int n_ctx = 0;
    bool output_embeddings_enabled = false;
    llama_pos prefill_n_past = 0;
    llama_pos final_n_past = 0;
    std::string stop_reason;
    std::vector<uocr_trace_chunk> chunks;
    std::vector<uocr_trace_embedding> embeddings;
    std::vector<uocr_trace_output_embedding> output_embeddings;
    std::vector<uocr_trace_top_logit> prefill_top_logits;
    std::vector<uocr_trace_generation_step> generation;
};

static llama_tokens env_tokens(const char * name, llama_tokens fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    llama_tokens tokens;
    const char * cursor = value;
    while (*cursor != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(cursor, &end, 10);
        if (end == cursor) {
            break;
        }
        tokens.push_back((llama_token) parsed);
        cursor = end;
        while (*cursor == ',' || *cursor == ' ' || *cursor == '\t' || *cursor == '\n') {
            cursor++;
        }
    }
    return tokens;
}

static std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int) ch << std::dec;
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

static void json_write_tokens(std::ostream & out, const llama_tokens & tokens) {
    out << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << tokens[i];
    }
    out << "]";
}

static void json_write_top_logits(std::ostream & out, const std::vector<uocr_trace_top_logit> & logits) {
    out << "[";
    for (size_t i = 0; i < logits.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << "{\"token_id\":" << logits[i].token_id
            << ",\"logit\":" << std::setprecision(9) << logits[i].logit
            << ",\"piece\":\"" << json_escape(logits[i].piece) << "\"}";
    }
    out << "]";
}

static std::vector<uocr_trace_top_logit> collect_top_logits(
        llama_context * lctx,
        const llama_vocab * vocab,
        int top_k) {
    std::vector<uocr_trace_top_logit> top;
    if (top_k <= 0) {
        return top;
    }

    llama_synchronize(lctx);
    const float * logits = llama_get_logits_ith(lctx, -1);
    if (logits == nullptr) {
        return top;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    top.reserve((size_t) std::min(top_k, n_vocab));
    for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
        const float logit = logits[token_id];
        if ((int) top.size() < top_k) {
            top.push_back({token_id, logit, ""});
            continue;
        }
        auto min_it = std::min_element(top.begin(), top.end(), [](const auto & a, const auto & b) {
            return a.logit < b.logit;
        });
        if (min_it != top.end() && logit > min_it->logit) {
            *min_it = {token_id, logit, ""};
        }
    }

    std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) {
        return a.logit > b.logit;
    });
    for (auto & item : top) {
        item.piece = common_token_to_piece(vocab, item.token_id, true);
    }
    return top;
}

static int rank_in_top_logits(const std::vector<uocr_trace_top_logit> & logits, llama_token token_id) {
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i].token_id == token_id) {
            return (int) i;
        }
    }
    return -1;
}

static std::string chunk_type_name(mtmd_input_chunk_type type) {
    switch (type) {
        case MTMD_INPUT_CHUNK_TYPE_TEXT:  return "text";
        case MTMD_INPUT_CHUNK_TYPE_IMAGE: return "image";
        case MTMD_INPUT_CHUNK_TYPE_AUDIO: return "audio";
    }
    return "unknown";
}

static std::vector<uocr_trace_decoder_pos> sample_decoder_positions(const mtmd_input_chunk * chunk) {
    std::vector<uocr_trace_decoder_pos> sample;
    const mtmd_image_tokens * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
    if (image_tokens == nullptr) {
        return sample;
    }

    const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    const size_t n_head = std::min<size_t>(4, n_tokens);
    for (size_t i = 0; i < n_head; ++i) {
        sample.push_back({i, mtmd_image_tokens_get_decoder_pos(image_tokens, 0, i)});
    }
    if (n_tokens > n_head) {
        const size_t tail_start = n_tokens > 4 ? n_tokens - 4 : n_head;
        for (size_t i = tail_start; i < n_tokens; ++i) {
            sample.push_back({i, mtmd_image_tokens_get_decoder_pos(image_tokens, 0, i)});
        }
    }
    return sample;
}

static uocr_trace_embedding summarize_embedding(
        size_t chunk_index,
        const float * embd,
        size_t n_tokens,
        int n_embd) {
    uocr_trace_embedding summary;
    summary.chunk_index = chunk_index;
    summary.n_tokens = n_tokens;
    summary.n_embd = n_embd;
    if (embd == nullptr || n_tokens == 0 || n_embd <= 0) {
        return summary;
    }

    const size_t n_values = n_tokens * (size_t) n_embd;
    summary.min = embd[0];
    summary.max = embd[0];
    const size_t n_first = std::min<size_t>(8, n_values);
    summary.first_values.assign(embd, embd + n_first);
    for (size_t i = 0; i < n_values; ++i) {
        const float value = embd[i];
        summary.sum += value;
        summary.abs_sum += std::fabs(value);
        summary.min = std::min(summary.min, value);
        summary.max = std::max(summary.max, value);
    }
    return summary;
}

static uocr_trace_output_embedding summarize_output_embedding(
        const std::string & phase,
        int index,
        llama_token token_id,
        const float * embd,
        int n_embd) {
    uocr_trace_output_embedding summary;
    summary.phase = phase;
    summary.index = index;
    summary.token_id = token_id;
    summary.n_embd = n_embd;
    if (embd == nullptr || n_embd <= 0) {
        return summary;
    }

    summary.min = embd[0];
    summary.max = embd[0];
    const size_t n_first = std::min<size_t>(8, (size_t) n_embd);
    summary.first_values.assign(embd, embd + n_first);
    for (int i = 0; i < n_embd; ++i) {
        const float value = embd[i];
        summary.sum += value;
        summary.abs_sum += std::fabs(value);
        summary.min = std::min(summary.min, value);
        summary.max = std::max(summary.max, value);
    }
    return summary;
}

static void write_uocr_trace(const uocr_trace & trace) {
    if (!trace.enabled || trace.path.empty()) {
        return;
    }

    std::filesystem::path path(trace.path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        LOG_WRN("%s: failed to open trace path '%s'\n", __func__, trace.path.c_str());
        return;
    }

    size_t text_token_count = 0;
    size_t media_token_count = 0;
    for (const auto & chunk : trace.chunks) {
        if (chunk.type == "text") {
            text_token_count += chunk.n_tokens;
        } else {
            media_token_count += chunk.n_tokens;
        }
    }

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"engine\": \"llamacpp\",\n";
    out << "  \"tool\": \"" << json_escape(trace.tool) << "\",\n";
    out << "  \"model_path\": \"" << json_escape(trace.model_path) << "\",\n";
    out << "  \"mmproj_path\": \"" << json_escape(trace.mmproj_path) << "\",\n";
    out << "  \"chat_template\": \"" << json_escape(trace.chat_template) << "\",\n";
    out << "  \"n_predict\": " << trace.n_predict << ",\n";
    out << "  \"n_vocab\": " << trace.n_vocab << ",\n";
    out << "  \"n_embd\": " << trace.n_embd << ",\n";
    out << "  \"n_embd_inp\": " << trace.n_embd_inp << ",\n";
    out << "  \"n_ctx\": " << trace.n_ctx << ",\n";
    out << "  \"output_embeddings_enabled\": " << (trace.output_embeddings_enabled ? "true" : "false") << ",\n";
    out << "  \"add_special\": " << (trace.add_special ? "true" : "false") << ",\n";
    out << "  \"parse_special\": " << (trace.parse_special ? "true" : "false") << ",\n";
    out << "  \"formatted_prompt\": \"" << json_escape(trace.formatted_prompt) << "\",\n";
    out << "  \"prefill_n_past\": " << trace.prefill_n_past << ",\n";
    out << "  \"final_n_past\": " << trace.final_n_past << ",\n";
    out << "  \"text_token_count\": " << text_token_count << ",\n";
    out << "  \"media_token_count\": " << media_token_count << ",\n";
    out << "  \"stop_reason\": \"" << json_escape(trace.stop_reason) << "\",\n";

    out << "  \"prefill_top_logits\": ";
    json_write_top_logits(out, trace.prefill_top_logits);
    out << ",\n";

    out << "  \"chunks\": [\n";
    for (size_t i = 0; i < trace.chunks.size(); ++i) {
        const auto & chunk = trace.chunks[i];
        out << "    {\"index\":" << chunk.index
            << ",\"type\":\"" << json_escape(chunk.type) << "\""
            << ",\"n_tokens\":" << chunk.n_tokens
            << ",\"n_pos\":" << chunk.n_pos
            << ",\"id\":\"" << json_escape(chunk.id) << "\"";
        if (!chunk.text_tokens.empty()) {
            out << ",\"text_tokens\":";
            json_write_tokens(out, chunk.text_tokens);
        }
        if (!chunk.decoder_pos_sample.empty()) {
            out << ",\"decoder_pos_sample\":[";
            for (size_t j = 0; j < chunk.decoder_pos_sample.size(); ++j) {
                if (j) {
                    out << ",";
                }
                const auto & sample = chunk.decoder_pos_sample[j];
                out << "{\"index\":" << sample.index
                    << ",\"t\":" << sample.pos.t
                    << ",\"x\":" << sample.pos.x
                    << ",\"y\":" << sample.pos.y
                    << ",\"z\":" << sample.pos.z << "}";
            }
            out << "]";
        }
        out << "}" << (i + 1 == trace.chunks.size() ? "\n" : ",\n");
    }
    out << "  ],\n";

    out << "  \"embeddings\": [\n";
    for (size_t i = 0; i < trace.embeddings.size(); ++i) {
        const auto & embd = trace.embeddings[i];
        out << "    {\"chunk_index\":" << embd.chunk_index
            << ",\"n_tokens\":" << embd.n_tokens
            << ",\"n_embd\":" << embd.n_embd
            << ",\"sum\":" << std::setprecision(12) << embd.sum
            << ",\"abs_sum\":" << std::setprecision(12) << embd.abs_sum
            << ",\"min\":" << std::setprecision(9) << embd.min
            << ",\"max\":" << std::setprecision(9) << embd.max
            << ",\"first_values\":[";
        for (size_t j = 0; j < embd.first_values.size(); ++j) {
            if (j) {
                out << ",";
            }
            out << std::setprecision(9) << embd.first_values[j];
        }
        out << "]}" << (i + 1 == trace.embeddings.size() ? "\n" : ",\n");
    }
    out << "  ],\n";

    out << "  \"output_embeddings\": [\n";
    for (size_t i = 0; i < trace.output_embeddings.size(); ++i) {
        const auto & embd = trace.output_embeddings[i];
        out << "    {\"phase\":\"" << json_escape(embd.phase) << "\""
            << ",\"index\":" << embd.index
            << ",\"token_id\":" << embd.token_id
            << ",\"n_embd\":" << embd.n_embd
            << ",\"sum\":" << std::setprecision(12) << embd.sum
            << ",\"abs_sum\":" << std::setprecision(12) << embd.abs_sum
            << ",\"min\":" << std::setprecision(9) << embd.min
            << ",\"max\":" << std::setprecision(9) << embd.max
            << ",\"first_values\":[";
        for (size_t j = 0; j < embd.first_values.size(); ++j) {
            if (j) {
                out << ",";
            }
            out << std::setprecision(9) << embd.first_values[j];
        }
        out << "]}" << (i + 1 == trace.output_embeddings.size() ? "\n" : ",\n");
    }
    out << "  ],\n";

    out << "  \"generation\": [\n";
    for (size_t i = 0; i < trace.generation.size(); ++i) {
        const auto & step = trace.generation[i];
        out << "    {\"index\":" << step.index
            << ",\"token_id\":" << step.token_id
            << ",\"piece\":\"" << json_escape(step.piece) << "\""
            << ",\"is_eog\":" << (step.is_eog ? "true" : "false")
            << ",\"is_antiprompt\":" << (step.is_antiprompt ? "true" : "false")
            << ",\"raw_top_rank\":" << step.raw_top_rank
            << ",\"banned_tokens\":";
        json_write_tokens(out, step.banned_tokens);
        out << ",\"top_logits\":";
        json_write_top_logits(out, step.top_logits);
        out << "}" << (i + 1 == trace.generation.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

static llama_tokens banned_ngram_tokens(
        const llama_tokens & origin_tokens,
        const llama_tokens & generated_tokens,
        const mtmd_no_repeat_ngram & config) {
    llama_tokens banned;
    const int ngram_size = config.ngram_size;
    const int window_size = config.window_size;
    if (!config.enabled || ngram_size <= 0 || window_size <= 0) {
        return banned;
    }

    llama_tokens tokens;
    tokens.reserve(origin_tokens.size() + generated_tokens.size());
    tokens.insert(tokens.end(), origin_tokens.begin(), origin_tokens.end());
    tokens.insert(tokens.end(), generated_tokens.begin(), generated_tokens.end());

    if ((int) tokens.size() < ngram_size) {
        return banned;
    }

    const int search_start = std::max(0, (int) tokens.size() - window_size);
    const int search_end = (int) tokens.size() - ngram_size + 1;
    if (search_end <= search_start) {
        return banned;
    }

    std::unordered_set<llama_token> banned_set;
    for (int idx = search_start; idx < search_end; ++idx) {
        bool matches = true;
        for (int offset = 0; offset < ngram_size - 1; ++offset) {
            if (tokens[idx + offset] != tokens[tokens.size() - (ngram_size - 1) + offset]) {
                matches = false;
                break;
            }
        }
        if (matches) {
            banned_set.insert(tokens[idx + ngram_size - 1]);
        }
    }

    for (const llama_token token : config.whitelist_tokens) {
        banned_set.erase(token);
    }

    banned.assign(banned_set.begin(), banned_set.end());
    return banned;
}

static bool prune_decode_history(
        llama_context * lctx,
        const mtmd_prefill_aware_swa & config,
        llama_pos prefill_end,
        llama_pos n_past,
        llama_pos & removed_until) {
    if (!config.enabled || config.decode_window <= 0) {
        return true;
    }
    if (!config.legacy_kv_prune) {
        return true;
    }

    const llama_pos remove_start = std::max(prefill_end, removed_until);
    const llama_pos remove_end = n_past - config.decode_window;
    if (remove_end <= remove_start) {
        return true;
    }

    llama_memory_t mem = llama_get_memory(lctx);
    if (!llama_memory_seq_rm(mem, 0, remove_start, remove_end)) {
        LOG_WRN("%s: failed to prune generated KV positions [%d, %d)\n",
                __func__, (int) remove_start, (int) remove_end);
        return false;
    }

    removed_until = remove_end;
    return true;
}

struct mtmd_cli_context {
    mtmd::context_ptr ctx_vision;
    common_init_result_ptr llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    common_sampler    * smpl;
    llama_batch         batch;
    int                 n_batch;

    mtmd::bitmaps bitmaps;
    std::vector<mtmd_helper::video_ptr> videos;

    mtmd::batch_ptr mbatch;

    // chat template
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;
    // TODO: support for --system-prompt with /clear command

    // support for legacy templates (models not having EOT token)
    llama_tokens antiprompt_tokens;

    int n_threads    = 1;
    llama_pos n_past = 0;
    mtmd_no_repeat_ngram no_repeat_ngram;
    mtmd_prefill_aware_swa prefill_aware_swa;
    mtmd_min_new_tokens min_new_tokens;
    llama_token media_history_token = LLAMA_TOKEN_NULL;
    llama_tokens eog_tokens;
    llama_tokens no_repeat_origin_tokens;
    uocr_trace trace;

    common_debug_cb_user_data cb_data;

    mtmd_cli_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init->model();
        lctx = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1); // batch for next token generation
        n_batch = params.n_batch;
        no_repeat_ngram.enabled = env_enabled("LLAMA_DEEPSEEK_OCR_NO_REPEAT_NGRAM");
        no_repeat_ngram.ngram_size = env_int("LLAMA_DEEPSEEK_OCR_NGRAM_SIZE", 30);
        no_repeat_ngram.window_size = env_int("LLAMA_DEEPSEEK_OCR_NGRAM_WINDOW", 90);
        no_repeat_ngram.whitelist_tokens = env_tokens(
                "LLAMA_DEEPSEEK_OCR_NGRAM_WHITELIST",
                no_repeat_ngram.whitelist_tokens);
        prefill_aware_swa.enabled = env_enabled("LLAMA_DEEPSEEK_OCR_PREFILL_AWARE_SWA");
        prefill_aware_swa.legacy_kv_prune = env_enabled("LLAMA_DEEPSEEK_OCR_LEGACY_KV_PRUNE");
        prefill_aware_swa.decode_window = env_int("LLAMA_DEEPSEEK_OCR_DECODE_WINDOW", 128);
        min_new_tokens.n_tokens = env_int("LLAMA_DEEPSEEK_OCR_MIN_NEW_TOKENS", 0);
        const char * trace_path = std::getenv("LLAMA_UOCR_PARITY_DUMP");
        trace.enabled = trace_path != nullptr && trace_path[0] != '\0';
        if (trace.enabled) {
            trace.path = trace_path;
            trace.top_k = env_int("LLAMA_UOCR_PARITY_TOPK", 8);
            trace.output_embeddings_enabled = env_enabled("LLAMA_UOCR_PARITY_OUTPUT_EMBEDDINGS");
            trace.tool = "llama-uocr-parity";
            trace.model_path = params.model.path;
            trace.mmproj_path = params.mmproj.path;
            trace.chat_template = params.chat_template;
            trace.n_predict = params.n_predict;
        }
        if (no_repeat_ngram.enabled) {
            LOG_INF("%s: DeepSeek-OCR no-repeat ngram enabled, ngram_size=%d, window_size=%d, whitelist_tokens=%zu\n",
                    __func__, no_repeat_ngram.ngram_size, no_repeat_ngram.window_size,
                    no_repeat_ngram.whitelist_tokens.size());
        }
        if (prefill_aware_swa.enabled) {
            LOG_INF("%s: DeepSeek-OCR prefill-aware SWA flag enabled; core R-SWA handles masking, legacy KV prune=%d, decode_window=%d\n",
                    __func__, (int) prefill_aware_swa.legacy_kv_prune, prefill_aware_swa.decode_window);
        }
        if (min_new_tokens.n_tokens > 0) {
            LOG_INF("%s: DeepSeek-OCR min-new-tokens experiment enabled, n_tokens=%d\n",
                    __func__, min_new_tokens.n_tokens);
        }

        if (!model || !lctx) {
#ifdef UOCR_FFI_LIBRARY
            throw std::runtime_error("failed to load language model");
#else
            exit(1);
#endif
        }

        if (trace.enabled && trace.output_embeddings_enabled) {
            llama_set_embeddings(lctx, true);
        }

        media_history_token = llama_vocab_n_tokens(vocab) + 1000000;
        if (trace.enabled) {
            trace.n_vocab = llama_vocab_n_tokens(vocab);
            trace.n_embd = llama_model_n_embd(model);
            trace.n_embd_inp = llama_model_n_embd_inp(model);
            trace.n_ctx = llama_n_ctx(lctx);
        }
        for (llama_token token_id = 0; token_id < llama_vocab_n_tokens(vocab); token_id++) {
            if (llama_vocab_is_eog(vocab, token_id)) {
                eog_tokens.push_back(token_id);
            }
        }

        if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template.\n");
            LOG_ERR("  For old llava models, you may need to use '--chat-template vicuna'\n");
            LOG_ERR("  For MobileVLM models, use '--chat-template deepseek'\n");
            LOG_ERR("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
#ifdef UOCR_FFI_LIBRARY
            throw std::runtime_error("model does not have a chat template; pass --chat-template");
#else
            exit(1);
#endif
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
        use_jinja = params.use_jinja;
        chat_history.clear();
        LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());

        init_vision_context(params);

        // load antiprompt tokens for legacy templates
        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~mtmd_cli_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }

    void init_vision_context(common_params & params) {
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.print_timings    = true;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        if (std::getenv("MTMD_DEBUG_GRAPH") != nullptr) {
            mparams.cb_eval_user_data = &cb_data;
            mparams.cb_eval = common_debug_cb_eval;
        }
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
#ifdef UOCR_FFI_LIBRARY
            throw std::runtime_error("failed to load vision model");
#else
            exit(1);
#endif
        }
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    bool load_media(const std::string & fname) {
        auto res = mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname.c_str(), false);
        if (!res.bitmap) {
            return false;
        }
        bitmaps.entries.emplace_back(res.bitmap);
        if (res.video_ctx) {
            videos.emplace_back(res.video_ctx);
        }
        return true;
    }
};

static int generate_response(mtmd_cli_context & ctx, int n_predict) {
    llama_tokens generated_tokens;
    const llama_pos prefill_end = ctx.n_past;
    llama_pos decode_kv_removed_until = prefill_end;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            LOG("\n");
            if (ctx.trace.stop_reason.empty()) {
                ctx.trace.stop_reason = g_is_interrupted ? "interrupted" : "stopped";
            }
            break;
        }

        llama_tokens banned = banned_ngram_tokens(
                ctx.no_repeat_origin_tokens,
                generated_tokens,
                ctx.no_repeat_ngram);
        if (i < ctx.min_new_tokens.n_tokens) {
            banned.insert(banned.end(), ctx.eog_tokens.begin(), ctx.eog_tokens.end());
        }
        std::vector<uocr_trace_top_logit> top_logits;
        if (ctx.trace.enabled) {
            top_logits = collect_top_logits(ctx.lctx, ctx.vocab, ctx.trace.top_k);
        }
        llama_token token_id = banned.empty()
            ? common_sampler_sample(ctx.smpl, ctx.lctx, -1)
            : common_sampler_sample_with_banned(ctx.smpl, ctx.lctx, -1, banned);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);
        const bool is_eog = llama_vocab_is_eog(ctx.vocab, token_id);
        const bool is_antiprompt = ctx.check_antiprompt(generated_tokens);

        if (ctx.trace.enabled) {
            uocr_trace_generation_step step;
            step.index = i;
            step.token_id = token_id;
            step.piece = common_token_to_piece(ctx.lctx, token_id);
            step.is_eog = is_eog;
            step.is_antiprompt = is_antiprompt;
            step.raw_top_rank = rank_in_top_logits(top_logits, token_id);
            step.banned_tokens = banned;
            step.top_logits = std::move(top_logits);
            ctx.trace.generation.push_back(std::move(step));
        }

        if (is_eog || is_antiprompt) {
            if (ctx.trace.stop_reason.empty()) {
                ctx.trace.stop_reason = is_eog ? "eog" : "antiprompt";
            }
            LOG("\n");
            break; // end of generation
        }

        LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (g_is_interrupted) {
            LOG("\n");
            break;
        }

        // eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
        if (ctx.trace.enabled && ctx.trace.output_embeddings_enabled) {
            ctx.trace.output_embeddings.push_back(summarize_output_embedding(
                        "generation",
                        i,
                        token_id,
                        llama_get_embeddings_ith(ctx.lctx, -1),
                        llama_model_n_embd(ctx.model)));
        }

        if (!prune_decode_history(
                    ctx.lctx,
                    ctx.prefill_aware_swa,
                    prefill_end,
                    ctx.n_past,
                    decode_kv_removed_until)) {
            return 1;
        }
    }
    if (ctx.trace.enabled && ctx.trace.stop_reason.empty()) {
        ctx.trace.stop_reason = generated_tokens.size() >= (size_t) n_predict ? "length" : "stopped";
    }

    std::string generated_text = common_detokenize(ctx.lctx, generated_tokens);
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = generated_text;
    ctx.chat_history.push_back(std::move(msg));

    return 0;
}

static std::string chat_add_and_format(mtmd_cli_context & ctx, common_chat_msg & new_msg) {
    LOG_DBG("chat_add_and_format: new_msg.role='%s', new_msg.content='%s'\n",
        new_msg.role.c_str(), new_msg.content.c_str());
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history,
        new_msg, new_msg.role == "user",
        ctx.use_jinja);
    ctx.chat_history.push_back(new_msg);
    return formatted;
}

static int eval_message(mtmd_cli_context & ctx, common_chat_msg & msg) {
    inject_test_response_marker();

    bool add_bos = ctx.chat_history.empty();
    auto formatted_chat = chat_add_and_format(ctx, msg);
    LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.c_str());
    if (ctx.trace.enabled) {
        ctx.trace.formatted_prompt = formatted_chat;
        ctx.trace.add_special = add_bos;
        ctx.trace.parse_special = true;
    }

    mtmd_input_text text;
    text.text          = formatted_chat.c_str();
    text.add_special   = add_bos;
    text.parse_special = true;

    if (g_is_interrupted) return 0;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx.ctx_vision.get(),
                        chunks.ptr.get(), // output
                        &text, // text
                        bitmaps_c_ptr.data(),
                        bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

    ctx.bitmaps.entries.clear();
    ctx.videos.clear();

    // batch encode all media chunks, then decode each
    size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    if (ctx.trace.enabled) {
        ctx.trace.chunks.clear();
        ctx.trace.embeddings.clear();
        ctx.trace.output_embeddings.clear();
        ctx.trace.prefill_top_logits.clear();
        ctx.trace.generation.clear();
        ctx.trace.stop_reason.clear();
        for (size_t i = 0; i < n_chunks; ++i) {
            auto chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
            uocr_trace_chunk trace_chunk;
            trace_chunk.index = i;
            trace_chunk.type = chunk_type_name(mtmd_input_chunk_get_type(chunk));
            trace_chunk.n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            trace_chunk.n_pos = mtmd_input_chunk_get_n_pos(chunk);
            const char * id = mtmd_input_chunk_get_id(chunk);
            trace_chunk.id = id ? id : "";
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_text_tokens = 0;
                const llama_token * text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_text_tokens);
                trace_chunk.text_tokens.assign(text_tokens, text_tokens + n_text_tokens);
            } else if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                trace_chunk.decoder_pos_sample = sample_decoder_positions(chunk);
            }
            ctx.trace.chunks.push_back(std::move(trace_chunk));
        }
    }
    for (size_t i = 0; i < n_chunks; i++) {
        auto chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_text_tokens = 0;
            const llama_token * text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_text_tokens);
            ctx.no_repeat_origin_tokens.insert(
                    ctx.no_repeat_origin_tokens.end(),
                    text_tokens,
                    text_tokens + n_text_tokens);

            // decode text chunk
            llama_pos new_n_past = ctx.n_past;
            res = mtmd_helper_eval_chunk_single(ctx.ctx_vision.get(),
                        ctx.lctx,
                        chunk,
                        ctx.n_past,
                        0, // seq_id
                        ctx.n_batch,
                        i == n_chunks - 1, // logits_last
                        &new_n_past);
            if (res != 0) {
                LOG_ERR("Unable to eval text chunk %zu\n", i);
                return 1;
            }
            ctx.n_past = new_n_past;
        } else {
            ctx.no_repeat_origin_tokens.insert(
                    ctx.no_repeat_origin_tokens.end(),
                    mtmd_input_chunk_get_n_tokens(chunk),
                    ctx.media_history_token);

            // media chunk: try to get embd from existing batch, or create a new batch
            float * embd = nullptr;
            if (ctx.mbatch) {
                embd = mtmd_batch_get_output_embd(ctx.mbatch.get(), chunk);

                if (embd) {
                    LOG_DBG("found embd for media chunk %zu in existing batch\n", i);
                } else {
                    LOG_DBG("media chunk %zu not found in existing batch, creating new batch\n", i);
                }
            }

            if (!embd) {
                // create and encode a new batch with as many media chunks as possible
                ctx.mbatch.reset(mtmd_batch_init(ctx.ctx_vision.get()));
                res = mtmd_batch_add_chunk(ctx.mbatch.get(), chunk);
                GGML_ASSERT(res == 0); // first chunk must always succeed

                int n_added = 1;
                // add as many subsequent media chunks as possible
                for (size_t j = i + 1; j < n_chunks; j++) {
                    auto next_chunk = mtmd_input_chunks_get(chunks.ptr.get(), j);
                    auto next_type = mtmd_input_chunk_get_type(next_chunk);
                    if (next_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        break; // text chunk splits the batch
                    }
                    res = mtmd_batch_add_chunk(ctx.mbatch.get(), next_chunk);
                    if (res != 0) {
                        break; // batch full or incompatible
                    }
                    n_added++;
                }

                int64_t time_start = ggml_time_ms();
                LOG_INF("encoding mtmd batch, n_chunks = %d (done = %zu, total = %zu)\n", n_added, i, n_chunks);
                res = mtmd_batch_encode(ctx.mbatch.get());
                if (res != 0) {
                    LOG_ERR("Failed to encode mtmd batch, res = %d\n", res);
                    return 1;
                }
                LOG_INF("mtmd batch encoding done in %d ms\n", (int)(ggml_time_ms() - time_start));

                embd = mtmd_batch_get_output_embd(ctx.mbatch.get(), chunk);
            }

            GGML_ASSERT(embd != nullptr);
            if (ctx.trace.enabled) {
                ctx.trace.embeddings.push_back(summarize_embedding(
                            i,
                            embd,
                            mtmd_input_chunk_get_n_tokens(chunk),
                            llama_model_n_embd_inp(ctx.model)));
            }

            llama_pos new_n_past = ctx.n_past;
            res = mtmd_helper_decode_image_chunk(ctx.ctx_vision.get(),
                        ctx.lctx,
                        chunk,
                        embd,
                        ctx.n_past,
                        0, // seq_id
                        ctx.n_batch,
                        &new_n_past,
                        nullptr, // callback
                        nullptr  // user_data
                    );
            if (res != 0) {
                LOG_ERR("Unable to decode media chunk %zu\n", i);
                return 1;
            }
            ctx.n_past = new_n_past;
        }
    }
    if (ctx.trace.enabled) {
        ctx.trace.prefill_n_past = ctx.n_past;
        ctx.trace.prefill_top_logits = collect_top_logits(ctx.lctx, ctx.vocab, ctx.trace.top_k);
        if (ctx.trace.output_embeddings_enabled) {
            ctx.trace.output_embeddings.push_back(summarize_output_embedding(
                        "prefill_last",
                        0,
                        LLAMA_TOKEN_NULL,
                        llama_get_embeddings_ith(ctx.lctx, -1),
                        llama_model_n_embd(ctx.model)));
        }
    }

    LOG("\n");

    return 0;
}

#ifndef UOCR_FFI_LIBRARY
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    ggml_backend_load_all();

    mtmd_cli_context ctx(params);
    LOG_INF("%s: loading model: %s\n", __func__, params.model.path.c_str());

    bool is_single_turn = !params.prompt.empty() && !params.image.empty();

    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    // Ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (g_is_interrupted) return 130;

    auto eval_system_prompt_if_present = [&] {
        if (params.system_prompt.empty()) {
            return 0;
        }

        common_chat_msg msg;
        msg.role = "system";
        msg.content = params.system_prompt;
        return eval_message(ctx, msg);
    };

    LOG_WRN("WARN: This is an experimental CLI for testing multimodal capability.\n");
    LOG_WRN("      For normal use cases, please use the standard llama-cli\n");

    if (eval_system_prompt_if_present()) {
        return 1;
    }

    if (is_single_turn) {
        g_is_generating = true;
        if (params.prompt.find(mtmd_default_marker()) == std::string::npos) {
            for (size_t i = 0; i < params.image.size(); i++) {
                // most models require the marker before each image
                // ref: https://github.com/ggml-org/llama.cpp/pull/17616
                params.prompt = mtmd_default_marker() + params.prompt;
            }
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = params.prompt;
        for (const auto & image : params.image) {
            if (!ctx.load_media(image)) {
                return 1; // error is already printed by libmtmd
            }
        }
        if (eval_message(ctx, msg)) {
            return 1;
        }
        if (!g_is_interrupted && generate_response(ctx, n_predict)) {
            return 1;
        }

    } else {
        LOG("\n Running in chat mode, available commands:");
        if (mtmd_support_vision(ctx.ctx_vision.get())) {
            LOG("\n   /image <path>    load an image");
        }
        if (mtmd_support_audio(ctx.ctx_vision.get())) {
            LOG("\n   /audio <path>    load an audio");
        }
        if (mtmd_helper_support_video(ctx.ctx_vision.get())) {
            LOG("\n   /video <path>    load a video");
        }
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");

        std::string content;

        while (!g_is_interrupted) {
            g_is_generating = false;
            LOG("\n> ");
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            std::string line;
            console::readline(line, false);
            if (g_is_interrupted) break;
            console::set_display(DISPLAY_TYPE_RESET);
            line = string_strip(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/clear") {
                ctx.n_past = 0;
                ctx.chat_history.clear();
                llama_memory_clear(llama_get_memory(ctx.lctx), true);
                if (eval_system_prompt_if_present()) {
                    return 1;
                }
                LOG("Chat history cleared\n\n");
                continue;
            }
            g_is_generating = true;
            bool is_image = line == "/image" || line.find("/image ") == 0;
            bool is_audio = line == "/audio" || line.find("/audio ") == 0;
            bool is_video = line == "/video" || line.find("/video ") == 0;
            if (is_image || is_audio || is_video) {
                if (line.size() < 8) {
                    LOG_ERR("ERR: Missing media filename\n");
                    continue;
                }
                std::string media_path = line.substr(7);
                if (ctx.load_media(media_path)) {
                    LOG("%s %s loaded\n", media_path.c_str(), is_image ? "image" : is_audio ? "audio" : "video");
                    content += mtmd_default_marker();
                }
                // else, error is already printed by libmtmd
                continue;
            } else {
                content += line;
            }
            common_chat_msg msg;
            msg.role = "user";
            msg.content = content;
            int ret = eval_message(ctx, msg);
            if (ret) {
                return 1;
            }
            if (g_is_interrupted) break;
            if (generate_response(ctx, n_predict)) {
                return 1;
            }
            content.clear();
        }
    }
    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(ctx.lctx);
    if (ctx.trace.enabled) {
        ctx.trace.final_n_past = ctx.n_past;
        write_uocr_trace(ctx.trace);
    }
    return g_is_interrupted ? 130 : 0;
}
#endif // UOCR_FFI_LIBRARY

#ifdef UOCR_FFI_LIBRARY

struct uocr_ffi_session {
    common_params params;
    std::unique_ptr<mtmd_cli_context> ctx;
    std::string last_error;
    uocr_ffi_status last_status = UOCR_FFI_STATUS_OK;
    std::mutex mutex;
    uint64_t run_count = 0;
};

static thread_local std::string g_uocr_ffi_last_error;
static std::once_flag g_uocr_ffi_init_once;

static void uocr_ffi_set_error(uocr_ffi_session * session, uocr_ffi_status status, const std::string & message) {
    if (session) {
        session->last_status = status;
        session->last_error = message;
    } else {
        g_uocr_ffi_last_error = message;
    }
}

static const char * uocr_ffi_str(const char * value, const char * fallback = "") {
    return value && value[0] != '\0' ? value : fallback;
}

static void uocr_ffi_setenv(const char * name, const std::string & value, bool enabled) {
#if defined(_WIN32)
    _putenv_s(name, enabled ? value.c_str() : "");
#else
    if (enabled) {
        setenv(name, value.c_str(), 1);
    } else {
        unsetenv(name);
    }
#endif
}

static void uocr_ffi_apply_env(const uocr_ffi_params & input) {
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_GUNDAM", "1", input.gundam_mode != 0);
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_NO_IMAGE_END", "1", input.no_image_end != 0);

    const bool no_repeat = input.no_repeat_ngram != 0;
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_NO_REPEAT_NGRAM", "1", no_repeat);
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_NGRAM_SIZE", std::to_string(input.ngram_size > 0 ? input.ngram_size : 30), no_repeat);
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_NGRAM_WINDOW", std::to_string(input.ngram_window > 0 ? input.ngram_window : 90), no_repeat);
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_NGRAM_WHITELIST", uocr_ffi_str(input.ngram_whitelist, "128821,128822"), no_repeat);

    const bool prefill_swa = input.prefill_aware_swa != 0;
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_PREFILL_AWARE_SWA", "1", prefill_swa);
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_LEGACY_KV_PRUNE", "1", prefill_swa && input.legacy_kv_prune != 0);
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_DECODE_WINDOW", std::to_string(input.decode_window > 0 ? input.decode_window : 128), prefill_swa);
    uocr_ffi_setenv("LLAMA_DEEPSEEK_OCR_MIN_NEW_TOKENS", std::to_string(input.min_new_tokens), input.min_new_tokens > 0);
}

static common_params uocr_ffi_parse_params(const uocr_ffi_params & input) {
    if (!input.model_path || input.model_path[0] == '\0') {
        throw std::runtime_error("model_path is required");
    }
    if (!input.mmproj_path || input.mmproj_path[0] == '\0') {
        throw std::runtime_error("mmproj_path is required");
    }

    std::vector<std::string> args;
    auto add_arg = [&](const std::string & value) {
        args.push_back(value);
    };

    add_arg("uocr-ffi");
    add_arg("-m");
    add_arg(input.model_path);
    add_arg("--mmproj");
    add_arg(input.mmproj_path);
    add_arg("--chat-template");
    add_arg(uocr_ffi_str(input.chat_template, "deepseek-ocr"));
    add_arg("--temp");
    add_arg("0");
    add_arg("--top-k");
    add_arg("1");
    add_arg("-c");
    add_arg(std::to_string(input.ctx_size > 0 ? input.ctx_size : 32768));
    add_arg("-b");
    add_arg(std::to_string(input.n_batch > 0 ? input.n_batch : 2048));
    add_arg("-ngl");
    if (input.n_gpu_layers <= -2) {
        add_arg("all");
    } else if (input.n_gpu_layers == -1) {
        add_arg("auto");
    } else {
        add_arg(std::to_string(input.n_gpu_layers));
    }
    add_arg("--log-verbosity");
    add_arg(std::to_string(input.log_verbosity > 0 ? input.log_verbosity : 2));
    if (input.force_prompt_eos != 0) {
        add_arg("--override-kv");
        add_arg("tokenizer.ggml.add_eos_token=bool:true");
    }

    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto & arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }

    common_params params;
    if (!common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        throw std::runtime_error("failed to parse native OCR parameters");
    }
    if (params.mmproj.path.empty()) {
        throw std::runtime_error("mmproj_path is required");
    }
    return params;
}

static void uocr_ffi_runtime_init_once() {
    std::call_once(g_uocr_ffi_init_once, []() {
        std::setlocale(LC_NUMERIC, "C");
        ggml_time_init();
        common_init();
        mtmd_helper_log_set(common_log_default_callback, nullptr);
        ggml_backend_load_all();
    });
}

static void uocr_ffi_reset_context(mtmd_cli_context & ctx) {
    g_is_generating = false;
    g_is_interrupted = false;
    ctx.n_past = 0;
    ctx.chat_history.clear();
    ctx.no_repeat_origin_tokens.clear();
    ctx.bitmaps.entries.clear();
    ctx.videos.clear();
    ctx.mbatch.reset();
    common_sampler_reset(ctx.smpl);
    llama_memory_clear(llama_get_memory(ctx.lctx), true);
}

static bool uocr_ffi_emit_event(
        uocr_ffi_event_callback callback,
        void * user_data,
        uocr_ffi_event_type type,
        const std::string & text,
        uint64_t index) {
    if (!callback) {
        return true;
    }
    uocr_ffi_event event = {};
    event.struct_size = sizeof(event);
    event.type = (uint32_t) type;
    event.text_utf8 = text.c_str();
    event.text_len = text.size();
    event.index = index;
    return callback(&event, user_data) == 0;
}

static int uocr_ffi_generate_response(mtmd_cli_context & ctx, int n_predict, uocr_ffi_event_callback callback, void * user_data) {
    llama_tokens generated_tokens;
    const llama_pos prefill_end = ctx.n_past;
    llama_pos decode_kv_removed_until = prefill_end;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            if (ctx.trace.stop_reason.empty()) {
                ctx.trace.stop_reason = g_is_interrupted ? "interrupted" : "stopped";
            }
            break;
        }

        llama_tokens banned = banned_ngram_tokens(
                ctx.no_repeat_origin_tokens,
                generated_tokens,
                ctx.no_repeat_ngram);
        if (i < ctx.min_new_tokens.n_tokens) {
            banned.insert(banned.end(), ctx.eog_tokens.begin(), ctx.eog_tokens.end());
        }
        std::vector<uocr_trace_top_logit> top_logits;
        if (ctx.trace.enabled) {
            top_logits = collect_top_logits(ctx.lctx, ctx.vocab, ctx.trace.top_k);
        }
        llama_token token_id = banned.empty()
            ? common_sampler_sample(ctx.smpl, ctx.lctx, -1)
            : common_sampler_sample_with_banned(ctx.smpl, ctx.lctx, -1, banned);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);
        const bool is_eog = llama_vocab_is_eog(ctx.vocab, token_id);
        const bool is_antiprompt = ctx.check_antiprompt(generated_tokens);

        if (ctx.trace.enabled) {
            uocr_trace_generation_step step;
            step.index = i;
            step.token_id = token_id;
            step.piece = common_token_to_piece(ctx.lctx, token_id);
            step.is_eog = is_eog;
            step.is_antiprompt = is_antiprompt;
            step.raw_top_rank = rank_in_top_logits(top_logits, token_id);
            step.banned_tokens = banned;
            step.top_logits = std::move(top_logits);
            ctx.trace.generation.push_back(std::move(step));
        }

        if (is_eog || is_antiprompt) {
            if (ctx.trace.stop_reason.empty()) {
                ctx.trace.stop_reason = is_eog ? "eog" : "antiprompt";
            }
            break;
        }

        std::string piece = common_token_to_piece(ctx.lctx, token_id);
        if (!piece.empty()) {
            if (!uocr_ffi_emit_event(callback, user_data, UOCR_FFI_EVENT_TOKEN, piece, generated_tokens.size() - 1)) {
                ctx.trace.stop_reason = "cancelled";
                return 130;
            }
        }

        if (g_is_interrupted) {
            break;
        }

        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
        if (ctx.trace.enabled && ctx.trace.output_embeddings_enabled) {
            ctx.trace.output_embeddings.push_back(summarize_output_embedding(
                        "generation",
                        i,
                        token_id,
                        llama_get_embeddings_ith(ctx.lctx, -1),
                        llama_model_n_embd(ctx.model)));
        }

        if (!prune_decode_history(
                    ctx.lctx,
                    ctx.prefill_aware_swa,
                    prefill_end,
                    ctx.n_past,
                    decode_kv_removed_until)) {
            return 1;
        }
    }
    if (ctx.trace.enabled && ctx.trace.stop_reason.empty()) {
        ctx.trace.stop_reason = generated_tokens.size() >= (size_t) n_predict ? "length" : "stopped";
    }

    std::string generated_text = common_detokenize(ctx.lctx, generated_tokens);
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = generated_text;
    ctx.chat_history.push_back(std::move(msg));

    return 0;
}

extern "C" {

uint32_t uocr_ffi_abi_version(void) {
    return UOCR_FFI_ABI_VERSION;
}

const char * uocr_ffi_build_info(void) {
    return "uocr-ffi/1";
}

const char * uocr_ffi_media_marker(void) {
    return mtmd_default_marker();
}

uocr_ffi_session * uocr_ffi_create(const uocr_ffi_params * params) {
    g_uocr_ffi_last_error.clear();
    if (!params) {
        g_uocr_ffi_last_error = "params is null";
        return nullptr;
    }
    if (params->struct_size < sizeof(uocr_ffi_params)) {
        g_uocr_ffi_last_error = "unsupported uocr_ffi_params struct_size";
        return nullptr;
    }

    try {
        uocr_ffi_runtime_init_once();
        uocr_ffi_apply_env(*params);
        std::unique_ptr<uocr_ffi_session> session(new uocr_ffi_session());
        session->params = uocr_ffi_parse_params(*params);
        session->ctx.reset(new mtmd_cli_context(session->params));
        return session.release();
    } catch (const std::exception & exc) {
        g_uocr_ffi_last_error = exc.what();
    } catch (...) {
        g_uocr_ffi_last_error = "unknown error creating uocr ffi session";
    }
    return nullptr;
}

void uocr_ffi_destroy(uocr_ffi_session * session) {
    delete session;
}

uocr_ffi_status uocr_ffi_run_image(uocr_ffi_session * session, const uocr_ffi_request * request) {
    if (!session) {
        g_uocr_ffi_last_error = "session is null";
        return UOCR_FFI_STATUS_INVALID_ARGUMENT;
    }
    if (!request) {
        uocr_ffi_set_error(session, UOCR_FFI_STATUS_INVALID_ARGUMENT, "request is null");
        return UOCR_FFI_STATUS_INVALID_ARGUMENT;
    }
    if (request->struct_size < sizeof(uocr_ffi_request)) {
        uocr_ffi_set_error(session, UOCR_FFI_STATUS_UNSUPPORTED_ABI, "unsupported uocr_ffi_request struct_size");
        return UOCR_FFI_STATUS_UNSUPPORTED_ABI;
    }
    if (!request->image_path || request->image_path[0] == '\0') {
        uocr_ffi_set_error(session, UOCR_FFI_STATUS_INVALID_ARGUMENT, "image_path is required");
        return UOCR_FFI_STATUS_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> guard(session->mutex);
    session->last_error.clear();
    session->last_status = UOCR_FFI_STATUS_OK;
    try {
        mtmd_cli_context & ctx = *session->ctx;
        uocr_ffi_reset_context(ctx);

        std::string prompt = uocr_ffi_str(request->prompt, "document parsing.");
        if (prompt.find(mtmd_default_marker()) == std::string::npos) {
            prompt = std::string(mtmd_default_marker()) + prompt;
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = prompt;

        g_is_generating = true;
        if (!ctx.load_media(request->image_path)) {
            throw std::runtime_error("failed to load image");
        }
        if (eval_message(ctx, msg)) {
            throw std::runtime_error("failed to evaluate image prompt");
        }

        const int n_predict = request->max_tokens > 0 ? request->max_tokens : INT_MAX;
        if (!g_is_interrupted) {
            const int generation_status = uocr_ffi_generate_response(ctx, n_predict, request->event_callback, request->user_data);
            if (generation_status == 130) {
                throw std::runtime_error("cancelled");
            }
            if (generation_status) {
                throw std::runtime_error("failed during token generation");
            }
        }

        if (ctx.trace.enabled) {
            ctx.trace.final_n_past = ctx.n_past;
            write_uocr_trace(ctx.trace);
        }
        g_is_generating = false;
        session->run_count++;
        if (g_is_interrupted) {
            uocr_ffi_set_error(session, UOCR_FFI_STATUS_CANCELLED, "interrupted");
            return UOCR_FFI_STATUS_CANCELLED;
        }
        uocr_ffi_emit_event(request->event_callback, request->user_data, UOCR_FFI_EVENT_DONE, "", session->run_count);
        return UOCR_FFI_STATUS_OK;
    } catch (const std::exception & exc) {
        g_is_generating = false;
        const std::string message = exc.what();
        uocr_ffi_set_error(
                session,
                message == "cancelled" ? UOCR_FFI_STATUS_CANCELLED : UOCR_FFI_STATUS_RUN_FAILED,
                message);
    } catch (...) {
        g_is_generating = false;
        uocr_ffi_set_error(session, UOCR_FFI_STATUS_ERROR, "unknown error running OCR");
    }
    return session->last_status;
}

const char * uocr_ffi_last_error(uocr_ffi_session * session) {
    if (session) {
        return session->last_error.c_str();
    }
    return g_uocr_ffi_last_error.c_str();
}

uocr_ffi_status uocr_ffi_last_status(uocr_ffi_session * session) {
    return session ? session->last_status : UOCR_FFI_STATUS_ERROR;
}

uint64_t uocr_ffi_run_count(uocr_ffi_session * session) {
    return session ? session->run_count : 0;
}

} // extern "C"

#endif // UOCR_FFI_LIBRARY
