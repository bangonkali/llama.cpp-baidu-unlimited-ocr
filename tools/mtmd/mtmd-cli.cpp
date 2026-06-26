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

#include <vector>
#include <limits.h>
#include <cinttypes>
#include <clocale>
#include <cstdlib>
#include <cstring>
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
    int decode_window = 128;
};

struct mtmd_min_new_tokens {
    int n_tokens = 0;
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
        prefill_aware_swa.decode_window = env_int("LLAMA_DEEPSEEK_OCR_DECODE_WINDOW", 128);
        min_new_tokens.n_tokens = env_int("LLAMA_DEEPSEEK_OCR_MIN_NEW_TOKENS", 0);
        if (no_repeat_ngram.enabled) {
            LOG_INF("%s: DeepSeek-OCR no-repeat ngram enabled, ngram_size=%d, window_size=%d, whitelist_tokens=%zu\n",
                    __func__, no_repeat_ngram.ngram_size, no_repeat_ngram.window_size,
                    no_repeat_ngram.whitelist_tokens.size());
        }
        if (prefill_aware_swa.enabled) {
            LOG_INF("%s: DeepSeek-OCR prefill-aware SWA enabled, decode_window=%d\n",
                    __func__, prefill_aware_swa.decode_window);
        }
        if (min_new_tokens.n_tokens > 0) {
            LOG_INF("%s: DeepSeek-OCR min-new-tokens experiment enabled, n_tokens=%d\n",
                    __func__, min_new_tokens.n_tokens);
        }

        if (!model || !lctx) {
            exit(1);
        }

        media_history_token = llama_vocab_n_tokens(vocab) + 1000000;
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
            exit(1);
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
            exit(1);
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
            break;
        }

        llama_tokens banned = banned_ngram_tokens(
                ctx.no_repeat_origin_tokens,
                generated_tokens,
                ctx.no_repeat_ngram);
        if (i < ctx.min_new_tokens.n_tokens) {
            banned.insert(banned.end(), ctx.eog_tokens.begin(), ctx.eog_tokens.end());
        }
        llama_token token_id = banned.empty()
            ? common_sampler_sample(ctx.smpl, ctx.lctx, -1)
            : common_sampler_sample_with_banned(ctx.smpl, ctx.lctx, -1, banned);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
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

        if (!prune_decode_history(
                    ctx.lctx,
                    ctx.prefill_aware_swa,
                    prefill_end,
                    ctx.n_past,
                    decode_kv_removed_until)) {
            return 1;
        }
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

    LOG("\n");

    return 0;
}

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
    return g_is_interrupted ? 130 : 0;
}
