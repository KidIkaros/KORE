// Token-level reference oracle: greedy-decode a GGUF with llama.cpp and print
// the prompt ids and the generated ids.
//
//   build: see ./build_ref.sh in this directory
//   usage: ref_ids model.gguf "prompt" n_predict [n_ctx] [n_threads]
//   env:   REF_TOPK=n   also dump the top-n logits after every step
//
// Greedy (top-1) so the token stream is a deterministic function of the model,
// which makes it directly comparable against any other correct implementation.
//
// Why this exists (plan section 14.11): plausibility is not evidence. llama.cpp
// is the authority for *GGUF semantics* -- the file format's contract -- exactly
// as gguf-py is for quant layout. Every new arch or quant should be gated on a
// byte-identical token stream against this, and the top-k dump is what tells a
// real disagreement apart from a numerical near-tie.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

static void print_ids(const char * tag, const std::vector<llama_token> & t) {
    printf("%s:", tag);
    for (auto id : t) { printf(" %d", id); }
    printf("\n");
}

// `REF_TOPK=n` dumps the top-n logits at every step, so a token disagreement can
// be told apart from a numerical near-tie.
static int g_topk = 0;

static int cmp_logit(const float * lg, int a, int b) { return lg[a] > lg[b]; }

static void print_topk(llama_context * ctx, const char * tag) {
    if (g_topk <= 0) { return; }
    const float * lg = llama_get_logits_ith(ctx, -1);
    const int n = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    std::vector<int> idx((size_t) n);
    for (int i = 0; i < n; i++) { idx[(size_t) i] = i; }
    const int k = g_topk < n ? g_topk : n;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [lg](int a, int b) { return cmp_logit(lg, a, b); });
    printf("%s:", tag);
    for (int i = 0; i < k; i++) { printf(" %d=%.9g", idx[(size_t) i], (double) lg[idx[(size_t) i]]); }
    printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf \"prompt\" n_predict [n_ctx] [n_threads]\n", argv[0]);
        return 1;
    }
    const char * mpath = argv[1];
    const char * prompt = argv[2];
    const int n_predict = atoi(argv[3]);
    const int n_ctx = argc > 4 ? atoi(argv[4]) : 4096;
    const int n_threads = argc > 5 ? atoi(argv[5]) : 12;
    if (const char * tk = getenv("REF_TOPK")) { g_topk = atoi(tk); }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(mpath, mp);
    if (model == nullptr) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = 2048;
    cp.n_ubatch = 2048;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) { fprintf(stderr, "ctx init failed\n"); return 1; }

    const int plen = (int) strlen(prompt);
    int n = -llama_tokenize(vocab, prompt, plen, nullptr, 0, true, true);
    if (n <= 0) { fprintf(stderr, "tokenize sizing failed\n"); return 1; }
    std::vector<llama_token> toks((size_t) n);
    if (llama_tokenize(vocab, prompt, plen, toks.data(), n, true, true) < 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    print_ids("prompt_ids", toks);

    llama_sampler * smpl = llama_sampler_init_greedy();

    llama_batch batch = llama_batch_get_one(toks.data(), n);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prompt decode failed\n"); return 1; }
    print_topk(ctx, "topk_after_prompt");

    std::vector<llama_token> gen;
    gen.reserve((size_t) n_predict);
    for (int i = 0; i < n_predict; ++i) {
        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (id == llama_vocab_eos(vocab)) { break; }
        gen.push_back(id);
        llama_batch b1 = llama_batch_get_one(&gen.back(), 1);
        if (llama_decode(ctx, b1) != 0) { fprintf(stderr, "decode failed at %d\n", i); return 1; }
        char tag[64];
        snprintf(tag, sizeof(tag), "topk_step_%d", i + 1);
        print_topk(ctx, tag);
    }
    print_ids("gen_ids", gen);

    std::string out;
    for (auto t : gen) {
        char buf[512];
        const int k = llama_token_to_piece(vocab, t, buf, (int) sizeof(buf), 0, false);
        if (k > 0) { out.append(buf, (size_t) k); }
    }
    printf("gen_text:%s\n", out.c_str());
    return 0;
}
