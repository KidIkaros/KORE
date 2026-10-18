# llama.cpp reference oracle

The GGUF-**semantics** gate for KORE's inference engine. `ref_ids` greedy-decodes
a GGUF with a CPU-only llama.cpp and prints the token stream, so any arch or quant
KORE claims to support can be checked byte-for-byte against the format's own
implementation.

This is deliberately *not* a benchmark and deliberately CPU-only. The point is a
second, independent implementation of the file format's contract — a GPU oracle
would share KORE's failure modes and prove nothing. (The *quant-layout* oracle is
a different tool: `crates/kore-kernels/tests/k_quant_fixtures.rs`, checked against
`gguf-py`'s `gguf.quants`, with `k_quant.rs` as the written spec.)

## Build

Point `LLAMA_SRC` at a built llama.cpp checkout:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j

cd /path/to/KORE
LLAMA_SRC=/path/to/llama.cpp kore-tools/llama-oracle/build_ref.sh
```

## Use

```bash
# 8 greedy tokens, one thread count for the context and the batch
kore-tools/llama-oracle/ref_ids model.gguf "The capital of France is" 8

# dump the top-20 logits after the prompt and after every step, so a token
# disagreement can be told apart from a numerical near-tie
REF_TOPK=20 kore-tools/llama-oracle/ref_ids model.gguf "The capital of France is" 8
```

Then compare against KORE's own greedy lane:

```bash
cargo build --release -p kore-cli --example bonsai
./target/release/examples/bonsai model.gguf tokenizer.json "The capital of France is" 8 cpu
./target/release/examples/bonsai model.gguf tokenizer.json "The capital of France is" 8 gpu
```

Both must print the same `gen_ids` as the oracle, element for element.

## Discipline note

Read the probe's **entire** output. A top-10 logits print that was inspected
through `tail -14` once hid rank 1 and sent a real investigation down a false
trail while the engine had been correct all along (plan section 14.11).

## Worked example (Qwen3.6-35B-A3B, plan section 14.28)

```
prompt_ids: 760 6511 314 9338 369            ("The capital of France is")
gen_ids:    11751 13 198 760 6511 314 9338 369
gen_text:   "Paris.\n\nThe capital of France is"
```

That stream is locked into `crates/kore-nn/tests/qwen36_real.rs`, guarded on the
13 GiB checkpoint's presence. Use the same pattern for the next arch: record the
stream, assert it, and let the file's absence skip the test rather than fail it.
