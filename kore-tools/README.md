# KORE tools

Shared tooling for the KORE ML engine: the dry-run ladder
(`run-kore-dryrun.sh`), a repeatable measurement protocol for "massive
performance gains on improbable hardware", the GGUF reference oracle
(`llama-oracle/`), a die-cooling gate (`cool.sh`), and the analysis instruments
used by the inference-revamp plan (`docs/kore-inference-revamp.md`).

## `run-kore-dryrun.sh`

Protocol rehearsal and first measured baseline on a CUDA GPU.

```bash
# local / Kaggle T4 alike:
bash kore-tools/run-kore-dryrun.sh
bash kore-tools/run-kore-dryrun.sh --backend cpu   # cheapest protocol replay
bash kore-tools/run-kore-dryrun.sh --skip-build=1  # don't rebuild
```

Defaults: Bonsai-2 (Qwen35, Q2_0), 5-token prompt, 64 generated tokens, CUDA back-end.

## Kaggle deployment (T4 16 GB, free tier)

1. New Kaggle notebook → Settings →Accelerator → **GPU** (T4). Runtime → GPU.
2. Open a notebook cell and run (one-time setup):
   ```bash
   # clone KORE, install a C compiler + Rust toolchain
   apt-get update -y && apt-get install -y build-essential curl
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   git clone https://github.com/KidIkaros/KORE.git
   cd KORE
   ```
3. In the next cell, paste the dry-run ladder (or copy `kore-tools/run-kore-dryrun.sh`
   into the notebook and `chmod +x` it), then run it. First run builds the binary
   (~2–4 min); subsequent runs are fast.

Notes / constraints:
- KORE's `cuda` feature needs CUDA 12 headers (`runtime + devel` package);
  T4's CUDA 12.x meets it.
- Sessions are ephemeral and GPU quota is limited (~30 h/wk on the free tier).
  First build is the expensive step; reuse the same session / reuse the wheel
  cache when re-running.
- To measure "massive" fairly: run `run-kore-dryrun.sh` **before any optimization**
  (that's the baseline), then re-run after each change and diff the JSON.

## What a "massive gain" is measured against

The plan (see `docs/wayfinder/map.md`) pins "massive" as:

- raw end-to-end chat tokens/sec at batch size 1 on your GPU, **and**
- speedup vs the current KORE build on the same hardware,

with a correctness gate first. This dry-run ladder produces the raw number; the
serving-path harness (Phase 1 of the `benchmark-harness-baseline` ticket) turns it
into the repeatable, comparable measurement.

## `cool.sh` — the die-temperature gate

Every timed number in the inference-revamp plan was taken on a box where the die
state dominates the result: a heat-saturated die read 155.7 ms where the true cost
was ~64, and an early 351 ms/token decode turned out to be ~47–61 ms/token once the
box was quiet. Cool to the *same* temperature before every timed run.

```bash
kore-tools/cool.sh          # wait for <= 53 C (defaults: 53 300)
kore-tools/cool.sh 50 600   # 50 C, give up after 10 min
```

Reads `nvidia-smi`, `rocm-smi`, or the `amdgpu` hwmon node — whichever is present —
so the same script is valid on NVIDIA and AMD boxes.

## `llama-oracle/` — the GGUF semantics gate

A CPU-only llama.cpp that greedy-decodes a GGUF and prints the token stream, so any
arch or quant KORE claims to support can be checked byte-for-byte against the
format's own implementation. See `llama-oracle/README.md`.

```bash
LLAMA_SRC=/path/to/llama.cpp kore-tools/llama-oracle/build_ref.sh
kore-tools/llama-oracle/ref_ids model.gguf "The capital of France is" 8
REF_TOPK=20 kore-tools/llama-oracle/ref_ids model.gguf "The capital of France is" 8
```

## Analysis instruments

- `moe-route-locality.py` — routing-dump analysis (layer/expert locality, union
  size vs batch width). Consumes `KORE_MOE_ROUTE_DUMP=<path>`.
- `clifford-quant-probe.py` — KV-log-error probe behind the P6.6a rotated-KV
  verdict. `--kv` mode consumes a `kv_dump` `KVDU` v1 dump.

## Measurement protocol

Non-negotiable, and the reason two headline numbers in this effort were retracted:

1. cool the die (`cool.sh`) before every timed run;
2. interleave the arms **in one process** — a ratio across two processes is not a
   measurement on this class of hardware;
3. warm every arm before timing any of them (wgpu compiles pipelines lazily, so the
   first arm in round 0 pays for the whole run);
4. hold the page-cache state fixed for anything that streams weights from disk;
5. gate byte-identity per batch shape, and never gate a reassociated summation on a
   token diff — use exposed µs plus a written tolerance argument.

`docs/inference-handoff.md` §9 carries the long form.
