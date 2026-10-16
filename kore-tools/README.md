# KORE tools

Shared tooling for the KORE ML engine. Currently: the dry-run ladder
(`run-kore-dryrun.sh`), a repeatable measurement protocol for "massive
performance gains on improbable hardware".

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
