#!/usr/bin/env bash
# KORE dry-run ladder: protocol rehearsal -> first measured baseline.
#
# Self-contained: put it on any Linux box with a CUDA GPU (e.g. a Kaggle T4
# notebook) and run. Requires CUDA 12 headers (Kore's `cuda` feature needs them).
#
# Usage:
#   bash kore-tools/run-kore-dryrun.sh             # default: cuda, Bonsai Q2_0, 64 tok
#   bash kore-tools/run-kore-dryrun.sh --backend cpu # cheapest protocol replay
#   bash kore-tools/run-kore-dryrun.sh --skip-build=1
set -euo pipefail

REPO="${KORE_REPO:-$PWD}"
cd "$REPO"

MODEL="${KORE_MODEL:-$REPO/models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf}"
TOK="${KORE_TOK:-$REPO/models/bonsai2-2b/tokenizer.json}"
PROMPT="${KORE_PROMPT:-"The capital of France is"}"
N_NEW="${KORE_N_NEW:-64}"
BACKEND="${KORE_BACKEND:-cuda}"
SKIP_BUILD="${KORE_SKIP_BUILD:-0}"

command -v nvidia-smi >/dev/null 2>&1 && echo ">> GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

if [ "$SKIP_BUILD" != "1" ]; then
  echo ">> (re)installing cargo/rustc if missing..."
  if ! command -v cargo &>/dev/null; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cargo rustc
  fi
  echo ">> building kore (release)..."
  cargo build --release --example bonsai 2>&1 | tail -6
fi

echo ">> running bonsai dry-run ($BACKEND, $N_NEW tok, prompt [$PROMPT])"
if [ "$BACKEND" = "cuda" ]; then
  Gpu=1
else
  Gpu=0
fi
./target/release/examples/bonsai "$MODEL" "$TOK" "$PROMPT" "$N_NEW" "$Gpu"
