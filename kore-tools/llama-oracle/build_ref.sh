#!/bin/bash
# Build the token-level llama.cpp reference oracle (`ref_ids`).
#
#   LLAMA_SRC=/path/to/llama.cpp kore-tools/llama-oracle/build_ref.sh
#
# `LLAMA_SRC` must point at a llama.cpp checkout that has *already been built*
# with CMake, i.e. these exist:
#
#   $LLAMA_SRC/build/src/libllama.a
#   $LLAMA_SRC/build/ggml/src/libggml.a
#   $LLAMA_SRC/build/ggml/src/libggml-cpu.a
#   $LLAMA_SRC/build/ggml/src/libggml-base.a
#
# A CPU-only build is deliberate: the oracle exists to be *the* reference, and a
# second GPU implementation of the same thing would be a second variate, not a
# check. It is also the reason the oracle is affordable on any box.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="${LLAMA_SRC:-}"

if [ -z "$S" ]; then
    echo "set LLAMA_SRC to a built llama.cpp checkout, e.g." >&2
    echo "  LLAMA_SRC=\$HOME/llama.cpp $0" >&2
    exit 1
fi
if [ ! -d "$S/include" ] || [ ! -d "$S/ggml/include" ]; then
    echo "'$S' does not look like a llama.cpp checkout (no include/ + ggml/include/)" >&2
    exit 1
fi

B="$S/build"
LIBS=(
    "$B/src/libllama.a"
    "$B/ggml/src/libggml.a"
    "$B/ggml/src/libggml-cpu.a"
    "$B/ggml/src/libggml-base.a"
)
for l in "${LIBS[@]}"; do
    if [ ! -f "$l" ]; then
        echo "missing '$l'" >&2
        echo "build llama.cpp first:  cmake -B \$LLAMA_SRC/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build \$LLAMA_SRC/build -j" >&2
        exit 1
    fi
done

g++ -O2 -std=c++17 \
    -I "$S/include" -I "$S/ggml/include" \
    "$HERE/ref_ids.cpp" -o "$HERE/ref_ids" \
    "${LIBS[@]}" \
    -fopenmp -lm -lpthread -ldl

echo "built $HERE/ref_ids"
echo "try:  $HERE/ref_ids models/ornith-9b/Ornith-1.0-9B-Q4_K_M.gguf \"The capital of France is\" 8"
