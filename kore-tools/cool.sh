#!/bin/bash
# Wait for the GPU die to cool before a timed run on KORE's inference engine.
#
#   kore-tools/cool.sh [target_C] [max_wait_s]      # defaults: 53 300
#
# Why this exists. Every timed number in the inference-revamp plan was taken on a
# box where the die state dominates the result:
#
#   * a heat-saturated die inflated ms/forward badly -- one uncooled run read
#     155.7 ms where the true cost was ~64, and an early Ornith-9B reading of
#     351 ms/token turned out to be ~47-61 ms/token once the box was quiet
#     (plan section 14.33 retracts the slow number);
#   * a *cold-start* bias inflated the speculative-decode ratios by up to 1.41x
#     until `bench_draft` learned to warm every arm before timing any of them;
#   * the reference targets in the plan (Strata, sudoingX) all report cooled runs.
#
# So: cool to the same temperature before every timed run, and prefer interleaved
# A/B arms inside one process -- a ratio measured across two processes on this
# class of box is not a measurement.
#
# Works with NVIDIA (nvidia-smi), AMD (rocm-smi), or nothing at all (the amdgpu
# hwmon node), so the same script is valid on the next machine.
set -u

TARGET=${1:-53}
MAX=${2:-300}

read_temp() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d ' '
        return
    fi
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showtemp --csv 2>/dev/null \
            | awk -F, 'NR>1 { gsub(/[^0-9.]/, "", $2); if ($2 != "") { printf "%d\n", $2; exit } }'
        return
    fi
    # amdgpu exposes its edge temperature through hwmon; no tools required.
    local f
    for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input \
             /sys/class/drm/card*/device/hwmon/temp1_input; do
        if [ -r "$f" ]; then
            echo $(( $(cat "$f") / 1000 ))
            return
        fi
    done
    echo ""
}

T0=$(date +%s)
while :; do
    T=$(read_temp)
    NOW=$(date +%s)
    if [ -z "$T" ]; then
        echo "cool: no readable GPU temperature; skipping" >&2
        exit 0
    fi
    if [ "$T" -le "$TARGET" ]; then
        echo "cool: ${T}C after $((NOW - T0))s"
        exit 0
    fi
    if [ $((NOW - T0)) -ge "$MAX" ]; then
        echo "cool: gave up at ${T}C after $((NOW - T0))s"
        exit 0
    fi
    sleep 5
done
