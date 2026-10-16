## Question

How does KORE produce measured, reproducible evidence that "improbable hardware"
runs, so claims are defensible (the Strata BENCH + community PR-report model; the
sudoingX sweeps by PR)?

- Calibrated benchmark harness that writes per-card reports
- Community contribution pipeline (PRs with per-GPU rows)
- A stored "expected-quality" bar vs FP16 so "impressive" is a pinned number
  instead of a feeling

## Claimed by

(unassigned)

## Blocks / blocked by

- blocked by: `engine-12gb-card-support` (you cannot measure 12 G b cards until the
  engine runs on one)
- unblocked: no

## Type / mode

`wayfinder:task` (HITL). Ends when a working benchmark harness + report template is
in place and dogfooded with correct and incorrect inputs.
