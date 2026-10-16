## Question

Conversation-first: what number is "massive," and how is it measured, so the plan
decides on a metric instead of a feeling?

- **Raw or relative?** Raw tokens/sec on the user's hardware, or speedup vs the
  current KORE build (and vs a baseline model)? Both are legitimate; they answer
  different questions and must not be conflated.
- **Baseline model?** What are we speeding up against (current build? a reference
  FP16 vanilla implementation? vLLM?)
- **What is measured?** Per-kernel matmul time, end-to-end chat tok/s, decode tok/s
  at fixed context, prefill tok/s at fixed prompt. Which set defines "massive"?
- **How recorded?** One schema so before/after runs are apples-to-apples.
- **What is the candidate bar?** ">=2x the current build on the same hardware" vs
  ">=N tokens/sec raw" - decide.

## Claimed by

(unassigned)

## Type / mode

`wayfinder:grilling` (HITL). The user's answers, not the agent's, name the metric.
