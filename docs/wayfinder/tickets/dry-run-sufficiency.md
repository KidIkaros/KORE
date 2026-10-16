## Question

Where does a "dry run" stop being satisfactory for KORE's trajectory toward running
impressive models on improbable hardware, and what must a "dry run" mean for KORE
so the other tickets can be decided precisely?

Concretely:
- Can dry-runs (calibrated runs, quantized-op probes, benchmark dry-runs) validate the
  *decisions* and *product shape* of this plan?
- Where is the line where "dry run" stops being enough and a *real run on the real
  card* + a *verifiable evidence pipeline* becomes required?
- What does "sufficient" mean here: trustworthy decisions, credible hardware claims,
  and a tractable packaging story?

This is conversation-first (grilling) - the user defines what "dry run" means for
their own project. The answer is recorded here, not in a subagent.

## Claimed by

(unassigned)

## Blocks / blocked by

- blocked by: none
- blocks: `turnkey-packaging-and-trust`, `measured-benchmark-pipeline` (its answer
  is needed before those packaging/evidence questions are settled)

## Type / mode

`wayfinder:grilling` (HITL). The user's answers, not the agent's, close this one.
