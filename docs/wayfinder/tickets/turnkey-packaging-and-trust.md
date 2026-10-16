## Question

What package must KORE ship so a user goes from zero to a running model on an
improbable card with verifiable trust? Decide the shape (not the implementation):

- One-line install that auto-detects card, RAM, driver, and VRAM thresholds
- SHA256-verified model + bundle downloads
- Auto model-selection by VRAM/RAM
- Calibrated kernel selection persisted to the machine
- Command to print a local, OpenAI-compatible endpoint

## Claimed by

(unassigned)

## Blocks / blocked by

- blocked by: none
- unblocked

## Type / mode

`wayfinder:task` (HITL). Manual work that must happen before a packaging decision
can be made - it earns its place by unblocking a decision, not by delivering the
destination. Ends when the install/verification shape is designed and dogfooded
with both correct and incorrect inputs.
