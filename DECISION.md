# Decision

`pause_blocked`

## Basis

- **0 real main prompts**, no real pilot, no frozen protocol, no authorized main
  processes. The minimum16-prompt validity requirement is unmet.
- Recorded preflight found all eight GPUs busy; none was selected or displaced.
  No GPU model job ran; goal GPU usage is0/14,400 seconds.
- Exact upstream source, checkpoint and tokenizer inputs are accessible and hashed.
  The checkpoint is **not** the blocker.
- Tiny CPU controls, fresh-process agreement and independent analysis/audit pass.
  They establish harness correctness at tiny scope only, not a scientific
  continue/stop result for Qwen3-4B.

## Resume input and next action

Provide an idle, process-free GPU opportunity and ask to resume. Do not kill or
move another user's work. Re-run preflight in a new directory, then the existing
four-prompt non-decision pilot. Its controls must justify real numerical gates and
runtime before protocol/config freeze or main execution.

Pending evidence: `raw/gpu-jobs/pilot-001/data/result.json`, a real freeze receipt
`evidence/freeze.json`, finalized real analysis/audit, and two fresh >=16-prompt
main result sets (`main-001`, `main-002`; target32). No weaker experiment substitutes
for them. No model download, package change, threshold tuning on main results or
new serving infrastructure is authorized.

**Goal status: not complete.** See `PRECISION_STATE_REPORT.md` for findings and
`PRECISION_STATE_AUDIT.md` for every requirement and its evidence boundary.
