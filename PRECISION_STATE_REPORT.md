# Precision state report

## Decision and evidence level

**`pause_blocked` — real-model evidence is not yet available.** This is not a
negative finding about precision state. The active scientific goal is not achieved.
The checkpoint is present; the blocker is safe GPU availability before the pilot.
See `PRECISION_STATE_AUDIT.md` for the requirement-by-requirement completion audit.

## Confirmed

### Read-only inputs and environment

- Remote `weige15/qaq_baseline` HEAD/main resolved to the required commit
  `a5f4d5d52358c7b9740ecec448db5279c6af5ddc`. The neighboring checkout was at
  `ce71a0afa97f1d29550a3cbe861af4822951b776`, with pre-existing untracked optional
  study files. It was not checked out, fetched into, edited, repaired, or executed.
  Exact pinned source was fetched into this repository's ignored `inputs/qaq-pinned`.
- SHA256 inventory covers all76 pinned tree blobs; only two Python files are
  imported byte-for-byte into `vendor/qaq/`. Twenty local input/model/tokenizer
  files were hashed. All supplied expected hashes matched.
- Existing nested checkpoint:
  `/nfs/home/s314511048/qaq_baseline/results/core-v1/integration/quantized_model.pt`
  (4,525,416,138 bytes), SHA256
  `b63aeeed85e11cea5d2790b1aae068920626e5d93117548358f76ae3b2171b85`.
- Existing prompt records:
  `/nfs/home/s314511048/qaq_baseline/results/core-v1/frozen/examples.jsonl`, SHA256
  `5eba9877916a1a473fc9e4987b4a6e1dafbe8cbdbc6fb17519b83d98e3b3aafd`.
- Model/tokenizer revision:
  `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`.
  Runtime: existing Python3.12.3, torch2.5.1+cu121, Transformers5.16.1, NumPy1.26.4
  at `/nfs/home/s314511048/.venv/bin/python`. No install or upgrade was performed.
- **Nested codes reconstruct FP16 weights for torch linear computation, not a
  low-bit kernel.** The study does not quantize KV or claim physical low-bit compute.

Provenance is in `evidence/intake.json`, with original commands, package inventory,
GPU identity/process records, source and input hashes. No model download, closed
experiment regeneration or checkpoint conversion was performed.

### Tested cached-forward harness (tiny CPU only)

`src/precision_state/harness.py` applies W4/W8 between actual cached Qwen3 forwards.
It records projection pre-hook observations, input IDs/hashes, cache object IDs,
append lengths and every layer's KV shape/dtype/bytes. Teacher calls verify that
old K/V tensor prefixes remain exactly unchanged, rather than recomputing history.

Two fresh processes in `raw/tiny-001/data` and `raw/tiny-002/data` used two synthetic
17-token inputs on a random two-layer tiny model. They are **not real pilot or main
prompts**, and cannot contribute to the minimum16. Constant controls preceded
synthetic cross-schedule comparisons.

| Synthetic input | Constant W8 one-shot/chunk max difference | Constant W4 difference | Fresh-cache repeat difference |
|---|---:|---:|---:|
|0|0.0009765625|0.00054931640625|0|
|1|0.0009765625|0.0009765625|0|

Both bits pass the tiny-only0.002 absolute tolerance (rtol0), approximately two
FP16 ULP at unit scale. All28 saved logit arrays match exactly across fresh
processes. CPU tests also pass at the planned64+64+1 chunk lengths. This numerical
tolerance cannot be transferred to Qwen3-4B without its own pilot controls.

The synthetic teacher test runs84,48,88 and44 with two fresh caches each. Both
primary paths use identical tokens,8 W4+8 W8 history tokens and the same W8 probe.
Full probe arrays and metrics are preserved. Their small-model differences are
correctness fixtures, **not confirmation of a Qwen3-4B precision-history effect**.
No real-model practical/noise gate is applied to them.

The separate synthetic free test covers fixed4/6/8 and4→8/8→4. All actual saved
trajectories reached their8-output cap with no EOS and identical paths across
modes. Explicit tests separately force early EOS and check divergence/length-only
cases, so passing retention tests does not imply naturally observed EOS variation.
All saved full-length trajectories have12,288 peak post-forward KV storage bytes
and92,160 cumulative byte-forwards. This is a tiny-model accounting fact, not a
Qwen3-4B memory or workload finding.

Mixed synthetic decode exposure is4/4; total input exposure is20/4 or4/20 because
prefill uses the first precision. The proposed real study likewise has equal
**decode** segments but unequal total bit exposure. Neither is called an equal
realized total-bit budget. Free-running token differences, if later observed,
will not be attributed specifically to KV history.

### Independent checks

- Focused16 tests and full29 project CPU tests pass in the final validation record.
  Coverage includes bad schedules, balanced histories, observed switching, unchanged
  cache prefixes, exact backing/view byte accounting, deterministic tokens/selection,
  early EOS/caps, append-only paths, bootstrap arithmetic and all five decisions.
- `scripts/analyze_tiny.py` and `scripts/audit_tiny.py` independently recompute raw
  primary metrics and free per-prompt contrasts. The audit imports no harness or
  analysis code; all arrays, traces, IDs and metadata are checked. Both derive
  `pause_blocked` from zero real main prompts, not from tiny effect magnitude.
- A read-only reviewer found no invalidation of current CPU numerical evidence,
  but identified GPU interruption cleanup and two audit fail-closed gaps. Local
  fixes add owned-process-group termination/reaping, complete analysis-run identity
  checks and explicit rejection of optimized Python. CPU regressions exercise
  wrapper-only SIGINT with a dummy grandchild, timeout, missing analysis runs,
  `python -O`, serial locks, pending/failed budget charges and busy GPU refusal.
  Reviewer output is preserved; it was not a second attestation of the later fixes.
- One initial test fixture incorrectly assumed DynamicCache retained shared input
  views. Actual update concatenates/copies them. The fixture now explicitly installs
  shared backing views; the failed attempt remains in `raw/cpu-tests-001`.

## Blocker and unknowns

Recorded preflight `raw/gpu-jobs/preflight-001` returned3: all eight RTX3090s had
processes, with99–100% utilization. No GPU was selected and no model job launched.
**GPU work charged under this goal:0 seconds of14,400.** Other users were untouched.
This is a point-in-time observation, not a claim that GPUs will stay occupied.

There are **zero real pilot prompts and zero main prompts**. We do not yet know:

- real constant-precision chunking/repeat error or whether it overlaps a history signal;
- viable real deterministic formatting, EOS behavior or bounded runtime for16/32 prompts;
- a justified frozen real numerical gate or main practical results;
- whether84-vs48 survives both gates and both fresh main processes;
- real free-running divergence, EOS/length/cap differences, uncertainty or KV bytes.

`PRECISION_STATE_PROTOCOL.md` and `configs/precision_state_v1.json` are explicitly
**unfrozen drafts**. IDs/hashes are selected outcome-blind (4 pilot,32 main), but
numerical gates and runtime authorization are unset. Real analysis/audit entry
points and their final gate validation remain to be finalized after the pilot;
the existing tiny-only programs intentionally cannot certify real main evidence.

## Proposed next work (not results)

1. Obtain an idle, process-free GPU opportunity; do not terminate another job.
2. Re-run recorded preflight into a new directory. Verify current source/input
   hashes and CPU checks, then execute the four-prompt non-decision pilot only.
3. Inspect constant controls and formatting, justify the noise envelope and cost.
   If safe16-prompt evidence is unattainable, remain paused rather than shrink below16.
4. Finalize/test real analysis and independent audit; freeze protocol/config/code
   and exact prompt hashes before any main cross-schedule result. Only then run
   two serial fresh main processes and independently recompute the final branch.

No precision-state, latency, throughput, quality, novelty, or real-model memory
claim is made from the current CPU artifacts.
