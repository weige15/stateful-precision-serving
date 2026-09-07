# Precision state v1 — DRAFT, NOT FROZEN

**Execution status: blocked before real-model pilot. No main process is authorized.**
`configs/precision_state_v1.json` has four exact pilot IDs and 32 disjoint main IDs,
full token IDs and SHA256 hashes. Numerical gates and runtime authorization remain
unset. This document is NOT a retrospective preregistration or a completed study.

## Question and scope

Does the KV cache carry information about earlier weight precision after current
weights are set to W8, holding input tokens and W4/W8 history exposure fixed? Separately,
do deterministic free-running precision schedules change realized workload?

Use only `weige15/qaq_baseline` commit
`a5f4d5d52358c7b9740ecec448db5279c6af5ddc`. The existing neighboring checkout is at a
different commit; a shallow fetch into this repository obtained the exact pin.
Only `model.py` and `quantization.py` were imported, byte-for-byte, into `vendor/qaq/`.
The source-tree SHA256 inventory and checkpoint/tokenizer hashes are in
`evidence/intake.json` and its raw original. Never edit/repair upstream, redownload
weights, rebuild its closed checkpoint, or regenerate its old experiments.

Model/tokenizer: Qwen/Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`. Nested signed int8 codes, FP32 group scales,
group size128, W4/W6/W8 midpoint reconstruction; embeddings/head/norms remain FP16.
**All nested weights reconstruct FP16 torch linear computation. This is not a
low-bit kernel, and precision does not change the KV dtype.** No scheduler, router,
batching engine, KV quantizer, async loading, kernel or multi-GPU work is included.

## Already executed: CPU correctness, not pilot or main

Two separate fresh CPU processes used a seeded, random, two-layer tiny Qwen3 with
FP16 SDPA, vocabulary64, hidden128, two KV heads of dimension32. Their two synthetic
17-token inputs are unrelated to the 36 selected real prompts. Cache operations
are the actual installed Transformers DynamicCache API, not a mock causal model.

Controls run before synthetic cross-schedule calls: constant W8 and W4 one-shot
`use_cache=False` versus cached8+8+1, and a fresh-cache repeat of each. Tiny-only
absolute tolerance0.002 (rtol0) is approximately two FP16 ULP at unit scale; FP16
logits are losslessly converted to FP32. This is an engineering check for the
random model, NOT the real-model numerical gate. Observed maximum0.0009765625;
all fresh-cache/process arrays match exactly. Additional CPU tests cover64+64+1.

Tests and separate analysis/audit scripts cover validation, token identity,
observed projection precision, cache object continuity, unchanged old KV prefixes,
byte arithmetic including shared backing views, early EOS/cap retention,
deterministic selection/decoding, append-only artifacts and branch boundaries.

## Real non-decision pilot (pending; maximum four prompts)

1. Verify all imported hashes and use only the four configured pilot IDs. Candidate
   formatting is the already frozen WT2 token IDs verbatim, first129 tokens, with
   no chat template, retokenization, BOS/EOS insertion or answers. Pilot may establish
   whether this is viable deterministic corpus continuation. If not, record the
   reason and revise formatting using available inputs before freezing; never use
   mixed histories or main outputs for that choice.
2. On each pilot prompt, run both constant controls over all129 positions:
   one-shot versus64+64+1 and two fresh cached repeats. Record full arrays, per-layer
   metadata and actual schedule. Only then run fixed W4/W6/W8 greedy generation for
   runtime/formatting checks. **No 84-vs48 teacher probe and no mixed free schedule
   are collected in the pilot.** No pilot mixed comparison may select thresholds.
3. Report maximum absolute constant chunk/repeat discrepancy by bit, prompt and
   position; inspect outliers. The current0.125 pilot ceiling is only a conservative
   failure ceiling, not an accepted main tolerance. Passing it does not justify
   interpretation. Repeats are expected to be exact on this deterministic setup.
4. Proposed noise rule, to finalize using only controls: N=max(1e-6,10*C), where C is
   the maximum pilot chunk/repeat discrepancy. Freeze the actual value, observed
   logit scale, FP16 rounding justification and a separately justified constant
   control envelope. Large/unexplained discrepancies require pause, not widening a
   threshold. If noise could overlap the proposed signal, do not interpret it.
5. Project full cost including source/hash verification, checkpoint loading,
   artifact I/O, all constant controls, two fresh caches for every teacher schedule,
   and five free modes. Use at least2x measured per-prompt work plus measured load
   cost, and a conservative allowance for unmeasured switching/reconstruction.
   Each process must fit30 minutes and all attempts fit four GPU-hours. Target32;
   any reduction must be declared before freeze, remain >=16, and be justified by
   cost, never results. If even16 is unsafe, pause. No projection currently exists.

## Freeze boundary (not crossed)

Before any real main cross-schedule output is produced or inspected:

- Finalize this document and the config with exact IDs/hashes, final formatting,
  lengths, schedules, probe, decoder/EOS settings, seeds, metrics, both numerical
  and practical gates, uncertainty, exclusions, runtime bound and branch rules.
- Hash pilot arrays, controls, commands, environment, source, prompt/tokenizer
  evidence and all analysis/audit code. Inspect readiness of real-study analysis
  and independent audit; current `*_tiny.py` programs intentionally reject real runs.
- Implement/test the finalized real analysis/audit entry points and freeze their
  hashes. They must validate traces and recompute raw arrays, not trust summary flags.
- Commit protocol/config/code independently of subsequent main results; record
  commit/time and SHA256 in an append-only `evidence/freeze.json`. Set frozen status
  and authorize exactly two fresh main processes only after all preceding gates.
  No freeze marker exists now; changing a status field is not sufficient evidence.
- After main execution starts, never alter gates, schedules, IDs or exclusions.
  A validity failure leads to pause, not protocol repair using observed results.

## Controlled teacher-forced main (pending)

Each configured input has129 fixed token IDs. Positions0..63 and64..127 are the
history; position128 is the identical input probe token. The probe is forwarded
under W8 and its full vocabulary logits predict position129 (not the probe itself).
No generated token, sampled choice or retokenization enters this comparison.

| Name | First64 tokens | Next64 tokens | One probe token | Role |
|---|---|---|---|---|
|84|W8|W4|W8|primary history A|
|48|W4|W8|W8|primary history B|
|88|W8|W8|W8|constant-history control|
|44|W4|W4|W8|constant-history control (probe changes to W8)|

84 and48 each have exactly64 W4 and64 W8 history tokens. 88/44 are endpoint
controls, not equal-budget comparators. Separate constant-W4 one-shot controls
keep the probe W4 too; do not confuse them with44's W8 probe.

Every schedule gets two independently instantiated DynamicCaches per prompt per
process; cache object and all previous K/V tensor prefixes are reused unchanged
across its three forwards. Forward pre-hooks record the actual bit setting of
all seven projections per layer. Fail on missing calls, wrong bits, wrong positions,
cache replacement, missing layers, changed old KV prefixes or wrong append lengths.
Two fresh main processes repeat the identical sample list and computation.

Preserve full probe logits in lossless `.npy` FP32 (lossless promotion of FP16), all
control logits, complete IDs, hashes, trace and per-layer cache shapes/dtypes/strides/
logical/backing bytes. Metrics: max absolute logit difference, total variation
(TV=0.5*sum|softmax(a)-softmax(b)|), Jensen–Shannon divergence in nats, top1 identity/
change and top10 intersection fraction. Stable ties choose lowest token ID.
Primary contrast84-minus48; endpoint comparisons and top1/top10 are descriptive.

Proposed practical history gate: lower95% paired-bootstrap bound on mean TV>0.01,
separate from lower95% bound on mean max-absolute difference>N. Both must pass in
both processes, after constant controls pass and repeat discrepancies are within
the frozen repeat envelope. TV0.01 is a proposed practical floor of one percent
probability mass moved, not a quality or downstream-utility guarantee. Neither0.01
nor N is selected from mixed results. A scientific history claim needs both gates,
not merely unequal tensors/logits.

## Deterministic free-running main (pending)

Use the same129-token prompts. Fixed W4/W6/W8 apply that precision to prefill and
every decode forward. Mixed modes use first-precision prefill of128 tokens, then
**two equal16-token decode-input segments**, W4→W8 or W8→W4. The last prompt token is
the first decode input. All runs use greedy argmax, batch1, no beams/sampling/
penalties, maximum32 outputs, EOS IDs151645 or151643. EOS counts in output length;
the final sampled token, including EOS, has not yet entered the KV cache.

Equality is of the planned decode segments, **not total processed-token precision
exposure**: prefill adds128 inputs to the first bit width. Even full32-step mixed
runs do not have equal total W4/W8 budgets. Record both exposures; do not describe
these as matched total-bit experiments. Fixed endpoints remain separate. Early
EOS is retained even before the switch or equal decode exposure. A run may both
emit EOS at the cap and hit the cap; right-censored means cap without EOS.

Save complete output token IDs and traces. Report each mode vs fixed8 paired by
prompt: signed/absolute output-length change, EOS difference/discordance, cap
status, first divergence index (0-based, including length-only differences), and
peak/cumulative live KV bytes. No divergence is `null`, not zero. In future main
summaries give divergence incidence and restricted time-to-divergence (null mapped
to cap+1, explicitly censored), not a biased mean excluding identical pairs.

KV: record every layer's K/V shape and dtype. Logical bytes=sum(numel*element_size).
Live backing bytes=sum(unique storage.nbytes), deduplicating shared views. Peak is
maximum **post-forward** live cache backing; cumulative is its sum across prefill
and decode observations, in **byte-forwards**, not allocated traffic or peak CUDA
allocator memory. Transient concat/attention/workspace allocations are not measured.

Proposed free practical gate: at least one of four comparator-vs-fixed8 contrasts
has a bootstrap lower bound exceeding4 mean absolute output tokens,0.125 EOS
discordance, or0.10 mean absolute relative cumulative KV change. These proposed
workload floors mean12.5% of the32-token cap, one-eighth EOS disagreement, or10%
cache byte-forward change; they are not quality thresholds. Control familywise
confidence at95% across4*3 opportunities using **99.58333333333333% two-sided
intervals (Bonferroni12)**. Signed changes,
first divergence, cap incidence and peak bytes are descriptive with95% intervals.
Token divergence alone is not a practical workload gate. Free-running results
cannot specifically establish KV-history causality because token paths can diverge.

## Uncertainty, exclusions, decisions

Proposed uncertainty: deterministic paired percentile bootstrap of prompt-level
means, seed314159,10000 draws, linear quantiles. Average fresh-cache repetitions
within a prompt rather than treating them as independent samples; evaluate each
fresh process separately. This describes this fixed convenience sample, not a
random population of deployed requests. No significance/quality/novelty generalization.
No outcome exclusions: retain EOS/cap/degenerate continuations. Bad hashes, nonfinite
logits, broken traces, missing schedules, controls failing, unknown source,
insufficient cost/minimum or repeat disagreement invalidate the study and pause.

After execution, analysis and audit must independently recompute primary pairs,
per-prompt free contrasts, intervals and branch from raw artifacts and frozen rules.
Require same branch in both fresh main processes and explicit repeat agreement.

| Valid frozen >=16-prompt evidence in both fresh processes | History gate | Free gate | Only permitted decision |
|---|---|---|---|
|no|any|any|`pause_blocked`|
|yes|yes|no|`continue_precision_provenance`|
|yes|no|yes|`revise_endogenous_workload`|
|yes|yes|yes|`continue_joint_state_work`|
|yes|no|no|`stop_precision_state_direction`|

## GPU and artifact discipline

Use `scripts/gpu_job.py` only, one idle process-free GPU, two initial XML checks
plus a pre-CUDA check after hashing. Other users' processes are never signaled.
Local flock serializes jobs; unfinished reservations block further work. Every job
reserves1860 seconds against14400; completed/failed jobs charge actual wall time.
Child-owned GNU timeout1790s +5s hard kill and owned-group cleanup keep work within
30 minutes. Record UUID, driver, process list, command, environment, wall time and
SHA256 manifests. No real GPU job has been launched under this goal (charge0).

Raw artifacts use unique ignored directories and exclusive file creation; preserve
failed attempts. Tracked evidence receipts point to raw bytes. Tests, verifier or
manifest success are not a substitute for missing real-model evidence. See
`PRECISION_STATE_AUDIT.md` for the requirement-level status and resume boundary.
