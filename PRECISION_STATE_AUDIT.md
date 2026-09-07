# Precision state completion audit

## Verdict

**Not achieved. Decision: `pause_blocked`.** The safe next scientific step is the
real-model constant-control pilot; recorded GPU availability prevents that step.
Passing CPU tests and artifact checks does not satisfy the real-study requirements.
No goal-complete transition is justified.

## Concrete deliverables

1. A tested between-forward precision harness recording actual applied W4/W8.
2. A valid controlled Qwen3-4B teacher-forced comparison with identical tokens,
   balanced history, W8 probe, genuine cache reuse and constant controls.
3. A separate deterministic Qwen3-4B free-running study with all five schedules,
   full token trajectories, EOS/length/cap/divergence and actual KV bytes.
4. Frozen protocol/config before main results, two fresh main processes, independent
   analysis/audit, and evidence-backed report/audit/exactly one allowed decision.

Only the CPU correctness portion of1 and reporting of the blockage are complete.
The real pilot, freeze and requirements2–3 remain unachieved, not replaced by tiny data.

## Prompt-to-artifact checklist

Status vocabulary: **PASS** = directly checked at stated scope; **CPU ONLY** =
implemented/tested on tiny model, not real scientific evidence; **DRAFT** = specified
but not frozen; **BLOCKED** = required evidence absent; **CONSTRAINT** = preserved
boundary, not a positive scientific result.

| Requirement from objective | Evidence inspected / exact surface | Status |
|---|---|---|
|Inspect local environment first|`raw/intake-001/upstream-check.txt`, `raw/intake-002/intake.json` command/version/GPU inventory|PASS|
|Determine checkpoint/evidence accessibility|20 input files present and hashed; checkpoint4,525,416,138 bytes; local frozen examples and tokenizer exist|PASS|
|Pin exact `a5f4d5d52358c7b9740ecec448db5279c6af5ddc`|remote ls-remote, shallow source fetch, intake verifies FETCH_HEAD before import|PASS|
|Do not silently use neighboring HEAD|intake records `ce71a0a...`; imports obtained from exact fetched pin instead|PASS|
|Upstream read-only, never edit/repair|tool commands/source import limited to local new repo; before/after upstream status and selected input hashes in final validation|CONSTRAINT|
|Hash every imported source/checkpoint|`evidence/intake.json`:76 tree blobs, two vendor imports, all20 local artifacts; final checker rehashes imports|PASS|
|State FP16 reconstruction, not low-bit kernel|vendor module docstring/forward; report and protocol explicitly state this|PASS|
|No package install/upgrade or large model download|existing venv used; only small pinned Git source fetch; preserved command inventory|CONSTRAINT|
|If checkpoint absent, do not regenerate closed evidence|checkpoint is present; no regeneration or download occurred|CONSTRAINT (absence branch not needed)|
|Use repo for new code and persistent outputs|`src/`, `scripts/`, `tests/`, `vendor/`, `configs/`, tracked receipts; ignored `raw/` and `inputs/`|PASS|
|Tested W4/W8 changes across cached forwards, actual schedule|`CachedRunner.forward`, pre-hook observations in both tiny runs; switching/tamper tests|CPU ONLY|
|Validate schedules atomically|`validate_schedule`; invalid bit/length/bool/unbalanced/probe tests; no mutation on error|PASS|
|Equal W4/W8 history counts|teacher traces8+8 in tiny;64+64 draft; independent audit sums observed history tokens excluding probe|CPU ONLY|
|Identical teacher-forced token IDs|all traces flatten to the same17 IDs; hashes recomputed; tampered-token audit test rejects|CPU ONLY|
|Same W8 probe token|teacher final trace length1, bits8, identical position16 (draft128); full logits saved|CPU ONLY|
|Genuinely reuse cache, not reconstruction|same DynamicCache object; append lengths; old per-layer tensors `torch.equal`; cached-vs-recomputed test|CPU ONLY|
|Constant W8/W4 one-shot vs chunked controls first|all prompts' controls precede mixed calls in `run_workload`; `raw/tiny-*/data/prompt-*.json`|CPU ONLY|
|Justified tiny tolerance|0.002 absolute/rtol0 FP16 unit-scale engineering tolerance; observed max0.0009765625; long129-token CPU test also passes|PASS (tiny only)|
|Repeat real-model constant controls in pilot|no real pilot result exists|BLOCKED|
|Unexplained chunk discrepancy overlapping signal blocks interpretation|draft freeze/validity policy; no real noise estimate/interpretation|DRAFT/BLOCKED|
|Append-only result paths|exclusive mkdir/open; raw failed attempts preserved; collision/manifest mutation tests|PASS|
|Exact KV-byte accounting|logical `numel*element_size`; unique backing storage; shared-view and dense per-layer independent tests|CPU ONLY|
|Deterministic selection and decoding|hash-ranked prompts independent of input order; greedy tie rule; exact free path repeat tests|PASS at CPU scope|
|Decision boundaries|all5 branches,15-vs16, missing validity, fresh-process and strict-threshold unit tests|PASS|
|Pilot<=4, non-decision, no mixed-history tuning|config four IDs; pilot runner omits all mixed modes; no real pilot executed|PASS constraint / BLOCKED pilot|
|Pilot format viability, repeated controls, cost projection|candidate raw WT2 formatting only; no real formatting/cost evidence|BLOCKED|
|Freeze named protocol/config before any main result|both named files present but explicitly NOT frozen; no `evidence/freeze.json`|DRAFT/BLOCKED|
|Exact IDs/hashes, pilot/main separation|config full IDs/token arrays/source-record and token hashes;4+32 disjoint selection|PASS selection, not final freeze|
|Freeze prefix/segments/probe/decoder/EOS/seeds/metrics|candidate129-token prompt,64+64+1,16+16 decode, seeds1729/314159 in protocol/config|DRAFT|
|Freeze justified noise gate separately from practical gate|real numerical fields null; proposed TV/workload gates not accepted/frozen|BLOCKED|
|Freeze uncertainty/exclusions/rules, never change after main|proposed paired bootstrap10000 draws, prompt unit, familywise free correction; no main data|DRAFT|
|Target>=32, never scientific continue/stop from<16|32 main IDs selected; minimum guarded; actual main count0 => pause|PASS constraint / BLOCKED evidence|
|If projected16 unsafe, pause rather than shrink|no runtime projection; no silent shrink; no scientific branch|CONSTRAINT|
|Primary84-vs48, two equal cached chunks, W8 probe|tiny actual traces; real runner prepared but never launched|CPU ONLY|
|88/44 controls plus fresh-cache repeats|four synthetic schedules x2 caches per prompt; constant-W4 control distinguished from44/W8 probe|CPU ONLY|
|Full probe logits, lossless raw, tracked hashes|28 `.npy` arrays per tiny process plus manifests/receipts; FP16->FP32 lossless|PASS tiny artifact scope|
|Recomputable max-logit/TV/top1/top-k metrics|`analyze_tiny.py` vs independent `audit_tiny.py`; known-distribution and randomized cross-check tests|CPU ONLY|
|Per-layer KV shape/dtype/bytes and schedules|raw traces; independent audit derives expected dense shape/bytes from tiny config|CPU ONLY|
|History claim only above both frozen gates in both processes|no real gates frozen or main processes; report explicitly makes no such claim|BLOCKED, constraint preserved|
|Free fixed W4/W6/W8 and two-segment4→8/8→4|all5 synthetic modes; planned equal decode segments, not equal total prefill+decode bits|CPU ONLY|
|Same frozen prompts and deterministic decoding for real study|same candidate token arrays; real freeze/validation absent|DRAFT/BLOCKED|
|Full token IDs; first divergence, EOS, output length, cap hits|all synthetic raw records plus paired comparisons; early EOS/length divergence tests|CPU ONLY|
|Retain EOS before equal exposure and cap cases|tests force early EOS; saved synthetic outputs all cap; exposure and censor flags checked|PASS tiny scope|
|Peak/cumulative actual KV bytes|raw per-forward exact backing snapshots; independent shape arithmetic; excludes temporary concat/allocator allocations|CPU ONLY|
|Paired per-prompt changes AND uncertainty for all real outcomes|tiny per-prompt changes recomputed; bootstrap primitive independently tested; real paired intervals absent|BLOCKED real evidence|
|Do not claim equal realized bit budgets|separate decode/total exposure: tiny4/4 vs20/4; protocol/report explicitly disavow total-budget matching|PASS constraint|
|Do not attribute free-running variation to KV history|explicit report/protocol causal limitation|PASS constraint|
|Idle process-free GPU and recorded preflight before every job|`raw/gpu-jobs/preflight-001` refusal, all8 devices busy, selected=null; no job started|PASS refusal behavior|
|Serial jobs,30min bound,4 GPU-hour cap incl failures|flock, unfinished-reservation guard,1860s reservation,14400s ledger, child timeout/owned-group cleanup; CPU safety regressions|PASS CPU safeguards; no real launch|
|Never kill/displace other processes|real preflight only; signal tests target own CPU dummy group; no GPU processes signaled|CONSTRAINT|
|Record GPU identity, commands, env, revisions, wall time, SHA256|intake + XML; tiny command/environment/source manifests/results; real job metadata absent because no launch|PASS available scope|
|Two fresh main processes only after frozen authorization|two fresh CPU PIDs/28 exact arrays are not main; main rejects unfrozen config|BLOCKED real processes|
|Separate analysis and audit independently recompute primary/final branch|tiny primary/free/controls rederived; audit imports no production helpers and derives0<16 =>pause|PASS CPU/blockage scope|
|Focused tests, all project CPU tests, artifact/hash/repeat checks, `git diff --check`|`raw/validation-001/` final logs and `evidence/artifacts.json`; no upstream suite rerun|PASS after final checks|
|Required named report/audit/decision files|this file, `PRECISION_STATE_REPORT.md`, `DECISION.md`|PASS|
|Separate confirmed / unknown / next work|report sections and decision blocker/resume checklist|PASS|
|Only one allowed final decision|analysis, audit and DECISION all `pause_blocked`; not a continue/stop claim|PASS|
|No prohibited systems or unmeasured claims|scope rules, file inventory, read-only review; no real performance/quality/novelty claims|CONSTRAINT|
|Completion audit must inspect actual evidence, not green proxies|this table distinguishes tiny evidence/drafts from all missing real requirements|PASS audit, goal NOT complete|

## Verifier coverage and limits

`check_artifacts.py` verifies the selected artifact inventory, pinned vendor hashes,
checkpoint/tokenizer/source inputs, prompt selection, pause decision and absence of
real authorization. It does not establish real cache correctness or scientifically
validate any proposed threshold. `audit_tiny.py` independently recomputes28 arrays,
all synthetic traces/metrics/free comparisons and requires exactly two matching
analysis identities. Optimized Python is rejected explicitly, rather than allowing
assertions to disappear. It cannot certify a real main run.

`raw/validation-001` reruns the16 focused tests and all29 project CPU tests. The new
negative tests check altered token IDs, precision observations, cache IDs/prefixes,
KV bytes, missing analysis runs and `python -O`. Bootstrap quantiles are checked
against a separately implemented sorted-quantile calculation. Safety regressions
include an actual CPU-only wrapper SIGINT with a dummy grandchild; no GPU is used.

## Review and repairs

`raw/review-001/full-review.md` is the independent read-only review. It approved only
a CPU-based pause and blocked future GPU execution on an interruption lifecycle bug.
The parent repaired that path and the audit cardinality/optimization gaps, then
ran the regression tests. This is tested repair evidence, not an independent
re-review attestation. Source snapshots in original tiny runs predate these guard/
audit repairs; the cached-forward implementation generating their arrays is unchanged.
Original raw outputs, test failure and original audit are preserved, with a new
independent audit result at `raw/audit-002/data/audit.json`.

## Remaining unmet requirements and exact next boundary

Missing artifacts:

- `raw/gpu-jobs/pilot-001/data/result.json` and its real constant-control arrays,
  formatting check, numerical noise justification and runtime projection;
- the genuinely frozen `PRECISION_STATE_PROTOCOL.md`, `configs/precision_state_v1.json`
  and `evidence/freeze.json` (current named documents are drafts);
- finalized, tested real-main analysis/audit entry points, frozen alongside protocol;
- `raw/gpu-jobs/main-001/data/result.json` and `raw/gpu-jobs/main-002/data/result.json`,
  >=16 identical real prompts (target32), full raw comparisons and repeat agreement;
- independently recomputed real gates, all paired uncertainty summaries and branch.

The exact available checkpoint path is in the report; no source/model artifact needs
to be downloaded. The external input needed is **an idle, process-free GPU opportunity
and permission to resume then**, not permission to displace a current process.
Resume with fresh preflight, not main execution. Until valid pilot evidence exists,
freeze/readiness cannot be assumed and the goal must remain uncompleted.
