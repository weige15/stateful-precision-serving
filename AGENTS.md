# Repository instructions

## Mission

This repository investigates whether precision choices alter persistent
autoregressive state or the amount of future serving work. Treat that as an open,
falsifiable question. Do not begin from the assumption that a scheduler,
controller, or new quantizer is warranted.

## Evidence language

Every protocol, report, and decision must separate:

- **Confirmed:** directly supported by inspected code, manifests, or executed data.
- **Unknown:** not measured or blocked.
- **Proposed:** a next experiment or interpretation, not a result.

A negative result that passes the frozen checks is a valid completion. Never move
a threshold, replace prompts, remove harmful cases, or change a metric after
viewing main results.

## Upstream boundary

The only upstream implementation for Phase 1 is
`weige15/qaq_baseline@a5f4d5d52358c7b9740ecec448db5279c6af5ddc`.
It is read-only evidence. Verify and record its exact commit and the hashes of
every imported source or checkpoint. Do not edit, commit, clean, reset, or repair
that repository. New code and outputs belong here.

The upstream nested representation reconstructs FP16 tensors and is not a
low-bit compute kernel. Report logical W4/W6/W8 settings separately from physical
memory, latency, and compute. Do not infer a serving speedup from logical bits.

## Phase 1 scientific controls

1. The controlled history experiment must feed identical token IDs under every
   compared schedule and use identical current precision at the probe.
2. The W4/W8 history schedules being compared must contain the same number of
   W4 and W8 history tokens. Record the applied precision at every forward.
3. Prove that the same `past_key_values` object evolves across chunks; a set of
   independent forwards is not a history experiment.
4. Compare constant-precision chunked execution with one-shot execution before
   interpreting any mixed-history difference. Treat unexplained chunking error as
   a blocker.
5. Keep teacher-forced and free-running results separate. Teacher forcing isolates
   cache history while free-running generation includes token-path divergence.
6. A pilot may establish correctness, deterministic noise, viable prompt
   formatting, and runtime only. Freeze prompt identities, schedules, thresholds,
   analysis, and exclusions before inspecting main cross-schedule results.
7. Run main evidence twice in fresh processes. Preserve every prompt and every
   result, including zeros, regressions, early EOS, and length-cap hits.
8. Raw result paths are append-only. Never overwrite or silently resume into a
   completed run directory.

## Phase 1 exclusions

Do not implement a scheduler, continuous batching engine, KV quantizer, router,
custom CUDA/Triton kernel, async weight loader, or multi-GPU mechanism. Do not
claim throughput, tail-latency, or quality gains. These are later branches and
require an explicit new goal after Phase 1 evidence.

## Safety and verification

Before GPU work, inspect the host and use the upstream process-free GPU preflight
when available. Use one idle GPU, never kill or displace another process, never
bypass a refusal, and apply the per-job and cumulative limits in the active goal.
Do not install or upgrade packages merely to make a result run; record a missing
runtime as a blocker.

Run focused CPU tests before any GPU pilot, then the full tests and an independent
result audit at completion. Record commands, exit codes, package/model revisions,
seeds, GPU identity, elapsed time, and raw-artifact hashes. Run
`git diff --check` before declaring completion.

## Stop behavior

Pause with a concise blocker report when checkpoint access, cache correctness,
determinism, resource safety, or the frozen minimum sample size cannot be
established. State the exact evidence gathered, attempted paths, and the one
input or permission needed next. Do not replace a blocked causal experiment with
a weaker correlational claim.
