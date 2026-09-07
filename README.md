# Precision Is State

A falsification-first research repository for **stateful approximation in
autoregressive LLM serving**.

The project asks a larger question than which request should use 4, 6, or 8 bits:

> Is numerical precision merely a cost chosen for the current operation, or does
> it alter persistent inference state and therefore change the future work that
> an LLM service must perform?

This is a research question, not a claimed result.

## Starting evidence

The preceding whole-query endpoint study is frozen at
[`weige15/qaq_baseline@a5f4d5d`](https://github.com/weige15/qaq_baseline/tree/a5f4d5d52358c7b9740ecec448db5279c6af5ddc).
Its fixed4/fixed8 query-mixture opportunity failed the prespecified fixed6 gate
on all three tasks. See the
[report](https://github.com/weige15/qaq_baseline/blob/a5f4d5d52358c7b9740ecec448db5279c6af5ddc/QUERY_BUDGET_SIGNAL_REPORT.md)
and
[audit](https://github.com/weige15/qaq_baseline/blob/a5f4d5d52358c7b9740ecec448db5279c6af5ddc/QUERY_BUDGET_SIGNAL_AUDIT.md).

That result stops the bounded **whole-query endpoint-routing** direction. This
repository does not relabel it as evidence for a scheduler.

| Status | Statement |
|---|---|
| Confirmed | The upstream code has a nested W4/W6/W8 weight representation and can change a block profile between forwards, although it reconstructs FP16 weights and is not a low-bit kernel. |
| Confirmed | The closed upstream evaluation used batch size 1 and `use_cache=False`; it did not test persistent KV state, generation length, continuous batching, or serving throughput. |
| Unknown | Whether the order of earlier precision choices changes later logits when the token prefix and current precision are held fixed. |
| Unknown | Whether precision changes token paths, EOS, output length, KV growth, or queueing-relevant work by a meaningful amount. |
| Proposed next step | Build controlled measurements that can confirm, revise, or stop these two hypotheses before implementing a scheduler. |

## Phase 1 question

For the same model and same token prefix, compare histories with the same W4/W8
count but different order. At the probe token, hold the current precision fixed.
If the next-token distribution changes beyond measured repeat and chunking noise,
then precision history has left a consequential state in the KV cache.

A separate deterministic generation experiment asks whether fixed and scheduled
precisions change the first divergent token, EOS, output length, and peak KV
bytes. This second experiment measures an end-to-end service consequence; it
must not be used to claim the isolated KV mechanism because generated token
paths may already differ.

## Why this is larger than a scheduler

A conventional scheduler treats request length, KV demand, and future work as
inputs. This project tests whether a precision action can change those quantities.
If it can, scheduling becomes control of a changing system rather than grouping
requests with known costs.

No scheduler, KV quantizer, custom kernel, router, or throughput claim belongs in
Phase 1. Those are later branches that require evidence here.

## Decision states

| Evidence after the frozen main study | Decision |
|---|---|
| Controlled history effect is reproducible | Continue with precision provenance and state-consistency mechanisms. |
| Only free-running length/EOS/KV growth changes materially | Revise toward decision-dependent workload modeling. |
| Both effects are material | Continue toward a joint state-and-work controller. |
| Neither effect exceeds the frozen practical and repeat-noise gates | Stop this precision-state direction. |
| Cache correctness, checkpoint access, determinism, or cost cannot be established | Pause and report the exact missing evidence; do not substitute a weaker claim. |

Thresholds, sample identities, schedules, and analysis rules must be frozen after
a non-decision pilot and before main evidence is observed.

## Run the first study with pi-goal

Install [pi-goal](https://github.com/Michaelliv/pi-goal) once:

```bash
pi install git:github.com/Michaelliv/pi-goal
```

Clone this repository, start Pi from its root, then paste the complete
`/goal` command in [GOAL.md](GOAL.md). The command is an auditable completion
contract: a negative result is complete work, while an unavailable checkpoint or
invalid cache path is a documented pause rather than permission to improvise.
