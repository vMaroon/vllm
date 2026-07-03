# [RFC]: Profiling-driven, background loading from secondary KV tiers (storage → CPU) for waiting requests

> Draft for the GitHub RFC form (template sections: Motivation / Proposed Change /
> Feedback Period / CC List / Any Other Things). Follow-up to
> [#38260 — Multi-tier KV offloading via the offloading connector](https://github.com/vllm-project/vllm/issues/38260).

## Motivation

[#38260](https://github.com/vllm-project/vllm/issues/38260) introduces multi-tier KV
offloading: a CPU primary tier (the GPU gateway) plus pluggable secondary tiers
(storage, network). The current design (`vllm/v1/kv_offload/tiering/`) follows two
principles that are correct for *storing* but blunt for *loading*:

1. **"Always offload to all tiers"** — every stored block is cascaded to every
   secondary tier.
2. **"Primary tier is the gateway"** — a secondary-tier hit is promoted into CPU
   before the GPU can read it (staged promotion).

What's missing is a **criterion and a schedule for loading back**. Two open questions
fall out of this:

- **Is a hit on a secondary tier worth loading at all?** Loading *N* cached tokens from
  a slow storage/network tier only beats recomputing them if the transfer is faster than
  prefill for those tokens. With no break-even logic, we either load unconditionally
  (and can be *slower* than recompute on a slow tier) or never load (and waste the cache).
  In the thread, @dannyharnik already asked for the ability to *"selectively do loads
  directly from storage"* — but "selectively" was left undefined. This RFC defines it.

- **When should the load happen?** Promotion is triggered lazily, when the scheduler
  first looks the request up. For a slow tier this puts secondary-tier latency on (or
  near) the scheduling critical path. Requests sitting in the **waiting queue** are an
  opportunity: we can start `storage → CPU` promotion the moment a request is admitted,
  so the KV is staged in the CPU gateway by the time the request is scheduled —
  amortizing slow-tier latency against queueing delay instead of paying it inline.

Neither [#38260](https://github.com/vllm-project/vllm/issues/38260) nor the roadmap
[#33689](https://github.com/vllm-project/vllm/issues/33689) addresses tier-value
profiling, break-even/admission thresholds, or prefetch-while-waiting. This RFC proposes
all three.

## Proposed Change

Three pieces: (A) profile each tier's *value* at startup, (B) use it as a per-hit
**admission threshold**, (C) drive loads **in the background while requests wait**,
reusing the existing async connector hooks.

### A. Startup tier profiling — "what is this tier worth?"

A one-shot probe per secondary tier, run at engine startup (the worker already owns the
transfer machinery used for store/promote):

- Measure the **secondary → primary** read path: latency and effective bandwidth, both
  per-block and bulk; optionally the write/store path.
- Combine with the **primary → GPU** leg and the model's measured **prefill throughput**
  (already profiled during memory profiling) to derive, per tier, a **break-even hit
  length `L*`** — the minimum cached-prefix length for which
  `time_to_load(N) < time_to_recompute(N)`.
- Emit a **startup report + Prometheus gauges**: per tier `L*`, measured bandwidth, and
  expected ms-saved as a function of hit length. This is the concrete *"value you can get
  from this tier"* signal an operator can read.
- **Enablement gate:** if even a maximal hit does not beat recompute (tier slower than
  prefill end-to-end), mark the tier **load-disabled** — keep *storing* to it (capacity /
  cross-engine reuse) but never schedule loads. This is the fail-closed default for a tier
  that profiling shows cannot help.

### B. Admission threshold — "load this hit if it's useful"

At lookup, gate promotion on the profiled threshold instead of promoting unconditionally:

- Promote a secondary-tier hit only when `hit_length >= L*` **and** the tier's bandwidth
  budget (below) allows. Otherwise treat as a miss → recompute.
- Implemented in the scheduler-side manager so it has request/queue context (see C).
- A runtime **EWMA of observed tier latency** refines `L*` under load (startup profiling
  sets the prior). MVP ships the static threshold; adaptive refinement is a follow-up.

### C. Background loading while requests are waiting

The connector interface already supports asynchronous loads between scheduler steps —
this is the integration seam, not new infrastructure:

- `get_num_new_matched_tokens(request, num_computed_tokens) -> (num_tokens | None, async)`
  — returning `None` means *"need more time, query again later"*; the async bool flags a
  load that completes between steps. Promotion in `TieringOffloadingManager` is already
  async (`submit_store`/job tracking; `lookup()` returns `None` = "promotion started,
  retry later"; `ref_cnt` protects in-flight blocks).
- `on_new_request(request)` fires when a request is **admitted to the waiting queue** —
  the trigger for **prefetch-on-admission**: kick off `storage → CPU` promotion here.
- `update_state_after_alloc()` is already specified to be called twice for async loads
  (allocate landing blocks, then again after the transfer completes) — the staging
  handshake we need.

Flow for a waiting request with a profitable secondary-tier hit:

1. `on_new_request` → start background `storage → CPU` promotion for the matched prefix
   (subject to the §B threshold).
2. While the request waits, `get_num_new_matched_tokens` returns `None` (or partial) until
   promotion lands; `ref_cnt` protects the staged blocks.
3. **Recompute fallback:** if the load has not landed by the time the request would
   otherwise be scheduled (deadline derived from `L*`/measured latency), return the
   already-computed prefix and let the rest recompute — never stall the step on a slow tier.
4. Once staged in CPU, the existing primary → GPU path serves it normally.

**Bandwidth budget / QoS:** background promotions share device and host bandwidth with
the critical primary↔GPU path and offload-stores. Profiling supplies the measured
bandwidth used to budget/throttle background loads so they don't starve the hot path.

### Config & metrics

- `kv_connector_extra_config`: per-tier `load_enabled` (auto from profiling, overridable),
  `min_load_tokens` (override `L*`), `prefetch_on_admission` (bool),
  `background_load_bandwidth_fraction`.
- Metrics: per-tier `L*`, measured bandwidth, prefetch hit/late/fallback counts,
  background-load queue depth, bytes loaded vs. tokens recomputed-after-fallback.

## Feedback Period

Two weeks.

## CC List

@dannyharnik @orozery @ruihong123 — plus KV-offload / connector maintainers from
[#38260](https://github.com/vllm-project/vllm/issues/38260),
[#33689](https://github.com/vllm-project/vllm/issues/33689),
[#19854](https://github.com/vllm-project/vllm/issues/19854).

## Any Other Things

- **GDS interplay:** profiling can also choose *staged-via-primary vs. direct GDS* per tier
  from measured bandwidths — this connects to @dannyharnik's "selectively do loads directly
  from storage using GDS" while keeping the store path via the primary tier.
- **TP > 1:** profile per-rank; decide whether `L*` is per-rank or aggregated.
- **Speculative load-vs-recompute race** for near-break-even hits (start the load but don't
  block; use whichever wins) — flagged as future work; double-work risk needs a budget.
- **Scope:** this RFC is the *loading* side of multi-tier offloading and assumes the
  storing/cascade design from [#38260](https://github.com/vllm-project/vllm/issues/38260).
