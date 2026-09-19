# Review of upstream KV snapshot PR #50134

Reviewed [PR #50134](https://github.com/vllm-project/vllm/pull/50134) at
`c6f332839388a26e5e800db21a102e9d13f703a6` on 2026-09-19. It is a useful synchronous
baseline, but it does not yet cover the CPU-offload and reference-count behavior
required by this PoC. Its merge is not a dependency of this fork.

## Reproduced correctness gaps

The review loaded the PR's exact `kv_events.py` and exercised `_KVCacheState`
directly. These are local state-machine reproductions, not GPU runs of that PR.
The same three failures were then reproduced through the real llm-d decoder,
Pool, and index using this PoC's fixture checker: each reference had three routing
keys and each PR snapshot path had zero. The evicted-parent control passed with
two keys. The checker includes the GPU-scoped reset adaptation documented in
README.md; it does not change the PR's snapshot state.

1. **CPU-only blocks lose reconstruction metadata.** Store a GPU block with
   tokens, store the same hash on CPU without tokens, then remove the GPU copy.
   The snapshot contains the tokenless CPU store without the GPU source needed
   to reconstruct its routing key. The dependency graph only follows
   `parent_block_hash`; it does not retain token sources for offloaded blocks.
   [Source](https://github.com/vllm-project/vllm/blob/c6f332839388a26e5e800db21a102e9d13f703a6/vllm/distributed/kv_events.py#L243)
2. **One removal drops all duplicate copies.** Store the same GPU hash twice,
   then remove it once. Full event replay retains one reference; the snapshot
   is empty. `_active_keys` tracks membership, and a repeated store replaces the
   earlier source without preserving its count.
   [Source](https://github.com/vllm-project/vllm/blob/c6f332839388a26e5e800db21a102e9d13f703a6/vllm/distributed/kv_events.py#L233)
3. **A GPU reset drops CPU residency.** Store GPU and CPU copies, then apply
   `AllBlocksCleared`. The snapshot loses the CPU block too. vLLM's block-pool
   reset emits this event for GPU state; the PR clears every tier.
   [Source](https://github.com/vllm-project/vllm/blob/c6f332839388a26e5e800db21a102e9d13f703a6/vllm/distributed/kv_events.py#L178)

The evicted-parent reproduction passed. The PR retains inactive ancestors and
orders parent stores before children, including restored parents. Its explicit
ownership and locality keys are also useful; the PoC needed corresponding scope
preservation when rebased onto current main.

## Availability and protocol

Snapshot planning, encoding, and transport occupy the publisher thread. Live
publishing pauses; a full publisher queue can then block the engine. This is a
documented design tradeoff, acknowledged by the author. The reported CPU timings
exclude ROUTER transport. The PoC instead isolates recording and serving on a
separate thread with bounded input and an explicit unavailable response.

The PR reuses the replay endpoint and existing framing, which is simpler for
deployment. It does not add a publisher lifetime identity or idle heartbeat;
those protections must be supplied separately for reliable restart and lost-tail
detection. The PoC supplies them but requires an updated consumer.

Invalid dependency plans are validated before sending any snapshot frames.
Clients time out instead of receiving an explicit error. This is fail-closed;
the automated suggestion to send an end marker alone was correctly withdrawn.

## Scope

The PoC's default incremental-event contract remains necessary. Current main's
optional full request reporting emits reuse as another `BlockStored`, making
physical-copy counts ambiguous. Neither snapshot implementation repairs that
event-schema issue.

Keep the fork usable independently. Any separate upstream proposal should explain
the recorder-thread, bounded-input, lifetime, and offload differences to the
existing issue's participants; no public review or comment was posted here.
