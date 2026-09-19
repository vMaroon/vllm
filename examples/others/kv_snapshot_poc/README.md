# KV snapshot bootstrap PoC

This PoC rebuilds a fresh llm-d routing index from a compacted KV event history,
then hands it over to contiguous live events. The production changes are limited
to the KV event configuration, publisher, and the recorder module.

Upstream [PR #50134](https://github.com/vllm-project/vllm/pull/50134) also addresses
[issue #40363](https://github.com/vllm-project/vllm/issues/40363) and was open when
checked on 2026-09-19. It serves snapshots synchronously through the existing
replay endpoint. This PoC explores a separate recorder thread with bounded input,
publisher identities, and idle heartbeats. Coordinate with that PR before
proposing a separate upstream submission.

## State and reconstruction

The recorder tracks resident reference counts by
`(medium, group, locality, ownership, hash)`. It
retains whole source `BlockStored` events and their reconstruction dependencies.
An evicted ancestor's metadata survives while a live descendant or offloaded
block needs it. Reference counting releases unused sources at batch boundaries.

A snapshot replays retained source events in original order, restores duplicate
store counts, then removes excess residency. Consumers must apply this replay to
a **private index** and expose that index only after bootstrap succeeds. Ancestors
included for reconstruction must never temporarily appear in the serving index.

Whole source events preserve sparse token/hash alignment, LoRA fields, extra
keys, and group metadata. This preserves the existing consumer's interpretation;
it does not fix ambiguities in the original sparse-attention event schema.
GPU integer hashes and CPU byte hashes are normalized only for metadata lookup.
Wire events remain unchanged.

## PoC wire contract

Setting `snapshot_endpoint` enables the recorder, publisher identities, and idle
heartbeats. Without it, the existing publisher wire format is unchanged.

- Live PUB frames: `[topic, sequence, payload]`. `sequence` contains an unsigned
  8-byte big-endian number followed by a 16-byte publisher UUID. `payload` is the
  existing encoded `KVEventBatch`. Readers must decode the first eight sequence
  bytes separately; readers requiring exactly eight bytes need an update.
- Snapshot request: any single frame through REQ, or a DEALER request.
- Snapshot reply: `[sequence, publisher_uuid, chunk, ...]`. The sequence is an
  8-byte signed big-endian number. `-1` means no batches recorded; `-2` means
  unavailable. Chunks are ordinary `KVEventBatch` payloads containing stores and
  removals.
- An idle publisher emits an empty batch every second. These batches establish
  subscription delivery and expose a dropped final data batch.

`client.py` implements the reference handoff:

1. Receive a live message before requesting a snapshot, establishing SUB delivery.
2. Buffer live traffic while receiving the snapshot.
3. Build a private index from the snapshot and buffered batches above its sequence.
4. Accept only consecutive batches from the same publisher UUID.
5. On a gap, restart, timeout, or buffer overflow, mark the source unready and
   bootstrap again. This client does not use replay-buffer recovery.

The recorder receives the exact immutable payload before PUB sends it. A snapshot
processes a finite FIFO cut of that queue, so ongoing arrivals cannot indefinitely
postpone the reply. Only the recorder thread operates its ROUTER socket.

## Bounds and failure behavior

The PoC limits pending input to 4,096 batches and 64 MiB, accounted metadata to
256 MiB, resident references to one million, encoded replies to 256 MiB, and the
client's live bootstrap buffer to 64 MiB. Metadata accounting includes an estimate
for decoded objects; it is not a measured process RSS limit. Snapshot construction
shares the engine process and GIL, so this does not promise zero inference overhead.

Missing reconstruction metadata, an unsupported event, or a resource-limit breach
invalidates the recorder. It serves unavailable instead of a partial snapshot;
live publishing continues. **Recovery from lost recorder input requires a vLLM
restart in this PoC.** Automatic authoritative reseeding is not implemented.

The recorder must run from the beginning of the publisher's lifetime. Reattaching
it to an already populated cache cannot reconstruct previously unseen metadata.

## Real llm-d validation

`check_index.go` uses the actual vLLM adapter, event-processing Pool, token
processor, and in-memory index from the user's llm-d-kv-cache checkout. It compares
routing entries, including tiers and groups, after full replay versus snapshot
bootstrap and a live suffix. Explicit expected counts prevent the critical
fixtures from passing with two equally empty indexes. Other cases require a
nonempty reference result.

There is one explicit consumer adaptation: vLLM's `AllBlocksCleared` clears GPU
state, while the current llm-d Pool clears all tiers. The harness expands it into
scoped GPU removals before passing it to the real Pool. This adaptation must also
be included in production integration; these results do not validate the current
unmodified reset behavior.

`fixtures.py` creates deterministic dependency, CPU-only, duplicate, reset, sparse,
group, canonical-block-size, and cut-point cases. `gpu_capture.py` runs real
inference while six consumers join. It captures both their initial snapshots and
the live batches they actually received. `compact_capture.py` deduplicates the
captures only after checking that each observed trailing payload matches the
corresponding ground-truth payload byte for byte.

## Reproduce locally

Run from this vLLM worktree, using the existing repository virtual environment:

```bash
VLLM_PYTHON=.venv/bin/python
POC="$PWD/examples/others/kv_snapshot_poc"
mkdir -p "$POC/results"
"$VLLM_PYTHON" -m pytest tests/distributed/test_kv_events_snapshot.py tests/distributed/test_kv_snapshot_bootstrap.py tests/distributed/test_events.py tests/distributed/test_kv_cache_events.py --confcutdir=tests/distributed -q
PYTHONPATH=. "$VLLM_PYTHON" "$POC/fixtures.py" "$POC/results/unit-index-cases.json"
```

From the llm-d-kv-cache checkout, run `go run /absolute/path/to/check_index.go
/absolute/path/to/unit-index-cases.json`. The same checker accepts compact GPU
capture `.json.gz` files. No changes to that checkout are required.

`pod.yaml` records the original one-GPU kermit test: vLLM v0.22.0,
Qwen3-0.6B, 2,048 GPU blocks, 128 distinct 512-token prompts, 32 concurrent clients,
and 60 seconds of load. Its ConfigMap needs the three patched vLLM files,
`client.py`, and `gpu_capture.py` from commit `a422df16ae`. Do not overlay the
rebased main-branch modules onto that older image. A GPU run of the rebased code
requires an image built from its upstream base; this rebase has local validation
only. The original offload variant adds:

```text
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size":64,"cpu_bytes_to_use":1073741824}}'
```

The independent `check_residency.py` auditor compares every resident hash and
reference count, including CPU tiers, rather than only queried routing keys.
These GPU runs test correctness; their request counts are not a throughput
regression benchmark.

Use `--require-cpu-only` with the residency auditor to reject captures that never
exercise a block present on CPU but absent on GPU. The 8 GiB CPU-offload run uses
`cpu_bytes_to_use: 8589934592` and passes that check.

See [RESULTS.md](RESULTS.md) for measured outcomes. Local capture artifacts and
logs are stored in ignored `results/`. Production
EPP subscriber wiring and atomic installation into its shared serving index are
outside this standalone PoC; the reference client and private-index checks make
that integration contract executable.

Use the default incremental KV event reporting mode. Current upstream's optional
`full` request mode emits `BlockStored` again for reused blocks without a separate
reuse marker. Event reference counts cannot distinguish those reports from new
physical copies. This PoC reconstructs event-derived state and does not resolve
that upstream ambiguity. See [PR_REVIEW.md](PR_REVIEW.md) for the comparison with
the existing upstream snapshot proposal.
