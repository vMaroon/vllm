# KV snapshot PoC validation

## Rebase validation on current upstream

Rebased on upstream `41c4a3ed4e45792739bd8a5bbcd01269b06eae2c` on 2026-09-19.
The rebased code passes 84 focused tests and 47 freshly generated real llm-d
decoder/Pool/index comparisons. The tests include current upstream's publisher
and wire-compatibility suites. All pre-commit hooks pass on the complete PoC diff.
Two failing scope regressions and a failing
publisher-configuration assertion preceded fixes for ownership/locality isolation
and reporting the resolved snapshot endpoint.

The standalone residency auditor accepts current map events and historical array
events. Its compatibility tests pass, and it again passes all 12 comparisons on
the saved 8 GiB offload capture, including CPU-only residency. That check validates
the auditor against historical data; it is not a fresh GPU run of the rebased
recorder.

Commands run from this worktree:

```bash
.venv/bin/python -m pytest tests/distributed/test_kv_events_snapshot.py tests/distributed/test_kv_snapshot_bootstrap.py tests/distributed/test_events.py tests/distributed/test_kv_cache_events.py --confcutdir=tests/distributed -q
PYTHONPATH=. .venv/bin/python examples/others/kv_snapshot_poc/fixtures.py examples/others/kv_snapshot_poc/results/rebase-index-cases.json
```

The resulting fixtures were checked with `check_index.go` from llm-d-kv-cache
`15bb3b9219fb08f4292a8fcb43e565f4fc6477d1`. That checkout was not modified.
Current-main tests used an isolated environment with Transformers 5; the original
test environment and repository dependency files were not changed. The focused
run excludes unrelated top-level model-test fixtures and emits version/Torch
deprecation warnings.

No GPU execution on the new base or new inference-overhead measurement was run.
The GPU results below belong to pre-rebase commit `a422df16ae` and v0.22.0.
See [PR_REVIEW.md](PR_REVIEW.md) for reproduced gaps in upstream PR #50134.

## Original PoC validation

Validated on 2026-09-19. The original recorder reconstructs the tested llm-d routing
state and preserves the complete event-derived residency across bootstrap.

## Evidence

| Check | Result |
| --- | --- |
| Local snapshot, failure, handoff, and publisher tests | 56 passed |
| Deterministic real llm-d index comparisons | 47 passed |
| GPU-only snapshot and live-handoff comparisons | 12 passed |
| 1 GiB CPU-offload snapshot and live-handoff comparisons | 12 passed |
| 8 GiB CPU-offload snapshot and live-handoff comparisons | 12 passed |
| Independent complete-residency comparisons on all GPU captures | 36 passed |
| Repository pre-commit checks on changed files | Passed |

The Go checker uses the actual llm-d vLLM decoder, event Pool, token processor,
and in-memory index. The independent Python auditor checks every event-derived
resident `(medium, group, hash)` reference, including duplicates, not just routing
lookups. The Go checker tests every block in all 128 input prompts separately,
including keys beyond an evicted prefix, so an early prefix miss cannot hide a
missing child.

The original PoC failed all five new regression cases before the fix. Its
missing-ancestor snapshot also fails the real-index checker. A later real-offload
failure exposed mixed integer/byte hashes; a failing regression was added before
normalizing metadata lookup and rerunning the complete local suite and GPU test.

## Fresh GPU runs

All three runs used `vllm/vllm-openai:v0.22.0` with the three modified vLLM modules
copied into the image, Qwen3-0.6B, one GPU, 2,048 GPU blocks, 128 distinct 512-token
prompts, 32 concurrent clients, and 60 seconds of request generation. Six clients
joined at different points in each run. The reference subscriber was connected
before inference, verified an empty initial snapshot, and checked every subsequent
sequence number.

| Workload | Completed requests | Captured batches | Joiners | Index comparisons |
| --- | ---: | ---: | ---: | ---: |
| GPU eviction | 38,502 | 5,205 | 6/6 converged | 12/12 |
| GPU eviction plus 1 GiB CPU offload | 22,613 | 4,772 | 6/6 converged | 12/12 |
| GPU eviction plus 8 GiB CPU offload | 31,018 | 7,174 | 6/6 converged | 12/12 |

Each joiner supplied an initial snapshot comparison and a comparison after its
actual received live suffix. All three reference streams had zero sequence gaps.
At the final cut, the GPU-only trace held 1,929 distinct resident keys and 1,957
references. The offload trace held 2,081 distinct resident keys, with 1,960 GPU
references and 146 CPU references. Snapshots and handoffs matched these counts
exactly. The 1 GiB run's final CPU entries also existed on GPU, so a third run
increased CPU capacity to 8 GiB. That run held 2,967 resident keys at the final
cut, with 1,964 GPU references and 1,024 CPU references. Of those CPU hashes,
523 no longer existed on GPU. All six fresh indexes recovered those CPU-only
blocks; their initial snapshots exercised 525-583 CPU-only hashes.

Across the three runs, 18 joiners converged over 92,133 completed requests.
Request counts describe the correctness workload, not a throughput regression
benchmark.

The first offload attempt intentionally remains documented as a failure: it
reported unavailable at sequence 5 because offload hashes were raw bytes while
GPU hashes were integers. The successful offload results above are from the
fresh rerun after that fix.

## Coverage and limits

Local coverage includes dependency garbage collection, deep ancestry, duplicate
references, CPU-only blocks, GPU-only resets, sparse-event preservation, metadata
passthrough, deterministic record-before-send ordering, finite queue draining,
concurrent snapshot requests, DP port offsets/tags, lost-tail detection, publisher
restart and incompatible wire formats, input overflow, metadata/reply budgets, recorder exceptions, and endpoint
bind failure. Existing publisher tests still pass with snapshots disabled.

The deterministic real-index cases cover evicted parents, CPU-only blocks,
duplicates followed by partial and complete removal, resets before and after
bootstrap, sparse events, cache groups, different canonical block sizes, and
multiple snapshot cut points followed by the remaining live history.

Production EPP subscriber wiring and atomic replacement of its shared serving
index are not implemented here. The reference client returns a validated replay
for construction of a private index. The real-index checker builds and queries
that private index.

The existing llm-d Pool clears all tiers on `AllBlocksCleared`; the PoC adapter
explicitly expands that event into GPU-scoped removals. Production integration
must include the same semantic correction. Another existing decoder edge was
observed: its hash converter rejects msgpack `int8` hashes. The deterministic
fixtures use realistic 64-bit hash values; that decoder edge was not changed.

Missing metadata or a resource-limit breach disables snapshots until vLLM is
restarted. Live publishing continues, but automatic authoritative reseeding is
not implemented. Sparse-event replay preserves current consumer interpretation;
it does not repair the original event schema. No new inference-overhead benchmark,
GPU cache above 2,048 blocks, or rits-roce run was tested.

## Reproduction and artifacts

See [README.md](README.md) for the protocol, commands, resource limits, and test
manifest. The vLLM base is `8c296de63b`; the real-index checker used llm-d-kv-cache
commit `15bb3b9219fb08f4292a8fcb43e565f4fc6477d1`. That checkout was not modified.

Local artifacts are retained under `results/` (ignored by git):

- `pytest.log`, `pre-commit.log`, `unit-index.log`: verification output.
- `unit-index-cases.json`, `original-poc.json`: deterministic fixtures and the
  negative control.
- `gpu-churn.compact.json.gz`, `offload.compact.json.gz`, and
  `offload-large.compact.json.gz`: wire captures. Shared
  histories were deduplicated only after matching observed suffix payloads byte
  for byte.
- `gpu-index.log`, `offload-index.log`, `offload-large-index.log`: real-index
  comparisons.
- `gpu-residency.log`, `offload-residency.log`, `offload-large-residency.log`:
  complete-residency comparisons.
- `gpu-churn.log`, `offload.log`, `offload-large.log`, and the corresponding
  offload server logs: fresh cluster run output.
- `offload-first-run.log`: the original mixed-hash failure.

The dedicated test pods and ConfigMap were deleted.
