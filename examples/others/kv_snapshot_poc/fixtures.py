# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Emit reproducible wire fixtures for check_index.go."""

import json
import sys

import msgspec
import pybase64 as base64

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.distributed.kv_events_snapshot import KVCacheSnapshot


def stored(hashes, parent=None, tokens=None, medium="GPU", group=None):
    return BlockStored(
        block_hashes=[(1 << 40) + h for h in hashes],
        parent_block_hash=(1 << 40) + parent if parent is not None else None,
        token_ids=tokens
        if tokens is not None
        else [4 * (h - 1) + j for h in hashes for j in range(1, 5)],
        block_size=4 if medium == "GPU" else 0,
        lora_id=None,
        lora_name=None,
        medium=medium,
        group_idx=group,
    )


def removed(hashes, medium="GPU", group=None):
    return BlockRemoved(
        block_hashes=[(1 << 40) + h for h in hashes], medium=medium, group_idx=group
    )


def payload(events):
    return base64.b64encode(
        msgspec.msgpack.encode(KVEventBatch(ts=0, events=events))
    ).decode()


def case(name, batches, suffix=(), expected=None, block_size=4, tokens=None):
    snap = KVCacheSnapshot()
    for events in batches:
        snap.apply(events)
    return dict(
        name=name,
        full=[payload(b) for b in batches],
        snapshot=[payload(list(snap.export()))],
        suffix=[payload(b) for b in suffix],
        tokens=tokens or [list(range(1, 13))],
        block_size=block_size,
        expected=expected,
    )


def fixtures():
    yield case(
        "evicted parent",
        [[stored([1]), stored([2, 3], parent=1)], [removed([1])]],
        expected=2,
    )
    yield case(
        "CPU only",
        [
            [stored([1, 2, 3])],
            [stored([1, 2, 3], tokens=[], medium="CPU")],
            [removed([1, 2, 3])],
        ],
        expected=3,
    )
    yield case(
        "duplicate then one removal",
        [[stored([1, 2, 3]), stored([1, 2, 3])]],
        [[removed([1, 2, 3])]],
        expected=3,
    )
    yield case(
        "duplicate then all removals",
        [[stored([1, 2, 3]), stored([1, 2, 3])]],
        [[removed([1, 2, 3]), removed([1, 2, 3])]],
        expected=0,
    )
    yield case(
        "GPU reset preserves CPU",
        [
            [stored([1, 2, 3]), stored([1, 2, 3], tokens=[], medium="CPU")],
            [AllBlocksCleared()],
        ],
        expected=3,
    )
    yield case(
        "reset after bootstrap",
        [[stored([1, 2, 3]), stored([1, 2, 3], tokens=[], medium="CPU")]],
        [[AllBlocksCleared()]],
        expected=3,
    )
    for bs in (2, 4, 8):
        yield case(
            f"canonical {bs}",
            [[stored([1, 2, 3, 4])], [removed([2])]],
            block_size=bs,
            tokens=[list(range(1, 17))],
        )
    yield case(
        "sparse source unchanged",
        [[stored([1, 3], tokens=list(range(1, 17)))], [removed([1])]],
        expected=2,
        tokens=[list(range(1, 17))],
    )
    yield case(
        "groups",
        [
            [stored([1, 2, 3], group=0), stored([1, 2, 3], group=1)],
            [removed([2], group=0)],
        ],
        expected=3,
    )
    # Every cut and subsequent suffix of a deterministic, branching stream.
    batches = []
    for i in range(1, 101):
        start = 3 * i - 2
        batches.append([stored(list(range(start, start + 3)))])
        if i % 2 == 0:
            batches.append([stored([start + 2], tokens=[], medium="CPU")])
        batches.append([removed([start, start + 1])])
    tokens = [list(range(4 * (3 * i - 3) + 1, 4 * (3 * i) + 1)) for i in range(1, 101)]
    for cut in range(1, len(batches) + 1, 7):
        yield case(f"cut {cut}", batches[:cut], batches[cut:], tokens=tokens)


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(list(fixtures()), f)
