# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent reference-count audit of every hash in a compact capture."""

import gzip
import json
import sys
from collections import Counter

import msgspec
import pybase64 as base64


def apply(live, payload):
    batch = msgspec.msgpack.decode(base64.b64decode(payload))
    for event in batch[1]:
        if isinstance(event, list):
            fields = (
                [
                    "block_hashes",
                    "parent_block_hash",
                    "token_ids",
                    "block_size",
                    "lora_id",
                    "medium",
                    "lora_name",
                    "extra_keys",
                    "group_idx",
                    "kv_cache_spec_kind",
                    "kv_cache_spec_sliding_window",
                    "locality",
                    "ownership",
                ]
                if event[0] == "BlockStored"
                else ["block_hashes", "medium", "group_idx", "locality", "ownership"]
            )
            event = {"type": event[0], **dict(zip(fields, event[1:]))}
        tag = event["type"]
        scope = (
            event.get("medium"),
            event.get("group_idx"),
            event.get("locality"),
            event.get("ownership"),
        )
        if tag == "BlockStored":
            live.update((*scope, h) for h in event["block_hashes"])
        elif tag == "BlockRemoved":
            for h in event["block_hashes"]:
                key = (*scope, h)
                if live[key] > 1:
                    live[key] -= 1
                else:
                    live.pop(key, None)
        elif tag == "AllBlocksCleared":
            for key in list(live):
                if key[0] in ("GPU", None):
                    del live[key]
        else:
            raise AssertionError(tag)


def main():
    with gzip.open(sys.argv[1], "rt") as f:
        capture = json.load(f)
    history = capture["history"]
    cuts = {c["full_count"] for c in capture["cases"]}
    reference = {}
    live = Counter()
    for i, payload in enumerate(history, 1):
        apply(live, payload)
        if i in cuts:
            reference[i] = live.copy()
    max_cpu_only = 0
    for case in capture["cases"]:
        actual = Counter()
        for payload in case["snapshot"]:
            apply(actual, payload)
        for payload in history[case["tail_start"] : case["full_count"]]:
            apply(actual, payload)
        assert actual == reference[case["full_count"]], case["name"]
        tiers = Counter()
        for key, count in actual.items():
            tiers[key[0]] += count

        def normalize(h):
            return int.from_bytes(h[-8:], "big") if isinstance(h, bytes) else h

        gpu = {normalize(key[-1]) for key in actual if key[0] == "GPU"}
        cpu = {normalize(key[-1]) for key in actual if key[0] == "CPU"}
        max_cpu_only = max(max_cpu_only, len(cpu - gpu))
        print(
            f"PASS {case['name']}: {len(actual)} keys,",
            f"refs={dict(tiers)}, CPU-only={len(cpu - gpu)}",
        )
    if "--require-cpu-only" in sys.argv[2:]:
        assert max_cpu_only > 0, "capture did not exercise CPU-only residency"
    print(f"PASS: {len(capture['cases'])} complete residency comparisons")


if __name__ == "__main__":
    main()
