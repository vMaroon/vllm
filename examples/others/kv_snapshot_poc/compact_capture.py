# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deduplicate captured histories after checking every observed live suffix."""

import gzip
import json
import sys


def compact(cases):
    history = max((c["full"] for c in cases), key=len)
    result = []
    for case in cases:
        end = len(case["full"])
        assert case["full"] == history[:end]
        payloads = case["snapshot"]
        cursor = end
        length = len(payloads)
        while length and cursor and payloads[length - 1] == history[cursor - 1]:
            length -= 1
            cursor -= 1
        item = dict(case)
        del item["full"]
        item["full_count"] = end
        item["tail_start"] = cursor
        item["snapshot"] = payloads[:length]
        result.append(item)
    return dict(history=history, cases=result)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        capture = compact(json.load(f))
    with gzip.open(sys.argv[2], "wt") as f:
        json.dump(capture, f)
    print(f"Compacted {len(capture['cases'])} cases, {len(capture['history'])} batches")
