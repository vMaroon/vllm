# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture real vLLM snapshots and live handoffs as check_index.go fixtures."""

import concurrent.futures
import json
import random
import threading
import time
import urllib.request

import pybase64 as base64
import zmq
from client import SnapshotClient


def encode(payloads):
    return [base64.b64encode(p).decode() for p in payloads]


def main():
    url = "http://127.0.0.1:8000"
    deadline = time.monotonic() + 600
    while True:
        try:
            urllib.request.urlopen(url + "/health", timeout=2).close()
            break
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)
    history = []
    lock = threading.Lock()
    ready = threading.Event()
    stop = threading.Event()
    errors = []

    def collect():
        try:
            with zmq.Context.instance().socket(zmq.SUB) as sub:
                sub.setsockopt(zmq.SUBSCRIBE, b"")
                sub.connect("tcp://127.0.0.1:5557")
                previous = None
                while not stop.is_set():
                    if not sub.poll(100):
                        continue
                    _, seq, raw = sub.recv_multipart()
                    number = int.from_bytes(seq[:8], "big")
                    if previous is not None and number != previous + 1:
                        raise AssertionError("ground-truth stream lost a batch")
                    previous = number
                    with lock:
                        history.append((number, raw))
                    ready.set()
        except Exception as e:
            errors.append(repr(e))
            ready.set()

    collector = threading.Thread(target=collect)
    collector.start()
    assert ready.wait(10) and not errors, errors
    initial = SnapshotClient("tcp://127.0.0.1:5557", "tcp://127.0.0.1:5657")
    try:
        _, empty = initial.bootstrap()
        assert not empty, "capture must start before any inference"
    finally:
        initial.close()
    prompts = []
    rng = random.Random(712)
    for _ in range(128):
        prompts.append([1000] * 16 + [rng.randrange(1001, 5000) for _ in range(496)])
    load_stop = threading.Event()
    completed = []

    def load(worker):
        local = random.Random(worker)
        while not load_stop.is_set():
            body = json.dumps(
                dict(
                    model="Qwen/Qwen3-0.6B",
                    prompt=local.choice(prompts),
                    max_tokens=4,
                    ignore_eos=True,
                    temperature=0,
                )
            ).encode()
            req = urllib.request.Request(
                url + "/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                response.read()
            completed.append(1)

    target = [None]
    joins = []

    def joiner(delay):
        c = SnapshotClient("tcp://127.0.0.1:5557", "tcp://127.0.0.1:5657")
        try:
            time.sleep(delay)
            begin = time.monotonic()
            _, payloads = c.bootstrap()
            snapshot = list(payloads)
            cut = c.next_seq - 1
            latency = time.monotonic() - begin
            while target[0] is None or c.next_seq <= target[0]:
                payload = c.poll(100)
                if payload is not None:
                    payloads.append(payload)
            joins.append((cut, snapshot, payloads, c.next_seq - 1, latency))
        except Exception as e:
            errors.append(repr(e))
        finally:
            c.close()

    threads = [
        threading.Thread(target=joiner, args=(delay,))
        for delay in (3, 8, 15, 25, 40, 50)
    ]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        loads = [executor.submit(load, i) for i in range(32)]
        for t in threads:
            t.start()
        while time.monotonic() - started < 60 and not errors:
            time.sleep(1)
        load_stop.set()
        for f in loads:
            f.result()
    time.sleep(1)
    with lock:
        target[0] = history[-1][0]
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive(), "joiner did not catch up"
    stop.set()
    collector.join()
    assert not errors, errors
    cases = []
    for i, (cut, snapshot, actual, end, latency) in enumerate(joins):
        for name, seq, payloads in (
            ("snapshot", cut, snapshot),
            ("handoff", end, actual),
        ):
            cases.append(
                dict(
                    name=f"GPU {name} {i}",
                    full=encode(p for n, p in history if n <= seq),
                    snapshot=encode(payloads),
                    suffix=[],
                    tokens=prompts,
                    block_size=16,
                )
            )
        print(
            f"joiner {i}: cut={cut}, final={end}, bootstrap={latency * 1000:.1f}ms",
            flush=True,
        )
    with open("/tmp/kv-snapshot-cases.json", "w") as f:
        json.dump(cases, f)
    print(
        f"CAPTURE PASS: {len(completed)} requests, {len(history)} batches, "
        f"{len(joins)} joiners, {len(cases)} index fixtures",
        flush=True,
    )


if __name__ == "__main__":
    main()
