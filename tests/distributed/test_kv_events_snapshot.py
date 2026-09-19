# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import threading
import time
from collections import Counter

import msgspec
import pytest
import zmq

from examples.others.kv_snapshot_poc.client import ResyncRequired, SnapshotClient
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)
from vllm.distributed.kv_events_snapshot import KVCacheSnapshot, KVEventSnapshotRecorder


def stored(hashes, parent=None, medium="GPU", group=None):
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=[h * 4 + j for h in hashes for j in range(4)]
        if medium == "GPU"
        else [],
        block_size=4 if medium == "GPU" else 0,
        lora_id=None,
        lora_name=None,
        medium=medium,
        group_idx=group,
    )


def counts(events, live: Counter | None = None):
    if live is None:
        live = Counter()
    for e in events:
        if isinstance(e, BlockStored):
            live.update((e.medium, e.group_idx, h) for h in e.block_hashes)
        elif isinstance(e, BlockRemoved):
            for h in e.block_hashes:
                key = (e.medium, e.group_idx, h)
                if live[key] > 1:
                    live[key] -= 1
                else:
                    live.pop(key, None)
        else:
            for key in list(live):
                if key[0] in ("GPU", None):
                    del live[key]
    return live


def random_batches(seed, n=400):
    rng = random.Random(seed)
    live: Counter = Counter()
    for i in range(n):
        events = []
        if rng.random() < 0.5 or not live:
            start = 1 + 8 * rng.randrange(30)
            events.append(stored(list(range(start, start + 8))))
        else:
            key = rng.choice(list(live))
            if key[0] == "GPU" and rng.random() < 0.3:
                events.append(stored([key[2]], medium="CPU"))
            else:
                events.append(
                    BlockRemoved(block_hashes=[key[2]], medium=key[0], group_idx=key[1])
                )
        if rng.random() < 0.02:
            events.append(AllBlocksCleared())
        counts(events, live)
        yield events


@pytest.mark.parametrize("seed", range(20))
def test_snapshot_references_match_full_history(seed):
    snap = KVCacheSnapshot()
    expected: Counter = Counter()
    for i, events in enumerate(random_batches(seed)):
        counts(events, expected)
        snap.apply(events)
        if i % 31 == 0:
            assert counts(snap.export()) == expected
    assert counts(snap.export()) == expected


def test_metadata_collected_when_last_dependent_disappears():
    snap = KVCacheSnapshot()
    for h in range(1, 1001):
        snap.apply([stored([h])])
        snap.apply([BlockRemoved(block_hashes=[h], medium="GPU")])
        assert not snap._sources and not snap._known
        assert snap._metadata_bytes == 0


def test_dependencies_released_iteratively():
    snap = KVCacheSnapshot()
    for h in range(1, 1501):
        snap.apply([stored([h], parent=h - 1 if h > 1 else None)])
        if h > 1:
            snap.apply([BlockRemoved(block_hashes=[h - 1], medium="GPU")])
    assert len(snap) == 1
    assert len(snap._sources) == 1500
    snap.apply([BlockRemoved(block_hashes=[1500], medium="GPU")])
    assert not snap._sources and not snap._known


def test_transfer_within_batch_preserves_metadata():
    snap = KVCacheSnapshot()
    snap.apply([stored([1])])
    snap.apply(
        [BlockRemoved(block_hashes=[1], medium="GPU"), stored([1], medium="CPU")]
    )
    exported = list(snap.export())
    assert isinstance(exported[0], BlockStored) and exported[0].token_ids
    assert counts(exported) == Counter({("CPU", None, 1): 1})


def test_missing_metadata_fails_closed():
    with pytest.raises(ValueError, match="Missing reconstruction"):
        KVCacheSnapshot().apply([stored([2], parent=1)])


def test_store_metadata_is_preserved_verbatim():
    event = stored([1, 2])
    event.extra_keys = [("image", "salt"), None]
    event.lora_name = "adapter"
    event.kv_cache_spec_kind = "sliding_window"
    event.kv_cache_spec_sliding_window = 128
    snap = KVCacheSnapshot()
    snap.apply([event, BlockRemoved(block_hashes=[1], medium="GPU")])
    assert next(snap.export()) == event


@pytest.fixture
def publisher(random_port):
    def create():
        return ZmqEventPublisher(
            0,
            endpoint=f"tcp://*:{random_port}",
            snapshot_endpoint=f"tcp://*:{random_port + 1}",
        )

    pub = create()
    yield pub, random_port, create
    pub.shutdown()


def client(port):
    return SnapshotClient(f"tcp://127.0.0.1:{port}", f"tcp://127.0.0.1:{port + 1}")


def request(port):
    with zmq.Context.instance().socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{port + 1}")
        sock.send(b"snapshot")
        assert sock.poll(5000)
        return sock.recv_multipart()


def publish(pub, events):
    pub.publish(KVEventBatch(ts=time.time(), events=events))
    pub._event_queue.join()


def test_empty_snapshot_and_idle_subscription(publisher):
    pub, port, _ = publisher
    reply = request(port)
    assert int.from_bytes(reply[0], "big", signed=True) == -1
    assert reply[1] == pub._snapshot_stream_id
    with_client = client(port)
    try:
        seq, payloads = with_client.bootstrap()
        assert seq >= 0  # heartbeat establishes SUB delivery
        assert payloads == []
    finally:
        with_client.close()


def test_record_happens_before_send(publisher, monkeypatch):
    pub, port, _ = publisher
    entered, release = threading.Event(), threading.Event()
    record = pub._snapshot_recorder.record

    def blocked(seq, payload):
        record(seq, payload)
        entered.set()
        assert release.wait(5)

    c = client(port)
    c.bootstrap()
    monkeypatch.setattr(pub._snapshot_recorder, "record", blocked)
    try:
        pub.publish(KVEventBatch(ts=0, events=[stored([1])]))
        assert entered.wait(5)
        assert not c.sub.poll(100)
        reply = request(port)
        assert any(msgspec.msgpack.decode(chunk)[1] for chunk in reply[2:])
    finally:
        release.set()
        c.close()


def test_overflow_disables_snapshot_without_losing_live_batch(publisher, monkeypatch):
    pub, port, _ = publisher
    c = client(port)
    try:
        c.bootstrap()
        monkeypatch.setattr(pub._snapshot_recorder, "MAX_PENDING_BYTES", 1)
        publish(pub, [stored([1])])
        assert c.poll() is not None
        assert int.from_bytes(request(port)[0], "big", signed=True) == -2
    finally:
        c.close()


def test_fold_failure_is_unavailable(publisher, monkeypatch):
    pub, port, _ = publisher

    def boom(events):
        raise RuntimeError("injected")

    monkeypatch.setattr(pub._snapshot_recorder._snapshot, "apply", boom)
    publish(pub, [stored([1])])
    assert int.from_bytes(request(port)[0], "big", signed=True) == -2


@pytest.mark.parametrize("budget", ["metadata", "reply"])
def test_resource_budget_fails_closed(publisher, monkeypatch, budget):
    pub, port, _ = publisher
    recorder = pub._snapshot_recorder
    if budget == "metadata":
        monkeypatch.setattr(recorder._snapshot, "MAX_METADATA_BYTES", 1)
    else:
        monkeypatch.setattr(recorder, "MAX_REPLY_BYTES", 1)
    publish(pub, [stored([1])])
    reply = request(port)
    assert int.from_bytes(reply[0], "big", signed=True) == -2
    assert len(reply) == 2


def test_concurrent_requests(publisher):
    from concurrent.futures import ThreadPoolExecutor

    pub, port, _ = publisher
    publish(pub, [stored([1, 2, 3])])
    with ThreadPoolExecutor(max_workers=8) as executor:
        replies = list(executor.map(lambda _: request(port), range(32)))
    for reply in replies:
        assert int.from_bytes(reply[0], "big", signed=True) >= 0
        events = (
            event
            for chunk in reply[2:]
            for event in msgspec.msgpack.decode(chunk, type=KVEventBatch).events
        )
        assert counts(events) == Counter({("GPU", None, h): 1 for h in (1, 2, 3)})


def test_data_parallel_rank_offsets_and_tags(random_port):
    pub = ZmqEventPublisher(
        2,
        endpoint=f"tcp://*:{random_port}",
        snapshot_endpoint=f"tcp://*:{random_port + 1}",
    )
    try:
        assert (
            pub.get_publisher_config().snapshot_endpoint == f"tcp://*:{random_port + 3}"
        )
        publish(pub, [stored([1])])
        reply = request(random_port + 2)
        assert reply[1] == pub._snapshot_stream_id
        assert (
            msgspec.msgpack.decode(reply[2], type=KVEventBatch).data_parallel_rank == 2
        )
    finally:
        pub.shutdown()


def test_snapshot_bind_failure_does_not_fail_publisher(random_port):
    with zmq.Context.instance().socket(zmq.ROUTER) as occupied:
        occupied.bind(f"tcp://*:{random_port + 1}")
        pub = ZmqEventPublisher(
            0,
            endpoint=f"tcp://*:{random_port}",
            snapshot_endpoint=f"tcp://*:{random_port + 1}",
        )
        try:
            assert pub._snapshot_recorder._failed.is_set()
            publish(pub, [stored([1])])
            assert len(pub._buffer) == 1
        finally:
            pub.shutdown()


def test_drain_takes_finite_cut():
    recorder = KVEventSnapshotRecorder.__new__(KVEventSnapshotRecorder)
    import queue

    recorder._inbox = queue.Queue()
    recorder._lock = threading.Lock()
    recorder._failed = threading.Event()
    payload = msgspec.msgpack.encode(KVEventBatch(ts=0, events=[]))
    recorder._inbox.put((0, payload))
    recorder._pending_bytes = len(payload)

    class Replenishing:
        def apply(self, events):
            recorder._inbox.put((1, payload))
            recorder._pending_bytes += len(payload)

    recorder._snapshot = Replenishing()
    recorder._drain(msgspec.msgpack.Decoder(type=KVEventBatch))
    assert recorder._seq == 0
    assert recorder._inbox.qsize() == 1


@pytest.mark.parametrize("seed", range(3))
def test_midstream_bootstrap_converges(publisher, seed):
    pub, port, _ = publisher
    batches = list(random_batches(seed, 600))
    expected: Counter = Counter()
    for events in batches:
        counts(events, expected)

    def produce():
        for events in batches:
            publish(pub, events)
            time.sleep(0.001)

    producer = threading.Thread(target=produce)
    c = client(port)
    producer.start()
    try:
        time.sleep(0.1)
        _, payloads = c.bootstrap()
        producer.join()
        target = pub._buffer[-1][0] + 1
        while c.next_seq < target:
            payload = c.poll()
            if payload is not None:
                payloads.append(payload)
        actual: Counter = Counter()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        for payload in payloads:
            counts(decoder.decode(payload).events, actual)
        assert actual == expected
    finally:
        producer.join()
        c.close()


def test_gap_requires_fresh_snapshot(publisher):
    pub, port, _ = publisher
    c = client(port)
    try:
        c.bootstrap()
        publish(pub, [stored([1])])
        assert c.sub.poll(2000)
        c.sub.recv_multipart()  # lose the last data batch
        with pytest.raises(ResyncRequired, match="gap"):
            # The idle heartbeat exposes the lost tail.
            while True:
                c.poll(2000)
        assert not c.ready
        _, payloads = c.bootstrap()
        assert counts(
            e
            for p in payloads
            for e in msgspec.msgpack.decode(p, type=KVEventBatch).events
        ) == Counter({("GPU", None, 1): 1})
    finally:
        c.close()


def test_incompatible_publisher_marks_source_unready(publisher, monkeypatch):
    pub, port, _ = publisher
    c = client(port)
    try:
        c.bootstrap()
        monkeypatch.setattr(pub, "_snapshot_stream_id", b"")
        publish(pub, [stored([1])])
        with pytest.raises(ResyncRequired, match="identities"):
            c.poll()
        assert not c.ready
    finally:
        c.close()


def test_restart_changes_epoch(publisher):
    pub, port, create = publisher
    c = client(port)
    replacement = None
    try:
        c.bootstrap()
        old = c.stream_id
        pub.shutdown()
        replacement = create()
        with pytest.raises(ResyncRequired, match="restart"):
            while True:
                c.poll(2000)
        c.bootstrap()
        assert c.stream_id != old
    finally:
        c.close()
        if replacement:
            replacement.shutdown()
