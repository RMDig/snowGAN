"""Focused tests for the native-CPU-RAM probe (snowgan.memtrace).

The probe's value is its math and its record shape, both of which must be
correct without a GPU or a real leak to observe. These tests pin:
  * the least-squares leak-rate fit,
  * the per-substep delta keying (which phase a delta is attributed to),
  * the no-op contract (None unless SNOWGAN_MEMTRACE is set),
  * the JSONL record shape written by tick().
Native probes (mallinfo2/proc) degrade to None/-1 off Linux, so the record
shape is asserted in a platform-independent way.
"""
from __future__ import annotations

import json
import os

import pytest

from snowgan import memtrace as mt


def test_linfit_recovers_known_slope():
    # y = 3x + 7 -> slope 3, perfect fit.
    xs = [0, 1, 2, 3, 4]
    ys = [7, 10, 13, 16, 19]
    slope, r2 = mt._linfit(xs, ys)
    assert slope == pytest.approx(3.0)
    assert r2 == pytest.approx(1.0)


def test_linfit_undetermined():
    assert mt._linfit([1], [1]) == (None, None)        # too few points
    assert mt._linfit([2, 2, 2], [1, 2, 3]) == (None, None)  # no x-spread


def test_maybe_create_is_noop_without_env(monkeypatch):
    monkeypatch.delenv("SNOWGAN_MEMTRACE", raising=False)
    assert mt.maybe_create_memtrace("/tmp") is None


def test_substep_deltas_attributed_to_preceding_label(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWGAN_MEMTRACE", "1")
    monkeypatch.delenv("SNOWGAN_MEMTRACE_TRIM", raising=False)
    monkeypatch.delenv("SNOWGAN_MEMTRACE_PYTHON", raising=False)
    tracer = mt.maybe_create_memtrace(str(tmp_path))
    assert tracer is not None

    # Force a known RSS sequence so the deltas are deterministic regardless of
    # the real process: 100 -> 105 (disc) -> 106 (gen) -> 110 (post) -> 110.
    seq = iter([100.0, 105.0, 106.0, 110.0, 110.0])
    monkeypatch.setattr(mt, "_rss_mib", lambda: next(seq))
    for label in ("disc_loop", "gen_loop", "poststep", "end"):
        tracer.note(label)
    deltas = tracer._drain_substeps()
    assert deltas["disc_loop_mib"] == pytest.approx(5.0)
    assert deltas["gen_loop_mib"] == pytest.approx(1.0)
    assert deltas["poststep_mib"] == pytest.approx(4.0)
    assert deltas["total_mib"] == pytest.approx(10.0)
    # Drained: a second call with no new marks yields nothing.
    assert tracer._drain_substeps() is None
    tracer.close()


def test_tick_writes_wellformed_jsonl(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWGAN_MEMTRACE", "5")
    tracer = mt.maybe_create_memtrace(str(tmp_path))
    assert tracer.due(10) and not tracer.due(11)  # cadence honors the interval

    tracer.tick(10)
    tracer.tick(15)
    tracer.close()

    lines = [l for l in open(tracer.path).read().splitlines() if l]
    assert len(lines) == 2
    rec = json.loads(lines[0])
    # Platform-independent keys that must always be present.
    for key in ("ts", "batch", "rss_mib", "mallinfo2", "maps_count",
                "gc_objects", "leak_rate"):
        assert key in rec, key
    assert rec["batch"] == 10
    assert rec["leak_rate"]["window"] == 1   # first sample
    assert json.loads(lines[1])["leak_rate"]["window"] == 2
