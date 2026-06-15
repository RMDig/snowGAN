"""Regression test for atomic weight saves (UPGRADES #8).

The bug: ``save_model`` called ``model.save_weights(path)`` directly, which
truncates the live file the moment it starts writing. An OOM-kill or Ctrl-C
mid-write left a 0-variable skeleton and destroyed the last-good checkpoint
(observed: generator.weights.h5 reduced to 10 KB). The fix writes to a sibling
temp and ``os.replace()``s it into place, so an interrupted write leaves the
previous good file untouched.

These tests exercise ``Trainer._atomic_save_weights`` with a stub "model" — it
uses no instance state, so no (slow, GPU-JIT) Trainer construction is needed. A
stub whose ``save_weights`` writes a partial file then raises reproduces exactly
the interrupted-write scenario; under the OLD direct-write code the live file
would be gone.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from snowgan.trainer import Trainer


class _StubModel:
    """Stands in for a Keras model. Records the path save_weights received and
    optionally raises after writing a partial temp (simulating an interrupt)."""

    def __init__(self, payload: bytes, raise_after_write: bool = False):
        self.payload = payload
        self.raise_after_write = raise_after_write
        self.seen_path = None

    def save_weights(self, path):
        self.seen_path = path
        with open(path, "wb") as f:
            f.write(self.payload)
        if self.raise_after_write:
            raise KeyboardInterrupt("simulated Ctrl-C mid-write")


def _atomic(model, path):
    # Method uses no instance state; a bare namespace satisfies `self`.
    return Trainer._atomic_save_weights(SimpleNamespace(), model, path)


def test_interrupted_write_preserves_last_good_file(tmp_path):
    final = tmp_path / "generator.weights.h5"
    final.write_bytes(b"GOOD-CHECKPOINT")  # pre-existing last-good file

    stub = _StubModel(b"PARTIAL", raise_after_write=True)
    with pytest.raises(KeyboardInterrupt):
        _atomic(stub, str(final))

    # The live file must be byte-for-byte the previous good content...
    assert final.read_bytes() == b"GOOD-CHECKPOINT"
    # ...and the partial temp must not linger.
    assert list(tmp_path.glob("._tmp_*")) == []


def test_successful_write_replaces_atomically(tmp_path):
    final = tmp_path / "generator.weights.h5"
    final.write_bytes(b"OLD")

    stub = _StubModel(b"NEW-WEIGHTS")
    _atomic(stub, str(final))

    assert final.read_bytes() == b"NEW-WEIGHTS"
    assert list(tmp_path.glob("._tmp_*")) == []


def test_temp_path_keeps_weights_h5_suffix(tmp_path):
    # Keras infers format from the extension; the temp must end in .weights.h5
    # or save_weights would reject it. Assert on the path the stub actually saw.
    final = tmp_path / "discriminator.weights.h5"
    stub = _StubModel(b"W")
    _atomic(stub, str(final))

    assert stub.seen_path.endswith(".weights.h5")
    assert os.path.basename(stub.seen_path).startswith("._tmp_")
