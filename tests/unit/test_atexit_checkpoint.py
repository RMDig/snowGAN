"""Regression tests for removing the atexit checkpoint (UPGRADES #8).

The bug had two halves:
  1. The top-level "live" checkpoint (the one __init__ resumes from) was written
     ONLY by ``atexit.register(self.save_model)``. atexit does not fire on
     SIGKILL/OOM, so after an OOM-kill the top-level weights went stale relative
     to the batch counter recovered from the snapshot dirs; and an interrupt
     *during* the atexit save truncated the live file.
  2. The train loop persisted only ``batch_<N>/`` snapshots, never the top-level.

The fix removes the atexit registration and writes the top-level checkpoint on
the step interval inside ``train()`` (atomically). These assertions fail under
the old code (which imported atexit and had no no-arg ``save_model()`` in the
loop) and pass under the new code. They inspect structure rather than running a
full GPU training step, matching how the rest of the suite exercises Trainer.
"""
from __future__ import annotations

import inspect

from snowgan.trainer import Trainer


def test_trainer_module_does_not_register_atexit_checkpoint():
    # The old code did `import os, atexit, ...` and `atexit.register(self.save_model)`.
    # Removing the import drops the module attribute entirely.
    module = inspect.getmodule(Trainer)
    assert not hasattr(module, "atexit"), (
        "snowgan.trainer must not use atexit for checkpointing — atexit is not "
        "signal-safe (no SIGKILL/OOM) and an interrupt mid-save corrupts the "
        "live checkpoint (UPGRADES #8)."
    )
    assert "atexit.register" not in inspect.getsource(Trainer.__init__)


def test_train_loop_persists_top_level_checkpoint_on_interval():
    src = inspect.getsource(Trainer.train)
    # The rolling top-level live checkpoint (no path arg) must be written during
    # the run, not only at exit...
    assert "self.save_model()" in src, (
        "train() must persist the top-level checkpoint on the save interval so "
        "an OOM-kill leaves a current, consistent checkpoint."
    )
    # ...alongside the numbered snapshot (must not have been dropped).
    assert "self.save_model(f\"{self.save_dir}/batch_{batch}/\")" in src


def test_save_model_routes_weights_through_atomic_helper():
    # Every weight write must go through the atomic helper; a stray direct
    # save_weights would reintroduce the truncation vector.
    src = inspect.getsource(Trainer.save_model)
    assert "self._atomic_save_weights(" in src
    assert ".save_weights(" not in src, (
        "save_model must not call save_weights directly — route through "
        "_atomic_save_weights (tmp + os.replace)."
    )
