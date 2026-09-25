"""Integration test: one real train_step at 64x64 emits a usable run record.

Unit tests pin the pieces; this pins the wiring. It runs the actual
``Trainer.train_step`` — real tapes, real gradient penalty, real optimizer
updates — on a tiny model, and asserts the resulting ``metrics.jsonl`` row
answers the questions the scoreboard turns on:

  - is the critic Lipschitz?            -> grad_norm_interp / _real / _fake
  - does the penalty bind?              -> gp_weighted vs wasserstein
  - what were the live dynamics?        -> disc_steps / gen_steps / lrs

``DataManager`` is stubbed so the test needs no HF download (CLAUDE.md §3:
sub-second focused tests, no real downloads, 64x64 not 1024x1024).
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from snowgan.config import build, config_template
from snowgan.metrics import read_last_launch


RES = 64  # 16 * 2^(1+1) -> exactly one filter block


class _StubDataManager:
    """Minimal stand-in: fixed depth, deterministic synthetic batches."""

    def __init__(self, config):
        self.config = config
        self.pair_depth = 1
        self.seen_profiles = set()

    def next_batch(self, batch_size):
        return np.random.uniform(-1, 1, size=(batch_size, 1, RES, RES, 3)).astype("float32")

    def batch(self, batch_size, datatype=None, config=None):
        return self.next_batch(batch_size)

    def reset_seen_profiles(self):
        self.seen_profiles.clear()

    def derive_splits(self, *args, **kwargs):
        self.config.trained_pool = []
        self.config.validation_pool = []
        self.config.test_pool = []


def _make_config(tmp_path, architecture, **overrides):
    import copy

    data = copy.deepcopy(config_template)
    data.update({
        "architecture": architecture,
        "save_dir": str(tmp_path) + "/",
        "resolution": [RES, RES],
        "filter_counts": [8] if architecture == "generator" else [8],
        "latent_dim": 16,
        "batch_size": 2,
        "depth": 1,
        "modality": "magnified_profile",
        "fade": False,
        "grad_probe_interval": 1,
        "ema_decay": 0.0,
        "multiscale_disc": False,
        "adaptive_steps": False,
        "augment": False,
        "fid_interval": 0,
    })
    data.update(overrides)
    path = tmp_path / f"{architecture}_config.json"
    path.write_text(json.dumps(data))
    return build(str(path))


def _make_trainer(tmp_path, monkeypatch, **disc_overrides):
    # importlib, not `from snowgan import trainer`: the package __init__
    # re-exports the Trainer class under that name, shadowing the submodule.
    import importlib

    trainer_module = importlib.import_module("snowgan.trainer")
    from snowgan.models.generator import Generator
    from snowgan.models.discriminator import Discriminator

    monkeypatch.setattr(trainer_module, "DataManager", _StubDataManager)

    gen_cfg = _make_config(tmp_path, "generator", training_steps=1)
    disc_cfg = _make_config(tmp_path, "discriminator", training_steps=2, **disc_overrides)

    gen = Generator(gen_cfg)
    gen.model.build((None, gen_cfg.latent_dim))
    disc = Discriminator(disc_cfg)
    disc.model.build((None, 1, RES, RES, 3))

    return trainer_module.Trainer(gen, disc)


def _run_one_step(trainer):
    images = tf.random.normal([2, 1, RES, RES, 3])
    trainer.train_step(images)
    return read_last_launch(str(trainer.gen.config.save_dir) + "metrics.jsonl")


def test_train_step_emits_separated_loss_components(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch, lambda_gp=10.0, spectral_norm=False)
    rows = _run_one_step(trainer)

    steps = [r for r in rows if r.get("global_step") is not None and "event" not in r]
    assert len(steps) == 1
    record = steps[0]

    # disc_loss is the conflation; these are the pieces it hides.
    for field in ("wasserstein", "gp_weighted", "grad_norm_interp", "d_real", "d_fake"):
        assert field in record, f"missing instrument field {field}"
        assert np.isfinite(record[field]), f"{field} is not finite"

    # The identity that makes the split meaningful.
    assert record["disc_loss"] == pytest.approx(
        record["wasserstein"] + record["gp_weighted"], rel=1e-3, abs=1e-3)
    assert record["wasserstein"] == pytest.approx(
        record["d_fake"] - record["d_real"], rel=1e-3, abs=1e-3)


def test_lipschitz_probe_fires_and_is_independent_of_lambda_gp(tmp_path, monkeypatch):
    """The probe must survive lambda_gp == 0 — the spectral-norm-only arm is
    exactly where "is the critic Lipschitz?" is the question, and a GP-derived
    instrument goes silent there because the GP call is skipped entirely."""
    trainer = _make_trainer(tmp_path, monkeypatch, lambda_gp=0.0, spectral_norm=True)
    rows = _run_one_step(trainer)
    record = [r for r in rows if "event" not in r][-1]

    assert record["gp_weighted"] == pytest.approx(0.0)
    assert np.isnan(record["grad_norm_interp"]), "no GP -> no interpolate norm"
    # ...but the standalone probe still reports.
    assert np.isfinite(record["grad_norm_real"])
    assert np.isfinite(record["grad_norm_fake"])
    assert record["grad_norm_real"] > 0


def test_run_record_carries_live_dynamics(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch, lambda_gp=10.0, spectral_norm=False)
    record = [r for r in _run_one_step(trainer) if "event" not in r][-1]

    assert record["disc_steps"] == 2
    assert record["gen_steps"] == 1
    assert record["lambda_gp"] == 10.0
    assert record["gen_lr"] > 0 and record["disc_lr"] > 0
    # Gates are expressed in critic updates so they stay comparable across a
    # disc_steps sweep; train steps are not.
    assert record["critic_updates"] == record["global_step"] * 2


def test_probe_rng_does_not_perturb_the_training_stream(tmp_path, monkeypatch):
    """Instrumentation must be observation-only. If the probe drew from the
    global TF RNG it would shift every subsequent noise draw, making a run
    with instrumentation non-comparable to one without at the same seed."""
    from snowgan import trainer as trainer_module

    def weights_after(probe_interval):
        tf.keras.utils.set_random_seed(1234)
        sub = tmp_path / f"probe{probe_interval}"
        sub.mkdir()
        trainer = _make_trainer(sub, monkeypatch, lambda_gp=10.0, spectral_norm=False)
        trainer.grad_probe_interval = probe_interval
        tf.keras.utils.set_random_seed(99)
        trainer.train_step(tf.ones([2, 1, RES, RES, 3]))
        return [w.numpy().copy() for w in trainer.gen.model.trainable_variables]

    with_probe = weights_after(1)
    without_probe = weights_after(0)

    for a, b in zip(with_probe, without_probe):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_lambda_gp_ten_survives_spectral_norm(tmp_path, monkeypatch):
    """The silent clamp made this configuration inexpressible, which is why
    the lambda_gp hypothesis could never be tested."""
    trainer = _make_trainer(tmp_path, monkeypatch, lambda_gp=10.0, spectral_norm=True)
    assert trainer.disc.config.lambda_gp == 10.0

    record = [r for r in _run_one_step(trainer) if "event" not in r][-1]
    assert record["lambda_gp"] == 10.0
