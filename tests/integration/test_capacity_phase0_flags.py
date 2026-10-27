"""Capacity campaign Phase 0 switches (docs/plans/capacity_plan.md §3–§6).

Each switch is OFF by default and must leave the historical recipe bit-identical
when off: the E0 rerun and every arm are compared against runs trained before
these switches existed.
"""

import copy
import json
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from test_diffaugment_generator_step import RES, _make_trainer, _record_critic_inputs


class _CountingData:
    """next_batch returns a distinct batch per call, so reuse is detectable."""

    def __init__(self):
        self.calls = 0

    def next_batch(self, batch_size):
        self.calls += 1
        rng = np.random.default_rng(self.calls)
        return rng.uniform(-1, 1, (batch_size, 1, RES, RES, 3)).astype("float32")


class _ShortTailData(_CountingData):
    """First fetch returns a short tail batch, as at the end of a manifest pass."""

    def next_batch(self, batch_size):
        batch = super().next_batch(batch_size)
        return batch[:batch_size - 1] if self.calls == 1 else batch


def test_a_short_tail_batch_is_not_used(tmp_path, monkeypatch):
    """cap_o2 crashed at step 250: a 3-image tail batch met 4 fakes in the GP."""
    t = _make_trainer(tmp_path, monkeypatch, augment=False)
    t._disc_steps = 2
    t.dataset = _ShortTailData()
    t.fresh_critic_batches = True
    t.train_step(tf.random.normal([2, 1, RES, RES, 3]))  # must not raise
    assert t.dataset.calls == 2, "the short batch should be discarded and a full one fetched"


def _critic_real_inputs(trainer, monkeypatch, images):
    seen = _record_critic_inputs(trainer, monkeypatch)
    trainer.train_step(images)
    # disc_steps=2 in this fixture: per critic step the critic sees real, fake,
    # then the GP interpolate. The real input is the first of each triple.
    return seen


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    t = _make_trainer(tmp_path, monkeypatch, augment=False)
    t._disc_steps = 2
    t.dataset = _CountingData()
    return t


def test_off_by_default_reuses_one_batch_and_fetches_nothing(trainer, monkeypatch):
    assert trainer.fresh_critic_batches is False
    images = tf.random.normal([2, 1, RES, RES, 3])
    seen = _critic_real_inputs(trainer, monkeypatch, images)
    reals = [x for x in seen if x.shape[0] == 2 and np.allclose(x, images.numpy())]
    assert len(reals) >= 2, "both critic steps should see the passed-in batch"
    assert trainer.dataset.calls == 0


def test_on_draws_a_fresh_batch_for_the_second_critic_step(trainer, monkeypatch):
    trainer.fresh_critic_batches = True
    images = tf.random.normal([2, 1, RES, RES, 3])
    seen = _critic_real_inputs(trainer, monkeypatch, images)
    assert trainer.dataset.calls == 1
    expected = np.random.default_rng(1).uniform(-1, 1, (2, 1, RES, RES, 3)).astype("float32")
    assert any(np.allclose(x, expected) for x in seen), "second critic step did not use the fresh batch"
    assert any(np.allclose(x, images.numpy()) for x in seen), "first critic step must still use the passed batch"


def test_only_the_on_state_draws_extra_noise(tmp_path, monkeypatch):
    """Off must not draw from the global RNG (the historical stream); on draws
    one extra noise tensor per extra critic step, so the streams diverge."""
    def run(flag):
        t = _make_trainer(tmp_path / str(flag), monkeypatch, augment=False)
        t._disc_steps = 2
        t.dataset = _CountingData()
        t.fresh_critic_batches = flag
        tf.random.set_seed(3)
        t.train_step(tf.zeros([2, 1, RES, RES, 3]))
        return tf.random.normal([4]).numpy()

    (tmp_path / "False").mkdir()
    (tmp_path / "True").mkdir()
    assert not np.array_equal(run(False), run(True)), "flag on should consume extra RNG"


def test_exclude_sites_skips_only_listed_sites():
    from snowgan.data.dataset import DataManager
    dm = DataManager.__new__(DataManager)
    dm.config = SimpleNamespace(exclude_sites=[0])
    assert dm._is_excluded({"site": 0, "column": 1, "core": 1})
    assert not dm._is_excluded({"site": 3, "column": 1, "core": 1})
    dm.config = SimpleNamespace(exclude_sites=[])
    assert not dm._is_excluded({"site": 0})
    dm.config = SimpleNamespace()  # legacy config without the field
    assert not dm._is_excluded({"site": 0})


def test_antialias_removes_checkerboard_aliasing():
    """A 2-px checkerboard aliases through a plain 1024->299 resize; antialiasing low-passes it."""
    from snowgan.kid import inception_features
    yy, xx = np.mgrid[0:1024, 0:1024]
    board = np.where((yy + xx) % 2 == 0, 1.0, -1.0).astype("float32")
    image = np.repeat(board[None, None, :, :, None], 3, axis=-1)  # (1, 1, H, W, 3)
    spread = lambda chunk, training=False: tf.math.reduce_std(chunk, axis=[1, 2])  # noqa: E731
    plain = inception_features(image, model=spread)
    smooth = inception_features(image, model=spread, antialias=True)
    assert float(plain.mean()) > 10 * float(smooth.mean())


def test_ema_reload_failure_is_fatal(tmp_path, monkeypatch):
    t = _make_trainer(tmp_path, monkeypatch, augment=False)
    ema_path = tmp_path / "generator_ema.weights.h5"
    ema_path.write_bytes(b"not an h5 file")
    t.gen.config.checkpoint = str(tmp_path / "generator.weights.h5")
    t.ema_decay = 0.999
    with pytest.raises(RuntimeError, match="Refusing to continue"):
        t._init_ema()


def test_flags_reach_the_config_and_survive_a_resume(tmp_path):
    from snowgan.config import build, config_template, configure_generic
    from snowgan.utils import parse_args

    def args(argv):
        with mock.patch.object(sys, "argv", ["snowgan", "--mode", "train"] + argv):
            return parse_args()

    path = tmp_path / "config.json"
    path.write_text(json.dumps(copy.deepcopy(config_template)))
    cfg = build(str(path))
    assert (cfg.kid_antialias, cfg.fresh_critic_batches, cfg.exclude_sites) == (False, False, [])

    configure_generic(cfg, args(["--kid_antialias", "--fresh_critic_batches", "--exclude_sites", "0"]))
    path.write_text(json.dumps(cfg.dump()))
    cfg = build(str(path))
    configure_generic(cfg, args([]))
    assert (cfg.kid_antialias, cfg.fresh_critic_batches, cfg.exclude_sites) == (True, True, [0])

    configure_generic(cfg, args(["--no-fresh_critic_batches", "--exclude_sites"]))
    assert (cfg.fresh_critic_batches, cfg.exclude_sites) == (False, [])
