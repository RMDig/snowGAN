"""Best-KID checkpointing (UPGRADES #68).

The campaign's best generator was often not its last: control_1024 drifted late,
and Arm A's anneal acted as soft early stopping. The end-of-run sweep only sees
the newest 10 snapshots plus every 10k, so the trainer scores KID on a cadence
and keeps the best. These pin the parts a 28-hour run depends on: a worse score
never overwrites the best, the best survives a restart, the eval cannot perturb
the training RNG stream, and a failing eval cannot kill the run.

Inception and the real set are stubbed: what is under test is the bookkeeping,
not the estimator (tests/unit/test_kid.py covers that).
"""

import json
import os

import numpy as np
import pytest
import tensorflow as tf

import snowgan.kid as kid_module
from test_diffaugment_generator_step import RES, _make_trainer


def _stub_features(images, batch_size=8, model=None):
    # Per-image channel means: deterministic in the images, no RNG.
    arr = np.asarray(images, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1, arr.shape[-1]).mean(axis=1)


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    t = _make_trainer(tmp_path, monkeypatch, augment=False)
    t.kid_interval, t.kid_samples = 5, 8
    t._kid_real_features = np.random.default_rng(0).normal(size=(20, 3))
    t._kid_inception = object()
    monkeypatch.setattr(kid_module, "inception_features", _stub_features)
    return t


def _script_scores(monkeypatch, values):
    values = iter(values)
    monkeypatch.setattr(kid_module, "kid_score",
                        lambda *a, **k: {"kid_mean": next(values), "kid_std": 0.0, "kid_se": 0.01,
                                         "subsets": 10, "subset_size": 8})


def _record(trainer):
    with open(f"{trainer.save_dir}/best_kid/kid.json", encoding="utf-8") as handle:
        return json.load(handle)


def test_new_minimum_saves_and_a_worse_score_does_not_overwrite(trainer, monkeypatch):
    _script_scores(monkeypatch, [0.30, 0.20, 0.25])
    for step in (5, 10, 15):
        trainer.global_step = step
        trainer._maybe_evaluate_kid()

    record = _record(trainer)
    assert record["kid_mean"] == 0.20 and record["global_step"] == 10
    assert trainer.best_kid == 0.20
    assert os.path.exists(os.path.join(trainer.save_dir, "best_kid", "generator.weights.h5"))
    assert os.path.exists(os.path.join(trainer.save_dir, "best_kid", "generator_config.json"))


def test_warm_up_scores_are_logged_but_cannot_claim_best(trainer, monkeypatch):
    """Arm B: the step-1k checkpoint out-scored every checkpoint to 14k while
    looking visibly worse. Below kid_min_step a score is recorded, not saved."""
    trainer.kid_min_step = 10
    _script_scores(monkeypatch, [0.10, 0.30, 0.20])
    for step in (5, 10, 15):
        trainer.global_step = step
        trainer._evaluate_kid()

    record = _record(trainer)
    assert record["global_step"] == 15 and record["kid_mean"] == 0.20
    events = [json.loads(line) for line in open(f"{trainer.save_dir}/metrics.jsonl", encoding="utf-8")]
    kid = [e for e in events if e.get("event") == "kid"]
    assert [(e["global_step"], e["eligible"], e["best"]) for e in kid] == [
        (5, False, False), (10, True, True), (15, True, True)]


def test_cadence_skips_off_steps_and_step_zero(trainer, monkeypatch):
    calls = []
    monkeypatch.setattr(trainer, "_evaluate_kid", lambda: calls.append(trainer.global_step))
    for step in range(0, 16):
        trainer.global_step = step
        trainer._maybe_evaluate_kid()
    assert calls == [5, 10, 15]


def test_restart_keeps_the_bar(trainer, monkeypatch, tmp_path):
    """The RSS wrapper restarts long runs; a relaunch must not overwrite a
    better best_kid/ with the first, worse checkpoint it scores."""
    _script_scores(monkeypatch, [0.20])
    trainer.global_step = 5
    trainer._evaluate_kid()

    assert trainer._load_best_kid() == 0.20
    # Simulate the relaunched process: fresh state, same save_dir.
    trainer.best_kid = trainer._load_best_kid()
    _script_scores(monkeypatch, [0.40])
    trainer.global_step = 10
    trainer._evaluate_kid()
    assert _record(trainer)["global_step"] == 5


def test_no_record_means_no_bar(trainer):
    assert trainer._load_best_kid() == float("inf")


def test_eval_does_not_perturb_the_training_rng(trainer, monkeypatch):
    _script_scores(monkeypatch, [0.3])
    tf.random.set_seed(7)
    expected = tf.random.normal([4]).numpy()

    tf.random.set_seed(7)
    trainer.global_step = 5
    trainer._evaluate_kid()
    assert np.array_equal(tf.random.normal([4]).numpy(), expected)


def test_same_latents_every_evaluation(trainer):
    """Fixed z: successive scores differ only by the weights."""
    first = trainer._kid_fake_features()
    second = trainer._kid_fake_features()
    assert first.shape == (8, 3)
    assert np.array_equal(first, second)


def test_a_failing_eval_does_not_raise(trainer, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("OOM in the Inception pass")
    monkeypatch.setattr(kid_module, "kid_score", boom)
    trainer.global_step = 5
    assert trainer._evaluate_kid() is None
    assert trainer.kid_interval == 5  # transient: keep trying next time


def test_missing_image_root_disables_instead_of_downloading(trainer):
    trainer._kid_real_features = None
    trainer.gen.config.image_root = None
    trainer.global_step = 5
    assert trainer._evaluate_kid() is None
    assert trainer.kid_interval == 0
    events = [json.loads(line) for line in open(f"{trainer.save_dir}/metrics.jsonl", encoding="utf-8")]
    assert any(e.get("event") == "kid_disabled" for e in events)


def test_cli_flags_reach_the_config_and_survive_a_resume(tmp_path):
    """Set from the CLI, persisted, and left alone when a relaunch omits them."""
    import copy
    import sys
    from unittest import mock
    from snowgan.config import build, config_template, configure_generic
    from snowgan.utils import parse_args

    def args(argv):
        with mock.patch.object(sys, "argv", ["snowgan", "--mode", "train"] + argv):
            return parse_args()

    path = tmp_path / "config.json"
    path.write_text(json.dumps(copy.deepcopy(config_template)))
    cfg = build(str(path))
    assert (cfg.kid_interval, cfg.kid_samples) == (0, 200)

    configure_generic(cfg, args(["--kid_interval", "1000", "--kid_samples", "120"]))
    path.write_text(json.dumps(cfg.dump()))
    cfg = build(str(path))
    configure_generic(cfg, args([]))
    assert (cfg.kid_interval, cfg.kid_samples) == (1000, 120)
