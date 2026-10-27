"""Capacity campaign Tier B switches (capacity_plan §4): K1 and G2.

Both are off by default and must then build the historical models exactly.
G2's residual blocks are zero-initialised, so turning them on must not change
the function the generator computes at step 0.
"""

import numpy as np
import pytest
import tensorflow as tf

from snowgan.losses import compute_gradient_penalty
from test_diffaugment_generator_step import RES, _make_trainer


def _layer_names(model):
    return [layer.name for layer in model.layers]


def test_defaults_build_the_historical_models(tmp_path, monkeypatch):
    t = _make_trainer(tmp_path, monkeypatch, augment=False)
    assert not any(n.startswith("head_") for n in _layer_names(t.disc.model))
    assert not any(isinstance(layer, tf.keras.layers.Add) for layer in t.gen.model.layers)
    assert t.disc.model.get_layer("features").output.shape[-1] == (RES // 2) ** 2 * 8


def test_k1_head_is_nonlinear_and_penalty_is_finite(tmp_path, monkeypatch):
    t = _make_trainer(tmp_path, monkeypatch, augment=False, extra={"disc_head": "mlp"})
    names = _layer_names(t.disc.model)
    assert "head_squeeze" in names and "head_hidden" in names
    # features is now the squeezed grid: 64 channels instead of the last conv's 8.
    assert t.disc.model.get_layer("features").output.shape[-1] == (RES // 2) ** 2 * 64
    real = tf.random.uniform([2, 1, RES, RES, 3], -1, 1)
    fake = tf.random.uniform([2, 1, RES, RES, 3], -1, 1)
    assert t.disc.model(real).shape == (2, 1)
    gp, norm = compute_gradient_penalty(t.disc, real, fake)
    assert np.isfinite(float(gp)) and np.isfinite(float(norm))


def test_k1_rejects_an_unknown_head(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="disc_head"):
        _make_trainer(tmp_path, monkeypatch, augment=False, extra={"disc_head": "attention"})


def test_g2_starts_as_the_same_function(tmp_path, monkeypatch):
    """Zero-init second conv: perturbing the first conv of every residual block
    cannot change the output until training moves the second conv."""
    t = _make_trainer(tmp_path, monkeypatch, augment=False, extra={"gen_residual_blocks": 1},
                      gen_filters=(16, 8))
    adds = [layer for layer in t.gen.model.layers if isinstance(layer, tf.keras.layers.Add)]
    assert len(adds) == 2  # one block per upsampling stage
    z = tf.random.normal([2, 16], seed=1)
    before = t.gen.model(z, training=False).numpy()

    convs = [layer for layer in t.gen.model.layers if isinstance(layer, tf.keras.layers.Conv3D)
             and not isinstance(layer, tf.keras.layers.Conv3DTranspose)]
    zero_init = [c for c in convs if not np.any(c.kernel.numpy())]
    first = [c for c in convs if c not in zero_init]
    assert len(zero_init) == 2 and len(first) == 2
    for c in first:
        c.kernel.assign(tf.random.normal(c.kernel.shape))
    assert np.allclose(t.gen.model(z, training=False).numpy(), before)

    zero_init[0].kernel.assign(tf.random.normal(zero_init[0].kernel.shape) * 0.1)
    assert not np.allclose(t.gen.model(z, training=False).numpy(), before)


def test_both_on_run_a_training_step(tmp_path, monkeypatch):
    t = _make_trainer(tmp_path, monkeypatch, augment=False,
                      extra={"disc_head": "mlp", "gen_residual_blocks": 1})  # 1 stage -> 64 px
    zero_before = [layer.kernel.numpy().copy() for layer in t.gen.model.layers
                   if isinstance(layer, tf.keras.layers.Conv3D) and not np.any(layer.kernel.numpy())]
    t.train_step(tf.random.uniform([2, 1, RES, RES, 3], -1, 1))
    assert np.isfinite(t.loss["disc"][-1]) and np.isfinite(t.loss["gen"][-1])
    # The zero-initialised convs receive gradient and start to move.
    zero_after = [layer.kernel.numpy() for layer in t.gen.model.layers
                  if isinstance(layer, tf.keras.layers.Conv3D)
                  and layer.kernel.shape == zero_before[0].shape]
    assert any(np.any(k) for k in zero_after)


def test_cli_round_trip(tmp_path):
    import copy
    import json
    import sys
    from unittest import mock
    from snowgan.config import build, config_template, configure_generic
    from snowgan.utils import parse_args

    path = tmp_path / "c.json"
    path.write_text(json.dumps(copy.deepcopy(config_template)))
    cfg = build(str(path))
    assert (cfg.disc_head, cfg.gen_residual_blocks) == ("linear", 0)
    with mock.patch.object(sys, "argv", ["snowgan", "--mode", "train", "--disc_head", "mlp",
                                         "--gen_residual_blocks", "1"]):
        configure_generic(cfg, parse_args())
    path.write_text(json.dumps(cfg.dump()))
    cfg = build(str(path))
    assert (cfg.disc_head, cfg.gen_residual_blocks) == ("mlp", 1)
