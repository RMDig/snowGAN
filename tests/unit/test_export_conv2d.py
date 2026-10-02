"""The rank-4 export must compute exactly what the Conv3D model computes.

snowGradient ships these models to a phone, where TFLite's CONV_3D is a slow
CPU-only kernel. The export rebuilds them with Conv2D layers; these tests are the
equivalence gate (<= 1e-5 on fixed inputs) across every generator variant and
both critic variants, plus the cases the export must refuse rather than
approximate.
"""

import itertools
import types

import keras
import numpy as np
import pytest
import tensorflow as tf

from snowgan.export import export_tflite, to_conv2d
from snowgan.models.discriminator import Discriminator
from snowgan.models.generator import Generator

TOL = 1e-5


def _cfg(**overrides):
    cfg = types.SimpleNamespace(
        latent_dim=16, filter_counts=[8], kernel_size=[3, 3], kernel_stride=[2, 2], padding="same",
        batch_norm=False, gen_norm="none", negative_slope=0.25, channels=3, final_activation="tanh",
        depth=1, learning_rate=1e-4, beta_1=0.5, beta_2=0.9, resolution=[64, 64], spectral_norm=False,
        gen_upsampler="resize", gen_convs_per_resolution=2, fade=False, fade_step=0, fade_steps=0)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture(autouse=True)
def _seed():
    keras.utils.set_random_seed(0)


def _images(n=3, size=64, seed=1):
    return np.random.default_rng(seed).uniform(-1, 1, (n, 1, size, size, 3)).astype(np.float32)


@pytest.mark.parametrize("spectral_norm", [False, True])
def test_discriminator_export_matches(spectral_norm):
    d = Discriminator(_cfg(filter_counts=[8, 16], spectral_norm=spectral_norm))
    x = _images()
    if spectral_norm:
        for _ in range(3):  # power iteration rewrites the stored kernel
            d.model(x, training=True)
    m2 = to_conv2d(d)
    assert m2.input.shape[1:] == (64, 64, 3)
    diff = np.max(np.abs(d.model(x, training=False).numpy() - m2(x[:, 0], training=False).numpy()))
    assert diff <= TOL, diff


def test_discriminator_features_tap_survives_and_matches():
    d = Discriminator(_cfg(filter_counts=[8, 16]))
    m2 = to_conv2d(d)
    x = _images()
    f3 = keras.Model(d.model.input, d.model.get_layer("features").output)(x).numpy()
    f2 = keras.Model(m2.input, m2.get_layer("features").output)(x[:, 0]).numpy()
    assert f3.shape == f2.shape
    assert np.max(np.abs(f3 - f2)) <= TOL


@pytest.mark.parametrize("upsampler,convs,norm",
                         list(itertools.product(["transpose", "resize"], [1, 2], ["none", "pixel", "batch"])))
def test_generator_export_matches(upsampler, convs, norm):
    g = Generator(_cfg(gen_upsampler=upsampler, gen_convs_per_resolution=convs, gen_norm=norm,
                       batch_norm=(norm == "batch")))
    z = np.random.default_rng(2).standard_normal((3, 16)).astype(np.float32)
    if norm == "batch":
        for _ in range(3):  # move BN moving statistics off their init values
            g.model(z, training=True)
    m2 = to_conv2d(g)
    y3 = g.model(z, training=False).numpy()[:, 0]
    y2 = m2(z, training=False).numpy()
    assert y2.shape == y3.shape == (3, 64, 64, 3)
    assert np.max(np.abs(y3 - y2)) <= TOL


def test_export_contains_no_rank5_ops():
    m2 = to_conv2d(Generator(_cfg(gen_upsampler="transpose", gen_convs_per_resolution=1)))
    kinds = {type(l).__name__ for l in m2.layers}
    assert not any("3D" in k for k in kinds), kinds


# --- refusals ---------------------------------------------------------------

def test_refuses_depth_two_discriminator():
    # The Flatten->Dense head mixes depth slices: no single rank-4 equivalent.
    with pytest.raises(ValueError, match="depth"):
        to_conv2d(Discriminator(_cfg(filter_counts=[8], depth=2)))


def test_refuses_depth_two_generator():
    with pytest.raises(ValueError, match="depth"):
        to_conv2d(Generator(_cfg(depth=2)))


def test_refuses_mid_fade_generator():
    g = Generator(_cfg(filter_counts=[8, 4], fade=True, fade_step=10, fade_steps=100))
    with pytest.raises(ValueError, match="mid-fade"):
        to_conv2d(g)


def test_finished_fade_is_exportable():
    g = Generator(_cfg(filter_counts=[8, 4], fade=True, fade_step=100, fade_steps=100))
    to_conv2d(g)


def test_refuses_unknown_layer_instead_of_dropping_it():
    inputs = keras.Input((1, 16, 16, 3))
    x = keras.layers.Conv3D(4, (1, 3, 3), padding="same")(inputs)
    x = keras.layers.Dropout(0.5)(x)
    with pytest.raises(TypeError, match="Dropout"):
        to_conv2d(keras.Model(inputs, x))


def test_refuses_depth_mixing_kernel():
    inputs = keras.Input((1, 16, 16, 3))
    x = keras.layers.Conv3D(4, (1, 3, 3), strides=(2, 1, 1), padding="same")(inputs)
    with pytest.raises(ValueError, match="depth"):
        to_conv2d(keras.Model(inputs, x))


# --- TFLite -----------------------------------------------------------------

def _run_tflite(path, x):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    assert tuple(inp["shape"]) == x.shape
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    return interp.get_tensor(out["index"])


def test_tflite_discriminator_round_trip(tmp_path):
    d = Discriminator(_cfg(filter_counts=[8, 16], spectral_norm=True, resolution=[32, 32]))
    x = _images(n=1, size=32)
    d.model(x, training=True)
    path = export_tflite(to_conv2d(d), str(tmp_path / "d.tflite"))
    got = _run_tflite(path, x[:, 0])
    want = d.model(x, training=False).numpy()
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - want)) <= TOL  # measured <= 3e-8 on 2026-10-02


def test_tflite_generator_round_trip(tmp_path):
    g = Generator(_cfg(gen_upsampler="transpose", gen_convs_per_resolution=1))
    z = np.random.default_rng(3).standard_normal((1, 16)).astype(np.float32)
    path = export_tflite(to_conv2d(g), str(tmp_path / "g.tflite"))
    got = _run_tflite(path, z)
    assert got.shape == (1, 64, 64, 3)
    assert np.max(np.abs(got - g.model(z, training=False).numpy()[:, 0])) <= TOL  # measured <= 3e-8 on 2026-10-02


# --- CLI --------------------------------------------------------------------

def test_cli_exports_from_sidecar_and_weights(tmp_path):
    import json

    from snowgan.config import build
    from snowgan.export import main

    sidecar = tmp_path / "discriminator_config.json"
    cfg = build(str(sidecar))  # template, then shrink
    cfg.filter_counts, cfg.resolution, cfg.spectral_norm = [8, 16], [32, 32], True
    cfg.save_config(str(sidecar))
    before = sidecar.read_text()
    d = Discriminator(build(str(sidecar)))
    weights = tmp_path / "discriminator.weights.h5"
    d.model.save_weights(str(weights))

    out = tmp_path / "d.tflite"
    main(["--kind", "discriminator", "--sidecar", str(sidecar), "--weights", str(weights), "--out", str(out)])
    x = _images(n=1, size=32)
    assert np.max(np.abs(_run_tflite(out, x[:, 0]) - d.model(x, training=False).numpy())) <= TOL
    assert sidecar.read_text() == before  # read-only: the sidecar is never rewritten
    assert json.loads(before)["filter_counts"] == [8, 16]
