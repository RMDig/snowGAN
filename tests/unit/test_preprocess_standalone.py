"""The standalone preprocess is the phone's reference implementation.

snowGradient's AvApp golden-image parity test calls
`snowgan.data.preprocess.preprocess_image` directly, without a DataManager (whose
constructor loads the HF dataset). These tests pin three things:

1. RGB output is bit-identical to the pre-extraction DataManager method, so
   moving it changed nothing for training.
2. Grayscale and RGBA frames become 3-channel BEFORE resize. Previously a rank-2
   frame raised inside tf.image.resize (the expand_dims after it was dead code)
   and RGBA passed through as 4 channels.
3. The module's own code depends only on NumPy and TensorFlow (importing it
   still runs the package __init__; the module docstring says so).
4. It works inside tf.data / tf.function, where a decoded PNG's channel count
   is unknown until run time.
"""

import ast
import inspect
import types

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from snowgan.data import preprocess as pp
from snowgan.data.dataset import DataManager


def _frame(h=40, w=56, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3)).astype(np.uint8)
    img[:, : w // 4] = (40, 90, 200)  # a band of board blue, so masking does work
    return img


def _old_method(image, resolution, mask_board):
    """DataManager.preprocess_image as it was before extraction (dataset.py@59dce14)."""
    image = tf.convert_to_tensor(np.array(image))
    image = tf.image.resize(image, resolution)
    if mask_board:
        image = pp.mask_blue_board(image)
    return (tf.cast(image, tf.float32) / 127.5) - 1.0


@pytest.mark.parametrize("mask_board", [False, True])
def test_rgb_output_is_bit_identical_to_the_old_method(mask_board):
    img = _frame()
    new = pp.preprocess_image(img, [16, 24], mask_board=mask_board).numpy()
    old = _old_method(img, [16, 24], mask_board).numpy()
    assert new.shape == (16, 24, 3) and new.dtype == np.float32
    assert np.array_equal(new, old)


def test_datamanager_method_delegates_with_its_config():
    dm = object.__new__(DataManager)
    dm.config = types.SimpleNamespace(resolution=[16, 24], mask_board=True)
    img = _frame()
    assert np.array_equal(dm.preprocess_image(img).numpy(),
                          pp.preprocess_image(img, [16, 24], mask_board=True).numpy())


@pytest.mark.parametrize("gray", [np.full((40, 56), 200, np.uint8),
                                  np.full((40, 56, 1), 200, np.uint8)])
def test_grayscale_becomes_rgb(gray):
    out = pp.preprocess_image(gray, [16, 16]).numpy()
    expected = pp.preprocess_image(np.full((40, 56, 3), 200, np.uint8), [16, 16]).numpy()
    assert out.shape == (16, 16, 3)
    assert np.array_equal(out, expected)


def test_rgba_drops_alpha():
    rgb = _frame()
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 7, np.uint8)], axis=-1)
    assert np.array_equal(pp.preprocess_image(rgba, [16, 16]).numpy(),
                          pp.preprocess_image(rgb, [16, 16]).numpy())


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_pil_modes_match_the_rgb_conversion(mode):
    rgb = Image.fromarray(_frame())
    converted = rgb.convert(mode)
    expected = pp.preprocess_image(np.array(converted.convert("RGB")), [16, 16]).numpy()
    assert np.array_equal(pp.preprocess_image(converted, [16, 16]).numpy(), expected)


@pytest.mark.parametrize("bad", [np.zeros((8, 8, 2)), np.zeros((2, 8, 8, 3)),
                                 np.zeros((8, 8, 3), np.uint16)])
def test_unsupported_shapes_raise(bad):
    with pytest.raises(ValueError):
        pp.preprocess_image(bad, [4, 4])


def test_16_bit_pil_modes_are_refused_not_clipped():
    with pytest.raises(ValueError, match="8-bit"):
        pp.preprocess_image(Image.fromarray(np.full((8, 8), 40000, np.uint16)), [4, 4])


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_graph_mode_with_unknown_channels(channels):
    rgb = _frame(16, 16)
    arr = {1: rgb[..., :1], 3: rgb, 4: np.concatenate([rgb, rgb[..., :1]], -1)}[channels]
    png = tf.io.encode_png(arr)
    ds = tf.data.Dataset.from_tensors(png).map(
        lambda b: pp.preprocess_image(tf.io.decode_png(b), [8, 8]))  # channels=0: unknown
    got = next(iter(ds)).numpy()
    assert got.shape == (8, 8, 3)
    assert np.array_equal(got, pp.preprocess_image(arr, [8, 8]).numpy())


def test_contract_version_is_exposed():
    assert pp.PREPROCESS_VERSION == 1


def test_module_own_imports_are_only_numpy_and_tensorflow():
    tree = ast.parse(inspect.getsource(pp))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"numpy", "tensorflow"}, imported
