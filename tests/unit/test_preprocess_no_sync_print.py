"""Regression: preprocess_image must not print per image.

The removed `print(f"Max - {tf.reduce_max(image).numpy()} ...")` forced two
device syncs per image (~0.5 ms/img on an RTX 5080) in the hottest data-pipeline
loop, defeating GPU pipelining, and flooded stdout (a calibration run logged
1,682 such lines). This pins that preprocess_image stays silent — a per-image
print here is a performance regression, not just noise — and that it still
normalizes correctly (UPGRADES #51).
"""

import types

import numpy as np

from snowgan.data.dataset import DataManager


def _shell_dm(resolution):
    """A DataManager with only what preprocess_image reads, bypassing __init__
    (which would call load_dataset)."""
    dm = object.__new__(DataManager)
    dm.config = types.SimpleNamespace(resolution=list(resolution))
    return dm


def test_preprocess_image_prints_nothing(capsys):
    dm = _shell_dm([16, 16])
    dm.preprocess_image(np.full((32, 32, 3), 255, dtype=np.float32))
    out = capsys.readouterr().out
    assert out == "", f"preprocess_image emitted per-image stdout (sync/flood regression): {out!r}"


def test_preprocess_image_normalizes_to_unit_range(capsys):
    dm = _shell_dm([16, 16])
    # 255 -> +1, 0 -> -1 after the /127.5 - 1 rescale.
    hi = dm.preprocess_image(np.full((32, 32, 3), 255, dtype=np.float32)).numpy()
    lo = dm.preprocess_image(np.zeros((32, 32, 3), dtype=np.float32)).numpy()
    assert hi.shape == (16, 16, 3)
    assert np.allclose(hi.max(), 1.0, atol=1e-4)
    assert np.allclose(lo.min(), -1.0, atol=1e-4)
    # Removing the print must not have changed behavior — still silent here too.
    assert capsys.readouterr().out == ""
