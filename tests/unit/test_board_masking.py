"""Focused test: blue-board masking isolates snow and fills the board with grey.

Core photos are a snow sample on a blue ruler board; the board/ruler/text dominate
every frame and are a confound for the downstream avalanche-risk task (a GAN
collapses onto them). mask_blue_board zeroes the chromatic-blue board to neutral
grey while keeping achromatic snow. The board hue is lighting-stable (measured peak
168-172 on PIL's 0-255 scale across the whole split); these pin that a blue region
is removed, achromatic pixels survive, and the fill is neutral grey (not a tanh
rail — see the grey-vs-black uncertainty in docs/UPGRADES.md #49).
"""

import copy
import json

import numpy as np
import tensorflow as tf

from snowgan.config import build, config_template
from snowgan.data.dataset import mask_blue_board, _BOARD_FILL_255


def _rgb(colour, h=32, w=32):
    return tf.constant(np.tile(np.array(colour, np.float32), (h, w, 1)))


def test_blue_board_becomes_neutral_grey():
    board = _rgb([30, 40, 200])  # saturated blue, like the measurement board
    out = mask_blue_board(board).numpy()
    assert np.allclose(out, _BOARD_FILL_255), (
        f"blue board not masked to grey (got mean {out.mean():.1f}, want {_BOARD_FILL_255})"
    )
    # grey fill sits at tanh-linear centre after the /127.5 - 1 rescale, NOT a rail.
    assert abs((_BOARD_FILL_255 / 127.5 - 1.0)) < 1e-3


def test_achromatic_snow_is_kept():
    for grey in ([230, 235, 240], [90, 92, 88], [255, 255, 255]):  # white/grey snow
        out = mask_blue_board(_rgb(grey)).numpy()
        assert np.allclose(out, np.array(grey, np.float32)), (
            f"achromatic snow {grey} was altered -> {out.reshape(-1,3)[0]}"
        )


def test_mixed_image_masks_only_the_board():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, :8] = [30, 40, 200]     # left half: blue board
    img[:, 8:] = [235, 235, 235]   # right half: snow
    out = mask_blue_board(tf.constant(img)).numpy()
    assert np.allclose(out[:, :8], _BOARD_FILL_255)          # board -> grey
    assert np.allclose(out[:, 8:], 235.0)                    # snow untouched


def test_non_rgb_is_a_noop():
    """Grayscale/other modalities lack the colour the hue test needs."""
    gray = tf.constant(np.full((16, 16, 1), 120.0, np.float32))
    assert np.allclose(mask_blue_board(gray).numpy(), 120.0)


def test_config_gates_masking(tmp_path):
    data = copy.deepcopy(config_template)
    data["mask_board"] = True
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    cfg = build(str(p))
    assert cfg.mask_board is True
    assert cfg.dump()["mask_board"] is True
    # Default (legacy config without the key) stays off.
    data2 = copy.deepcopy(config_template); data2.pop("mask_board", None)
    p2 = tmp_path / "legacy.json"; p2.write_text(json.dumps(data2))
    assert build(str(p2)).mask_board is False
