"""Focused test: the shallow (proven) generator stack stays alive without norm.

`gen_convs_per_resolution=1` reproduces the pre-2026-06-13 generator — the only
architecture with a proven result (the magnified_profiles v0.1.0 backbone). Its
defining property is that a ~6-layer stack keeps its activations alive with NO
normalization, which is why that model trained with `batch_norm=false` and no
PixelNorm.

Commit b4e0677 doubled the stack to ~11 layers, and an un-normalized 11-layer
stack *vanishes*: output std ~8e-4, i.e. a uniform mid-grey frame (observed
2026-07-22 on a real run). Depth here is load-bearing, not a free capacity knob —
these pin that so the grey-frame configuration can't be shipped again unnoticed.
"""

import copy
import json

import numpy as np
import tensorflow as tf

from snowgan.config import build, config_template
from snowgan.models.generator import Generator


def _gen_config(tmp_path, **overrides):
    data = copy.deepcopy(config_template)
    data.update({
        "architecture": "generator",
        "latent_dim": 100,
        "depth": 1,
        "channels": 3,
        "kernel_size": [3, 3],
        "kernel_stride": [2, 2],
        # 2 blocks -> 16 * 2^(2+1) = 128; small enough for CPU.
        "filter_counts": [16, 8],
        "gen_upsampler": "transpose",
        "gen_norm": "none",
        "batch_norm": False,
    })
    data.update(overrides)
    cfg_path = tmp_path / "generator_config.json"
    cfg_path.write_text(json.dumps(data))
    return build(str(cfg_path))


def _output_std(cfg, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    gen = Generator(cfg)
    out = gen.model(tf.random.normal([4, cfg.latent_dim]), training=False).numpy()
    return float(out.std()), gen


def test_deeper_unnormalized_stack_attenuates_more(tmp_path):
    """Doubling convs/resolution without normalization costs signal, and the loss
    compounds with depth.

    This runs at a 2-block toy scale where the gap is modest (~2x). At the real
    5-block 1024² config the same mechanism is catastrophic — measured 2026-07-22:
    the un-normalized deep stack outputs std 8e-4 (a uniform mid-grey frame), while
    the shallow stack carrying the trained magnified_profiles weights outputs std
    0.506 (matching the data's ~0.50). Hence the CLI warning that
    `--gen_convs_per_resolution 2` requires `--gen_norm pixel`.
    """
    shallow, _ = _output_std(_gen_config(tmp_path, gen_convs_per_resolution=1))
    deep, _ = _output_std(_gen_config(tmp_path, gen_convs_per_resolution=2))
    assert shallow > 0.0, "shallow stack produced no signal at all"
    assert deep < shallow, (
        f"expected the un-normalized deep stack to attenuate more than the shallow "
        f"one (shallow={shallow:.4f}, deep={deep:.6f})"
    )


def test_shallow_stack_has_one_transpose_per_resolution(tmp_path):
    """Structure check: 1 Conv3DTranspose per filter block + the toRGB head, and
    no extra stride-1 Conv3D capacity layers."""
    cfg = _gen_config(tmp_path, gen_convs_per_resolution=1)
    gen = Generator(cfg)
    transposes = [l for l in gen.model.layers if isinstance(l, tf.keras.layers.Conv3DTranspose)]
    plain_convs = [l for l in gen.model.layers
                   if isinstance(l, tf.keras.layers.Conv3D)
                   and not isinstance(l, tf.keras.layers.Conv3DTranspose)
                   and l.name.startswith("conv")]
    assert len(transposes) == len(cfg.filter_counts) + 1  # blocks + toRGB
    assert plain_convs == [], f"shallow stack should have no extra convs, got {plain_convs}"


def test_gen_convs_per_resolution_round_trips(tmp_path):
    cfg = _gen_config(tmp_path, gen_convs_per_resolution=1)
    assert cfg.gen_convs_per_resolution == 1
    assert cfg.dump()["gen_convs_per_resolution"] == 1
    # Default stays 2 so existing configs are unchanged.
    assert build_default(tmp_path).gen_convs_per_resolution == 2


def build_default(tmp_path):
    data = copy.deepcopy(config_template)
    data.pop("gen_convs_per_resolution", None)  # legacy config without the key
    p = tmp_path / "legacy_config.json"
    p.write_text(json.dumps(data))
    return build(str(p))
