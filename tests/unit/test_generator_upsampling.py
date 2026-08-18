"""Focused tests for the resize-convolution generator upsampler.

Regression target: every upsampling block (including the toRGB head) was a
stride-2 Conv3DTranspose with a kernel (3) not divisible by the stride (2) —
the textbook checkerboard-artifact generator (Odena et al. 2016), which
produced the grid texture in v0.1 core samples. v0.2 upsamples with
UpSampling3D + a stride-1 Conv3D instead. These tests pin that no transposed
conv survives and that the resolution coupling is unchanged.
"""

import types

import numpy as np
import tensorflow as tf

from snowgan.models.generator import Generator


def _tiny_config(**over):
    cfg = types.SimpleNamespace(
        latent_dim=16,
        filter_counts=[8],
        kernel_size=[3, 3],
        kernel_stride=[2, 2],
        padding="same",
        batch_norm=False,
        negative_slope=0.25,
        channels=3,
        final_activation="tanh",
        depth=1,
        learning_rate=1e-4,
        beta_1=0.5,
        beta_2=0.9,
    )
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def test_no_transposed_conv_layers():
    # Default upsampler is "resize" -> no transposed conv (no checkerboard).
    tf.random.set_seed(0)
    gen = Generator(_tiny_config())
    names = [type(layer).__name__ for layer in gen.model.layers]
    assert not any("Transpose" in name for name in names), names


def test_transpose_upsampler_uses_conv3dtranspose():
    # The "transpose" upsampler restores learned upsampling (recovers texture);
    # core uses kernel 4 / stride 2 to keep checkerboard mild.
    tf.random.set_seed(0)
    gen = Generator(_tiny_config(gen_upsampler="transpose", kernel_size=[4, 4]))
    names = [type(layer).__name__ for layer in gen.model.layers]
    assert any("Conv3DTranspose" in name for name in names), names
    out = gen(tf.random.normal([2, 16]), training=False).numpy()
    assert out.shape == (2, 1, 64, 64, 3)  # coupling preserved across upsamplers
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_output_resolution_coupling_preserved():
    # gen output = 16 * 2^(len(filter_counts)+1). One filter -> 16*2^2 = 64.
    tf.random.set_seed(0)
    gen = Generator(_tiny_config())
    out = gen(tf.random.normal([2, 16]), training=False)
    assert out.shape == (2, 1, 64, 64, 3), out.shape


def test_two_blocks_double_resolution():
    # Two filters -> 16*2^3 = 128, and fade endpoints (len(feats) >= 2) build.
    tf.random.set_seed(0)
    gen = Generator(_tiny_config(filter_counts=[16, 8]))
    out = gen(tf.random.normal([2, 16]), training=False)
    assert out.shape == (2, 1, 128, 128, 3), out.shape
    assert gen.fade_endpoints is not None


def test_output_in_tanh_range():
    tf.random.set_seed(0)
    gen = Generator(_tiny_config())
    out = gen(tf.random.normal([2, 16]), training=False).numpy()
    assert out.min() >= -1.0 and out.max() <= 1.0
