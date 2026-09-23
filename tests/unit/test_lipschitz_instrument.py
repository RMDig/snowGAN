"""Focused tests: the Lipschitz instrument and the lambda_gp clamp (plan 0.2/0.3).

Two defects these pin:

**The instrument never existed.** ``compute_gradient_penalty`` computed the
critic's input-gradient norm and threw it away, and ``disc_loss`` is the sum of
the Wasserstein term and ``lambda_gp * GP`` (plus ``0.5 * W_lowres`` when the
multiscale head is on). So the one number that says whether the 1-Lipschitz
constraint actually binds was never available, and every verdict in
docs/experiments.md was reached without it.

**The clamp was silent.** The trainer rewrote ``lambda_gp -> 1.0`` whenever
spectral norm was on, then persisted it — so the config stopped describing the
run, and ``lambda_gp > 1`` became inexpressible under SN. That matters because
lambda_gp is the variable that separates every run in this repo that produced
structure (>= 10, SN off) from every run that collapsed (<= 1, SN on).
"""

import numpy as np
import pytest
import tensorflow as tf
import keras

from snowgan.losses import compute_gradient_penalty, critic_input_gradient_norm
from snowgan.trainer import Trainer


def _linear_critic(weight_value, shape=(1, 4, 4, 3)):
    """A critic with an exactly known input gradient.

    ``D(x) = sum(w * x)`` for constant ``w``, so ``dD/dx = w`` everywhere and
    ``||dD/dx||_2 = |w| * sqrt(numel)`` — an analytic target the probe must hit.
    """
    inputs = keras.Input(shape=shape)
    flat = keras.layers.Flatten()(inputs)
    dense = keras.layers.Dense(1, use_bias=False, dtype="float32")
    outputs = dense(flat)
    model = keras.Model(inputs, outputs)
    n = int(np.prod(shape))
    dense.set_weights([np.full((n, 1), weight_value, dtype="float32")])
    return model, n


def test_probe_matches_analytic_gradient_norm():
    weight = 0.25
    critic, n = _linear_critic(weight)
    x = tf.random.normal([2, 1, 4, 4, 3])

    measured = float(critic_input_gradient_norm(critic, x))
    expected = weight * np.sqrt(n)

    assert measured == pytest.approx(expected, rel=1e-4)


def test_probe_detects_a_non_lipschitz_critic():
    """A critic 40x over the 1-Lipschitz target must read ~40, not ~1.

    This is the whole point of the instrument: SpectralNormalization on a
    Conv3D bounds the *reshaped kernel*, not the convolution's operator norm,
    so 'SN is on' is not evidence the constraint binds.
    """
    critic, n = _linear_critic(40.0 / np.sqrt(4 * 4 * 3))
    x = tf.random.normal([2, 1, 4, 4, 3])
    assert float(critic_input_gradient_norm(critic, x)) == pytest.approx(40.0, rel=1e-3)


def test_probe_does_not_mutate_spectral_norm_state():
    """An instrument must not perturb what it measures.

    The probe runs the critic with training=False so power iteration does not
    advance. (The generator step still runs the critic with training=True — a
    separate, known defect tracked as an increment, not fixed here.)
    """
    inputs = keras.Input(shape=(1, 4, 4, 3))
    conv = keras.layers.SpectralNormalization(keras.layers.Conv3D(2, (1, 3, 3), padding="same"))
    x = conv(inputs)
    flat = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, dtype="float32")(flat)
    critic = keras.Model(inputs, outputs)

    before = [w.numpy().copy() for w in conv.weights]
    critic_input_gradient_norm(critic, tf.random.normal([2, 1, 4, 4, 3]))
    after = [w.numpy() for w in conv.weights]

    for b, a in zip(before, after):
        np.testing.assert_allclose(b, a, rtol=0, atol=0)


def test_gradient_penalty_returns_the_norm_it_used_to_discard():
    critic, n = _linear_critic(1.0 / np.sqrt(4 * 4 * 3))
    real = tf.random.normal([2, 1, 4, 4, 3])
    fake = tf.random.normal([2, 1, 4, 4, 3])

    penalty, norm = compute_gradient_penalty(critic, real, fake)

    # w chosen so ||grad|| == 1 exactly -> a perfectly 1-Lipschitz critic
    # yields zero penalty, and the reported norm is 1.
    assert float(norm) == pytest.approx(1.0, rel=1e-4)
    assert float(penalty) == pytest.approx(0.0, abs=1e-6)


def test_gradient_penalty_is_applied_exactly_once():
    """UPGRADES #1 regression guard: the penalty returned here is unscaled, and
    Discriminator.get_loss multiplies by lambda_gp once. A double application
    would make the effective weight lambda^2."""
    from snowgan.models.discriminator import Discriminator

    critic, n = _linear_critic(3.0 / np.sqrt(4 * 4 * 3))
    real = tf.random.normal([2, 1, 4, 4, 3])
    fake = tf.random.normal([2, 1, 4, 4, 3])
    penalty, norm = compute_gradient_penalty(critic, real, fake)

    # ||grad|| == 3 -> raw penalty == (3-1)^2 == 4, unscaled.
    assert float(penalty) == pytest.approx(4.0, rel=1e-3)

    loss = Discriminator.get_loss(None, tf.zeros([2, 1]), tf.zeros([2, 1]), penalty, 10.0)
    assert float(loss) == pytest.approx(40.0, rel=1e-3)  # 10 * 4, not 100 * 4


# --- lambda_gp resolution -----------------------------------------------------

def test_lambda_gp_is_not_clamped_by_default_under_spectral_norm():
    resolved, note = Trainer._resolve_lambda_gp(10.0, spectral_norm=True, clamp_under_sn=False)
    assert resolved == 10.0
    assert note is not None and "honored as configured" in note


def test_lambda_gp_clamps_only_when_explicitly_requested():
    resolved, note = Trainer._resolve_lambda_gp(10.0, spectral_norm=True, clamp_under_sn=True)
    assert resolved == 1.0
    assert "clamp_gp_under_sn" in note


def test_lambda_gp_untouched_without_spectral_norm():
    resolved, note = Trainer._resolve_lambda_gp(10.0, spectral_norm=False, clamp_under_sn=True)
    assert resolved == 10.0
    assert note is None


def test_lambda_gp_zero_survives():
    """The SN-only critic (lambda_gp == 0) must stay at 0, not be coerced."""
    resolved, note = Trainer._resolve_lambda_gp(0.0, spectral_norm=True, clamp_under_sn=True)
    assert resolved == 0.0
    assert note is None
