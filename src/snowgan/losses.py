import tensorflow as tf


def critic_input_gradient_norm(discriminator, images):
    """Mean L2 norm of the critic's gradient with respect to its *input*.

    This is the quantity the WGAN-GP penalty exists to drive toward 1.0, and
    the only direct read on whether the Lipschitz constraint actually binds.

    It is deliberately a standalone probe rather than a by-product of
    :func:`compute_gradient_penalty`, because the train step skips the penalty
    entirely when ``lambda_gp == 0`` — so a by-product instrument would vanish
    in exactly the spectral-norm-only configuration whose whole question is
    "is this critic 1-Lipschitz?".

    Callers pass whichever manifold they want measured: reals, fakes, or
    interpolates. They are different numbers and all three are worth logging —
    the penalty only ever constrains the interpolate path (Gulrajani et al.
    2017), so agreement between reals/fakes and interpolates is evidence the
    constraint generalizes off that path, and disagreement is evidence it does
    not.

    Uses ``training=False`` so the probe cannot mutate the critic's
    spectral-norm power-iteration vectors: an instrument must not perturb what
    it measures.

    Args:
        discriminator: the critic wrapper (callable; ``__call__`` forwards to
            the Keras model).
        images (tf.Tensor): batch to measure at, rank-5 ``(B, D, H, W, C)``.

    Returns:
        tf.Tensor: scalar float32 mean over the batch of ``||dD(x)/dx||_2``.
    """
    # float32 throughout: critic scores are unbounded and fp16 overflows the
    # backward pass under a mixed_float16 policy (UPGRADES #15).
    x = tf.cast(images, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = tf.cast(discriminator(x, training=False), tf.float32)
    grads = tape.gradient(pred, x)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]) + 1e-12)
    return tf.reduce_mean(norm)


def compute_gradient_penalty(discriminator, real_images, fake_images):
    """
    Computes gradient penalty for WGAN-GP.

    Args:
        discriminator (tf.keras.Model): The discriminator model
        real_images (tf.Tensor): Real images batch
        fake_images (tf.Tensor): Fake images batch

    Returns:
        tuple[tf.Tensor, tf.Tensor]: ``(penalty, mean_norm)``.
        ``penalty`` is the unscaled ``mean((||grad|| - 1)^2)`` — the caller
        applies lambda_gp exactly once. ``mean_norm`` is the mean interpolate
        gradient norm, returned rather than discarded so the run log can show
        whether the constraint binds (it was computed and thrown away for the
        whole history of this repo, which is why no past run could be judged
        on it).
    """
    batch_size = tf.shape(real_images)[0]
    # Compute GP in float32 for numerical stability (important with mixed precision)
    real_f32 = tf.cast(real_images, tf.float32)
    fake_f32 = tf.cast(fake_images, tf.float32)
    # Match alpha to the full image rank for broadcasting (B, D, H, W, C)
    alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0., 1.)
    interpolated = alpha * real_f32 + (1 - alpha) * fake_f32


    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
        pred = tf.cast(pred, tf.float32)

    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]) + 1e-12)
    penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return penalty, tf.reduce_mean(norm)
