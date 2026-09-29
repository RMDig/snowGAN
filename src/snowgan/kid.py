"""Kernel Inception Distance — a GAN metric that works at our sample size.

Why not FID. `Trainer._compute_fid` estimates a 2048x2048 covariance from 64
samples; the estimate is rank-63 and the Fréchet term is numerically degenerate.
FID is also *biased* at small n, and the bias depends on n — so two runs scored
with different sample counts are not comparable, and a single run's FID cannot
be compared to a published one. Chong & Forsyth (CVPR 2020) quantify this.

KID (Bińkowski et al., ICLR 2018, "Demystifying MMD GANs") is the squared
maximum-mean-discrepancy between Inception features under a polynomial kernel,
using the **unbiased** MMD² estimator. Unbiased means its expectation does not
depend on the sample size, so small-n estimates are noisy but not systematically
wrong — which is the property we need with 344 held-out images.

    k(x, y) = (x·y / d + 1)^3        d = feature dimension (2048)

    MMD²_u =   1/(m(m-1)) Σ_{i≠j} k(xᵢ,xⱼ)
             + 1/(n(n-1)) Σ_{i≠j} k(yᵢ,yⱼ)
             - 2/(mn)     Σ_{i,j}  k(xᵢ,yⱼ)

Reported as mean ± standard error over several random subsets, which is how the
paper recommends using it and what makes it a usable discriminator rather than a
single number of unknown precision.

**Reals come from the held-out pools only.** Scoring against training images
would measure how well the generator reproduces what it was shown, which is the
question `memorization_check.py` answers — not whether it models the
distribution.
"""

import numpy as np


def polynomial_mmd2_unbiased(features_x, features_y, degree=3, gamma=None, coef0=1.0):
    """Unbiased estimator of MMD² under a polynomial kernel.

    Args:
        features_x: (m, d) real features.
        features_y: (n, d) generated features.
        degree, gamma, coef0: kernel parameters. ``gamma=None`` uses 1/d, the
            standard choice — it keeps the kernel scale independent of feature
            dimension so values are comparable across extractors.

    Returns:
        float: MMD² estimate. Can be slightly negative — that is *expected* for
        an unbiased estimator when the true MMD² is near zero, and clipping it
        to 0 would reintroduce the bias the estimator exists to avoid.
    """
    features_x = np.asarray(features_x, dtype=np.float64)
    features_y = np.asarray(features_y, dtype=np.float64)
    m, n = len(features_x), len(features_y)
    if m < 2 or n < 2:
        raise ValueError("MMD² needs at least 2 samples per side")
    if gamma is None:
        gamma = 1.0 / features_x.shape[1]

    k_xx = (gamma * (features_x @ features_x.T) + coef0) ** degree
    k_yy = (gamma * (features_y @ features_y.T) + coef0) ** degree
    k_xy = (gamma * (features_x @ features_y.T) + coef0) ** degree

    # Exclude the diagonal: k(xi, xi) is the self-similarity term whose presence
    # is exactly what makes the naive estimator biased.
    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    sum_xy = k_xy.sum() / (m * n)
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def kid_score(real_features, fake_features, subsets=10, subset_size=100, seed=0):
    """KID as mean ± standard error over random subsets.

    A single MMD² estimate on one subset has unknown precision. Averaging over
    subsets and reporting the standard error is what turns KID into something
    you can rank two runs with — and the SE is what says whether a difference
    between two runs is resolvable at all.

    Args:
        real_features: (m, d) features of held-out real images.
        fake_features: (n, d) features of generated images.
        subsets: number of random subsets to average over.
        subset_size: samples drawn per side per subset. Capped at the smaller
            of the two corpora.

    Returns:
        dict: ``{"kid_mean", "kid_std", "kid_se", "subsets", "subset_size"}``.
    """
    real_features = np.asarray(real_features)
    fake_features = np.asarray(fake_features)
    size = int(min(subset_size, len(real_features), len(fake_features)))
    if size < 2:
        raise ValueError("not enough samples for a KID subset")

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(subsets):
        idx_r = rng.choice(len(real_features), size=size, replace=False)
        idx_f = rng.choice(len(fake_features), size=size, replace=False)
        values.append(polynomial_mmd2_unbiased(real_features[idx_r], fake_features[idx_f]))

    values = np.asarray(values, dtype=np.float64)
    return {
        "kid_mean": float(values.mean()),
        "kid_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "kid_se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        "subsets": int(subsets),
        "subset_size": size,
    }


def inception_features(images, batch_size=8, model=None):
    """InceptionV3 pool3 features (2048-d) for a batch of images in [-1, 1].

    Accepts rank-4 ``(N, H, W, C)`` or rank-5 ``(N, D, H, W, C)``; the depth
    axis is folded into the batch so each modality slice is scored on its own,
    matching how the rest of the codebase treats depth.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

    owns_model = model is None
    if owns_model:
        model = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))

    out = []
    try:
        for start in range(0, len(images), batch_size):
            chunk = tf.convert_to_tensor(images[start:start + batch_size], dtype=tf.float32)
            if len(chunk.shape) == 5:
                shape = tf.shape(chunk)
                chunk = tf.reshape(chunk, [shape[0] * shape[1], shape[2], shape[3], shape[4]])
            chunk = tf.image.resize(chunk, (299, 299))
            chunk = preprocess_input((chunk + 1.0) * 127.5)
            out.append(model(chunk, training=False).numpy())
    finally:
        if owns_model:
            del model
    return np.concatenate(out, axis=0)
