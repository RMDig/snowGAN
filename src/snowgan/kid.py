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


class RealSetError(ValueError):
    """The real set cannot be assembled without scoring against seen data."""


def pool_groups(config, pools):
    """The (site, column, core) groups recorded in the config's named pools."""
    groups = set()
    for pool in pools:
        for entry in (getattr(config, pool, None) or []):
            groups.add(tuple(int(v) for v in entry))
    return groups


def check_sites_unseen(config, sites):
    """Refuse sites the run's sidecar shows it trained, validated or tested on."""
    seen = pool_groups(config, ("trained_pool", "validation_pool", "test_pool"))
    if not seen:
        raise RealSetError(
            "This run's config records no split pools, so there is no record of which "
            "sites it saw. Refusing to treat any site as unseen.")
    overlap = sorted({g[0] for g in seen} & set(sites))
    if overlap:
        raise RealSetError(f"--sites {sorted(sites)} includes site(s) {overlap} that appear in this "
                           f"run's split pools; those images are not unseen.")


def select_rows(frame, wanted, held=None, sites=None):
    """Manifest row indices of datatype `wanted` in the `held` groups or `sites`."""
    from snowgan.data.dataset import normalize_datatype

    rows = []
    for index, datatype in enumerate(frame["datatype"]):
        if normalize_datatype(datatype) != wanted:
            continue
        key = (int(frame["site"][index]), int(frame["column"][index]), int(frame["core"][index]))
        if sites is not None and key[0] not in sites:
            continue
        if held is not None and key not in held:
            continue
        rows.append((index, key))
    return rows


def real_features(config, image_root, limit, inception, sites=None, seed=0, chunk=32):
    """Inception features of the real set, plus the group key per image.

    Shared by ``scripts/kid_check.py`` and the trainer's best-KID checkpointing,
    so an in-training KID and an end-of-run sweep score against the same reals
    by the same path and their numbers are directly comparable.

    Images are loaded and featurised `chunk` at a time. Holding the whole set
    at 1024^2 float32 first needed ~25 GB for sites 3-6 (995 images) against
    a 31 GB WSL. Each image goes PIL bilinear to the run's resolution, then
    through `inception_features`.

    Default: the run's validation+test pools. With `sites`: every group from
    those sites, after `check_sites_unseen`.

    Raises:
        RealSetError: no held-out record, a seen site, or < 2 images resolved.
    """
    import os
    from PIL import Image
    from datasets import load_dataset
    from snowgan.data.dataset import normalize_datatype

    dataset = load_dataset(getattr(config, "dataset", "rmdig/rocky_mountain_snowpack"))["train"]
    frame = dataset.to_pandas().drop(columns=["image", "audio"], errors="ignore")
    wanted = normalize_datatype(getattr(config, "modality", "magnified_profile"))

    if sites is not None:
        check_sites_unseen(config, sites)
        rows = select_rows(frame, wanted, sites=sites)
    else:
        held = pool_groups(config, ("validation_pool", "test_pool"))
        if not held:
            raise RealSetError(
                "This run's config records no validation/test pools, so there is no "
                "held-out set to score against. Refusing to score against training data.")
        rows = select_rows(frame, wanted, held=held)

    root = os.path.expanduser(image_root)
    rows = [(i, k) for i, k in rows if os.path.exists(os.path.join(root, str(frame["file_path"][i])))]
    if limit and len(rows) > limit:
        # A random subset, not the first N in manifest order (which is one site).
        pick = np.random.default_rng(seed).choice(len(rows), size=limit, replace=False)
        rows = [rows[i] for i in sorted(pick)]

    if len(rows) < 2:
        raise RealSetError(f"only {len(rows)} real images resolved under {root}")

    height, width = int(config.resolution[0]), int(config.resolution[1])
    features, keys = [], [key for _, key in rows]
    for start in range(0, len(rows), chunk):
        images = []
        for index, _ in rows[start:start + chunk]:
            with Image.open(os.path.join(root, str(frame["file_path"][index]))) as handle:
                arr = np.asarray(handle.convert("RGB").resize((width, height), Image.BILINEAR),
                                 dtype=np.float32)
            images.append(arr / 127.5 - 1.0)
        features.append(inception_features(np.stack(images)[:, None, ...], model=inception))
    return np.concatenate(features, axis=0), keys


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
