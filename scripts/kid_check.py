#!/usr/bin/env python3
"""Score a run's checkpoints with KID against HELD-OUT real images.

Replaces `dist_err` as the campaign's quality metric. `dist_err` compares only
the first two moments of the output, and the 2026-09-29 noise probe measured it
swinging 0.245 -> 1.808 across eight consecutive 1k-spaced checkpoints of one
run — noise larger than any effect an arm was likely to produce.

Reals are drawn from `validation_pool + test_pool` only. The run honoured
splits, so those groups were never trained on; scoring against training images
would measure reproduction, which is `memorization_check.py`'s question.

Usage:
    # one checkpoint (the run's final weights)
    python scripts/kid_check.py keras/snowgan/control_1024/

    # sweep, matching the campaign's sampling design
    python scripts/kid_check.py keras/snowgan/control_1024/ --sweep 30000-39000:1000
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402


def _held_out_reals(config, image_root, limit):
    """Images from validation+test groups, as a (N, 1, H, W, C) array in [-1, 1]."""
    from PIL import Image
    from datasets import load_dataset
    from snowgan.data.dataset import normalize_datatype

    dataset = load_dataset(getattr(config, "dataset", "rmdig/rocky_mountain_snowpack"))["train"]
    frame = dataset.to_pandas().drop(columns=["image", "audio"], errors="ignore")
    wanted = normalize_datatype(getattr(config, "modality", "magnified_profile"))

    held = set()
    for pool in ("validation_pool", "test_pool"):
        for entry in (getattr(config, pool, None) or []):
            held.add(tuple(entry))
    if not held:
        raise SystemExit(
            "This run's config records no validation/test pools, so there is no "
            "held-out set to score against. Refusing to score against training data.")

    root = os.path.expanduser(image_root)
    height, width = int(config.resolution[0]), int(config.resolution[1])
    images = []
    for index, datatype in enumerate(frame["datatype"]):
        if normalize_datatype(datatype) != wanted:
            continue
        key = (frame["site"][index], frame["column"][index], frame["core"][index])
        if key not in held:
            continue
        path = os.path.join(root, str(frame["file_path"][index]))
        if not os.path.exists(path):
            continue
        with Image.open(path) as handle:
            arr = np.asarray(handle.convert("RGB").resize((width, height), Image.BILINEAR),
                             dtype=np.float32)
        images.append(arr / 127.5 - 1.0)
        if limit and len(images) >= limit:
            break
    if len(images) < 2:
        raise SystemExit(f"only {len(images)} held-out images resolved under {root}")
    return np.stack(images)[:, None, ...]


def _generate(checkpoint_dir, config_overrides, count, seed, batch=4):
    import tensorflow as tf
    from snowgan.config import build
    from snowgan.models.generator import Generator
    from snowgan.checkpoint import resolve_weights_path

    scratch = "/tmp/_kid_cfg"
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)
    shutil.copy(os.path.join(checkpoint_dir, "generator_config.json"),
                os.path.join(scratch, "generator_config.json"))
    config = build(os.path.join(scratch, "generator_config.json"))
    for key, value in config_overrides.items():
        if value is not None:
            setattr(config, key, value)

    weights = resolve_weights_path(os.path.join(checkpoint_dir, "generator.weights.h5"))
    if weights is None:
        return None, config

    generator = Generator(config)
    generator.model.build((None, config.latent_dim))
    generator.model.load_weights(weights)

    tf.keras.utils.set_random_seed(seed)
    z = tf.random.normal([count, config.latent_dim])
    out = []
    for start in range(0, count, batch):       # chunked: 1024^2 wide forwards OOM
        out.append(generator.model(z[start:start + batch], training=False).numpy())
    del generator
    tf.keras.backend.clear_session()
    return np.concatenate(out, axis=0), config


def _checkpoints(run_dir, spec):
    """Resolve --sweep 'START-END:STEP' to (step, dir) pairs, or the run itself."""
    if not spec:
        return [("final", run_dir)]
    span, _, stride = spec.partition(":")
    start, _, end = span.partition("-")
    stride = int(stride or 1000)
    out = []
    for step in range(int(start), int(end) + 1, stride):
        path = os.path.join(run_dir, f"batch_{step}")
        if os.path.isdir(path):
            out.append((step, path))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--image_root", default="~/rmdig-cache-1024")
    parser.add_argument("--n_gen", type=int, default=200, help="generated samples per checkpoint")
    parser.add_argument("--n_real", type=int, default=0, help="cap held-out reals (0 = all)")
    parser.add_argument("--subsets", type=int, default=10)
    parser.add_argument("--subset_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", default=None, help="'START-END:STEP', e.g. 30000-39000:1000")
    parser.add_argument("--gen_upsampler", default=None, choices=["resize", "transpose"])
    parser.add_argument("--gen_convs_per_resolution", type=int, default=None, choices=[1, 2])
    parser.add_argument("--gen_norm", default=None, choices=["pixel", "batch", "none"])
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from snowgan.config import build
    from snowgan.kid import inception_features, kid_score

    run_dir = args.run_dir.rstrip("/\\")
    overrides = {"gen_upsampler": args.gen_upsampler,
                 "gen_convs_per_resolution": args.gen_convs_per_resolution,
                 "gen_norm": args.gen_norm}

    scratch = "/tmp/_kid_base"
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)
    shutil.copy(os.path.join(run_dir, "generator_config.json"),
                os.path.join(scratch, "generator_config.json"))
    base_config = build(os.path.join(scratch, "generator_config.json"))

    print("Loading held-out reals (validation+test groups only)...")
    reals = _held_out_reals(base_config, args.image_root, args.n_real)
    print(f"  {len(reals)} held-out images at {list(base_config.resolution)}")

    from tensorflow.keras.applications.inception_v3 import InceptionV3
    inception = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))
    real_features = inception_features(reals, model=inception)
    print(f"  real features {real_features.shape}")

    rows = []
    print(f"\n{'step':>9} {'KID':>10} {'+/- SE':>9} {'subsets':>8}")
    for step, path in _checkpoints(run_dir, args.sweep):
        fakes, _ = _generate(path, overrides, args.n_gen, args.seed)
        if fakes is None:
            continue
        fake_features = inception_features(fakes, model=inception)
        score = kid_score(real_features, fake_features,
                          subsets=args.subsets, subset_size=args.subset_size,
                          seed=args.seed)
        rows.append((step, score))
        print(f"{str(step):>9} {score['kid_mean']:>10.5f} {score['kid_se']:>9.5f} "
              f"{score['subsets']:>8}")

    if len(rows) > 1:
        values = np.array([s["kid_mean"] for _, s in rows])
        print(f"\n  across {len(values)} checkpoints: mean {values.mean():.5f}  "
              f"std {values.std(ddof=1):.5f}  SE {values.std(ddof=1)/np.sqrt(len(values)):.5f}")
        print(f"  coefficient of variation = {values.std(ddof=1)/abs(values.mean()):.3f}"
              "   (lower = a more resolvable metric)")
        best = min(rows, key=lambda r: r[1]["kid_mean"])
        print(f"  best checkpoint: {best[0]} at KID {best[1]['kid_mean']:.5f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
