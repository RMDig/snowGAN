#!/usr/bin/env python3
"""Score a run's checkpoints with KID against HELD-OUT real images.

Replaces `dist_err` as the campaign's quality metric. `dist_err` compares only
the first two moments of the output, and the 2026-09-29 noise probe measured it
swinging 0.245 -> 1.808 across eight consecutive 1k-spaced checkpoints of one
run — noise larger than any effect an arm was likely to produce.

Reals are drawn from `validation_pool + test_pool` only. The run honoured
splits, so those groups were never trained on; scoring against training images
would measure reproduction, which is `memorization_check.py`'s question.

`--sites` scores against every group from the named sites instead. That is the
only valid real set for a run that did NOT honour its splits (both v0.1.0
releases trained on their own pools, UPGRADES #54); the sidecar's pools must
show no group from those sites, or the script refuses.

`--real_vs_real` first scores two group-disjoint halves of the real set against
each other. That is the floor: the KID two samples of real images reach, at
this sample size, with nothing generated involved. A generator's KID means
something only relative to it.

Usage:
    # one checkpoint (the run's final weights)
    python scripts/kid_check.py keras/snowgan/control_1024/

    # sweep, matching the campaign's sampling design
    python scripts/kid_check.py keras/snowgan/control_1024/ --sweep 30000-39000:1000

    # a v0.1.0 release, against sites it never saw, with the real-vs-real floor
    python scripts/kid_check.py <release_dir> --sites 3,4,5,6 --real_vs_real
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402


def _pool_groups(config, pools):
    groups = set()
    for pool in pools:
        for entry in (getattr(config, pool, None) or []):
            groups.add(tuple(int(v) for v in entry))
    return groups


def _check_sites_unseen(config, sites):
    """Refuse sites the run's sidecar shows it trained, validated or tested on."""
    seen = _pool_groups(config, ("trained_pool", "validation_pool", "test_pool"))
    if not seen:
        raise SystemExit(
            "This run's config records no split pools, so there is no record of which "
            "sites it saw. Refusing to treat any site as unseen.")
    overlap = sorted({g[0] for g in seen} & set(sites))
    if overlap:
        raise SystemExit(f"--sites {sorted(sites)} includes site(s) {overlap} that appear in this "
                         f"run's split pools; those images are not unseen.")


def _select_rows(frame, wanted, held=None, sites=None):
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


def _group_halves(keys, seed):
    """Two boolean masks splitting images into group-disjoint halves.

    Split by (site, column, core) group, not by image: images of one group are
    near-duplicates, and an image-level split would put the same group on both
    sides and understate the floor.
    """
    groups = sorted(set(keys))
    if len(groups) < 2:
        raise SystemExit(f"real-vs-real needs at least 2 groups, found {len(groups)}")
    order = np.random.default_rng(seed).permutation(len(groups))
    half_a = {groups[i] for i in order[::2]}
    mask_a = np.array([k in half_a for k in keys])
    return mask_a, ~mask_a


def _real_features(config, image_root, limit, inception, sites=None, seed=0, chunk=32):
    """Inception features of the real set, plus the group key per image.

    Images are loaded and featurised `chunk` at a time. Holding the whole set
    at 1024^2 float32 first needed ~25 GB for sites 3-6 (995 images) against
    a 31 GB WSL. Each image goes through the same path as before (PIL bilinear
    to the run's resolution, then `inception_features`), so scores stay
    comparable with earlier campaign numbers.

    Default: the run's validation+test pools. With `sites`: every group from
    those sites, after `_check_sites_unseen`.
    """
    from PIL import Image
    from datasets import load_dataset
    from snowgan.data.dataset import normalize_datatype

    dataset = load_dataset(getattr(config, "dataset", "rmdig/rocky_mountain_snowpack"))["train"]
    frame = dataset.to_pandas().drop(columns=["image", "audio"], errors="ignore")
    wanted = normalize_datatype(getattr(config, "modality", "magnified_profile"))

    if sites is not None:
        _check_sites_unseen(config, sites)
        rows = _select_rows(frame, wanted, sites=sites)
    else:
        held = _pool_groups(config, ("validation_pool", "test_pool"))
        if not held:
            raise SystemExit(
                "This run's config records no validation/test pools, so there is no "
                "held-out set to score against. Refusing to score against training data.")
        rows = _select_rows(frame, wanted, held=held)

    root = os.path.expanduser(image_root)
    rows = [(i, k) for i, k in rows if os.path.exists(os.path.join(root, str(frame["file_path"][i])))]
    if limit and len(rows) > limit:
        # A random subset, not the first N in manifest order (which is one site).
        pick = np.random.default_rng(seed).choice(len(rows), size=limit, replace=False)
        rows = [rows[i] for i in sorted(pick)]

    if len(rows) < 2:
        raise SystemExit(f"only {len(rows)} real images resolved under {root}")

    from snowgan.kid import inception_features

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


def _generate(checkpoint_dir, config_overrides, count, seed, weights_file="generator.weights.h5", batch=4):
    import tensorflow as tf
    from snowgan.config import build
    from snowgan.models.generator import Generator
    from snowgan.checkpoint import resolve_weights_path

    # A private copy: build() must not touch the run's (or the HF cache's)
    # sidecar, and a fixed path would let two concurrent runs read each other's.
    scratch = tempfile.mkdtemp(prefix="kid_cfg_")
    shutil.copy(os.path.join(checkpoint_dir, "generator_config.json"),
                os.path.join(scratch, "generator_config.json"))
    config = build(os.path.join(scratch, "generator_config.json"))
    shutil.rmtree(scratch, ignore_errors=True)
    for key, value in config_overrides.items():
        if value is not None:
            setattr(config, key, value)

    weights = resolve_weights_path(os.path.join(checkpoint_dir, weights_file))
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


def _floor_verdict(score, floor):
    """The pre-registered criterion (docs/plans/research_preview_asks.md, 2026-10-02).

    Pass iff |KID - floor| <= 2 * sqrt(SE_kid^2 + SE_floor^2). Stated once, as
    this formula.
    """
    gap = score["kid_mean"] - floor["kid_mean"]
    bound = 2.0 * float(np.hypot(score["kid_se"], floor["kid_se"]))
    verdict = "PASS" if abs(gap) <= bound else "FAIL"
    return f"{verdict}: KID - floor = {gap:+.5f}, bound +/-{bound:.5f}"


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
    parser.add_argument("--sites", default=None,
                        help="Comma-separated site ids to score against instead of the run's "
                             "held-out pools, e.g. 3,4,5,6. Refused if the run's pools include them.")
    parser.add_argument("--real_vs_real", action="store_true",
                        help="Also score two group-disjoint halves of the real set against each "
                             "other: the floor a generator's KID must be read against.")
    parser.add_argument("--weights_file", default="generator.weights.h5",
                        help="Generator weights to score, e.g. generator_ema.weights.h5 for a "
                             "release's EMA shadow. Default: the primary weights.")
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

    scratch = tempfile.mkdtemp(prefix="kid_base_")
    shutil.copy(os.path.join(run_dir, "generator_config.json"),
                os.path.join(scratch, "generator_config.json"))
    base_config = build(os.path.join(scratch, "generator_config.json"))
    shutil.rmtree(scratch, ignore_errors=True)

    from tensorflow.keras.applications.inception_v3 import InceptionV3
    inception = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))

    sites = {int(v) for v in args.sites.split(",")} if args.sites else None
    print(f"Loading reals ({f'sites {sorted(sites)}' if sites else 'validation+test groups only'})...")
    real_features, keys = _real_features(base_config, args.image_root, args.n_real, inception,
                                         sites=sites, seed=args.seed)
    print(f"  {len(keys)} images from {len(set(keys))} groups at {list(base_config.resolution)}; "
          f"features {real_features.shape}")
    print(f"  scoring {args.weights_file}")

    if args.real_vs_real:
        mask_a, mask_b = _group_halves(keys, args.seed)
        floor = kid_score(real_features[mask_a], real_features[mask_b],
                          subsets=args.subsets, subset_size=args.subset_size, seed=args.seed)
        print(f"\n  real-vs-real floor ({mask_a.sum()} vs {mask_b.sum()} images, group-disjoint): "
              f"KID {floor['kid_mean']:.5f} +/- {floor['kid_se']:.5f} "
              f"(subset_size {floor['subset_size']})")
    else:
        floor = None

    rows = []
    print(f"\n{'step':>9} {'KID':>10} {'+/- SE':>9} {'subsets':>8}")
    for step, path in _checkpoints(run_dir, args.sweep):
        fakes, _ = _generate(path, overrides, args.n_gen, args.seed, weights_file=args.weights_file)
        if fakes is None:
            continue
        fake_features = inception_features(fakes, model=inception)
        score = kid_score(real_features, fake_features,
                          subsets=args.subsets, subset_size=args.subset_size,
                          seed=args.seed)
        rows.append((step, score))
        print(f"{str(step):>9} {score['kid_mean']:>10.5f} {score['kid_se']:>9.5f} "
              f"{score['subsets']:>8}")
        if floor is not None:
            print(f"{'':>9} {_floor_verdict(score, floor)}")

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
