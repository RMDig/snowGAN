#!/usr/bin/env python3
"""Capacity campaign Phase 0 diagnostics (docs/plans/capacity_plan.md §3).

Three read-only measurements on finished runs; nothing here trains.

  scale     P0.1/P0.2  KID per scale, normalised by a train-vs-held-out reference,
                       with positive controls and a leave-one-group-out jackknife.
  critic    P0.3       ADA-style critic overfitting ratio across snapshots.
  spectrum  P0.4       2-D power spectra, fakes vs reals, at native resolution
                       and on the 299² tensors Inception sees.

Examples (EMA weights, the seed-42 Arm D window):

    python scripts/capacity_diagnostics.py scale keras/snowgan/arm_d_ema \\
        --steps 70000-79000:1000 --n_fake 1000 --n_train 1000
    python scripts/capacity_diagnostics.py critic keras/snowgan/arm_d_ema \\
        --steps 10000-70000:10000,79000 --n 100
    python scripts/capacity_diagnostics.py spectrum keras/snowgan/arm_d_ema \\
        --steps 70000-79000:1000 --n 200 --out /tmp/spectra

Images are streamed in chunks: 1,000 fakes at 1024² float32 would be 12.6 GB.
Snapshots are addressed by batch_N directory name; read their fade_step if the
exact global step matters (names lag after a restart).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

SCALES = ("full", "global128", "crop")


# --------------------------------------------------------------------------- io

def _config(run_dir, name="generator_config.json"):
    from snowgan.config import build
    scratch = tempfile.mkdtemp(prefix="capdiag_cfg_")
    try:
        shutil.copy(os.path.join(run_dir, name), os.path.join(scratch, name))
        return build(os.path.join(scratch, name))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _snapshot_dirs(run_dir, spec):
    """'START-END:STEP[,N,...]' over batch_N directory names; missing ones skipped."""
    steps = []
    for part in spec.split(","):
        if "-" in part:
            span, _, stride = part.partition(":")
            start, _, end = span.partition("-")
            steps += list(range(int(start), int(end) + 1, int(stride or 1000)))
        else:
            steps.append(int(part))
    out = []
    for step in steps:
        path = os.path.join(run_dir, f"batch_{step}")
        if os.path.isdir(path):
            out.append((step, path))
        else:
            print(f"  (skipping batch_{step}: not retained)")
    if not out:
        raise SystemExit(f"no retained snapshots in {run_dir} for {spec}")
    return out


def _manifest(config):
    from datasets import load_dataset
    from snowgan.data.dataset import normalize_datatype
    frame = load_dataset(getattr(config, "dataset", "rmdig/rocky_mountain_snowpack"))["train"]
    frame = frame.to_pandas().drop(columns=["image", "audio"], errors="ignore")
    return frame, normalize_datatype(getattr(config, "modality", "magnified_profile"))


def _real_rows(config, image_root, pools, limit, seed):
    from snowgan.kid import pool_groups, select_rows
    frame, wanted = _manifest(config)
    groups = pool_groups(config, pools)
    if not groups:
        raise SystemExit(f"config records no {pools}")
    root = os.path.expanduser(image_root)
    rows = [(i, k) for i, k in select_rows(frame, wanted, held=groups)
            if os.path.exists(os.path.join(root, str(frame["file_path"][i])))]
    if limit and len(rows) > limit:
        pick = np.random.default_rng(seed).choice(len(rows), size=limit, replace=False)
        rows = [rows[i] for i in sorted(pick)]
    paths = [os.path.join(root, str(frame["file_path"][i])) for i, _ in rows]
    return paths, [k for _, k in rows]


def _iter_reals(paths, resolution, chunk=16):
    from PIL import Image
    height, width = resolution
    for start in range(0, len(paths), chunk):
        batch = []
        for path in paths[start:start + chunk]:
            with Image.open(path) as handle:
                arr = np.asarray(handle.convert("RGB").resize((width, height), Image.BILINEAR),
                                 dtype=np.float32)
            batch.append(arr / 127.5 - 1.0)
        yield np.stack(batch)


def _load_generator(snapshot, weights_file="generator_ema.weights.h5"):
    from snowgan.models.generator import Generator
    config = _config(snapshot)
    generator = Generator(config)
    generator.model.build((None, config.latent_dim))
    weights = os.path.join(snapshot, weights_file)
    if not os.path.exists(weights):
        raise SystemExit(f"{weights} missing")
    generator.model.load_weights(weights)
    return generator, config


def _iter_fakes(snapshots, total, seed, chunk=4, weights_file="generator_ema.weights.h5"):
    """`total` EMA fakes spread evenly over the window's snapshots (a window-level set)."""
    import tensorflow as tf
    per = int(np.ceil(total / len(snapshots)))
    made = 0
    for n, (_, path) in enumerate(snapshots):
        generator, config = _load_generator(path, weights_file)
        z = tf.random.Generator.from_seed(seed + n).normal([per, config.latent_dim])
        for start in range(0, per, chunk):
            if made >= total:
                break
            out = generator.model(z[start:start + chunk], training=False).numpy()[:, 0]
            made += len(out)
            yield out
        del generator
        tf.keras.backend.clear_session()


# ------------------------------------------------------------- transformations

def _corrupt(batch, kind, rng):
    """Positive-control corruptions of real images in [-1, 1], shape (B, H, W, 3)."""
    import tensorflow as tf
    if kind is None:
        return batch
    if kind == "tileshuffle":  # global: 4x4 tiles permuted, texture intact
        b, h, w, c = batch.shape
        th, tw = h // 4, w // 4
        tiles = batch[:, :th * 4, :tw * 4].reshape(b, 4, th, 4, tw, c).transpose(0, 1, 3, 2, 4, 5)
        tiles = tiles.reshape(b, 16, th, tw, c)
        out = np.empty_like(tiles)
        for i in range(b):
            out[i] = tiles[i, rng.permutation(16)]
        out = out.reshape(b, 4, 4, th, tw, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, th * 4, tw * 4, c)
        return out
    if kind == "blur":  # texture: gaussian sigma 3 px at native resolution
        x = np.arange(-9, 10, dtype=np.float32)
        g = np.exp(-x ** 2 / 18.0)
        g /= g.sum()
        kernel = np.outer(g, g)[:, :, None, None] * np.ones((1, 1, 3, 1), np.float32)
        return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], "SAME").numpy()
    if kind == "checker":  # texture: injected 2-px checkerboard, amplitude 0.15
        h, w = batch.shape[1:3]
        yy, xx = np.mgrid[0:h, 0:w]
        board = np.where((yy + xx) % 2 == 0, 0.15, -0.15).astype(np.float32)[None, :, :, None]
        return np.clip(batch + board, -1.0, 1.0)
    raise ValueError(kind)


def _scale_views(batch, rng):
    """{scale: (B, 299, 299, 3)} in [-1, 1]. Crops: one random 299² crop per image."""
    import tensorflow as tf
    x = tf.convert_to_tensor(batch)
    views = {
        "full": tf.image.resize(x, (299, 299), antialias=True),
        "global128": tf.image.resize(tf.image.resize(x, (128, 128), antialias=True), (299, 299)),
    }
    h, w = batch.shape[1:3]
    crops = []
    for img in batch:
        top, left = rng.integers(0, h - 299 + 1), rng.integers(0, w - 299 + 1)
        crops.append(img[top:top + 299, left:left + 299])
    views["crop"] = tf.convert_to_tensor(np.stack(crops))
    return views


def _features(chunks, inception, seed, corrupt=None):
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    rng = np.random.default_rng(seed)
    out = {s: [] for s in SCALES}
    for batch in chunks:
        batch = _corrupt(batch, corrupt, rng)
        for scale, view in _scale_views(batch, rng).items():
            out[scale].append(inception(preprocess_input((view + 1.0) * 127.5), training=False).numpy())
    return {s: np.concatenate(v) for s, v in out.items()}


# ----------------------------------------------------------------------- scale

def _kid(real, fake, subsets, seed):
    from snowgan.kid import kid_score
    return kid_score(real, fake, subsets=subsets, subset_size=100, seed=seed)["kid_mean"]


def _jackknife(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    return float(np.sqrt((n - 1) / n * np.sum((values - values.mean()) ** 2)))


def cmd_scale(args):
    """Per-scale excess KID, calibrated by positive controls.

    Excess at scale s:  D_s(X) = KID_s(held, X) - KID_s(held, train reals),
    i.e. how much worse X matches the held-out reals than a perfect generator
    of the training distribution would. Raw KIDs are not comparable across
    scales, and a ratio to KID_s(held, train) explodes at crop scale, where
    training and held-out crops (mostly blue grid card) are nearly identical
    (smoke test: crop ratios of -1294 and -3019). So each scale's excess is
    expressed in units of a known corruption:

        G = D_global128(fakes) / D_global128(tile shuffle)   layout deficit
        T = D_crop(fakes)      / D_crop(blur)                texture deficit

    G/T > 2 favours H1 (layout), < 0.5 favours H2/H3 (texture); see
    capacity_plan P0.2. Uncertainty: leave-one-held-group-out jackknife, with
    every term recomputed per fold. Controls must be selective first: the tile
    shuffle must score < 0.5 in T units and the blur < 0.5 in G units.
    """
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    config = _config(args.run_dir)
    resolution = tuple(int(v) for v in config.resolution)
    inception = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))

    held_paths, held_keys = _real_rows(config, args.image_root, ("validation_pool", "test_pool"), 0, args.seed)
    train_paths, _ = _real_rows(config, args.image_root, ("trained_pool",), args.n_train, args.seed)
    print(f"held-out {len(held_paths)} images / {len(set(held_keys))} groups; train sample {len(train_paths)}")

    held = _features(_iter_reals(held_paths, resolution), inception, args.seed)
    train = _features(_iter_reals(train_paths, resolution), inception, args.seed + 1)
    controls = {kind: _features(_iter_reals(held_paths, resolution), inception, args.seed + 2, corrupt=kind)
                for kind in ("tileshuffle", "blur", "checker")}
    fakes = None
    if not args.controls_only:
        snapshots = _snapshot_dirs(args.run_dir, args.steps)
        fakes = _features(_iter_fakes(snapshots, args.n_fake, args.seed), inception, args.seed + 3)

    keys = np.array([str(k) for k in held_keys])
    folds = [np.ones(len(keys), bool)] + [keys != g for g in sorted(set(keys))]

    def excess(scale, mask, name):
        ref = _kid(held[scale][mask], train[scale], args.subsets, args.seed)
        other = fakes[scale] if name == "fakes" else controls[name][scale][mask]
        return _kid(held[scale][mask], other, args.subsets, args.seed) - ref

    names = list(controls) + (["fakes"] if fakes is not None else [])
    # table[name][scale] = [full-data value, fold 1, ..., fold k]
    table = {n: {sc: [excess(sc, m, n) for m in folds] for sc in SCALES} for n in names}

    def calibrated(name, scale, unit):
        return np.array(table[name][scale]) / np.array(table[unit][scale])

    def show(label, values):
        print(f"   {label:<40} {values[0]:>8.3f} ± {_jackknife(values[1:]):.3f}")
        return {"value": float(values[0]), "jackknife_se": _jackknife(values[1:])}

    report = {"run_dir": args.run_dir, "steps": args.steps, "n_fake": args.n_fake,
              "excess": {n: {sc: [float(v) for v in table[n][sc]] for sc in SCALES} for n in names}}

    print("\nexcess KID over a perfect generator, D_s(X) = KID_s(held, X) - KID_s(held, train):")
    print(f"{'set':>14} " + " ".join(f"{sc:>12}" for sc in SCALES))
    for n in names:
        print(f"{n:>14} " + " ".join(f"{table[n][sc][0]:>12.4f}" for sc in SCALES))

    print("\ncontrol selectivity (must both be < 0.5):")
    report["selectivity"] = {
        "tileshuffle_in_T_units": show("tile shuffle, texture units (crop/blur)", calibrated("tileshuffle", "crop", "blur")),
        "blur_in_G_units": show("blur, layout units (global128/tileshuffle)", calibrated("blur", "global128", "tileshuffle")),
    }
    if fakes is not None:
        g = calibrated("fakes", "global128", "tileshuffle")
        t = calibrated("fakes", "crop", "blur")
        print("\nfakes:")
        report["G"] = show("G = layout deficit (tile-shuffle units)", g)
        report["T"] = show("T = texture deficit (blur units)", t)
        report["G_over_T"] = show("G / T", g / t)
        report["checker_T"] = show("checkerboard control, texture units", calibrated("checker", "crop", "blur"))
        print("\nPre-registered (capacity_plan P0.2): H1 if G/T > 2 and its 95% CI excludes 1 at both "
              "seeds; H2/H3 if G/T < 0.5 likewise; otherwise inconclusive.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.out}")


# ---------------------------------------------------------------------- critic

def _critic_scores(discriminator, chunks):
    out = []
    for batch in chunks:
        out.append(discriminator.model(batch[:, None], training=False).numpy().ravel())
    return np.concatenate(out)


def _auc(higher, lower):
    """P(score of a `higher` item > score of a `lower` item), ties counted half.

    Scale-free, so it is immune to the drift of WGAN critic scores between
    snapshots, and has no denominator to blow up. The ADA-style ratio
    (D_train - D_held) / (D_train - D_fake) does both: at arm_d_ema 79k the
    critic scored fakes and training reals almost equally (-48.7 vs -47.4) and
    r came out at -21.
    """
    a, b = np.asarray(higher)[:, None], np.asarray(lower)[None, :]
    return float(np.mean((a > b) + 0.5 * (a == b)))


def cmd_critic(args):
    from snowgan.models.discriminator import Discriminator
    config = _config(args.run_dir)
    resolution = tuple(int(v) for v in config.resolution)
    train_paths, _ = _real_rows(config, args.image_root, ("trained_pool",), args.n, args.seed)
    held_paths, _ = _real_rows(config, args.image_root, ("validation_pool", "test_pool"), args.n, args.seed)
    train_imgs = np.concatenate(list(_iter_reals(train_paths, resolution)))
    held_imgs = np.concatenate(list(_iter_reals(held_paths, resolution)))

    rows = []
    print(f"{'step':>7} {'D(train)':>10} {'D(held)':>10} {'D(fake)':>10} {'r':>8} "
          f"{'AUC tr>held':>12} {'AUC tr>fake':>12}")
    for step, path in _snapshot_dirs(args.run_dir, args.steps):
        dconf = _config(path, "discriminator_config.json")
        disc = Discriminator(dconf)
        disc.model.build((None, dconf.depth, *[int(v) for v in dconf.resolution], dconf.channels))
        disc.model.load_weights(os.path.join(path, "discriminator.weights.h5"))
        # The critic trained against the RAW generator; scoring it on EMA fakes
        # (smoother, better) confounds how well it separates real from fake.
        fakes = np.concatenate(list(_iter_fakes([(step, path)], args.n, args.seed,
                                                weights_file=args.fake_weights)))
        s_train = _critic_scores(disc, [train_imgs[i:i + 8] for i in range(0, len(train_imgs), 8)])
        s_held = _critic_scores(disc, [held_imgs[i:i + 8] for i in range(0, len(held_imgs), 8)])
        s_fake = _critic_scores(disc, [fakes[i:i + 8] for i in range(0, len(fakes), 8)])
        d_train, d_held, d_fake = float(s_train.mean()), float(s_held.mean()), float(s_fake.mean())
        r = (d_train - d_held) / (d_train - d_fake)
        auc_held, auc_fake = _auc(s_train, s_held), _auc(s_train, s_fake)
        rows.append({"step": step, "d_train": d_train, "d_held": d_held, "d_fake": d_fake, "r": r,
                     "auc_train_vs_held": auc_held, "auc_train_vs_fake": auc_fake})
        print(f"{step:>7} {d_train:>10.3f} {d_held:>10.3f} {d_fake:>10.3f} {r:>8.3f} "
              f"{auc_held:>12.3f} {auc_fake:>12.3f}")
        del disc

    auc = np.array([x["auc_train_vs_held"] for x in rows])
    ranks = np.argsort(np.argsort(auc))
    rho = float(np.corrcoef(np.arange(len(auc)), ranks)[0, 1]) if len(auc) > 2 else float("nan")
    growth = float(auc[-1] - auc[0])
    print(f"\nAUC(train > held) growth from first snapshot: {growth:+.3f}   Spearman rho(step, AUC): {rho:+.2f}")
    print("Pre-registered (capacity_plan P0.3, revised): H5 supported if AUC growth > 0.10 and rho > 0.6.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"rows": rows, "auc_growth": growth, "rho": rho}, handle, indent=2)


# -------------------------------------------------------------------- spectrum

def _mean_power(chunks, size=None, antialias=False):
    import tensorflow as tf
    total, n = None, 0
    for batch in chunks:
        x = batch
        if size is not None:
            x = tf.image.resize(x, (size, size), antialias=antialias).numpy()
        gray = x.mean(axis=-1)
        power = np.abs(np.fft.fftshift(np.fft.fft2(gray), axes=(-2, -1))) ** 2
        total = power.sum(0) if total is None else total + power.sum(0)
        n += len(gray)
    return total / n


def cmd_spectrum(args):
    config = _config(args.run_dir)
    resolution = tuple(int(v) for v in config.resolution)
    paths, _ = _real_rows(config, args.image_root, ("trained_pool",), args.n, args.seed)
    snapshots = _snapshot_dirs(args.run_dir, args.steps)
    os.makedirs(args.out, exist_ok=True)

    report = {}
    for label, size, aa in (("native", None, False), ("inception299_plain", 299, False),
                            ("inception299_antialias", 299, True)):
        real = _mean_power(_iter_reals(paths, resolution), size, aa)
        fake = _mean_power(_iter_fakes(snapshots, args.n, args.seed), size, aa)
        ratio = np.log10(fake + 1e-12) - np.log10(real + 1e-12)
        np.save(os.path.join(args.out, f"logratio_{label}.npy"), ratio)
        c = ratio.shape[0] // 2
        axis = {}
        for period in (2, 4, 8, 16, 32):
            offset = int(round(ratio.shape[0] / period))
            if offset <= c:
                idx = min(c + offset, ratio.shape[0] - 1)
                axis[f"period_{period}"] = {"x": float(ratio[c, idx]), "y": float(ratio[idx, c])}
        report[label] = axis
        print(f"{label}: log10(P_fake / P_real) along the axes at stride-period frequencies")
        for k, v in axis.items():
            print(f"   {k:>10}: x {v['x']:+.2f}   y {v['y']:+.2f}")
        try:
            from PIL import Image
            lim = 2.0
            img = np.clip((ratio + lim) / (2 * lim), 0, 1)
            Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(args.out, f"logratio_{label}.png"))
        except Exception as exc:  # the .npy is the record; the png is a convenience
            print(f"   (png not written: {exc})")
    with open(os.path.join(args.out, "spectrum.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {args.out}")


# ------------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("run_dir")
        p.add_argument("--image_root", default="~/rmdig-cache-1024")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default=None)

    p = sub.add_parser("scale", help="P0.1/P0.2 per-scale KID ratios with positive controls")
    common(p)
    p.add_argument("--steps", default="70000-79000:1000")
    p.add_argument("--n_fake", type=int, default=1000)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--subsets", type=int, default=50)
    p.add_argument("--controls_only", action="store_true")
    p.set_defaults(func=cmd_scale)

    p = sub.add_parser("critic", help="P0.3 critic overfitting ratio across snapshots")
    common(p)
    p.add_argument("--steps", default="10000-70000:10000,79000")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--fake_weights", default="generator.weights.h5",
                   help="Generator weights for the fakes the critic scores. Default: the raw "
                        "weights the critic trained against (EMA fakes confound separation).")
    p.set_defaults(func=cmd_critic)

    p = sub.add_parser("spectrum", help="P0.4 power spectra, fakes vs reals")
    common(p)
    p.add_argument("--steps", default="70000-79000:1000")
    p.add_argument("--n", type=int, default=200)
    p.set_defaults(func=cmd_spectrum)

    args = parser.parse_args()
    if args.cmd == "spectrum" and not args.out:
        parser.error("spectrum needs --out DIR")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    args.func(args)


if __name__ == "__main__":
    main()
