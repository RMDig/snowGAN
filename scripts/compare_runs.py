#!/usr/bin/env python3
"""Emit the pre-registered campaign scoreboard for one or more runs.

Defined in docs/plans/increment_campaign.md §2, fixed before any arm ran. Every
checkpoint is evaluated with **identical latents**, so differences between
checkpoints are model movement rather than sampling noise.

    dist_err   |mean - (-0.22)|/0.22 + |std - 0.63|/0.63
               quality proxy: does output match the data's first two moments.
               Crude by construction -- see the plan's stated limitation; sample
               inspection remains the arbiter.

    osc        std of dist_err over the last 3 checkpoints. PRIMARY metric: the
               defect being fixed is that the baseline orbits its target instead
               of settling. Also serves as the noise floor -- an arm whose
               improvement is smaller than the baseline's own osc has not
               demonstrated an effect.

    grad_norm  mean ||dD/dx|| late, and its slope per 10k steps, read from
               metrics.jsonl. Validity, not quality: once the critic escapes,
               the loss is no longer a Wasserstein distance.

    diversity  mean pairwise |G(zi) - G(zj)|. GATE, not a score.

Usage:
    python scripts/compare_runs.py keras/snowgan/control_1024/
    python scripts/compare_runs.py keras/snowgan/{control_1024,arm_a_lranneal}/ --every 10000
"""

import argparse
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

DATA_MEAN, DATA_STD = -0.22, 0.63


def dist_err(mean, std):
    return abs(mean - DATA_MEAN) / abs(DATA_MEAN) + abs(std - DATA_STD) / DATA_STD


def _checkpoint_steps(run_dir, every):
    """(global_step, path) for each batch_* snapshot, sorted by REAL step.

    Keyed on fade_step, not the directory name: the batch counter is recovered
    by globbing snapshot dirs and rewinds on restart, so batch_N is not N.
    """
    found = []
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        cfg = os.path.join(path, "generator_config.json")
        if not name.startswith("batch_") or not os.path.exists(cfg):
            continue
        try:
            with open(cfg) as handle:
                found.append((int(json.load(handle).get("fade_step", 0)), path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    found.sort()
    if every:
        kept, last = [], -10 ** 9
        for step, path in found:
            if step - last >= every:
                kept.append((step, path))
                last = step
        found = kept
    return found


def _sample_stats(path, n, latent_seed, overrides):
    import tensorflow as tf
    from snowgan.config import build
    from snowgan.models.generator import Generator
    from snowgan.checkpoint import resolve_weights_path

    scratch = "/tmp/_cmp_cfg"
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)
    shutil.copy(os.path.join(path, "generator_config.json"),
                os.path.join(scratch, "generator_config.json"))
    config = build(os.path.join(scratch, "generator_config.json"))
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    weights = resolve_weights_path(os.path.join(path, "generator.weights.h5"))
    if weights is None:
        return None

    generator = Generator(config)
    generator.model.build((None, config.latent_dim))
    generator.model.load_weights(weights)

    # Identical latents at every checkpoint and every run.
    tf.keras.utils.set_random_seed(latent_seed)
    images = []
    for _ in range(n):
        z = tf.random.normal([1, config.latent_dim])
        images.append(generator.model(z, training=False).numpy()[0])
    arr = np.stack(images)

    pairwise = [float(np.mean(np.abs(arr[i] - arr[j])))
                for i in range(len(arr)) for j in range(i + 1, len(arr))]
    del generator
    tf.keras.backend.clear_session()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "saturated": float(np.mean(np.abs(arr) > 0.99)),
        "diversity": float(np.mean(pairwise)) if pairwise else float("nan"),
    }


def _grad_stats(run_dir):
    """Late ||dD/dx|| and its slope per 10k steps, from metrics.jsonl."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return {}
    steps, norms = [], []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get("grad_norm_interp")
            if record.get("event") or value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            steps.append(record["global_step"])
            norms.append(value)
    if len(steps) < 20:
        return {}
    steps_a, norms_a = np.array(steps, dtype=float), np.array(norms, dtype=float)
    late = norms_a[steps_a >= steps_a.max() - 2000]
    slope = np.polyfit(steps_a, norms_a, 1)[0] * 10000
    return {"grad_norm_final": float(late.mean()) if len(late) else float("nan"),
            "grad_slope_per_10k": float(slope),
            "final_step": int(steps_a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--every", type=int, default=10000, help="min step gap between checkpoints")
    parser.add_argument("--n", type=int, default=16, help="samples per checkpoint")
    parser.add_argument("--latent_seed", type=int, default=0)
    parser.add_argument("--gen_upsampler", default=None, choices=["resize", "transpose"])
    parser.add_argument("--gen_convs_per_resolution", type=int, default=None, choices=[1, 2])
    parser.add_argument("--gen_norm", default=None, choices=["pixel", "batch", "none"])
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    overrides = {"gen_upsampler": args.gen_upsampler,
                 "gen_convs_per_resolution": args.gen_convs_per_resolution,
                 "gen_norm": args.gen_norm}

    summary = []
    for run in args.runs:
        run = run.rstrip("/\\")
        name = os.path.basename(run)
        print(f"\n=== {name} ===")
        checkpoints = _checkpoint_steps(run, args.every)
        if not checkpoints:
            print("  no batch_* checkpoints with a config; skipping")
            continue

        print(f"  {'step':>8} {'mean':>8} {'std':>7} {'sat%':>6} {'divers':>8} {'dist_err':>9}")
        errors = []
        for step, path in checkpoints:
            stats = _sample_stats(path, args.n, args.latent_seed, overrides)
            if stats is None:
                continue
            err = dist_err(stats["mean"], stats["std"])
            errors.append((step, err, stats))
            print(f"  {step:>8} {stats['mean']:>8.3f} {stats['std']:>7.3f} "
                  f"{100*stats['saturated']:>6.2f} {stats['diversity']:>8.3f} {err:>9.3f}")

        if not errors:
            continue
        tail = [e for _, e, _ in errors[-3:]]
        osc = float(np.std(tail)) if len(tail) > 1 else float("nan")
        grads = _grad_stats(run)
        summary.append({
            "run": name,
            "osc": osc,
            "dist_err_final": errors[-1][1],
            "dist_err_best": min(e for _, e, _ in errors),
            "best_step": min(errors, key=lambda r: r[1])[0],
            "diversity_final": errors[-1][2]["diversity"],
            **grads,
        })

    if not summary:
        return 1
    print("\n\n=== campaign scoreboard ===")
    header = (f"{'run':>26} {'osc':>7} {'dist_err':>9} {'best':>7} {'@step':>8} "
              f"{'gradN':>7} {'slope':>8} {'divers':>7}")
    print(header)
    print("-" * len(header))
    for row in summary:
        print(f"{row['run']:>26} {row['osc']:>7.3f} {row['dist_err_final']:>9.3f} "
              f"{row['dist_err_best']:>7.3f} {row['best_step']:>8} "
              f"{row.get('grad_norm_final', float('nan')):>7.3f} "
              f"{row.get('grad_slope_per_10k', float('nan')):>8.3f} "
              f"{row['diversity_final']:>7.3f}")
    print("\nosc = std(dist_err) over the last 3 checkpoints -- PRIMARY, and the noise")
    print("floor: an arm improving by less than the baseline's osc has shown nothing.")
    print("gradN near 1.0 = critic still Lipschitz. divers < 0.049 = dead (gate).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
