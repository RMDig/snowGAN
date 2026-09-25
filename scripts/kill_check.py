#!/usr/bin/env python3
"""The CLAUDE.md §9 kill-check, as a tool instead of a snippet to paste.

Loads a generator from a save_dir and reports the scoreboard's cheap tier:
latent diversity, output saturation, output std — plus an optional
nearest-neighbour memorization probe, which the scoreboard has never had.

Usage:
    # Calibrate against the known-good release (plan 0.6). ALWAYS use --copy-to
    # for a release directory: any snowgan run pointed at one rewrites its
    # config sidecars, and those sidecars are the release artifact.
    python scripts/kill_check.py keras/snowgan/magnified_profiles/ \
        --gen_upsampler transpose --gen_convs_per_resolution 1 --gen_norm none \
        --copy-to /tmp/magprof_ref

    # Gate a live run.
    python scripts/kill_check.py keras/snowgan/exp0_control_1024/

Why raw weights by default: every preview the trainer writes goes through an
EMA swap, and with ema_decay 0.999 the shadow still retains 61% of the random
init at step 500 and 13% at step 2,000. Gating an early run on EMA samples
fails healthy runs. Pass --ema to read the shadow deliberately.

Reference values from the real data, for comparison (docs/experiments.md):
    mean ~ -0.22, std ~ 0.63, ~0% saturated, in [-1, 1].
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402


def _load_generator(save_dir, overrides, use_ema):
    import tensorflow as tf  # noqa: F401  (import for side effects / policy)
    from snowgan.config import build
    from snowgan.models.generator import Generator
    from snowgan.checkpoint import resolve_weights_path

    cfg_path = os.path.join(save_dir, "generator_config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"No generator_config.json in {save_dir}")
    config = build(cfg_path)

    # Release configs predate gen_upsampler / gen_convs_per_resolution /
    # gen_norm, so build() supplies today's defaults (resize / 2 / none) and
    # silently constructs a DIFFERENT architecture than the weights were
    # trained under. The correct values are recoverable only from git history
    # at the release SHA, which is why they must be passed explicitly here.
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    generator = Generator(config)
    generator.model.build((None, config.latent_dim))

    weights = None
    if use_ema:
        ema = os.path.join(save_dir, "generator_ema.weights.h5")
        weights = ema if os.path.exists(ema) else None
        if weights is None:
            print("WARNING: --ema requested but no generator_ema.weights.h5; using raw weights.")
    if weights is None:
        weights = resolve_weights_path(os.path.join(save_dir, "generator.weights.h5"))
    if weights is None:
        raise SystemExit(f"No generator weights found in {save_dir}")

    generator.model.load_weights(weights)
    print(f"Loaded {weights}")
    return generator, config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("save_dir")
    parser.add_argument("--n", type=int, default=8, help="latents to sample (default 8)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema", action="store_true",
                        help="read the EMA shadow instead of raw weights (meaningless below ~10k steps)")
    parser.add_argument("--copy-to", default=None,
                        help="copy save_dir here first and read the copy. REQUIRED for release "
                             "directories: loading one in place rewrites its config sidecars.")
    parser.add_argument("--gen_upsampler", default=None, choices=["resize", "transpose"])
    parser.add_argument("--gen_convs_per_resolution", type=int, default=None, choices=[1, 2])
    parser.add_argument("--gen_norm", default=None, choices=["pixel", "batch", "none"])
    args = parser.parse_args()

    save_dir = args.save_dir
    if args.copy_to:
        if os.path.exists(args.copy_to):
            shutil.rmtree(args.copy_to)
        os.makedirs(os.path.dirname(args.copy_to) or ".", exist_ok=True)
        shutil.copytree(save_dir, args.copy_to,
                        ignore=shutil.ignore_patterns("batch_*", "synthetic_images", "*.png"))
        print(f"Working on a copy at {args.copy_to} (source left untouched)")
        save_dir = args.copy_to

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    generator, config = _load_generator(save_dir, {
        "gen_upsampler": args.gen_upsampler,
        "gen_convs_per_resolution": args.gen_convs_per_resolution,
        "gen_norm": args.gen_norm,
    }, args.ema)

    z = tf.random.normal([args.n, config.latent_dim])
    images = []
    for i in range(args.n):  # one at a time: 1024^2 previews OOM in a wide forward
        images.append(generator.model(z[i:i + 1], training=False).numpy()[0])
    images = np.stack(images)

    std = float(images.std())
    mean = float(images.mean())
    saturated = float(np.mean(np.abs(images) > 0.99))
    pairwise = [float(np.mean(np.abs(images[i] - images[j])))
                for i in range(args.n) for j in range(i + 1, args.n)]
    diversity = float(np.mean(pairwise))

    print(f"\n  samples        {args.n} at {list(config.resolution)}, depth {config.depth}")
    print(f"  output mean    {mean:+.4f}      (real data ~ -0.22)")
    print(f"  output std     {std:.4f}       (real data ~  0.63)")
    print(f"  saturated      {100 * saturated:.2f}%      frac(|x| > 0.99)")
    print(f"  latent diversity {diversity:.6f}   mean pairwise |G(zi) - G(zj)|")
    print(f"  min pairwise   {min(pairwise):.6f}   (0 for any pair => collapse)")

    print("\n  verdict")
    if diversity < 1e-4:
        print("    DEAD   - full mode collapse: the generator is ignoring z.")
    elif saturated > 0.5:
        print("    DEAD   - tanh-rail collapse: over half the output is pinned at +/-1.")
    elif std < 0.05:
        print("    DEAD   - vanished: output is a near-constant frame.")
    else:
        print("    ALIVE  - passes the cheap tier. This is necessary, not sufficient:")
        print("             open synthetic_images/ and look, then check memorization.")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
