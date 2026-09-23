#!/usr/bin/env python3
"""Phase 1z: can this code overfit a GAN to 8 images? (~10 minutes)

The cheapest decisive test in the plan. A GAN that cannot memorize 8 fixed
images at low resolution is broken as *code*; the recipe is not the variable.
Run this before committing ~20 hours to the Phase 1a control — if it fails,
skip straight to the bisect over the 51 commits since fde5671.

It exercises the real ``Trainer.train_step`` — real tapes, real gradient
penalty, real optimizer updates — against a frozen 8-image batch drawn from
the actual dataset, so it tests the code path the control will use.

Usage:
    python scripts/overfit_sanity.py
    python scripts/overfit_sanity.py --steps 3000 --resolution 64
    python scripts/overfit_sanity.py --spectral_norm --disc_lambda_gp 1.0   # the suspect recipe

Reading the result:
    latent diversity ~ 0        -> generator ignoring z: mode collapse
    NN distance not falling     -> not fitting the 8 images at all
    NN distance -> ~0 + diverse -> code is sound; the control's outcome is
                                   about the recipe, not the implementation
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402


def build_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--resolution", type=int, default=64,
                        help="64 (1 filter block) or 128 (2). Keep it small — this is a "
                             "code check, not a quality run.")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--modality", default="magnified_profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--disc_lr", type=float, default=1e-5)
    parser.add_argument("--gen_steps", type=int, default=3)
    parser.add_argument("--disc_steps", type=int, default=2)
    parser.add_argument("--disc_lambda_gp", type=float, default=10.0)
    parser.add_argument("--spectral_norm", action="store_true")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--report_every", type=int, default=250)
    return parser.parse_args()


def main():
    args = build_args()

    blocks = {64: 1, 128: 2, 256: 3}.get(args.resolution)
    if blocks is None:
        raise SystemExit("--resolution must be 64, 128 or 256 (the 16*2^(N+1) coupling)")

    save_dir = args.save_dir or os.path.join("keras", "snowgan", "overfit_sanity")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    import tensorflow as tf
    from snowgan.config import build, config_template
    from snowgan.models.generator import Generator
    from snowgan.models.discriminator import Discriminator
    from snowgan.data.dataset import DataManager

    tf.keras.utils.set_random_seed(args.seed)

    gen_filters = [1024 // (2 ** i) for i in range(blocks)]
    disc_filters = list(reversed(gen_filters))

    def make_config(architecture, filters, lr, steps):
        data = dict(config_template)
        data.update({
            "architecture": architecture,
            "save_dir": save_dir + "/",
            "checkpoint": None,
            "resolution": [args.resolution, args.resolution],
            "filter_counts": filters,
            "latent_dim": 100,
            "batch_size": args.batch_size,
            "depth": 1,
            "modality": args.modality,
            "learning_rate": lr,
            "training_steps": steps,
            "lambda_gp": args.disc_lambda_gp,
            "spectral_norm": args.spectral_norm,
            "gen_norm": "none",
            "gen_upsampler": "transpose",
            "gen_convs_per_resolution": 1,
            "fade": False,
            "augment": False,
            "adaptive_steps": False,
            "multiscale_disc": False,
            "ada_target": 0.0,
            "grad_clip_norm": 0.0,
            "ema_decay": 0.0,
            "lr_decay": None,
            "fid_interval": 0,
            "grad_probe_interval": args.report_every,
            "seed": args.seed,
        })
        path = os.path.join(save_dir, f"{architecture}_config.json")
        with open(path, "w") as handle:
            json.dump(data, handle, indent=2)
        return build(path)

    gen_config = make_config("generator", gen_filters, args.gen_lr, args.gen_steps)
    disc_config = make_config("discriminator", disc_filters, args.disc_lr, args.disc_steps)

    # --- the fixed 8 images -------------------------------------------------
    print(f"Loading {args.images} real {args.modality} images at {args.resolution}px...")
    manager = DataManager(gen_config)
    fixed = manager.next_batch(args.images)
    if fixed is None:
        raise SystemExit("Could not load a batch — is the dataset cached?")
    fixed = tf.convert_to_tensor(fixed, dtype=tf.float32)
    print(f"  batch {tuple(fixed.shape)}  mean {float(tf.reduce_mean(fixed)):+.3f}  "
          f"std {float(tf.math.reduce_std(fixed)):.3f}")

    class _FixedData:
        """Always serves the same images. The point is to memorize them."""

        def __init__(self, config, batch):
            self.config = config
            self.pair_depth = 1
            self.seen_profiles = set()
            self._batch = batch

        def next_batch(self, batch_size):
            idx = tf.random.uniform([batch_size], 0, int(self._batch.shape[0]), dtype=tf.int32)
            return tf.gather(self._batch, idx).numpy()

        def batch(self, batch_size, datatype=None, config=None):
            return self.next_batch(batch_size)

        def reset_seen_profiles(self):
            pass

        def derive_splits(self, *a, **k):
            self.config.trained_pool = []
            self.config.validation_pool = []
            self.config.test_pool = []

    import importlib

    trainer_module = importlib.import_module("snowgan.trainer")
    original = trainer_module.DataManager
    trainer_module.DataManager = lambda config: _FixedData(config, fixed)
    try:
        generator = Generator(gen_config)
        generator.model.build((None, gen_config.latent_dim))
        discriminator = Discriminator(disc_config)
        discriminator.model.build((None, 1, args.resolution, args.resolution, 3))
        trainer = trainer_module.Trainer(generator, discriminator)
    finally:
        trainer_module.DataManager = original

    probe_z = tf.random.normal([6, gen_config.latent_dim])
    reals = fixed.numpy()

    def report(step):
        samples = np.stack([generator.model(probe_z[i:i + 1], training=False).numpy()[0]
                            for i in range(probe_z.shape[0])])
        pairwise = [float(np.mean(np.abs(samples[i] - samples[j])))
                    for i in range(len(samples)) for j in range(i + 1, len(samples))]
        # Nearest-neighbour distance to the 8 targets: the direct read on
        # "is it fitting these images?", and the memorization check the
        # scoreboard has never had.
        nn = float(np.mean([min(np.mean(np.abs(s - r)) for r in reals) for s in samples]))
        saturation = float(np.mean(np.abs(samples) > 0.99))
        print(f"  step {step:>6}  nn_dist {nn:.4f}  diversity {np.mean(pairwise):.6f}  "
              f"std {samples.std():.4f}  sat {100*saturation:.1f}%")
        return nn, float(np.mean(pairwise)), saturation

    print(f"\nOverfitting {args.images} images for {args.steps} steps "
          f"(disc {args.disc_steps} : gen {args.gen_steps}, lambda_gp "
          f"{args.disc_lambda_gp}, SN {args.spectral_norm})...")
    nn_start, _, _ = report(0)
    started = time.time()

    # Track the BEST nn_dist, not the last. At step 0 the generator emits a
    # near-zero frame, so nn_dist starts at roughly mean|real| and legitimately
    # RISES while the generator expands its output range before it starts
    # matching anything. Gating on the final value alone calls a healthy early
    # trajectory a failure.
    best_nn = nn_start
    last = (nn_start, 0.0, 0.0)
    for step in range(1, args.steps + 1):
        trainer.train_step(tf.convert_to_tensor(trainer.dataset.next_batch(args.batch_size)))
        if step % args.report_every == 0:
            last = report(step)
            best_nn = min(best_nn, last[0])

    elapsed = time.time() - started
    nn_end, diversity_end, saturation_end = last

    print(f"\n  {args.steps} steps in {elapsed/60:.1f} min "
          f"({elapsed/args.steps:.2f}s/step)")
    print(f"  nn_dist  start {nn_start:.4f}  best {best_nn:.4f}  final {nn_end:.4f}\n")
    print("  verdict")
    if diversity_end < 1e-4:
        print("    FAIL   - mode collapse: the generator ignores z even on 8 images.")
        print("             The code cannot train this architecture. Bisect fde5671..HEAD.")
        return 1
    if saturation_end > 0.5:
        print(f"    FAIL   - tanh-rail collapse: {100*saturation_end:.0f}% of output pinned.")
        print("             Output range ran away before the model fit anything.")
        return 1
    if best_nn < nn_start * 0.9:
        print(f"    PASS   - fitting ({nn_start:.4f} -> {best_nn:.4f}) and diverse "
              f"({diversity_end:.4f}).")
        print("             The implementation trains. Phase 1a's outcome is about the recipe.")
        return 0
    if best_nn < nn_start:
        print(f"    INCONCLUSIVE - moving in the right direction ({nn_start:.4f} -> "
              f"{best_nn:.4f}) but not far enough to call it.")
        print("             Re-run with more --steps before reading anything into it.")
        return 2
    print(f"    FAIL   - no progress: nn_dist never improved on {nn_start:.4f}.")
    print("             The code cannot train this architecture. Bisect fde5671..HEAD.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
