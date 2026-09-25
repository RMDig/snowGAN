#!/usr/bin/env python3
"""Measure real `Trainer.train_step` throughput on a device.

Exists because the plan's cost estimates (and therefore its gate placement)
were fitted from historical checkpoint mtimes, and because this host JIT-compiles
CUDA kernels from PTX for its Blackwell GPU (sm_120a) — so "does the GPU help?"
is an empirical question here, not an assumption. See docs/UPGRADES.md #45.

Uses synthetic batches so it measures compute, not the ~12 minutes the HF
manifest load costs at startup.

Usage:
    python scripts/bench_train_step.py --device gpu --resolution 1024 --disc_steps 2
    python scripts/bench_train_step.py --device cpu --resolution 64
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--disc_steps", type=int, default=2)
    parser.add_argument("--gen_steps", type=int, default=3)
    parser.add_argument("--disc_lambda_gp", type=float, default=10.0)
    parser.add_argument("--spectral_norm", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--warmup", type=int, default=3,
                        help="steps excluded from timing (PTX JIT + autotune land here)")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    # Must precede the tensorflow import: TF caches device config at import.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    blocks = {64: 1, 128: 2, 256: 3, 512: 4, 1024: 5}.get(args.resolution)
    if blocks is None:
        raise SystemExit("--resolution must be one of 64/128/256/512/1024 (16*2^(N+1))")

    import tensorflow as tf
    from snowgan.config import build, config_template
    from snowgan.models.generator import Generator
    from snowgan.models.discriminator import Discriminator

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"device requested: {args.device}   TF sees GPUs: {[g.name for g in gpus]}")

    save_dir = os.path.join("keras", "snowgan", "_bench")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    gen_filters = [1024 // (2 ** i) for i in range(blocks)]

    def make_config(architecture, filters, lr, steps):
        data = dict(config_template)
        data.update({
            "architecture": architecture, "save_dir": save_dir + "/", "checkpoint": None,
            "resolution": [args.resolution, args.resolution], "filter_counts": filters,
            "latent_dim": 100, "batch_size": args.batch_size, "depth": 1,
            "modality": "magnified_profile", "learning_rate": lr, "training_steps": steps,
            "lambda_gp": args.disc_lambda_gp, "spectral_norm": args.spectral_norm,
            "gen_norm": "none", "gen_upsampler": "transpose", "gen_convs_per_resolution": 1,
            "fade": False, "augment": False, "adaptive_steps": False,
            "multiscale_disc": False, "ada_target": 0.0, "grad_clip_norm": 0.0,
            "ema_decay": 0.0, "lr_decay": None, "fid_interval": 0,
            "grad_probe_interval": 0,  # off: benchmark the train step, not the instrument
        })
        path = os.path.join(save_dir, f"{architecture}_config.json")
        with open(path, "w") as handle:
            json.dump(data, handle)
        return build(path)

    gen_config = make_config("generator", gen_filters, 1e-4, args.gen_steps)
    disc_config = make_config("discriminator", list(reversed(gen_filters)), 1e-5, args.disc_steps)

    class _Synthetic:
        def __init__(self, config):
            self.config = config
            self.pair_depth = 1
            self.seen_profiles = set()

        def next_batch(self, batch_size):
            return tf.random.uniform([batch_size, 1, args.resolution, args.resolution, 3],
                                     -1, 1).numpy()

        def batch(self, batch_size, datatype=None, config=None):
            return self.next_batch(batch_size)

        def reset_seen_profiles(self):
            pass

        def derive_splits(self, *a, **k):
            self.config.trained_pool = self.config.validation_pool = self.config.test_pool = []

    import importlib

    trainer_module = importlib.import_module("snowgan.trainer")
    original = trainer_module.DataManager
    trainer_module.DataManager = lambda config: _Synthetic(config)
    try:
        generator = Generator(gen_config)
        generator.model.build((None, gen_config.latent_dim))
        discriminator = Discriminator(disc_config)
        discriminator.model.build((None, 1, args.resolution, args.resolution, 3))
        trainer = trainer_module.Trainer(generator, discriminator)
    finally:
        trainer_module.DataManager = original

    gen_params = generator.model.count_params()
    disc_params = discriminator.model.count_params()
    print(f"generator {gen_params/1e6:.1f}M params   critic {disc_params/1e6:.1f}M params")

    batch = tf.random.uniform([args.batch_size, 1, args.resolution, args.resolution, 3], -1, 1)

    print(f"\nwarmup {args.warmup} steps (PTX JIT + cuDNN autotune land here)...")
    warm_started = time.time()
    for _ in range(args.warmup):
        trainer.train_step(batch)
    print(f"  warmup took {time.time() - warm_started:.1f}s")

    print(f"timing {args.steps} steps...")
    started = time.time()
    for _ in range(args.steps):
        trainer.train_step(batch)
    elapsed = time.time() - started

    per_step = elapsed / args.steps
    print(f"\n  {per_step:.3f} s/step   ({args.steps} steps in {elapsed:.1f}s)")
    print(f"  disc {args.disc_steps} : gen {args.gen_steps} at {args.resolution}px, "
          f"batch {args.batch_size}, lambda_gp {args.disc_lambda_gp}, SN {args.spectral_norm}")
    for gate, updates in (("4k critic updates", 4000), ("20k critic updates", 20000)):
        steps_needed = updates / max(1, args.disc_steps)
        print(f"  {gate:>20}: {steps_needed:>8.0f} steps = {steps_needed*per_step/3600:6.2f} h")

    shutil.rmtree(save_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
