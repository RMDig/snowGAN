#!/usr/bin/env python3
"""Isolation bench for the native CPU-RAM leak localized to the disc loop.

memtrace localized a ~+3.7 MiB/batch native (glibc live-arena) leak to the
discriminator training loop on the real run. This bench reproduces ONLY the
disc forward/backward — the real Discriminator, eager, on the GPU — so we can
A/B spectral normalization in isolation and watch glibc uordblks (live arena).

    # leading hypothesis: SpectralNormalization's per-forward assign-in-tape
    python scripts/bench_sn_leak.py --sn     --steps 400
    python scripts/bench_sn_leak.py --no-sn  --steps 400

Compare the "uord" slope between the two runs. If --sn climbs and --no-sn is
flat, the Keras SpectralNormalization eager update is the leak. Run in the WSL
ml-env (GPU) — the leak is host-side allocation during GPU op dispatch and may
not show CPU-only. Pure native counters, no memtrace file written.
"""
import argparse
import types

import tensorflow as tf

from snowgan.models.discriminator import Discriminator
from snowgan.memtrace import _mallinfo2, _rss_mib


def _disc_config(sn, resolution, channels=3, depth=1):
    # Minimal config object the Discriminator builder reads (mirrors the real
    # core run: filter_counts, kernel 3 / stride 2, SN toggle).
    return types.SimpleNamespace(
        spectral_norm=sn,
        resolution=[resolution, resolution],
        channels=channels,
        depth=depth,
        filter_counts=[64, 128, 256, 512, 1024],
        kernel_size=[3, 3],
        kernel_stride=[2, 2],
        padding="same",
        negative_slope=0.25,
        final_activation="tanh",
        learning_rate=1e-4,
        beta_1=0.5,
        beta_2=0.9,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sn", dest="sn", action="store_true", help="spectral_norm ON")
    g.add_argument("--no-sn", dest="sn", action="store_false", help="spectral_norm OFF")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--report", type=int, default=50, help="print every N steps")
    args = ap.parse_args(argv)

    cfg = _disc_config(args.sn, args.resolution)
    disc = Discriminator(cfg)
    shape = (args.batch, cfg.depth, args.resolution, args.resolution, cfg.channels)
    print(f"BENCH spectral_norm={args.sn} steps={args.steps} batch={args.batch} "
          f"res={args.resolution} | uord = glibc live-arena MiB", flush=True)

    base_uord = None
    for step in range(args.steps + 1):
        real = tf.random.normal(shape)
        fake = tf.random.normal(shape)
        with tf.GradientTape() as tape:
            out_real = disc.model(real, training=True)
            out_fake = disc.model(fake, training=True)
            # Plain Wasserstein critic loss (no GP — matches lambda_gp=0 run).
            loss = tf.reduce_mean(out_fake) - tf.reduce_mean(out_real)
        grads = tape.gradient(loss, disc.model.trainable_variables)
        disc.optimizer.apply_gradients(zip(grads, disc.model.trainable_variables))
        del tape, grads, out_real, out_fake, loss, real, fake

        if step % args.report == 0:
            mi = _mallinfo2() or {}
            uord = mi.get("uordblks_mib")
            if base_uord is None and uord is not None:
                base_uord = uord
            delta = (uord - base_uord) if (uord is not None and base_uord is not None) else float("nan")
            per_step = (delta / step) if step else 0.0
            print(f"  step={step:>5} rss={_rss_mib():>7.0f} uord={uord if uord is None else round(uord,1)} "
                  f"delta={delta:+.1f}MiB per_step={per_step:+.3f}MiB", flush=True)


if __name__ == "__main__":
    main()
