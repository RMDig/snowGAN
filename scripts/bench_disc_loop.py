#!/usr/bin/env python3
"""Isolation bench that bisects the native CPU-RAM leak in the disc loop.

memtrace localized a ~+3.7 MiB/batch native (glibc live-arena) leak to the
discriminator training loop on the real run. A first bench proved the disc
forward/backward is CLEAN in isolation (SN on or off) -- so the leak is in what
ELSE runs per disc iteration: the GENERATOR forward (to make the fakes) and/or
diff_augment. This bench mirrors the real disc inner loop and lets you toggle
each component so glibc uordblks (live arena) names the leaker.

    python scripts/bench_disc_loop.py --full                 # gen + augment + sn
    python scripts/bench_disc_loop.py --gen --no-augment     # generator only
    python scripts/bench_disc_loop.py --no-gen --augment     # augment only
    python scripts/bench_disc_loop.py --no-gen --no-augment  # disc only (baseline; flat)

Watch the uord per_step slope. Whichever toggle makes it climb is the leak.
Run in the WSL ml-env (GPU): the leak is host-side allocation during GPU op
dispatch and won't show CPU-only. Pure native counters, no memtrace file.
"""
import argparse
import types

import numpy as np
import tensorflow as tf

from snowgan.models.discriminator import Discriminator
from snowgan.models.generator import Generator
from snowgan.augment import augment as diff_augment
from snowgan.memtrace import _mallinfo2, _rss_mib


def _disc_config(sn, resolution, channels=3, depth=1):
    return types.SimpleNamespace(
        spectral_norm=sn, resolution=[resolution, resolution], channels=channels,
        depth=depth, filter_counts=[64, 128, 256, 512, 1024], kernel_size=[3, 3],
        kernel_stride=[2, 2], padding="same", negative_slope=0.25,
        final_activation="tanh", learning_rate=1e-4, beta_1=0.5, beta_2=0.9,
    )


def _gen_config(resolution, channels=3, depth=1, latent_dim=100):
    # Mirrors the real core generator: descending filters -> 16*2^6 = 1024,
    # PixelNorm, kernel 3 / stride 2 (resize-conv upsampler builds from these).
    return types.SimpleNamespace(
        resolution=[resolution, resolution], channels=channels, depth=depth,
        latent_dim=latent_dim, filter_counts=[1024, 512, 256, 128, 64],
        kernel_size=[3, 3], kernel_stride=[2, 2], padding="same",
        batch_norm=False, gen_norm="pixel", negative_slope=0.25,
        final_activation="tanh", learning_rate=1e-4, beta_1=0.5, beta_2=0.9,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="shorthand for --gen --augment --sn")
    ap.add_argument("--gen", dest="gen", action="store_true", default=None, help="generate fakes via the real Generator")
    ap.add_argument("--no-gen", dest="gen", action="store_false")
    ap.add_argument("--augment", dest="augment", action="store_true", default=None, help="apply diff_augment to real+fake")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--sn", dest="sn", action="store_true", default=True, help="spectral_norm on the critic")
    ap.add_argument("--no-sn", dest="sn", action="store_false")
    ap.add_argument("--numpy-real", action="store_true",
                    help="feed the real images as a NumPy array (like next_batch's np.stack) "
                         "instead of a tf.Tensor -- reproduces the real disc loop's input")
    ap.add_argument("--train-gen", action="store_true",
                    help="also run a full generator training step (tracked gen forward -> disc "
                         "-> gen backprop -> apply) each iteration, as the real train_step does")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--report", type=int, default=50)
    args = ap.parse_args(argv)

    use_gen = True if (args.full or args.train_gen) else (args.gen if args.gen is not None else False)
    use_aug = True if args.full else (args.augment if args.augment is not None else False)

    disc = Discriminator(_disc_config(args.sn, args.resolution))
    gen = Generator(_gen_config(args.resolution)) if use_gen else None
    img_shape = (args.batch, 1, args.resolution, args.resolution, 3)
    latent = 100
    print(f"BENCH gen={use_gen} augment={use_aug} sn={args.sn} steps={args.steps} "
          f"batch={args.batch} res={args.resolution} | uord = glibc live-arena MiB", flush=True)

    base = None
    for step in range(args.steps + 1):
        if args.numpy_real:
            real = np.random.randn(*img_shape).astype("float32")  # NumPy, like next_batch
        else:
            real = tf.random.normal(img_shape)
        if use_gen:
            noise = tf.random.normal([args.batch, latent])
            fake = tf.stop_gradient(gen(noise, training=True))
        else:
            fake = tf.random.normal(img_shape)
        if use_aug:
            real = diff_augment(real, p=0.5)
            fake = diff_augment(fake, p=0.5)
        with tf.GradientTape() as tape:
            out_real = disc.model(real, training=True)
            out_fake = disc.model(fake, training=True)
            loss = tf.reduce_mean(out_fake) - tf.reduce_mean(out_real)
        grads = tape.gradient(loss, disc.model.trainable_variables)
        disc.optimizer.apply_gradients(zip(grads, disc.model.trainable_variables))
        del tape, grads, out_real, out_fake, loss, real, fake

        # Generator training step: gen forward TRACKED by the tape, through the
        # disc, then backprop through the whole generator + apply. The real
        # train_step does this every batch; no prior bench did (the disc bench
        # only ran the disc update with stop_gradient'd fakes).
        if args.train_gen and gen is not None:
            gnoise = tf.random.normal([args.batch, latent])
            with tf.GradientTape() as gtape:
                synth = gen(gnoise, training=True)
                sout = disc.model(synth, training=True)
                gloss = -tf.reduce_mean(sout)
            gvars = gen.model.trainable_variables
            ggrads = gtape.gradient(gloss, gvars)
            gen.optimizer.apply_gradients(zip(ggrads, gvars))
            del gtape, ggrads, gvars, synth, sout, gloss, gnoise

        if step % args.report == 0:
            mi = _mallinfo2() or {}
            uord = mi.get("uordblks_mib")
            if base is None and uord is not None:
                base = uord
            delta = (uord - base) if (uord is not None and base is not None) else float("nan")
            per = (delta / step) if step else 0.0
            print(f"  step={step:>5} rss={_rss_mib():>7.0f} uord={uord if uord is None else round(uord, 1)} "
                  f"delta={delta:+.1f}MiB per_step={per:+.3f}MiB", flush=True)


if __name__ == "__main__":
    main()
