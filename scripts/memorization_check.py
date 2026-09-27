#!/usr/bin/env python3
"""Is the generator reproducing training images, or modelling the distribution?

The CLAUDE.md §9 scoreboard's cheap tier (latent diversity, saturation) is a
property of the generator *in isolation*: a model that perfectly memorized 2,000
images passes all of it. This is the check that separates the two, and it is the
one that decides whether more training or more data is the right next spend.

Method (the standard GAN memorization probe): for each generated sample, find
its nearest training image and record that distance. The number is meaningless
alone — what matters is the comparison to the **real-to-real** nearest-neighbour
distance over the same set. If generated samples sit *closer* to the training
set than training images sit to each other, the model is reproducing rather
than generalizing.

    gen->real NN  ~=  real->real NN     healthy: samples are as novel as the
                                        data is diverse
    gen->real NN  <<  real->real NN     memorizing
    gen->real NN  >>  real->real NN     off-manifold: not modelling the data

Distances are computed at a reduced resolution (default 128 px) because the
question is "is this the same snow sample", not "is this pixel-identical" — and
a full 1024 px all-pairs comparison over thousands of images is needlessly
expensive for that.

Usage:
    python scripts/memorization_check.py keras/snowgan/control_1024/ \
        --image_root ~/rmdig-cache-1024 --n 16
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402


def _load_real_corpus(config, image_root, probe_size, limit):
    """Downscaled training images as a single array, plus their identities."""
    from PIL import Image
    from datasets import load_dataset

    dataset = load_dataset(getattr(config, "dataset", "rmdig/rocky_mountain_snowpack"))["train"]
    frame = dataset.to_pandas().drop(columns=["image", "audio"], errors="ignore")

    from snowgan.data.dataset import normalize_datatype

    wanted = normalize_datatype(getattr(config, "modality", "magnified_profile"))
    rows = [(i, frame["file_path"][i]) for i, d in enumerate(frame["datatype"])
            if normalize_datatype(d) == wanted]

    # Honour the split the run honoured: comparing against groups the model was
    # never shown would understate memorization.
    held_out = set()
    for pool in ("validation_pool", "test_pool"):
        for entry in (getattr(config, pool, None) or []):
            held_out.add(tuple(entry))
    if held_out and getattr(config, "honor_splits", True):
        keys = {i: (frame["site"][i], frame["column"][i], frame["core"][i])
                for i, _ in rows}
        rows = [(i, fp) for i, fp in rows if keys[i] not in held_out]

    if limit:
        rows = rows[:limit]

    images, ids = [], []
    root = os.path.expanduser(image_root) if image_root else None
    for index, relative in rows:
        path = os.path.join(root, str(relative)) if root else None
        if not path or not os.path.exists(path):
            continue
        with Image.open(path) as handle:
            arr = np.asarray(
                handle.convert("RGB").resize((probe_size, probe_size), Image.BILINEAR),
                dtype=np.float32)
        images.append(arr / 127.5 - 1.0)
        ids.append(relative)
    if not images:
        raise SystemExit("No local training images resolved; pass --image_root.")
    return np.stack(images), ids


def _nearest(query, corpus):
    """(min distance, argmin) per query row, mean-absolute-difference metric."""
    flat_q = query.reshape(len(query), -1)
    flat_c = corpus.reshape(len(corpus), -1)
    best_d = np.full(len(flat_q), np.inf, dtype=np.float32)
    best_i = np.zeros(len(flat_q), dtype=np.int64)
    for start in range(0, len(flat_c), 256):          # chunked: corpus can be large
        block = flat_c[start:start + 256]
        d = np.abs(flat_q[:, None, :] - block[None, :, :]).mean(axis=2)
        idx = d.argmin(axis=1)
        val = d[np.arange(len(d)), idx]
        better = val < best_d
        best_d[better] = val[better]
        best_i[better] = start + idx[better]
    return best_d, best_i


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("save_dir")
    parser.add_argument("--image_root", default="~/rmdig-cache-1024")
    parser.add_argument("--n", type=int, default=16, help="generated samples")
    parser.add_argument("--probe_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="cap corpus size (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gen_upsampler", default=None, choices=["resize", "transpose"])
    parser.add_argument("--gen_convs_per_resolution", type=int, default=None, choices=[1, 2])
    parser.add_argument("--gen_norm", default=None, choices=["pixel", "batch", "none"])
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from PIL import Image
    from snowgan.config import build
    from snowgan.models.generator import Generator
    from snowgan.checkpoint import resolve_weights_path

    # Read the config from a copy: loading a save_dir must not rewrite it.
    scratch = "/tmp/_memcheck_cfg"
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)
    shutil.copy(os.path.join(args.save_dir, "generator_config.json"),
                os.path.join(scratch, "generator_config.json"))
    config = build(os.path.join(scratch, "generator_config.json"))
    for key, value in (("gen_upsampler", args.gen_upsampler),
                       ("gen_convs_per_resolution", args.gen_convs_per_resolution),
                       ("gen_norm", args.gen_norm)):
        if value is not None:
            setattr(config, key, value)

    tf.keras.utils.set_random_seed(args.seed)
    generator = Generator(config)
    generator.model.build((None, config.latent_dim))
    weights = resolve_weights_path(os.path.join(args.save_dir, "generator.weights.h5"))
    generator.model.load_weights(weights)
    print(f"Loaded {weights}")

    print(f"Loading training corpus at {args.probe_size}px...")
    corpus, ids = _load_real_corpus(config, args.image_root, args.probe_size, args.limit)
    print(f"  {len(corpus)} training images (splits honoured: "
          f"{getattr(config, 'honor_splits', True)})")

    samples = []
    for i in range(args.n):
        z = tf.random.normal([1, config.latent_dim])
        img = generator.model(z, training=False).numpy()[0]
        if img.ndim == 4:
            img = img[0]
        pil = Image.fromarray(np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8))
        samples.append(
            np.asarray(pil.resize((args.probe_size, args.probe_size), Image.BILINEAR),
                       dtype=np.float32) / 127.5 - 1.0)
    samples = np.stack(samples)

    gen_d, gen_i = _nearest(samples, corpus)

    # Real-to-real baseline: each training image's nearest OTHER training image.
    # This is the reference the generated distance only means something against.
    rng = np.random.default_rng(args.seed)
    subset = rng.choice(len(corpus), size=min(200, len(corpus)), replace=False)
    real_d = []
    for i in subset:
        others = np.delete(np.arange(len(corpus)), i)
        d, _ = _nearest(corpus[i:i + 1], corpus[others])
        real_d.append(d[0])
    real_d = np.array(real_d)

    print(f"\n  generated -> nearest training image")
    print(f"    mean {gen_d.mean():.4f}   min {gen_d.min():.4f}   max {gen_d.max():.4f}")
    print(f"  training  -> nearest OTHER training image  (the reference)")
    print(f"    mean {real_d.mean():.4f}   min {real_d.min():.4f}   p5 {np.percentile(real_d,5):.4f}")

    ratio = gen_d.mean() / real_d.mean()
    print(f"\n  ratio gen/real = {ratio:.3f}")
    distinct = len(set(gen_i.tolist()))
    print(f"  {distinct}/{len(gen_i)} samples have distinct nearest neighbours")

    print("\n  verdict")
    if gen_d.min() < np.percentile(real_d, 5) * 0.5:
        print("    MEMORIZING - at least one sample is far closer to a training image")
        print("                 than training images are to each other.")
    elif ratio < 0.8:
        print("    SUSPECT    - samples sit systematically closer to the training set")
        print("                 than the training set sits to itself. More data or")
        print("                 augmentation before more steps.")
    elif ratio > 1.5:
        print("    OFF-MANIFOLD - samples are far from any training image; the model")
        print("                   is not reproducing the data distribution.")
    else:
        print("    HEALTHY    - samples are about as far from the training set as")
        print("                 training images are from each other: modelling, not")
        print("                 reproducing.")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
