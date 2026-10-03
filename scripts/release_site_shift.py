"""How much of the magnified-profile release's KID (0.386 / 0.338 EMA vs sites
3-6) is site difference rather than the generator? Written 2026-10-03, before
running. Descriptive, no pass/fail: the pre-registered verdict stands either way.

  A. KID(real sites 0-2, real sites 3-6): the site shift alone, no generator.
  B. KID(generator, real sites 0-2): fit to its own training sites. These are
     training images, so B measures fit, not generalisation.

Reading, fixed in advance: B near 0.38 -> the gap is the generator.
A near 0.38 with B small -> the gap is mostly site shift.
Same estimator, subsets (10 x 100), seed and image path as kid_check.py.

Run from the snowGAN-preview worktree:
    PYTHONPATH=src python /mnt/d/Models/snowGAN/release_scoring/site_shift.py
"""

import glob
import importlib.util
import os
import sys

import numpy as np

WT = "/mnt/d/GitSpot/snowGAN-preview"
spec = importlib.util.spec_from_file_location("kid_check", f"{WT}/scripts/kid_check.py")
kc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kc)

from PIL import Image  # noqa: E402
from datasets import load_dataset  # noqa: E402
from tensorflow.keras.applications.inception_v3 import InceptionV3  # noqa: E402
from snowgan.config import build  # noqa: E402
from snowgan.kid import inception_features, kid_score  # noqa: E402

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--RMDig--snowGAN-magnified-profile/snapshots/dc935c1*/"))[0]
ROOT = os.path.expanduser("~/rmdig-cache-1024")
KW = dict(subsets=10, subset_size=100, seed=0)


def site_features(frame, sites, inception, height=1024, width=1024, chunk=32):
    rows = [(i, k) for i, k in kc._select_rows(frame, 2, sites=sites)
            if os.path.exists(os.path.join(ROOT, str(frame["file_path"][i])))]
    feats = []
    for start in range(0, len(rows), chunk):
        imgs = []
        for i, _ in rows[start:start + chunk]:
            with Image.open(os.path.join(ROOT, str(frame["file_path"][i]))) as h:
                imgs.append(np.asarray(h.convert("RGB").resize((width, height), Image.BILINEAR),
                                       dtype=np.float32) / 127.5 - 1.0)
        feats.append(inception_features(np.stack(imgs)[:, None, ...], model=inception))
    return np.concatenate(feats), len({k for _, k in rows})


def main():
    frame = load_dataset("rmdig/rocky_mountain_snowpack")["train"].to_pandas().drop(
        columns=["image", "audio"], errors="ignore")
    inception = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))
    f_old, g_old = site_features(frame, {0, 1, 2}, inception)
    f_new, g_new = site_features(frame, {3, 4, 5, 6}, inception)
    print(f"sites 0-2: {len(f_old)} images, {g_old} groups; sites 3-6: {len(f_new)} images, {g_new} groups")

    a = kid_score(f_old, f_new, **KW)
    print(f"A  KID(real 0-2, real 3-6)       {a['kid_mean']:.5f} +/- {a['kid_se']:.5f}")
    for weights in ("generator.weights.h5", "generator_ema.weights.h5"):
        fakes, _ = kc._generate(SNAP, {}, 200, 0, weights_file=weights)
        f_gen = inception_features(fakes, model=inception)
        b = kid_score(f_old, f_gen, **KW)
        c = kid_score(f_new, f_gen, **KW)
        print(f"B  KID(gen, real 0-2)  {weights:26s} {b['kid_mean']:.5f} +/- {b['kid_se']:.5f}"
              f"   (vs 3-6, re-check: {c['kid_mean']:.5f} +/- {c['kid_se']:.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
