#!/usr/bin/env python3
"""Build a local, downscaled image mirror for `--image_root`.

Why this exists: the HF manifest's `image` column is URL-backed, so
`dataset['train'][i]['image']` issues an HTTP GET for a ~16 MB PNG on *every*
access — once per image per epoch. Measured 2.03 s/image, which made the data
pipeline ~97% of a 1024 px train step against 0.54 s of GPU compute. A local
mirror takes that to 0.007 s/image. See docs/UPGRADES.md #60.

The mirror keeps the manifest's directory layout, so `--image_root <dir>`
resolves `<dir>/<file_path>` directly:

    <out>/preprocessed/magnified_profiles/image_47.png

Sources each image from the local full-resolution clone when present and
downloads it otherwise, so a partial clone costs bandwidth, not correctness.

Aspect ratio is preserved (short side = --size), matching the existing
~/rmdig-cache-512 convention. The trainer squashes to a square itself; going
via an aspect-preserving intermediate that is >= the target in both dimensions
means no upsampling happens anywhere in the chain.

Usage:
    python scripts/build_image_cache.py --size 1024 --modality magnified_profile
    python scripts/build_image_cache.py --size 1024 --modality all --workers 16

Resume-safe: existing outputs of the right size are skipped, so an interrupted
build can simply be re-run.
"""

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from PIL import Image  # noqa: E402

_BASE_URL = "https://huggingface.co/datasets/RMDig/rocky_mountain_snowpack/resolve/main/"
_MODALITIES = {"core": 0, "profile": 1, "magnified_profile": 2, "crystal_card": 3}


def _resize_short_side(image, size):
    """Downscale so the SHORT side is `size`, preserving aspect. Never upscales."""
    width, height = image.size
    short = min(width, height)
    if short <= size:
        return image.convert("RGB")
    scale = size / float(short)
    return image.convert("RGB").resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)


def build(args):
    from datasets import load_dataset

    print("Loading manifest...")
    dataset = load_dataset(args.dataset)["train"]
    frame = dataset.to_pandas().drop(columns=["image", "audio"], errors="ignore")
    if "file_path" not in frame.columns:
        raise SystemExit("Manifest has no file_path column; cannot build a keyed mirror.")

    if args.modality == "all":
        wanted = set(_MODALITIES.values())
    else:
        wanted = {_MODALITIES[args.modality]}
    rows = [frame["file_path"][i] for i, d in enumerate(frame["datatype"]) if d in wanted]
    print(f"{len(rows)} images for modality={args.modality}")

    out_root = os.path.expanduser(args.out)
    local_root = os.path.expanduser(args.local_source) if args.local_source else None

    todo, skipped = [], 0
    for relative in rows:
        destination = os.path.join(out_root, relative)
        if os.path.exists(destination) and not args.force:
            skipped += 1
            continue
        todo.append(relative)
    print(f"{skipped} already present, {len(todo)} to build")
    if not todo:
        return 0

    import httpx

    counters = {"local": 0, "downloaded": 0, "failed": 0, "bytes": 0}
    started = time.time()

    def one(relative):
        destination = os.path.join(out_root, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        source = os.path.join(local_root, relative) if local_root else None
        if source and os.path.exists(source):
            with Image.open(source) as handle:
                _resize_short_side(handle, args.size).save(destination, optimize=False)
            return "local", 0

        url = _BASE_URL + relative.replace(os.sep, "/")
        last_error = None
        for attempt in range(args.retries):
            try:
                response = httpx.get(url, timeout=args.timeout, follow_redirects=True)
                response.raise_for_status()
                payload = response.content
                with Image.open(io.BytesIO(payload)) as handle:
                    _resize_short_side(handle, args.size).save(destination, optimize=False)
                return "downloaded", len(payload)
            except Exception as error:  # network flakiness is expected at this volume
                last_error = error
                time.sleep(min(2 ** attempt, 10))
        # Leave no partial file behind for a resume to mistake for a good one.
        if os.path.exists(destination):
            try:
                os.remove(destination)
            except OSError:
                pass
        raise RuntimeError(f"{relative}: {last_error}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, relative): relative for relative in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                kind, size = future.result()
                counters[kind] += 1
                counters["bytes"] += size
            except Exception as error:
                counters["failed"] += 1
                print(f"  FAILED {error}")
            if index % 50 == 0 or index == len(todo):
                elapsed = time.time() - started
                rate = index / elapsed
                remaining = (len(todo) - index) / rate if rate else 0
                print(f"  {index}/{len(todo)}  local={counters['local']} "
                      f"dl={counters['downloaded']} fail={counters['failed']}  "
                      f"{rate:.1f} img/s  eta {remaining/60:.1f} min", flush=True)

    elapsed = time.time() - started
    print(f"\nDone in {elapsed/60:.1f} min: {counters['local']} from local clone, "
          f"{counters['downloaded']} downloaded ({counters['bytes']/1e9:.1f} GB), "
          f"{counters['failed']} failed")
    print(f"Mirror at {out_root}")
    print(f"Use it with:  --image_root {out_root}")
    return 1 if counters["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=1024, help="short side, px (default 1024)")
    parser.add_argument("--modality", default="magnified_profile",
                        choices=sorted(_MODALITIES) + ["all"])
    parser.add_argument("--out", default=None,
                        help="output root (default ~/rmdig-cache-<size>)")
    parser.add_argument("--local_source", default="/mnt/d/GitSpot/snowGAN/data/rmsnow",
                        help="local full-res clone to prefer over downloading")
    parser.add_argument("--dataset", default="rmdig/rocky_mountain_snowpack")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--force", action="store_true", help="rebuild existing entries")
    args = parser.parse_args()
    if args.out is None:
        args.out = f"~/rmdig-cache-{args.size}"
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
