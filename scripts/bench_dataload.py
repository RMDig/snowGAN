#!/usr/bin/env python3
"""Model-free isolation of the disc-loop native leak: the data pipeline.

The model benches (disc / generator / augment / full) were all flat -- the leak
is NOT in training. The only thing the real run does that those benches faked
with tf.random.normal is load real images via DataManager.next_batch, whose
hot line is `self.dataset['train'][idx]` (HF datasets Image-column indexed
access, known to accumulate decoded buffers in process RAM: huggingface/datasets
#4883 / #7180). This bench loops next_batch with NO model and watches glibc
uordblks (live arena).

    python scripts/bench_dataload.py --steps 200

If uord climbs ~one image's worth per batch, the data pipeline is the leak.
Run in the WSL ml-env. Pure native counters, no memtrace file.
"""
import argparse
import contextlib
import io
import os
import types

from snowgan.data.dataset import DataManager
from snowgan.memtrace import _mallinfo2, _rss_mib


def _config(resolution, modality):
    return types.SimpleNamespace(
        dataset="rmdig/rocky_mountain_snowpack",
        modality=modality,
        seen_profiles=[],
        resolution=[resolution, resolution],
        channels=3,
        train_ind=0,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--modality", default="core")
    ap.add_argument("--report", type=int, default=20)
    args = ap.parse_args(argv)

    dm = DataManager(_config(args.resolution, args.modality))
    print(f"BENCH dataload modality={args.modality} steps={args.steps} batch={args.batch} "
          f"res={args.resolution} | uord = glibc live-arena MiB", flush=True)

    base = None
    devnull = io.StringIO()
    for step in range(args.steps + 1):
        # Silence the loader's per-sample prints; we only want the uord trend.
        with contextlib.redirect_stdout(devnull):
            out = dm.next_batch(args.batch)
            if out is None:  # exhausted -> cycle so the same rows get re-accessed
                dm.config.train_ind = 0
                out = dm.next_batch(args.batch)
        del out
        devnull.truncate(0)
        devnull.seek(0)

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
