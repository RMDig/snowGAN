#!/usr/bin/env python3
"""Evaluate the Phase 1 gates against a run's ``metrics.jsonl``.

Reads only the run log — no TensorFlow, no weights — so it is instant and can
be run against a live training directory from another shell.

Usage:
    python scripts/check_gates.py keras/snowgan/exp0_control_1024/
    python scripts/check_gates.py <save_dir> --at 4000     # gate at N critic updates

Gates come from docs/plans/training_dynamics_recovery_plan.md §4. Three notes
on why they are shaped the way they are:

**Critic updates, not train steps.** A gate at "step 500" is 1,000 critic
updates at ``disc_steps 2`` and 23,500 at ``disc_steps 47``. Expressing gates
in train steps makes them incomparable across the very ratio sweep they are
supposed to govern.

**Last launch only.** ``global_step`` is persisted every 50 steps and rewinds
on restart; the ``batch`` counter rewinds further. Reading the whole file would
evaluate a gate on rows replayed after a restart.

**The Wasserstein ceiling is derived, not guessed.** A genuinely 1-Lipschitz
critic cannot separate two images in [-1, 1] by more than their L2 distance,
which is bounded by ``2*sqrt(d)`` for ``d = depth*H*W*C``. At 256x256x3 that is
887 — so the |W| ~ 440 that looked like a "runaway" in the failed runs is about
85% of what an *optimal* 1-Lipschitz critic would report. It is evidence the
generator is bad, not that the critic is unconstrained. The number that speaks
to the constraint is the gradient norm.
"""

import argparse
import importlib.util
import math
import os

# Load metrics.py directly rather than `from snowgan.metrics import ...`:
# snowgan/__init__.py imports Trainer, which imports TensorFlow, which costs
# ~40s of startup and defeats the point of a log-only checker you can run
# against a live training directory.
_METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "src", "snowgan", "metrics.py")
_spec = importlib.util.spec_from_file_location("_snowgan_metrics", _METRICS_PATH)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
read_last_launch = _metrics.read_last_launch


def _mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else float("nan")


def _fmt(value, width=10, places=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a".rjust(width)
    return f"{value:>{width}.{places}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("save_dir", help="run directory containing metrics.jsonl")
    parser.add_argument("--at", type=int, default=None,
                        help="evaluate using the last N critic updates (default: whole launch)")
    parser.add_argument("--window", type=int, default=200,
                        help="number of trailing records to average (default 200)")
    args = parser.parse_args()

    path = os.path.join(args.save_dir, "metrics.jsonl")
    rows = read_last_launch(path)
    if not rows:
        print(f"No records in {path}. Has the run started?")
        return 2

    start = next((r for r in rows if r.get("event") == "run_start"), None)
    steps = [r for r in rows if "event" not in r]
    if not steps:
        print("Run log has a start event but no train steps yet.")
        return 2

    if args.at is not None:
        steps = [r for r in steps if r.get("critic_updates", 0) <= args.at]
        if not steps:
            print(f"No records at or below {args.at} critic updates.")
            return 2

    window = steps[-args.window:]
    latest = steps[-1]

    print(f"\n=== {args.save_dir} ===")
    if start:
        print(f"recipe: disc {start.get('disc_steps')} : gen {start.get('gen_steps')}  |  "
              f"lambda_gp {start.get('lambda_gp')}  SN {start.get('spectral_norm')}  |  "
              f"clip {start.get('grad_clip_norm')}  adaptive {start.get('adaptive_steps')}  |  "
              f"splits honored {start.get('honor_splits')}  seed {start.get('seed')}")
        resolution = start.get("resolution")
    else:
        resolution = None

    # The ceiling is only meaningful at the run's real resolution (it scales
    # with sqrt(d)), so guess nothing — a defaulted 1024 would report a
    # 16x-too-large ceiling for a 256 run and pass a critic that should fail.
    if resolution:
        d = resolution[0] * resolution[1] * 3
        ceiling = 2.0 * math.sqrt(d)
    else:
        d = ceiling = None
        print("  (no run_start record - resolution unknown, Wasserstein gate skipped;\n"
              "   pass --resolution or run via the trainer's train() entry point)")

    print(f"global_step {latest.get('global_step')}   critic_updates {latest.get('critic_updates')}   "
          f"(averaging last {len(window)} records)\n")

    gn_interp = _mean([r.get("grad_norm_interp") for r in window])
    gn_real = _mean([r.get("grad_norm_real") for r in window])
    gn_fake = _mean([r.get("grad_norm_fake") for r in window])
    wass = _mean([r.get("wasserstein") for r in window])
    gp = _mean([r.get("gp_weighted") for r in window])
    gen_loss = _mean([r.get("gen_loss") for r in window])
    d_real = _mean([r.get("d_real") for r in window])
    d_fake = _mean([r.get("d_fake") for r in window])

    ceiling_note = f"(1-Lipschitz ceiling +/-{ceiling:.0f})" if ceiling else "(ceiling unknown)"
    print(f"  ||dD/dx|| interp {_fmt(gn_interp)}   real {_fmt(gn_real)}   fake {_fmt(gn_fake)}")
    print(f"  Wasserstein      {_fmt(wass)}   {ceiling_note}")
    print(f"  lambda*GP        {_fmt(gp)}   D(real) {_fmt(d_real)}   D(fake) {_fmt(d_fake)}")
    print(f"  gen_loss         {_fmt(gen_loss)}")
    gen_lr, disc_lr = latest.get("gen_lr"), latest.get("disc_lr")
    if gen_lr is not None and disc_lr is not None:
        print(f"  lr gen {gen_lr:.2e}  disc {disc_lr:.2e}\n")
    else:
        print()

    verdicts = []

    # Gate 1 -- does the Lipschitz constraint bind?
    probe = gn_real if not math.isnan(gn_real) else gn_interp
    if math.isnan(probe):
        verdicts.append(("Lipschitz", "NO DATA", "no gradient-norm probe recorded; "
                                                 "is --grad_probe_interval 0?"))
    elif 0.5 <= probe <= 2.0:
        verdicts.append(("Lipschitz", "PASS", f"||dD/dx|| = {probe:.3f}, constraint binds"))
    else:
        verdicts.append(("Lipschitz", "CHECK", f"||dD/dx|| = {probe:.3f} is outside [0.5, 2.0]. "
                                               f"Compare against the 0.6 reference before "
                                               f"calling this a kill -- the band is provisional."))

    # Gate 2 -- is the critic saturating its own 1-Lipschitz ceiling?
    if ceiling is None or math.isnan(wass):
        verdicts.append(("Wasserstein", "NO DATA", "resolution unknown" if ceiling is None else ""))
    elif abs(wass) > ceiling:
        verdicts.append(("Wasserstein", "FAIL", f"|W| = {abs(wass):.0f} exceeds the "
                                                f"1-Lipschitz ceiling {ceiling:.0f}: the critic "
                                                f"is provably not 1-Lipschitz."))
    elif abs(wass) > 0.8 * ceiling:
        verdicts.append(("Wasserstein", "CHECK", f"|W| = {abs(wass):.0f} is {100*abs(wass)/ceiling:.0f}% "
                                                 f"of ceiling -- near-optimal separation, i.e. the "
                                                 f"generator is still producing garbage."))
    else:
        verdicts.append(("Wasserstein", "PASS", f"|W| = {abs(wass):.0f}, "
                                                f"{100*abs(wass)/ceiling:.0f}% of ceiling"))

    # Gate 3 -- is the Wasserstein estimate converging?
    #
    # Deliberately NOT the |gen_loss|/|W| ratio the plan's first draft proposed.
    # gen_loss is -E[D(fake)], an absolute critic level, and that level is free
    # gauge: SpectralNormalization normalizes the kernel and leaves the bias
    # unconstrained, so adding a constant to the critic's output changes
    # gen_loss arbitrarily while changing nothing about training. Dividing a
    # gauge-dependent level by a gauge-invariant difference is meaningless --
    # observed live on a healthy 64x64 control, where the ratio read 19.2
    # precisely BECAUSE |W| had converged toward 0.
    #
    # |W| itself is gauge-invariant and is the actual progress signal: as the
    # generator improves, the critic's best separation of the two
    # distributions shrinks.
    half = max(1, len(window) // 2)
    early = _mean([abs(r.get("wasserstein", float("nan"))) for r in window[:half]])
    late = _mean([abs(r.get("wasserstein", float("nan"))) for r in window[half:]])
    if math.isnan(early) or math.isnan(late):
        verdicts.append(("Convergence", "NO DATA", ""))
    elif len(window) < 20:
        verdicts.append(("Convergence", "NO DATA", "need a longer window"))
    elif late <= early:
        verdicts.append(("Convergence", "PASS",
                         f"|W| {early:.2f} -> {late:.2f} across the window (shrinking)"))
    else:
        verdicts.append(("Convergence", "CHECK",
                         f"|W| {early:.2f} -> {late:.2f} (growing): the critic is pulling "
                         f"further ahead of the generator."))

    # Gate 4 -- is the penalty strong enough to matter?
    #
    # The share of the loss is NOT the test on its own: a well-constrained
    # critic sits near ||dD/dx|| == 1, where the penalty term is small by
    # construction precisely because it is working. Measured on a healthy
    # 64x64 control at lambda_gp 10: norm 1.30, GP only 2.7% of |loss|.
    # The failure mode is a small share AND a norm far from 1 -- that is a
    # penalty too weak to pull the critic back, which is what lambda_gp 1.0
    # against an O(400) Wasserstein term produces.
    if not math.isnan(gp) and not math.isnan(wass) and abs(wass) > 1e-9:
        share = abs(gp) / (abs(gp) + abs(wass))
        detail = f"lambda*GP is {100*share:.1f}% of |loss|"
        if math.isnan(probe):
            verdicts.append(("GP strength", "NO DATA", detail))
        elif abs(probe - 1.0) <= 0.5:
            verdicts.append(("GP strength", "PASS",
                             f"{detail}; small share is expected while ||dD/dx|| ~ 1"))
        elif share < 0.05:
            verdicts.append(("GP strength", "FAIL",
                             f"{detail} AND ||dD/dx|| = {probe:.2f} is off target -- the "
                             f"penalty is too weak to bind. Raise --disc_lambda_gp."))
        else:
            verdicts.append(("GP strength", "CHECK",
                             f"{detail}; penalty is substantial but ||dD/dx|| = {probe:.2f} "
                             f"has not converged."))

    width = max(len(name) for name, _, _ in verdicts)
    print("  gates")
    for name, status, detail in verdicts:
        print(f"    {name:<{width}}  {status:<8} {detail}")

    print("\n  Not checkable from the log -- run scripts/kill_check.py for these:")
    print("    latent diversity, output saturation, sample inspection, NN memorization check\n")

    return 0 if all(s == "PASS" for _, s, _ in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
