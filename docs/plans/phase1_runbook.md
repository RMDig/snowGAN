# Phase 1 runbook — the control run

Operational companion to
[training_dynamics_recovery_plan.md](training_dynamics_recovery_plan.md). That
document says *why*; this one says *what to type* and *how to read the answer*.

---

## 0. Environment

TensorFlow is **only** in the WSL venv:

```bash
/mnt/d/Envs/WSL/ml-env/bin/python        # Python 3.12.3, TF 2.21.0
```

The Windows Python has no TF. The repo is `/mnt/d/GitSpot/snowGAN` from WSL
(`D:\GitSpot\snowGAN` from Windows). The HF dataset is cached (~39 GB under
`~/.cache/huggingface`): 4,040 rows — 2,350 magnified_profile, 845 core, 845
profile. Do **not** set `HF_HUB_OFFLINE=1`; the cached repo id is case-mismatched
and offline resolution fails. Let the cache serve normally.

### GPU

The host has an **RTX 5080 (Blackwell, sm_120a)**, 16 GB, driver 591.59. TF 2.21
sees it but ships **no cubins for sm_120a**, so kernels JIT from PTX on first
use. The driver caches them in `~/.nv/ComputeCache` (already ~424 MB warm), which
is why warmup is seconds rather than the "30 minutes or longer" TF warns about.

**Two things must be set, and `train_exp0_control.sh` now sets both:**

```bash
export PATH="/usr/local/cuda-12.8/bin:$PATH"   # NOT the default /usr/bin/ptxas
export CUDA_CACHE_MAXSIZE=4294967296           # don't evict + re-JIT mid-run
```

`/usr/bin/ptxas` is **12.0.140** and wins on the default PATH; TF warns that
"CUDA versions 12.x.y up to and including 12.6.2 miscompile certain edge cases
around clamping." This architecture clamps on every step — the tanh head,
`clip_by_value` in `augment`, gradient clipping — so that is a correctness risk
in a run you intend to interpret, not a cosmetic warning.
`/usr/local/cuda-12.8/bin/ptxas` is 12.8.93. Measured cost of switching:
**none** (identical s/step).

Do **not** pass `CUDA_VISIBLE_DEVICES=-1` to training runs. It belongs on the
test suite (which is CPU-only by design and faster that way), and it is what
made the first 1z runs CPU-bound.

Measured throughput (`scripts/bench_train_step.py`, disc 2 : gen 3, batch 4):

| | 64 px | 1024 px |
|---|---|---|
| CPU | 0.66 s/step | — |
| GPU | **0.081 s/step** (8×) | **0.537 s/step** |

At 1024 px the models are 41.9 M (generator) and 7.3 M (critic) parameters, and
1024² fits comfortably in 16 GB at batch 4.

Suite (should be **222 passed**; keep this one on CPU):

```bash
cd /mnt/d/GitSpot/snowGAN
TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 \
  /mnt/d/Envs/WSL/ml-env/bin/python -m pytest tests/ -q
```

---

## 1. Run 1z first — the ten-minute test

```bash
/mnt/d/Envs/WSL/ml-env/bin/python scripts/overfit_sanity.py --steps 2500 --report_every 250
```

Can this code overfit a GAN to 8 fixed images at 64×64? It drives the real
`Trainer.train_step`, so it tests the code path the control will use.

| verdict | meaning | next |
|---|---|---|
| `PASS` | fitting and diverse | the implementation trains; run 1a |
| `INCONCLUSIVE` | improving but not far enough | re-run with more `--steps` |
| `FAIL` | collapse, tanh-rail, or no progress | **stop.** Bisect `fde5671..HEAD` (51 commits) using this script as the test |

Run the suspect recipe through it too — it is the cheapest A/B of the whole
hypothesis:

```bash
# the recipe every collapsed run used
/mnt/d/Envs/WSL/ml-env/bin/python scripts/overfit_sanity.py \
  --steps 2500 --spectral_norm --disc_lambda_gp 1.0
```

If the default (λ_gp 10, no SN) fits and this one collapses, the hypothesis is
supported at a cost of ~20 minutes instead of ~20 hours.

---

## 2. Run 1a — the control

```bash
scripts/train_exp0_control.sh
```

Override via env: `SAVE_DIR`, `MAX_STEPS`, `MAX_RSS_MB`. Extra flags are
forwarded (`scripts/train_exp0_control.sh --max_steps 500` for a smoke).

**What it is.** The recipe reconstructed from the `batch_*/` snapshots for the
era that actually trained the released backbone — `disc 2 : gen 3`, `λ_gp 10`,
**no spectral norm**, no clipping, no LR decay, no EMA, no ADA, no multiscale —
at 1024², from scratch. **Not** the released terminal config, which describes a
state reached ~130k steps later and entirely at `lr 1e-7`.

**What it decides.** λ_gp/SN separates every working run from every dead one,
but co-varies exactly with code era. This run separates those:

- **trains** → the recipe was the cause; the v0.2 audit's SN + λ_gp-1 default is
  the regression. Go to Phase 2A.
- **does not** → a code regression in the 51 commits since `fde5671`. Bisect.

**Cost — measured on this GPU, not extrapolated.** `scripts/bench_train_step.py
--device gpu --resolution 1024 --disc_steps 2` gives **0.537 s/step**, so:

| gate | train steps | wall clock |
|---|---|---|
| 4,000 critic updates | 2,000 | **~0.3 h** |
| 20,000 critic updates | 10,000 | **~1.5 h** |

An earlier draft of this runbook said ~19 hours, fitted from the released run's
checkpoint mtimes (`t ≈ 6.4 s + 0.40 s × disc_steps`). That fit is ~13× off
against a working GPU toolchain — whatever those historical runs were paying
(CPU fallback, PTX thrashing, the cuDNN issues in UPGRADES #45), it was not the
cost of the arithmetic. **Do not size future runs from those mtimes; benchmark.**

At ~0.5 s/step the CPU-RAM leak (~3.6 MiB/batch) also arrives ~13× sooner in
wall-clock: expect to hit a 24 GB ceiling around batch ~6,000, i.e. one or two
restarts across the run. That is what `--max_rss_mb` is for, and each restart
re-pays the ~10 min manifest load.

Run it twice at different seeds before concluding failure — at 1.5 h a seed
repeat is cheap, and a false negative costs a 51-commit bisect.

---

## 3. Reading it

From any shell, against a live directory:

```bash
# loss-side gates — no TF, ~0.3 s, safe to run from Windows
python scripts/check_gates.py keras/snowgan/exp0_control_1024/

# generator-side kill-check — needs TF
/mnt/d/Envs/WSL/ml-env/bin/python scripts/kill_check.py keras/snowgan/exp0_control_1024/
```

Gates, in critic updates (not train steps — a train-step gate is not comparable
across a `disc_steps` sweep):

| critic updates | gate | kill if |
|---|---|---|
| 1,000 | `‖dD/dx‖` near 1 | far off ⇒ the constraint is not binding |
| 4,000 | `\|W\|` well under `2·sqrt(d)`, and trending **down** | growing ⇒ the critic is pulling away from the generator |
| 4,000 | latent diversity > **0.049** | below ⇒ mode collapse |
| 20,000 | two latents give two different, non-saturated images with structure; NN check says they are not training images | no ⇒ dead or memorizing |

**Not** `|gen_loss| / |W|`, which an earlier draft of the plan proposed. `gen_loss`
is `−E[D(fake)]`, an absolute critic level, and that level is free gauge —
`SpectralNormalization` normalizes the kernel and leaves the bias unconstrained,
so a constant added to the critic's output changes `gen_loss` arbitrarily while
changing nothing about training. Observed live on a healthy 64² control: the
ratio read 19.2 precisely *because* `|W|` had converged toward 0. `|W|` is
gauge-invariant and is the real progress signal.

Calibration the thresholds come from (measured 2026-09-23, raw weights, n=6):

| run | std | saturated | latent diversity |
|---|---|---|---|
| `magnified_profiles` v0.1.0 (good) | 0.533 | 3.63% | **0.4898** |
| `exp0_magprof_control_256` (dead) | 0.841 | — | **0.000026** |
| `core_masked_256` (dead) | 1.000 | — | **0.000000** |

**Two numbers that are easy to misread.**

*A large `|W|` is not a runaway critic.* A 1-Lipschitz critic can separate two
images in [−1,1] by up to their L2 distance, bounded by `2·sqrt(depth·H·W·C)` —
887 at 256², 3547 at 1024². The `|W| ≈ 440` that looked alarming in the failed
runs is ~85% of what an *optimal* 1-Lipschitz critic would report: evidence the
generator is bad, not that the critic is unconstrained. `check_gates.py` prints
the ceiling alongside.

*A small `λ·GP` share is not a weak penalty.* A well-constrained critic sits near
`‖dD/dx‖ = 1`, where the penalty term is small precisely because it is working
(measured on a healthy 64² control at λ_gp 10: norm 1.30, GP 2.7% of `|loss|`).
The failure mode is a small share **and** a norm far from 1.

---

## 4. Logging the result

CLAUDE.md §9: a row in [../experiments.md](../experiments.md) **before** the next
run starts. Record base commit, the single change, the scoreboard result, and the
verdict. The `run_start` record at the top of `metrics.jsonl` has the resolved
recipe — copy from that, not from the config file, which the trainer mutates.

---

## 5. Known traps

- **`--rebuild` must never reach `train_with_restarts.sh`.** It replays argv on
  every restart, so the run would wipe its own weights each relaunch.
- **`--max_rss_mb` is a no-op off Linux.** `_process_rss_mb` reads
  `/proc/self/status` only, so on macOS the trainer never exits with code 75 and
  the wrapper cannot pre-empt the OOM-kill.
- **`fade_steps` offsets the cosine LR schedule even when `fade` is off** —
  `post_fade_step = global_step − fade_steps`. Dropping the flag as a tidy-up
  shifts the whole schedule. Irrelevant for the control (`--lr_decay none`),
  load-bearing for increment 4.
- **`adaptive_steps` is not reproducible.** It persists its adjustment into
  `training_steps`, re-read as the next launch's base, so the ratio ratchets
  across restarts (2 → 61 in the released run). Never use it in a run you intend
  to interpret.
- **Never point a run at a release directory.** Any `snowgan` invocation rewrites
  the config sidecars, and those sidecars *are* the release artifact AvAI
  rebuilds from. `kill_check.py --copy-to` exists for this.
- **`batch_N` directory numbering is not chronological.** The counter restarts
  after cleanup. Sort by `fade_step` (which is `global_step`) instead.
