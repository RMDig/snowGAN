# Increment campaign — damping the oscillation

**Status:** metrics validated 2026-09-29, arms not started. Metric revised after the
noise probe found the original one produced false positives — see §2.
**Governs:** the next three training runs. Operates under
[CLAUDE.md](../../CLAUDE.md) §9 (one piece per campaign, judged by a fixed scoreboard,
logged before the next run starts).
**Reference:** [training_dynamics_recovery_plan.md](training_dynamics_recovery_plan.md)
for how we got here; [experiments.md](../experiments.md) for the baseline result.

---

## 1. What the baseline established, and what it left open

`control_1024` (git `8bcdbe4`, 100,000 steps, 1024², 34.9 h) is the first snowGAN run
with a trustworthy reference curve. It produced good crystal-mesh output and it is
**not** data-limited: the memorization check puts generated samples 1.41× *farther*
from the training set than training images are from each other, with 2,006 unique
images seen ~200× each and augmentation off.

Two defects it exposed, both measured:

**A. The run oscillates instead of converging.** Kill-checked at four checkpoints with
identical latents, so the spread is model movement, not sampling noise:

| global_step | output mean | output std | latent diversity |
|---|---|---|---|
| 25,000 | −0.248 | 0.454 | 0.446 |
| **50,000** | **−0.226** | **0.599** | 0.517 |
| 75,706 | −0.476 | 0.329 | 0.307 |
| 100,000 | +0.147 | 0.563 | 0.547 |
| *real data* | *−0.22* | *0.63* | — |

The best match to the data's statistics is step 50,000. The trajectory crosses the
target rather than settling on it.

**B. The critic escapes its Lipschitz constraint, monotonically, all run.**

| window | ‖∇D‖ | λ·GP | \|W\| |
|---|---|---|---|
| 1–10k | 0.815 | 1.79 | 9.1 |
| 40–50k | 1.330 | 2.21 | 27.5 |
| 90–100k | **1.913** | **9.40** | 55.9 |

λ_gp = 10 is being outrun — the penalty grows 5× while the constraint it enforces gets
steadily looser. Nothing in the recipe damps either process: `--lr_decay none` was
deliberate, to match the structure era.

**The campaign tests three candidate dampers, one per arm.**

---

## 2. Pre-registered metrics

Fixed **before** any arm runs, because choosing how to judge after seeing results is how
you fool yourself — and this repo has a documented history of unfalsifiable verdicts.

**Revised 2026-09-29 after the noise probe.** The original design used `dist_err` (a
first-two-moments match) sampled at 10k spacing. Both halves of that were wrong, and the
probe caught it before the 83 h was spent:

- **The sampling aliased the signal.** At 1k spacing `dist_err` swings 0.245 → 1.808
  across eight consecutive checkpoints of one run. Sampling every 10k measured a fast
  signal too slowly; the reported `osc = 0.403` was an undersampled artifact.
- **`dist_err` produces false positives.** Scored on `control_1024` (seed 42) against
  `noise_seed43` — two runs differing *only* by seed, so the correct answer is
  "indistinguishable" — `dist_err` reported 0.688 ± 0.095 vs 1.211 ± 0.175, **|t| = 2.64**.
  It would have "found" an effect between arms that was pure seed noise.

### The metric

**KID** (Kernel Inception Distance; Bińkowski et al., ICLR 2018) against **held-out**
images only — the 344 magnified_profile images in `validation_pool + test_pool`, which the
run never trained on. Unbiased at small n, which FID is not: `_compute_fid` estimates a
2048×2048 covariance from 64 samples and its bias depends on n, so two runs scored at
different sample counts are not comparable (UPGRADES #58).

Measured over **10 checkpoints at 1k spacing** across the run's last 10k steps, 200
generated samples each, 10 subsets of 100. Reported as mean ± SE over checkpoints.

On the same seed-42/seed-43 pair, KID gives 0.400 ± 0.033 vs 0.402 ± 0.021 — **|t| = 0.05**,
correctly indistinguishable.

| metric | role |
|---|---|
| **`KID_mean`** | **PRIMARY.** Mean over the 10 checkpoints. Lower is better |
| `KID_se` | SE over checkpoints. An arm is distinguishable only at >2× pooled SE |
| `KID_cv` | std/mean across checkpoints — how much the run wanders, i.e. the oscillation, now on a metric that can measure it |
| `grad_norm_final` | validity: once the critic escapes, the loss is not a Wasserstein distance |
| `grad_slope` | is the escape still accelerating |
| `diversity` | **gate, not a score.** < 0.049 ⇒ dead, arm discarded |
| `mem_ratio` | gate. < 0.8 ⇒ memorizing, arm discarded |
| `dist_err` | **demoted to descriptive.** Retained for continuity with the baseline log; never decisive |

**Minimum detectable effect:** 2 × pooled SE ≈ **0.078 KID** on a base of 0.40 (~20%
relative). An arm inside that is a tie, and a tie is broken by a seed replicate — not by
preference.

**Measurement precision vs model movement.** Each checkpoint's KID carries its own SE of
~0.002–0.005 (≈1% of the value), so the 0.065–0.105 spread *across* checkpoints is real
model movement, not measurement noise. `dist_err` could not separate those two things,
which is why its swing was uninterpretable.

**What KID still cannot do.** It scores Inception features, which are tuned for natural
images, not snow crystal structure. A model could improve KID while looking worse to a
domain eye. Sample inspection stays the arbiter per §9, and the downstream probe — which
still does not exist — remains the real objective.

## 3. The reference

**Reuse `control_1024`.** It is the same code (`8bcdbe4`), same seed, same recipe, with
checkpoints every 1,000 steps — so an arm run to 80,000 steps compares against the
baseline's own 80,000-step checkpoint. This saves ~28 h of GPU and removes a confound.

**Comparability rule.** An arm is comparable only if it differs from the baseline by
exactly one flag group *and* runs code where every other path is bit-identical. Verified
for the one code change in this campaign: both `diff_augment` call sites are gated on
`self.use_augment` ([trainer.py:902-903](../../src/snowgan/trainer.py#L902)), so the
generator-step fix is inert under `--no-augment` and cannot perturb arms A or B.

Any arm violating this is not evidence and does not get logged as such.

---

## 4. The arms

All inherit the baseline recipe: 1024², `batch_size 4`, `disc 2 : gen 3`, `gen_lr 1e-4`,
`disc_lr 1e-5`, no SN, no clipping, no EMA, no ADA, no multiscale, `seed 42`,
`--image_root ~/rmdig-cache-1024`, `--max_rss_mb 24000`, **80,000 steps**.

### Arm A — LR anneal

**Hypothesis.** The oscillation is under-damped training at constant LR. Annealing
shrinks the step size as the run proceeds, so the orbit should contract rather than
persist. This is the hypothesis Denny raised at the outset; it was unsupported then and
is directly supported now by defect A.

```
--lr_decay cosine --lr_decay_steps 80000 --lr_min 0.000001 --fade_steps 1
```

**`--fade_steps 1` is part of this arm, not a second variable.** `_update_learning_rates`
computes `post_fade_step = global_step − fade_steps`, so the default 50,000 would delay
the anneal until step 50k — past half the run. With `fade=False`, `fade_steps` has no
other effect, so setting it to 1 only removes the offset. This is exactly the trap that
froze the released run's LRs at `lr_min` from step 250k.

**Prediction if true:** `KID_mean` drops by more than 0.078 (the minimum detectable
effect), `KID_cv` falls as the orbit contracts, `grad_slope` flattens late. **If false:**
`KID_mean` inside 0.078 of baseline — the oscillation is driven by something other than
step size.

### Arm B — stronger Lipschitz penalty

**Hypothesis.** Defect B is λ_gp = 10 being outrun. Doubling the penalty should hold
‖∇D‖ nearer 1.

```
--disc_lambda_gp 20
```

**Prediction if true:** `grad_norm_final` closer to 1.0, `grad_slope` flatter. **If
false:** ‖∇D‖ drifts the same, meaning the escape is not a matter of penalty weight and
the next move is R1/R2 (queue rank 6) rather than a bigger λ.

**Watch for:** an over-constrained critic. The 1z-B A/B measured SN + λ_gp=1 pinning
‖∇D‖ to 0.35–0.42 — wrong in the other direction. If ‖∇D‖ undershoots ~0.7, λ=20 is too
strong and the tuning runs go downward, not up.

### Arm C — DiffAugment, actually working

**Hypothesis.** 2,006 images with no augmentation is the most obvious regularization gap,
and DiffAugment was designed for exactly this regime (Zhao et al. 2020: usable GANs from
100 images).

**Requires a code fix first.** `--augment` has never worked correctly here: augmentation
is applied in the critic update ([trainer.py:902-903](../../src/snowgan/trainer.py#L902))
but not the generator update, so the two optimize against different distributions. The
fix lands with its own focused test before this arm runs. `augment.py` also has **no
translation**, DiffAugment's strongest component for small data; adding it is a second
change and therefore a *separate tuning run within this arm*, not part of the first.

```
--augment                      # run C1: fix only, existing ops
--augment  (+ translation)     # run C2: after C1, if C1 is not worse
```

**Prediction if true:** `KID_mean` drops by more than 0.078, `mem_ratio` rises (more
novelty). **If false:** no change, or diversity drops below the gate — augmentation too
strong for the generator's capacity.

---

## 5. What each arm cannot tell us

- None of these touch the **downstream probe**, which §9 calls the real objective and
  which still does not exist. Every verdict here is about generative behaviour.
- All arms train on the current 7-site manifest, so none produces a release candidate
  for snowGradient (their `assert_sites_unseen` will reject it, correctly).
- One seed per arm. See §2.1.

---

## 6. Early stopping and checkpoint selection

Denny's observation, made concrete: these metrics should feed back into training, not
just post-hoc analysis. They are **two signals with different jobs**, and conflating them
is the mistake to avoid.

**‖∇D‖ is a validity gate, not a quality signal.** It says whether the loss still means
what it claims. When it escapes, `disc_loss` stops being a Wasserstein distance and any
loss-based decision after that point is unfounded. It does **not** say quality stopped
improving — in the baseline, diversity and std kept improving through 100k while ‖∇D‖
drifted to 1.9. Framing it as "stop training here" would have thrown away real progress.

**`dist_err` is the quality signal** — weakly, per §2's limitation.

Together they give checkpoint selection the baseline needed and did not have:

> Select the checkpoint minimizing `dist_err`, **among those where ‖∇D‖ was still within
> [0.7, 1.4]**.

On the baseline that rule selects **step 50,000** — which is the checkpoint that actually
best matches the data, and which a naive "take the last checkpoint" would have missed.

**Proposed, pending campaign results:** implement as `--early_stop_patience` (stop when
`dist_err` has not improved for N evaluations) plus a `best_dist_err/` checkpoint
mirroring the existing `best_fid/` path. Not built yet — the thresholds should be
calibrated on three more runs before being wired into the trainer, or we would be
encoding a rule derived from a single run.

---

## 7. Budget — needs sign-off

Measured rate on this host: **1.25 s/step** at 1024² with the local mirror.

| | steps | wall clock |
|---|---|---|
| Baseline | — | **reuse `control_1024`** (saves ~28 h) |
| Arm A — LR anneal | 80,000 | 27.8 h |
| Arm B — λ_gp 20 | 80,000 | 27.8 h |
| Arm C1 — DiffAugment | 80,000 | 27.8 h |
| **total** | | **~83 h ≈ 3.5 days** |

This fits the stated budget exactly, and 80,000 is chosen to fit it — arms are compared
against the baseline's 80,000-step checkpoint, not its final one.

**What 3.5 days does not buy.** §9 expects a piece to need several tuning runs before a
verdict ("judge a piece by its best-tuned run"). This budget funds **one run per piece**.
So the honest output is a *screen*, not a verdict: it ranks three candidate dampers and
identifies which deserve tuning. Arm C2 (translation), any λ sweep, and seed repeats are
all follow-on spend.

Each arm also costs ~2 RSS restarts (~434 KiB/batch, 24 GB ceiling) — validated, 11
consecutive restarts resumed cleanly in the restart test.

---

## 8. Risks

| risk | mitigation |
|---|---|
| Difference smaller than run-to-run noise | Measured: 2x pooled SE = 0.078 KID (§2). Ties broken by seed repeat, not preference |
| KID scores Inception features, not snow structure | Sample inspection remains the arbiter (§9); any win that looks worse is re-judged by eye |
| Arm C's code change contaminates A and B | Verified inert under `--no-augment` (§3); A and B run on `8bcdbe4` regardless |
| `_cleanup_saved_batches` no-op (UPGRADES #66) | 3 arms × ~45 GB ≈ 135 GB. 1.5 TB free, so tolerable — but fix it before a fourth |
| Two trainers on one `save_dir` (UPGRADES #39) | Confirm zero `bin/snowgan` processes before each launch; arms run sequentially |
| A dead arm burns 28 h | Kill gates at 4,000 critic updates (diversity, ‖∇D‖) as in Phase 1 |

---

## 9. Sequence

1. ~~Land the DiffAugment generator-step fix + test.~~ **Done** (`f3d093d`).
2. ~~Build the campaign comparison tool.~~ **Done.** `scripts/compare_runs.py` (descriptive
   metrics, no GPU) and `scripts/kid_check.py` (the primary metric; ~4 min of GPU per run).
3. Arm A (27.8 h) → log to experiments.md → Arm B (27.8 h) → log → Arm C1 (27.8 h) → log.
4. Report the ranking, and name what deserves tuning spend.

Steps 1 and 2 can start immediately and burn no GPU.
