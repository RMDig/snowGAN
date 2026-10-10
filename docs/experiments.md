# snowGAN experiment log

The running record of training experiments. One row per run. Fill a row **before**
starting the next run — this log is the memory that stops us re-deriving what we already
tried. Governed by [CLAUDE.md §9](../CLAUDE.md).

## The scoreboard (applied identically to every run)

Never judge by loss curves — WGAN loss magnitude is not quality. In order:

1. **Kill-check** (minutes, before letting a run continue):
   - **Latent diversity:** mean pairwise `|G(zᵢ) − G(zⱼ)|` over ~8 samples. Must be well
     above 0. `≈ 0.000` ⇒ full mode collapse (generator ignoring z) ⇒ dead run.
   - **Saturation:** `frac(|output| > 0.99)` low, output std near the data's (~0.5).
     `std ≈ 1.0 / 100% pinned` ⇒ tanh-rail collapse. `std ≈ 0` ⇒ vanished (grey frame).
   - Real core data reference: mean ≈ −0.22, std ≈ 0.63, ~0% saturated (in [−1,1]).
2. **Sample inspection:** open `synthetic_images/*.png`; two latents must give two
   *different*, non-saturated images.
3. **Downstream probe (the objective):** linear probe on avalanche-risk labels. Feature
   transferability, not image beauty. Gated on labels (TBD).

The kill-check is now a tool rather than a snippet — `scripts/kill_check.py`:

```
# gate a live run (raw weights; the EMA shadow is ~13% random init at step 2,000)
python scripts/kill_check.py keras/snowgan/<run>/

# calibrate against the release. --copy-to is REQUIRED for a release dir:
# loading one in place rewrites its config sidecars, which ARE the release artifact.
python scripts/kill_check.py keras/snowgan/magnified_profiles/ \
  --gen_upsampler transpose --gen_convs_per_resolution 1 --gen_norm none \
  --copy-to /tmp/magprof_ref
```

The rebuild flags are mandatory for the release: its config predates
`gen_upsampler` / `gen_convs_per_resolution` / `gen_norm`, so `build()` supplies
today's defaults (`resize` / 2 / `none`) and silently constructs a **different
architecture** than the weights were trained under.

And the loss-curve half of the scoreboard is now readable from
`metrics.jsonl` — `python scripts/check_gates.py <save_dir>` reports the
critic's input-gradient norm, the Wasserstein term separated from λ·GP, and the
1-Lipschitz ceiling `2·sqrt(depth·H·W·C)`. Note what that ceiling implies: a |W|
of ~440 at 256² is ~85% of what an *optimal* 1-Lipschitz critic would report, so
it is evidence the generator is bad, not that the critic is unconstrained.

## Protocol

- One architectural piece per campaign; multiple tuning runs on that piece are expected
  (vary only its own knobs). Judge a piece by its best-tuned run. See CLAUDE.md §9.
- Ground truth = a baseline that **provably trains from scratch**, not one that only loads
  old weights.

## Ground truth

- **Proven architecture** (commit `7b487a0`, pre-2026-06-13): Conv3DTranspose per
  resolution, **one** conv per block, **no** normalization, kernel 3 / stride 2, tanh
  head. This trained the magnified_profiles v0.1.0 backbone (genuine crystal-mesh
  output). Reproduced on `feat/shallow-generator-baseline` via
  `--gen_convs_per_resolution 1 --gen_norm none --gen_upsampler transpose`.
- **Proven *recipe*** (corrected 2026-09-23, see the note below): `disc 2 : gen 3`,
  `gen_lr 1e-4`, `disc_lr 1e-5`, **`lambda_gp 10.0`**, **spectral norm OFF**, no
  gradient clipping, no LR decay, no EMA, no ADA, no multiscale critic.
- **Static validation done:** loading `magnified_profiles/generator.weights.h5` into the
  reproduced architecture regenerates the mesh image. Re-measured 2026-09-23 with
  `scripts/kill_check.py` (raw weights, n=6, seed 0): mean −0.270, **std 0.533**,
  **3.63% saturated**, **latent diversity 0.4898**. Dead runs for contrast:
  `exp0_magprof_control_256` diversity **0.000026**, `core_masked_256` **0.000000**
  (std exactly 1.0 — full tanh rail). Four orders of magnitude of separation, so the
  scoreboard's diversity threshold is unambiguous.
- **Dynamic validation OUTSTANDING:** we have NOT confirmed the current code *trains* this
  architecture from scratch. That is experiment 0 — now specified in
  [plans/training_dynamics_recovery_plan.md](plans/training_dynamics_recovery_plan.md) §4
  and launchable via `scripts/train_exp0_control.sh`.

> **Correction (2026-09-23) — the recipe was read off the wrong artifact.**
> A run's terminal `*_config.json` describes the *end* of the run, not the era that
> trained it: `adaptive_steps` rewrites `training_steps` and persists it, and the
> pre-2026-09 SN clamp rewrote `lambda_gp`. The per-snapshot configs in `<run>/batch_*/`
> carry `fade_step`, which **is** `global_step`; sorted by it, the released
> `magnified_profiles` run reads:
>
> | global_step | disc_steps | lr_decay | spectral_norm | grad_clip | λ_gp |
> |---|---|---|---|---|---|
> | ≤ ~288,613 | **2** | absent | **absent** | **absent** | **10.0** |
> | 289,614 | 2 | cosine on | SN on | absent | 10.0 |
> | 295,614–296,614 | 2 | cosine | True | 1.0 | **1.0** |
> | 298,623 → 425,644 | 1 → 2 → 4 → … → **61** | cosine | True | 1.0 | 1.0 |
>
> So "disc-heavy (disc 47 : gen 3)" in the retro takeaways below is **wrong**: 47 is a
> terminal value of the `adaptive_steps` ratchet, reached ~130k steps *after* the model
> acquired its structure. And cosine was switched on at global_step 289,614 under the
> pre-`7ab94a8` hard-coded 200k horizon — `post_fade_step = 289,614 − 50,000 = 239,614 >
> 200,000` — so **both LRs jumped straight to `lr_min = 1e-7` the moment decay was
> enabled**. Every late-era change (SN on, clipping on, the ratchet to 61) was applied to
> a model that had stopped moving. Note also that `batch_N` directory numbering is not
> chronological: the counter restarts after cleanup, so `batch_157000` predates
> `batch_1000` here by months.

## Log

Verdict key: ✅ trains/structure · ⚠️ partial · ❌ collapsed · ⏳ pending

| # | Date | Base | Single change under test | Data / res | Scoreboard result | Verdict |
|---|------|------|--------------------------|-----------|-------------------|---------|
| retro-A | ~06→07 | deep stack + PixelNorm + transpose | (many at once — uninterpretable) | core 1024 | 332k batches, tanh-saturated throughout (std 2.26 pre-tanh, 37% pinned) | ❌ |
| retro-B | 07-20 | + OutputGain (#47), SN+GP | (stacked) | core 256 | gain settled ~2.0, still saturated | ❌ |
| retro-C | 07-20 | + SN-only (GP off) | (stacked) | core 256 | gain settled 1.09 (healthy!) but collapsed to constant | ❌ |
| retro-D | 07-20 | + minibatch-stddev (#48) | (stacked) | core 256 | collapsed by 2k, saturated | ❌ |
| retro-E | 07-22 | shallow stack, no norm, no OutputGain | (arch swap) | core 256 | current weights std 1.0, 100% pinned | ❌ |
| retro-F | 07-25 | + board masking (#49, grey fill) | (masking) | core 256 masked | std 1.0, 100% pinned, pairwise_div **0.0000** | ❌ |
| **1z-A** | 09-23 | HEAD + Phase 0 | **can the code overfit 8 images at all?** (structure-era recipe: disc 2 : gen 3, λ_gp 10, **no SN**) | magnified_profile 64, 8 imgs, 2500 steps | nn_dist **0.570 → 0.263**; diversity 0.007 → **0.374**; output std → **0.643** vs data 0.640; saturation peaked 43.9% @250 then self-corrected to 5.4%; ‖dD/dx‖ → **1.000** | ✅ |
| **1z-B** | 09-23 | 1z-A | **only change: `--spectral_norm --disc_lambda_gp 1.0`** (the collapsed runs' recipe) | same | nn_dist **0.570 → 0.292**; diversity **0.372**; ‖dD/dx‖ settles **0.35–0.42** (over-smoothed) with λ·GP stuck at 0.33–0.47 | ✅ fits — hypothesis **not** confirmed at this scale; mechanism corrected (see above) |
| **0** | 09-23 | HEAD + Phase 0 | **control: does current code train the structure-era recipe from scratch?** disc 2 : gen 3, λ_gp 10, **no SN**, no clip, no decay, no EMA, no ADA, no multiscale, seed 42 | magnified_profile **512**, 10,000 steps = 20,000 critic updates, 133 min on RTX 5080 | **‖dD/dx‖ 1.046 (real) / 1.030 (interp) / 1.081 (fake)**; \|W\| 1.0 vs 1774 ceiling; λ·GP 25.4% of loss; **latent diversity 0.609** (released model 0.490, dead runs 0.00003); std 0.583; saturated 3.35%; samples show the crystal-mesh motif, two latents visibly different | **✅ PASSES** |

**Arm D — generator EMA 0.999 (2026-10-10): every checkpoint becomes a good one;
the gain over the baseline is probable, not established.** `arm_d_ema`: control_1024's
recipe plus `--ema_decay 0.999`. Seed 42 (same split, same 344 reals), 80k steps, 28 h,
one RSS restart, commit 4b90eed. Pre-registered in increment_campaign §4. Windows matched
on `fade_step` (D 70,702–79,702 vs control 70,706–79,706).

Window KID at each scoring seed:

| scoring seed | EMA shadow | raw weights (same run) | control_1024 | EMA − control |
|---|---|---|---|---|
| 0 | 0.2415 ± 0.0014 | 0.3238 ± 0.0133 | 0.2987 ± 0.0240 | −0.057 |
| 1 | 0.2608 ± 0.0012 | 0.3469 ± 0.0151 | 0.3043 ± 0.0244 | −0.044 |
| 2 | 0.2639 ± 0.0011 | 0.3382 ± 0.0154 | 0.3047 ± 0.0227 | −0.041 |
| **3-seed** | **0.2554 ± 0.0071** | 0.3363 ± 0.0161 | 0.3026 ± 0.0238 | **−0.047** |

(3-seed SE per the §2 amendment: `sqrt(SE_ckpt² + σ_seed²/3)`.)

- **Primary (EMA vs control): borderline.** As pre-registered (seed 0 only) it passes:
  0.057 > 0.048. With the sample-set noise the protocol omitted, 0.047 < 0.050
  (|t| = 1.9) and it narrowly fails. Recorded as **probable, not established**. The
  amendment was written after seeing this result, so it is applied going forward and
  not used to re-decide Arm D.
- **Secondary (window spread < 0.038): decisive pass at every seed.** EMA std
  0.0034–0.0043 vs control 0.072–0.077, about 20× smaller. All 10 EMA checkpoints at
  seed 0 lie between 0.233 and 0.247.
- **Within-run control: the gain is pure averaging.** EMA beats the same run's raw
  weights by 0.074–0.086 at every seed. The raw weights tie control (0.336 vs 0.303),
  so training dynamics were baseline-like.
- **The peak is not higher.** EMA's best_kid (57k) re-scores at 0.234 (seed 1).
  control_1024's best surviving snapshot (77k) is 0.222 (seed 1). EMA does not find a
  better model; it makes *every* checkpoint about as good as the raw run's lucky one.
  It was never beaten after 57k (in-training KID 70–80k: 0.244–0.266).

**Reading.** The checkpoint-to-checkpoint KID spread that dominated Arms A and B is
fast weight jitter around a good region, and a ~1k-step average removes it. That
answers the §4 "if false" branch: it is not slow drift. The practical effect is
reliability. Any late EMA checkpoint is near the best, so the run no longer depends on
checkpoint selection. Replicate at seed 43 vs `noise_seed43` launched 2026-10-10
(`arm_d_ema_s43`, commit 4b90eed), scored under the §2 amendment.

**Arm B — `lambda_gp 20` (2026-10-07): critic escape damped, KID tie.**
`arm_b_gp20`: control_1024's recipe with one change, `--disc_lambda_gp 20.0`. Flat LR,
seed 42 (same split, same 344 reals), 80k steps, 27 h, one RSS restart, commit 6ae5d70.
Windows matched on `fade_step`: B 70,572–79,572, control 70,706–79,706. control_1024's
window was re-scored and reproduced exactly (0.2987, ρ +0.15).

| | window KID mean ± SE | ρ | ‖∇D‖ 70–80k | weight movement /1k | best checkpoint (re-scored, `--seed 1`) |
|---|---|---|---|---|---|
| control_1024 (λ 10) | 0.2987 ± 0.0240 | +0.15 | 1.595 | 0.167 | 77k: 0.222 (from 39 surviving snapshots) |
| **Arm B (λ 20)** | 0.2825 ± 0.0168 | −0.21 | **1.354** | 0.182 | 69k: 0.192 (from 61 eligible in-training evals) |

Δ −0.016, 2 × pooled SE 0.059, |t| 0.55 → **tie**.

‖∇D‖ by decade, B vs control: 20k 1.048/1.052, 40k 1.115/1.330, 60k 1.316/1.496, 70k
1.354/1.595. The penalty slows the escape (about 15% lower late) but does not stop it.
Weight movement matches the baseline, so unlike Arm A there is no freezing confound.

The best-checkpoint gap (0.192 vs 0.222) is not evidence. B's best is the minimum over
more candidates (61 vs 39), and the minimum of more noisy draws is lower by
construction.

**Reading — the Lipschitz lever is ruled out at this range.** Two independent
interventions have now lowered late ‖∇D‖ by 0.24–0.33: the anneal (A1-s43, 1.56 vs 1.88)
and λ_gp 20 (1.35 vs 1.60). Both gave a KID tie. Critic escape in the 1.3–1.9 range is not
what limits sample quality here. What dominates every run is checkpoint-to-checkpoint KID
spread (window std 0.05–0.09), which none of the three arms reduced without freezing the
model.

Early-step artifact found during this run: the step-1k checkpoint held `best_kid/` until
21k (0.329) despite looking visibly worse. Fixed by `--kid_min_step` (UPGRADES #68
follow-up).

**A1 replicate, paired verdict (2026-10-06): TIE at seed 43. The anneal's effect is
small at best.** `noise_seed43` was extended 40k → 80k (flat LR, same seed/split,
`--cleanup_milestone 0` so its 31–39k probe snapshots survive; commit 8ca8aae) to give
A1-s43 a same-split baseline. Windows matched on recorded `fade_step` (snapshot names lag
the step after a restart: A1-s43 scored 70,409–79,409, baseline 70,000–79,000).

| seed | A1 window KID | baseline window KID | Δ (A1 − base) | 2 × pooled SE | verdict | ρ A1 / base |
|---|---|---|---|---|---|---|
| 42 | 0.2270 ± 0.0162 | 0.2987 ± 0.0240 | −0.0717 | 0.058 | A1 better (|t| 2.47) | −0.95 / +0.15 |
| 43 | 0.3108 ± 0.0217 | 0.3251 ± 0.0272 | −0.0144 | 0.070 | **tie** (|t| 0.41) | +0.72 / +0.33 |

A stronger, fully paired read comes from the in-training KID, which both seed-43 runs
logged at the same global steps with **identical latents** (dedicated RNG seeded off
seed 43) and identical reals. Over the 40 shared checkpoints (41k–80k): A1 − baseline =
**−0.021 ± 0.015**, with A1 lower at 19 of 40 steps. That is 41–60k −0.017 ± 0.024 and
60–80k −0.034 ± 0.020.

**Reading.**
- Across two seeds the anneal's effect runs from about −0.07 down to −0.01. The sign is
  consistent, but it is not resolvable at seed 43. A1-s42's headline result was mostly
  that seed's draw.
- **Best-KID selection matters more than the anneal.** The flat-LR baseline's best
  in-training checkpoint (74k, **0.222**) beats A1-s43's best (56k, 0.243). Picking the
  right checkpoint from a plain run did as well as the schedule did.
- ‖∇D‖ over 70–80k: A1 1.555 vs baseline **1.880** at seed 43; 1.319 vs 1.595 at seed 42.
  That is about 0.3 lower in both seeds. This revives, at modest strength, the
  Lipschitz-damping claim retracted above: two seeds now agree in sign and size. Weight
  movement: baseline 0.189/1k vs A1 0.0115.
- At seed 43, a 0.33 gap in ‖∇D‖ came with a KID tie. So no measurement yet links critic
  escape to sample quality. Arm B is the direct test of whether it matters; until then
  "escape drives late degradation" remains a hypothesis.

**A1 replicate at seed 43 (2026-10-05): A1's trend signature did NOT replicate.**
`arm_a1_s43`: A1's exact recipe (`--lr_decay cosine --lr_decay_steps 80000 --lr_min 1e-6
--fade_steps 1`), `SEED=43`, 80k steps, 27 h, one RSS restart. First run with best-KID
checkpointing (`--kid_interval 1000`). Commit 73700f8.

Seed 43 also changes the split (see increment_campaign §2 correction): 609 held-out images
from 9 groups. **Absolute KID is not comparable to the seed-42 rows below.** The intended
paired comparison (vs `noise_seed43`) is not possible either: `noise_seed43` stopped at
39k, so there is no seed-43 baseline window at 70–79k.

What *is* comparable is the within-run signature A1 was credited with:

| | window KID mean ± SE | Spearman ρ | last (79k) | weight movement /1k | ‖∇D‖ 70–80k |
|---|---|---|---|---|---|
| A1, seed 42 | 0.2270 ± 0.0162 | **−0.95** | 0.166 (its best) | 0.0096 | 1.319 |
| **A1, seed 43** | 0.3108 ± 0.0217 | **+0.72** | 0.385 (9th of 10) | 0.0115 | 1.555 |

Per-checkpoint (kid_check, seed 0): 0.215, 0.266, 0.326, 0.239, 0.282, 0.445, 0.333,
0.285, 0.332, 0.385. The in-training scores (different latents) track these at r = 0.76.

**Reading.**
- The monotone decline that was A1's strongest evidence (ρ −0.95, ending at its best) did
  not recur. At seed 43, KID *rises* across the window at the same near-zero LR.
- Weight movement was as frozen as A1-s42 (0.0115 vs 0.0096 per 1k, 15× below baseline),
  yet KID still swung 0.21 → 0.44 between adjacent checkpoints. **Freezing the weights
  did not stabilize the output distribution.** That undercuts the "soft early stopping"
  reading of A1: a frozen model was supposed to be a stable one.
- Window std 0.069, vs A1-s42 0.051 and baseline 0.076. The checkpoint-to-checkpoint KID
  spread barely depends on the LR, so the A1-s42 ρ = −0.95 is plausibly one draw from a
  wide distribution. A1's effect is **unreplicated**; treat Arm A as unresolved.
- ‖∇D‖ rose to 1.555 under a near-zero LR, as A1-s42's did (1.103 → 1.319). This is the
  one feature both seeds share: the critic escapes even when the step size cannot explain
  it, which still points at Arm B.

**Best-KID checkpointing earned its keep.** Best was step 56,000 at 0.243 in training.
Re-scored with `kid_check.py --seed 1` it reads **0.234 ± 0.002**, so the winner's-curse
bias was small here. The final 80k checkpoint re-scores at 0.352. Selecting by KID beat
"take the last checkpoint" by 0.12, and beat the whole 70–79k window mean by 0.08.

**Arm A2 — `lr_min 1e-5` (2026-10-03): the higher floor loses A1's benefit.**
Identical to A1 except a 10x higher LR floor, to test whether A1 froze too early.
KID over 70–79k vs the same windows:

| run | KID mean | SE | ρ (trend) | last | weight movement /1k |
|---|---|---|---|---|---|
| baseline (flat LR) | 0.2987 | 0.0240 | +0.15 | 0.371 | 0.167 |
| A1 (`lr_min 1e-6`) | **0.2270** | 0.0162 | **−0.95** | **0.166** | 0.0096 |
| A2 (`lr_min 1e-5`) | 0.2722 | 0.0100 | **+0.83** | 0.296 | 0.0318 |

A2 vs baseline: +0.027, 2×SE 0.052 → **tie**. A2 vs A1: +0.045, 2×SE 0.038 → **A2
distinguishably worse** (narrowly). A2's KID *rises* across the window (ρ = +0.83) while
the model keeps moving at 3.3× A1's rate.

**Reading.** Keeping the model in motion did not let it keep improving — at `lr 1e-5` it
drifted *worse* late in training. Only near-zero LR stopped the degradation. So A1's win
looks less like "the anneal finds a better optimum" and more like **"the anneal acts as
soft early stopping"**: it freezes the model before late-training drift degrades it. That
is the outcome flagged in advance as the one that would undercut the preferred
interpretation, and it is recorded as such.

It also points at *what* drives the late degradation: something keeps pushing the model
away from good solutions whenever it is allowed to move. The critic's Lipschitz escape is
the prime suspect, which is what Arm B tests.

**A2 as an accidental replicate.** Through the first half A1 and A2's schedules differ by
≤13%, yet their ‖∇D‖ differs by up to 48% (40–50k: 1.151 vs 1.705). The "A1 holds ‖∇D‖
~0.28 below baseline" claim logged for Arm A is **retracted** — it sits inside that
run-to-run spread and never had error bars. Rule going forward: no number supports a claim
without an error estimate.

**Baseline late window (90–99k):** 0.2825 ± 0.0126, ρ = −0.01. The baseline decelerates
but never converges. A1 at 80k still beats the baseline's *final* window (diff 0.0555 vs
2×SE 0.041).

**Arm A — LR anneal (2026-10-01): promising, untuned, and confounded by freezing.**
80,000 steps at 1024px, identical to `control_1024` except `--lr_decay cosine
--lr_decay_steps 80000 --lr_min 1e-6 --fade_steps 1`. Judged on KID against the 344
held-out images, 10 checkpoints at 1k spacing over 70,000–79,000, vs the baseline's same
window.

| | KID mean | std | SE | Spearman ρ (step vs KID) | first → last |
|---|---|---|---|---|---|
| baseline | 0.2987 | 0.0760 | 0.0240 | **+0.15** | 0.255 → 0.371 |
| **Arm A** | **0.2270** | 0.0511 | 0.0162 | **−0.95** | 0.259 → **0.166** |

difference +0.0717, pooled SE 0.0290, |t| = 2.47.

**The pre-registration disagrees with itself and the result is borderline on the mean.**
§2 states the criterion both as a formula ("difference > 2 × pooled SE" = 0.058 here →
PASS) and as a number ("MDE ≈ 0.078" → TIE). Recorded as ambiguous rather than resolved
in the favourable direction. The number was estimated from the seed-42/43 pair whose
pooled SE (0.039) was larger than this comparison's (0.029).

**The trend is not borderline.** Arm A declines almost monotonically (ρ = −0.95) and ends
at its own best checkpoint; the baseline oscillates (ρ = +0.15) and ends at its 9th-best
of 10. Final checkpoints: 0.166 vs 0.371. That is the predicted effect — the hypothesis
was never "annealing lowers KID" but "annealing stops the orbiting."

**But weight movement confounds it.** Relative generator weight movement per 1,000 steps
over the same window: baseline **0.167**, Arm A **0.0096** — 17× less. A model whose LR
has collapsed to 2.3e-6 produces stable output because it has stopped moving, so "the
oscillation stopped" is partly tautological.

What survives the confound: Arm A reached KID **0.166**, better than the baseline's best
anywhere in its window (0.219), while moving 17× less. More movement did not buy more
progress — the baseline got *worse* across the same window (0.255 → 0.371). The defensible
claim is "annealing settles at a better point than the oscillating baseline occupies at
equal step count", not "annealing improves the attainable optimum".

**Lipschitz: damped, not cured.** ‖∇D‖ held ~0.28 below baseline throughout the back half
(70–80k: 1.319 vs 1.595), but bottomed at 1.103 around 50–60k and then *rose* while the LR
was 2.3e-6. Step size cannot explain a rise at a near-frozen LR, so the escape is only
partly step-size-driven — which is the case for Arm B attacking the constraint directly.

**Gates (final model):** latent diversity 0.552, min pairwise 0.310, saturation 1.15%,
output mean −0.090 / std 0.540. All pass.

**Open, by §9's own standard:** one run, untuned. `lr_min 1e-6` may freeze too early and
give up further progress; the anneal's own knobs (`lr_min`, horizon, schedule shape) have
not been varied. A piece is judged by its best-tuned run, and this is its first.

**Mechanism correction — SN *over*-constrains the critic (1z-B, 2026-09-23).** The A/B below
ran the same 8-image overfit under the collapsed runs' recipe (`--spectral_norm
--disc_lambda_gp 1.0`). **It also fits** (nn_dist 0.570 → 0.292, diversity 0.372), so
λ_gp/SN is *not* fatal at small scale and short horizons. But the new instrument shows a
large, monotone, measurable difference — in the **opposite direction** from the one the
recovery plan originally claimed:

| steps | ‖dD/dx‖ λ_gp 10, no SN | ‖dD/dx‖ λ_gp 1 + SN | λ·GP (A) | λ·GP (B) |
|---|---|---|---|---|
| 0–250 | 0.959 | 0.679 | 1.975 | 0.141 |
| 500–1000 | 0.832 | 0.388 | 0.287 | 0.378 |
| 1500–2000 | 0.992 | 0.368 | 0.002 | 0.399 |
| 2000–2500 | **1.000** | **0.424** | 0.001 | 0.332 |

Arm A converges to exactly 1-Lipschitz and its penalty goes to zero — nothing left to
correct. Arm B settles at **~0.35–0.42**, i.e. the critic is **over-smoothed to roughly a
third of the intended Lipschitz constant**, and its penalty term *stays elevated* (0.33–0.47)
because λ_gp=1 is too weak to pull it back up against spectral normalization. Measured at
reals rather than interpolates: A ends at 0.990, B at 0.444.

So the earlier framing — "λ_gp=1 is too weak to bind, so the critic is unconstrained" — was
**wrong in sign**. The critic is over-constrained, not under-constrained. Same root cause
(SN + a weak GP put the critic at the wrong Lipschitz constant), opposite direction, and a
different predicted failure mode: an over-smoothed critic passes weak, uninformative
gradients to the generator. That is survivable on 8 images at 64² and is a plausible
mechanism for collapse at 1024² over 100k+ steps — but that step is **not yet evidence**,
and experiment 0 is what would supply it.

**The implementation is not broken (1z-A, 2026-09-23).** Before spending ~19 GPU-hours on
experiment 0, `scripts/overfit_sanity.py` asked the cheaper question: can this code overfit
a GAN to 8 fixed images at 64×64? Under the structure-era recipe (disc 2 : gen 3, λ_gp 10,
**no spectral norm**) it fits (nearest-neighbour distance 0.570 → 0.263), stays diverse
(0.374, four orders above the collapse threshold), and its output std converges to the
data's (0.643 vs 0.640) — with the critic held at ‖dD/dx‖ ≈ 0.99. A transient rail-out at
step 250 (43.9% saturated) self-corrected to ~5%.

**So the failures since the releases are about the recipe, not a broken implementation.**
That does not settle which recipe variable, but it removes "a regression landed in the 51
commits since `fde5671`" as the leading explanation and makes experiment 0 worth running.

**The variable that actually separates (added 2026-09-23).** Across all thirteen runs in
both artifact trees, one split is perfect:

| | λ_gp | spectral_norm | disc:gen | outcome |
|---|---|---|---|---|
| `magnified_profiles` (structure era) | **10.0** | **off** | 2 : 3 | ✅ crystal mesh |
| `core_v0.1.0_forensic` | **10.0** | **off** | 2 : 3 | ✅ blobby but structured |
| `snowgan_slow_progressive` (legacy tree) | **10.0** | **off** | 2 : 4, `gen_lr 1e-3` | ✅ best image in the corpus, at 19k steps |
| `exp0_magprof_control_256` | 1.0 | on | 1 : 3 | ❌ |
| `core_masked_256` | 1.0 | on | 1 : 3 | ❌ |
| `core_proven_v2` | 1.0 | on | **2 : 3** | ❌ |
| `core_proven_recipe` | 1.0 | on | 2 : 3 | ❌ |
| `core_sanity_256` | 1.0 | on | 5 : 3 | ❌ |
| `core_sanity_256_snonly` | **0.0** | on | 5 : 3 | ❌ |
| `core_sanity_256_v3` | **0.0** | on | 5 : 3 | ❌ |
| `core/` (post-v0.2) | 1.0 | on | 39 : 1 | ❌ |

Mechanism: the trainer silently clamped `lambda_gp → 1.0` whenever spectral norm was on,
which also made `lambda_gp > 1` inexpressible under SN — so this could never be tested.
At 256²×3 the Wasserstein term is O(400), so a penalty weighted 1.0 is numerically
negligible against it (Gulrajani uses λ=10 where |W| is O(10)). The clamp is removed as of
2026-09-23 (`--clamp_gp_under_sn` restores it).

**Confound — the reason experiment 0 exists.** This split co-varies exactly with code era:
every λ_gp-10/no-SN run predates the v0.2 audit, every λ_gp-1/SN run postdates it. Config
forensics cannot separate "the recipe changed" from "a regression landed in the 51 commits
since `fde5671`". The control run disambiguates them.

**Things that do NOT separate, so stop reaching for them:** the critic/generator step ratio
(`core_proven_v2` died with the same 2:3 and the same LRs as both releases), and
`grad_clip_norm` (the structure era had clipping off).

**Long-horizon baseline (2026-09-25/27): the run oscillates, it does not converge.**
100,000 steps at 1024px on the exp-0 recipe (`control_1024`, git 8bcdbe4, 34.9 h,
one RSS restart, 200,000 critic updates). Kill-checked at four checkpoints with the
**same latents** (seed 0, n=8), so the differences are model changes, not sampling noise:

| global_step | output mean | output std | saturated | latent diversity |
|---|---|---|---|---|
| 25,000 | −0.248 | 0.454 | 0.31% | 0.446 |
| **50,000** | **−0.226** | **0.599** | 4.43% | 0.517 |
| 75,706 | −0.476 | 0.329 | — | 0.307 |
| 100,000 | +0.147 | 0.563 | 0.05% | 0.547 |
| *real data* | *−0.22* | *0.63* | *~0%* | — |

The closest match to the data's global statistics is **step 50,000**, not the final
model. The trajectory wanders across the target rather than settling on it.

Meanwhile the critic drifts monotonically away from 1-Lipschitz for the whole run:

| window | ‖dD/dx‖ | λ·GP | \|W\| |
|---|---|---|---|
| 1–10k | 0.815 | 1.79 | 9.1 |
| 40–50k | 1.330 | 2.21 | 27.5 |
| 70–80k | 1.595 | 4.32 | 32.3 |
| 90–100k | **1.913** | **9.40** | 55.9 |

λ_gp = 10 is being outrun: the penalty term grows 5× while the constraint it enforces
gets steadily looser. Nothing in this recipe damps either process — `--lr_decay none`
was deliberate (the structure era had no schedule), so the models orbit at constant LR.

**Not a data ceiling.** `scripts/memorization_check.py` on the final model: generated
samples sit at mean distance **0.473** from their nearest training image, while training
images sit at **0.335** from each other — ratio **1.41**, 13/16 with distinct nearest
neighbours. With 2,006 unique images seen ~200× each and augmentation **off**,
memorization is the expected failure and it is not happening.

**Reading:** the binding constraint is optimization dynamics, not data volume and not
training length. See the increment queue — ranks 3 (DiffAugment, never correctly enabled)
and 4 (LR anneal) now have direct evidence behind them.

**Retro takeaways (what NOT to repeat):**
- Saturation/collapse survived *every* generator-side change and the data-confound removal
  (masking). ⇒ the driver is **training dynamics**, not generator architecture or the
  board confound.
- retro-C is the tell: a *healthy gain* (1.09, un-saturated head) still collapsed to a
  constant ⇒ the failure is the generator ignoring z (mode collapse), separable from tanh
  saturation, and not fixed by head tuning.
- ~~retro-F (masked) reached pairwise diversity **exactly 0** with `gen_steps 3 :
  disc_steps 1` — an *inverted* WGAN ratio (generator 3× the critic). The proven
  mag_profiles run was disc-heavy (disc 47 : gen 3). Critic/generator step ratio is a
  prime suspect and has never been isolated.~~ **Retracted 2026-09-23.** The diversity-0
  observation stands (re-measured: 0.000000). The *explanation* does not: the proven run
  was **disc 2 : gen 3** during the era that trained it, not 47 : 3 — see the correction
  under "Ground truth". `core_proven_v2` then ran that exact 2 : 3 ratio and died anyway,
  so the step ratio does not separate working runs from dead ones and is no longer the
  prime suspect. λ_gp / spectral norm is (see "The variable that actually separates").
- Every run changed multiple variables — none is clean evidence. Hence experiment 0 and
  §9.

## Candidate increments (research-derived 2026-07-26; test one at a time)

Repo inventory + web research (WGAN small-data stabilization) converged. Apply only
after experiment 0 establishes a training baseline; one piece per campaign (multiple
tuning runs per piece allowed — CLAUDE.md §9). Verdict column filled as we test.

| Rank | Increment | Cost | Evidence | Verdict |
|------|-----------|------|----------|---------|
| 1 | **Critic-heavy ratio**: `--disc_steps 5 --gen_steps 1 --no-adaptive_steps`; sweep disc 5→8→15. Optional TTUR (D_lr 4e-4, G_lr 1e-4). | none (flags) | Both repo+web #1. Fresh defaults are inverted (steps 2:3, lr 10:1); the proven run was ~47:3; `adaptive_steps` drove core *gen*-heavy (retro-F). WGAN needs a near-optimal critic. | ⏳ |
| 2 | **DiffAugment** real+fake (`--augment`, color+translation+cutout) | none (wired) | Zhao NeurIPS 2020 — usable GANs from 100 imgs. Verify rank-5 depth-consistency first. | ⏳ |
| 3 | **SN-only, drop GP** (`--disc_lambda_gp 0`) | none | Resolves SN+GP double-Lipschitz; SN "often makes GP unnecessary" (Miyato 2018). | ⏳ |
| 4 | **ADA** (`--ada_target 0.6`, needs #2) | none | StyleGAN2-ADA (Karras 2020), the ≥1k-img reference. | ⏳ |
| 5 | **Minibatch-stddev** (NOT in this branch — parked on `fix/disc-minibatch-stddev`) | merge/code | ProGAN anti-collapse; reduce over BATCH axis, not depth. | ⏳ |
| 6 | **R1+R2 + relativistic (R3GAN)** | code, baseline swap | Huang NeurIPS 2024 — modern small-data recipe. Sequence last. | ⏳ |

**Strategic (architectural call, Denny's):** GAN-disc-as-backbone is defensible but
transfers only modestly (~+5%; Xiang & Li 2020) and its quality is gated by
not-collapsing. SSL does NOT need a big ViT — SimCLR/BYOL on a small *conv* encoder is
phone-deployable and often beats ViTs / supervised transfer on small & medical imaging.
Recommended posture: fix the GAN (1→4, generator wanted anyway) AND stand up a
**SimCLR-on-the-same-conv-encoder arm as the baseline to beat** before GPU-months.
Middle path if committed to disc-as-backbone: **FastGAN** (self-supervised decoder on
the discriminator; stable from ≤100 imgs, Liu ICLR 2021).

Sources: DiffAugment (arxiv 2006.10738), StyleGAN2-ADA (NVlabs), R3GAN (arxiv
2501.05441), TTUR (arxiv 1706.08500), Spectral Norm (arxiv 1802.05957), FastGAN
(openreview 1Fqg133qRaI), "Is Discriminator a Good Feature Extractor?" (arxiv
1912.00789), SSL low-data (sciencedirect S0925231224019702).
