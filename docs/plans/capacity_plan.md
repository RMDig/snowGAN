# Capacity campaign: where is the ceiling, and what moves it?

**Status:** REVISION 2 (2026-10-15). Draft 1 was reviewed in parallel through four lenses
(fact-checking against artifacts, GAN architecture, statistics, execution). §9 records
what it got wrong. Nothing here has run. **Budget and arm order need sign-off (§7).**
**Builds on:** [increment_campaign.md](increment_campaign.md) and
[../experiments.md](../experiments.md).

---

## 0. Why this campaign exists

- **EMA 0.999 is adopted.** It replicated at two seeds (−0.047 at 1.9σ, −0.088 at
  3.0σ), and the gain is pure averaging.
- **Training length and LR are not the lever.** EMA KID plateaus by 50–60k at both
  seeds; 80k → 160k tied. The anneal did not replicate, and λ_gp 20 tied.
- The recipe plateaus at an EMA KID of about **0.24–0.25**. This campaign asks what
  moves that number: the critic, the generator, the optimizer, or the data regime.

## 1. Facts

All verified against weight files, configs, code and the dataset manifest
(2026-10-15).

**F1. Generator.**
- 41,922,176 params.
- `Dense(100 → 16·16·1024)`, 26.2M params, no bias, **no activation before the first
  ConvTranspose**. It is therefore a rank ≤ 100 linear embedding, not representational
  capacity.
- 5 × ConvTranspose (1024→64, k3, s2, same), LReLU, no normalization, one conv per
  resolution. Conv stack 15.7M.
- **No conv operates at 1024².** The last layer is a k3/s2 transpose straight to RGB, so
  checkerboard texture is written directly into pixels.
- No per-layer noise input. All stochastic detail must come through z.
- The dense layer's width is `filter_counts[0]`, so widening the first stage also widens
  the dense layer.

**F2. Critic.**
- 7,319,169 params, 5.7× smaller than G.
- 5 × Conv3D (64→1024, k3, s2), LReLU, then `Flatten(name="features")` → `Dense(1)`.
- Theoretical receptive field 63 px; the effective RF is smaller.
- The head is a **position-specific** linear readout over 32×32×1024 features. Position-
  dependent first-order statistics (e.g. vignette darkness) are linearly detectable.
  **Interactions between patches (geometry, placement, layout consistency) are not.**
- AvAI reads `get_layer("features")`. Changing the critic changes that tensor.

**F3. Optimization.**
- disc_lr 1e-5 vs gen_lr 1e-4, i.e. a 10× *slower* critic, and 2 critic : 3 generator
  steps. This is the reverse of TTUR / standard WGAN-GP (n_critic 5).
- Batch 4.
- **Both critic steps reuse the same real batch and the same noise** (`trainer.py`
  train_step: noise drawn once before the critic loop).
- No equalized LR: layer fan-ins range from 100 to 9,216.

**F4. Data.**
- 2,350 magnified_profile images in 45 groups, sites 0–6.
- Seed 42 trains on 2,006 / 36 groups; seed 43 on 1,741 / 36. **All seven sites are in
  every campaign run's training pool.**
- **Site 0 (134 images, 1 group, always in training) is annotated in the manifest:**
  "Pilot … took too large segment for snow profile picturing, advised not to use in
  higher level models."
- **Held-out sets are lopsided.** Seed 42: 344 images, no site 0 or 4. Seed 43: 609
  images, one group = 201 (33%).

**F5. The data-limit evidence is weak.**
- The memorization ratio of 1.41 came from control_1024's *raw* final weights with **16
  samples**.
- A ratio above 1 is also what a model far from the data produces.
- It tests generator copying. At ~2k images the known failure is **critic
  overfitting** (Karras et al. 2020, ADA), which this ratio cannot see.
- DiffAugment (Arm C) was fixed but **never run**.

**F6. Metric.**
- `kid.py` resizes 1024 → 299 with `tf.image.resize` bilinear, **`antialias=False`**.
- A 2-px checkerboard aliases into low-frequency energy that Inception sees. Current KID
  is mostly a global/mid-scale metric with an aliasing artifact.

**F7. Noise.**
- Within-run plateau wander (SD of 10k-window means, 50–160k): **0.012 at s42, 0.014 at
  s43**.
- Run-to-run variance at fixed seed **has never been measured for EMA**. The one hint is
  that raw weights of two seed-42 runs with identical dynamics gave 0.336 vs 0.303.
- Window SE alone understates the comparison noise. The draft-1 rule would fire at about
  1σ (≈30% false positives).

**F8. Cost (FLOP-scaled from the measured 1.18 s/step with EMA).**

| change | s/step | 60k steps |
|---|---|---|
| base | 1.18 | 20 h |
| deeper critic | ~1.3 | 22 h |
| generator conv at 1024² / k4 / extra depth | ~1.9–2.0 | 32–33 h |
| width (keeping the dense layer fixed) | ~1.6 | 27 h |

**F9. Artifacts.**
- arm_d_ema keeps the 50k, 60k and 70–79k snapshots plus all of 80–160k; 57k exists only
  as `best_kid/`.
- arm_d_ema_s43's best is 80k.
- `kid_check.py` defaults to *raw* weights; EMA needs `--weights_file
  generator_ema.weights.h5`.

## 2. Hypotheses

| | hypothesis | prior after review |
|---|---|---|
| **H5** | **Critic overfitting / data regime.** At ~2k images the critic memorizes, its gradients degrade, and quality plateaus (ADA). Critic escape (‖∇D‖ → 1.9) is consistent with this. | **High.** Literature plus never-tested |
| **H6** | **Critic under-optimized / gradient noise.** 10× slower critic, batch 4, reals and noise reused across critic steps. Extra critic capacity cannot help a critic that is not fitted. | **High.** Cheap to test |
| **H1** | **Critic cannot model cross-patch structure.** Position-specific linear head, local RF. | Medium |
| **H2** | **Generator lacks depth/stochasticity, not width.** One conv per resolution, none at 1024², no noise injection. Wider than StyleGAN2 at low resolution already. | Medium |
| **H3** | **Checkerboard** from k3/s2 written straight to RGB. Possibly inflated in KID by aliasing (F6). | Medium |
| **H7** | **Data quality.** Site 0 is a flagged pilot in every run. | Low–medium; cheap |

H5 and H6 come first. Under the draft's ordering, a null on a capacity arm would have
been uninterpretable if the critic was either overfitting or under-optimized.

## 3. Phase 0: fix the instruments, then measure (≈ 1.5 days dev + ≈ 1 day GPU)

**P0.0 Measure run-to-run variance (E0). Required; 60k, ≈ 20 h.**
- Rerun arm_d_ema's exact recipe at seed 42 on the campaign commit, with
  `--cleanup_milestone 0`.
- This gives `σ_run`. Raw weights already diverge between same-seed runs, so a rerun is
  a valid replicate.
- It also anchors code drift since 4b90eed. Every arm compares against E0 (same commit),
  not against arm_d_ema.

**P0.1 Metric fixes (code only).**
- `inception_features(..., antialias=True)` as an option. Report both versions for one
  transition run, then switch the primary to antialiased.
- Scoring defaults for the campaign: `--n_gen 1000 --subsets 50`. Sample-set noise
  currently equals the checkpoint SE for EMA, and generation is cheap.
- Positive controls before trusting any per-scale reading:
  - a **global** corruption (4×4 tile shuffle of held-out reals);
  - a **texture** corruption (blur; injected 2-px checkerboard).
  - The global-scale score must move only for the first, the crop score only for the
    second.

**P0.2 Per-scale gap (no training, ≈ 2 h).**
- Scales:
  - (a) full frame, antialiased → 299;
  - (b) full frame → 128 antialiased → 299 (layout);
  - (c) native 299² crops, at most one crop per image per subset (texture).
- Reference at each scale: `F_s = KID_s(train reals, held-out reals)`, the score of a
  perfect generator of the training distribution. Not the half-vs-half floor.
- `R_s = KID_s(EMA fakes, held-out) / F_s`. CI by leave-one-group-out jackknife over the
  held-out groups.
- **Pre-registered:**
  - H1 favoured if `R_b / R_c > 2`, with the 95% CI excluding 1, at **both seeds**.
  - H2/H3 favoured if `R_b / R_c < 0.5` under the same conditions.
  - Otherwise inconclusive, and nothing is ranked by it.
- Scored on arm_d_ema 70–79k and arm_d_ema_s43 70–79k (EMA weights).

**P0.3 Critic overfitting (no training, ≈ 1 h).**
- For each retained arm_d_ema snapshot, `r = (D(train) − D(held)) / (D(train) − D(fake))`.
  This is ADA's heuristic, normalized because WGAN scores have no zero.
- Held-out groups are genuinely shifted (F4), so only **growth** of r from the 10k
  snapshot counts, not its level.
- **H5 supported** if r rises by > 0.3 from 10k and Spearman ρ(step) > 0.6.
- Snapshots are every 10k up to 70k, then dense. That is coarse but sufficient for a trend.

**P0.4 Spectrum (no training, ≈ 30 min).**
- 2D log power spectra, fakes vs reals, at native 1024² **and** on the 299² tensors
  Inception sees.
- Look along the axes at stride-period frequencies. The grid card is itself periodic, so
  compare against reals, never against flat.

**P0.5 (optional, ≈ 4 h) Projection of held-out reals** into the trained EMA generator
(optimize z, LPIPS + L1).
- What it cannot reconstruct shows what the generator's range lacks.
- Replaces draft 1's overfit-64 probe, which was invalid: a 26M linear embedding can
  memorize 64 layouts, the script only supports ≤ 256 px, and its default `save_dir` is
  `rmtree`'d, which would delete experiment 1z's artifacts.

## 4. Phase 1: arms (each a single change on the adopted recipe)

**Base:** `train_exp0_control.sh --ema_decay 0.999 --cleanup_milestone 0`, seed 42,
**60k** steps, `MAX_STEPS=60000` set explicitly (the script defaults to 100000).

**Tier A: cheap and most likely (run first, in this order).**

| arm | single change | tests | code |
|---|---|---|---|
| **C1** | DiffAugment (color + cutout), applied in both critic and generator steps (fixed in f3d093d) | H5 | exists; `--augment` |
| **O1** | disc_lr 1e-5 → 3e-5 | H6 | config |
| **O2** | fresh real batch + fresh noise for each critic step | H6 | small trainer change behind a flag, default off |
| **D1** | drop site 0 from training (still not held out) | H7 | small: an exclude-groups option |

**Tier B: architecture, run by Phase 0 evidence.**

| arm | single change | tests | code |
|---|---|---|---|
| **K1** | nonlinear critic head: conv 3×3 → dense 512 → LReLU → dense 1 (ProGAN-style) | H1 | small; changes `features` |
| **G1** | one stride-1 conv at 1024² after the final upsample (generator only) | H3 | small |
| **G2** | zero-init residual conv block per resolution (last conv initialised at 0, so the network starts identical to today's) | H2 depth | small |
| **G3** | per-layer noise injection, learned per-channel scales initialised at 0 | H2 stochasticity | small |

**Tier C: only if B is positive.**
- **K2:** two more critic layers at 512 width (RF only, limited parameter growth).
- **E3′:** width `1024 1024 512 256 128`, keeping the dense layer fixed.
- **mbstd:** branch `fix/disc-minibatch-stddev` (f754cac). Audit the batch-axis bug noted
  in experiments.md first. It couples per-sample gradients under GP.

**Dropped:**
- draft-1 E3 (doubles the dense layer too, 115M params);
- E4 in both nets (changes the critic as well);
- E5 cpr 2 with no norm decision (replaced by G2).

**Reported by every arm:** step time, peak VRAM (`tf.config.experimental.get_memory_info`),
per-scale R_s (descriptive only, since it chose the arm), ‖∇D‖, diversity, memorization
ratio, P0.3's r.

## 5. Decision rule (pre-registered; replaces increment_campaign §2 for this campaign)

1. **Primary statistic: paired in-training EMA KID** against E0, step-matched over
   40–59k (20 points). Same latents and same reals, because both runs are seed 42. SE by
   block bootstrap (5k blocks), never the naive SE, because the series is
   autocorrelated.
2. **Distinguishable** if `|Δ| > 2·sqrt(2·σ_run² + SE_paired²)`. Until E0 is in, use
   `σ_run = σ_plateau = 0.013`, which gives a threshold of about 0.04. Reset it to the
   measured `|E0 − arm_d_ema| / √2` combined with σ_plateau once E0 completes. **MDE ≈
   0.04–0.06 — capacity changes smaller than that are invisible in one run.**
3. **Confirmation:** three-seed `kid_check` (n_gen 1000, subsets 50, antialiased) on the
   arm's 50–59k window against E0's.
4. **Robustness gate:** leave-one-group-out, with Δ of the same sign in ≥ 8 of 9
   jackknife recomputations.
5. **Multiplicity:** Holm correction across Tier A, and separately across Tier B.
6. **Replicate** every winner at seed 43 against an E0-equivalent at seed 43,
   one-sided in the original direction. Quote the effect size **from the replicate
   only** (winner's curse).
7. **Plateau guard:** if an arm's last 10k window has not settled (not within 2·σ_plateau
   of the previous window), extend it to 80k before declaring a tie.
8. The Spearman-ρ OR-criterion is **dropped** for this campaign (descriptive only).
9. Secondaries (pre-registered, directional): KID at the targeted scale; precision and
   recall (k=3).

## 6. Safety

- **Downstream.** K1/K2/mbstd change the critic's `features` tensor. Arms live on branches
  with defaults off. No arm's critic is released as a backbone without a snowGradient
  transfer check.
- **Editable install.** A worktree launch runs *main's* code unless `PYTHONPATH=<worktree>/src`
  is set. Launch arms from the main checkout on the arm branch, and keep it there for the
  run.
- **EMA reload.** `_init_ema` silently resets EMA on a load failure. Make it fail loudly
  before any long arm (small fix).
- **Scripts.** Never run `overfit_sanity.py` without `--save_dir`.

## 7. Budget (needs sign-off)

| phase | GPU | wall clock |
|---|---|---|
| Phase 0 code (metric options, P0.2–P0.4 tools, O2/D1 flags, EMA-reload guard) | — | ≈ 1.5 days dev |
| E0 (60k) | 20 h | |
| P0.2–P0.4 (+ P0.5) | 4–8 h | |
| Tier A: C1, O1, O2, D1 at 60k | 4 × ≈ 20 h | ≈ 3.5 days |
| Seed-43 E0 + replicates of Tier A winners | 20 h + 20 h each | ≈ 2–3 days |
| Tier B (2–3 arms by evidence) | 22–33 h each | ≈ 3–4 days |

**Total ≈ 10–12 GPU-days** (draft 1 claimed 6; the execution review measured 8–10 for
its arms alone).

## 8. What this campaign cannot tell us

- Whether a KID gain transfers to the downstream task. Critic changes alter AvAI's features.
- Effects below the MDE (≈ 0.04–0.06).
- Absolute quality against "real". With 9 lopsided held-out groups, only model-vs-model
  ranking is supported.

## 9. What draft 1 got wrong

1. **Deprioritized augmentation on the wrong evidence.** A generator-copying ratio from 16
   raw samples was read as "not data-limited". At ~2k images the known failure is critic
   overfitting, which that ratio cannot see.
2. **Missed the optimization confound.** The critic runs 10× slower than the generator,
   on batch 4, reusing reals and noise across its steps. A null capacity arm would have
   been uninterpretable.
3. **The decision rule was miscalibrated.** It ignored within-run plateau wander and
   run-to-run variance, so its false-positive rate was ≈ 30%. The MDE is ≈ 0.04–0.06, not
   ≈ 0.02.
4. **"2× width" doubled the 26M dense layer too** (→ 115M params, ≈ 49 h). It was not a
   single change.
5. **Treated the dense layer as capacity.** With no activation before the first conv it
   is a rank ≤ 100 linear map.
6. **Overstated the critic's limit.** A position-specific linear head *can* detect
   position-dependent statistics such as vignette. It cannot model cross-patch
   interactions.
7. **Misattributed #47/#48.** They are rows of the retro table in experiments.md, from a
   different modality, resolution and recipe, not UPGRADES entries.
   Minibatch-stddev already exists on a branch.
8. **E1 confounded RF with a 3.4× critic and a 16× smaller `features` tensor.** E4
   changed both nets.
9. **P0.1 named checkpoints that no longer exist, would have scored raw weights**, and
   relied on an aliased resize. It had no positive controls and compared against a
   half-vs-half floor that is not the right reference.
10. **The overfit-64 probe was invalid** (see P0.5) and would have deleted experiment 1z's
    artifacts.
11. **Budget was low by about 1.5–2×**, and the launcher's default of 100k steps was not
    noted.
12. **Missed two levers already in the data and code:** site 0 is a flagged pilot, and a
    multiscale-critic path exists but is broken (UPGRADES #57). The latter is not
    pursued here; K1 is the cleaner test of H1.
