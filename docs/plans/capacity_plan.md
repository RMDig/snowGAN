# Capacity campaign — where is the ceiling, and what moves it?

**Status:** DRAFT 1 (2026-10-15), pending parallel review. Nothing here has run.
**Builds on:** [increment_campaign.md](increment_campaign.md) and
[../experiments.md](../experiments.md) (Arms A–D and the EMA horizon run).

---

## 0. Why this campaign exists

The increment campaign ended with one adoption and one ceiling:

- **EMA 0.999 is adopted.** It was replicated at two seeds (EMA − baseline: −0.047 at
  1.9σ, −0.088 at 3.0σ). The gain is pure averaging: the same run's raw weights tie
  the baseline.
- **Training length and LR are not the lever.** EMA KID levels off by 50–60k steps at
  both seeds. 80k → 160k tied at both seeds (−0.003, −0.019). The anneal did not
  replicate. λ_gp 20 damped critic escape and tied on KID.

So the recipe plateaus at an EMA KID of about **0.24–0.25** on the held-out pools. The
remaining levers are the **model** and the **data**. This plan is about testing which
capacity dimension is binding, cheaply, before any 28 h arm.

## 1. Facts this plan rests on

Each is re-derivable from the code or artifacts named. Reviewers: check them.

**F1. Generator (`models/generator.py`, recipe in `scripts/train_exp0_control.sh`).**
- Latent 100 → `Dense(16·16·1024)` → reshape to (1, 16, 16, 1024).
- Then 5 × `Conv3DTranspose` (filters 1024, 512, 256, 128, 64; kernel (1,3,3); stride
  (1,2,2)), each followed by LeakyReLU, no normalization, `gen_convs_per_resolution 1`.
- Then a transpose-conv toRGB to 1024², tanh.
- **41.9M parameters, of which the dense layer is ~26.2M (63%).** The conv stack is about
  15.6M.
- The 3×3 kernel at stride 2 is not divisible by the stride, which is the textbook
  checkerboard setup. Checkerboard texture is visible in previews (Arm B, 14k).

**F2. Critic (`models/discriminator.py`).**
- 5 × `Conv3D` (64, 128, 256, 512, 1024; kernel (1,3,3); stride (1,2,2)), each followed
  by LeakyReLU.
- Then `Flatten` (named `"features"`) → `Dense(1)`.
- **7.3M parameters, 5.7× smaller than the generator.** The final feature map is
  32×32×1024, so the dense head is ~1.05M parameters.
- **Receptive field of each final feature: 63×63 px of a 1024² image.**
- **The only cross-patch interaction is the linear head**: the critic score is a learned
  linear combination of 1,024 local (63 px) patch features. It cannot nonlinearly judge
  global structure (vignette, grid geometry, crystal placement across the frame).
- **Downstream contract:** AvAI calls `get_layer("features")` on the loaded critic
  (UPGRADES #3/#7). Any critic change alters that tensor's shape.

**F3. Data.**
- 2,350 magnified_profile images in 45 (site, column, core) groups, sites 0–6.
- Seed 42 trains on **2,006 images / 36 groups, all seven sites**. Seed 43 trains on
  1,741 images / 36 groups.
- **Sites 3–6 are already in every campaign run's training pool.** Only the v0.1.0
  releases predate them. "Add sites 3–6" is not an untried lever for these runs.
- `memorization_check.py` on control_1024: ratio 1.41 (> 1 = novel output), so not
  memorizing.

**F4. Metric floors (measured 2026-10-15, `kid_check.py --real_vs_real`, group-disjoint
halves).**
- Seed 42: **0.128 ± 0.003** (206 vs 138 images).
- Seed 43: **0.033 ± 0.001** (390 vs 219 images).
- The floors differ 4×. With 9 held-out groups the floor is dominated by between-group
  variation, so it is a noisy reference. Even so, EMA KID ≈ 0.25 is clearly above both,
  so there is headroom.

**F5. Cost.**
- 1.12 s/step at batch 4 without EMA, 1.18 s with EMA. VRAM ~10 GB of 16 GB in training.
- 80k steps ≈ 27–28 h. EMA plateaus by 50–60k.
- Three-seed window scoring is about 15 min per run window.

## 2. Hypotheses: which capacity dimension binds?

| | hypothesis | predicts |
|---|---|---|
| **H1** | **Critic is too local / too weak.** A 63 px RF plus a linear head means the critic polices texture but not global structure, so the generator gets no gradient toward correct layout. | Patch-level statistics near the floor, global statistics far from it. Samples with realistic texture but wrong global composition. |
| **H2** | **Generator conv capacity is thin.** One conv per resolution and 15.6M conv params; most capacity is in a dense layer that only sets the 16×16 layout. | Texture/fine-detail statistics far from the floor even at patch level. Fails to overfit a small set at 1024². |
| **H3** | **Checkerboard artifacts** (k3/s2 transpose) put a high-frequency signature into every sample that Inception features pick up. | KID improves with kernel 4/stride 2 alone. Artifact visible in a spectral (FFT) comparison. |
| **H4** | **Data ceiling.** 2,006 images / 36 groups is the limit. | Critic overfits: its scores separate train reals from held-out reals. More capacity does not help. |

These are not exclusive. The point of Phase 0 is to rank them before spending GPU-days.

## 3. Phase 0 — diagnostics (no long training; ~1 day total)

**P0.1 Locate the gap by scale (no training, ~2 h GPU).** Score the best EMA checkpoints
(arm_d_ema 57k, arm_d_ema_s43 79k) and the real-vs-real floor at:
- (a) full frame downsampled to 299 (current KID);
- (b) full frame at 128² then up to 299 (global composition only);
- (c) random 299² crops at native 1024² (texture only; e.g. 10 crops per image).

Read: the gap-to-floor ratio per scale. Gap concentrated in (b) → H1. In (c) → H2/H3.
Both → no single dimension; go to P0.2/P0.3 results.

**P0.2 Generator representational probe (~4 h).** Overfit 64 fixed training images at
1024² with the current generator and a deliberately strong critic, as in experiment 1z,
for 10k steps. Measure nearest-neighbour distance and crop-KID against those 64.
- Fails to fit → the generator cannot represent the data at this size → H2 is binding.
- Fits → generator capacity is sufficient for the content; look at the critic or data.

**P0.3 Critic generalization (no training, ~1 h).** For snapshots across arm_d_ema, score
the critic on (i) training reals, (ii) held-out reals, (iii) EMA fakes.
- `D(train) − D(held-out)` growing over training → the critic is memorizing → H4.
- Near zero → the critic is not data-limited, consistent with H1 (too weak, not
  overfitted).

**P0.4 Spectral check (no training, ~30 min).** Azimuthally averaged power spectrum of
EMA fakes vs reals. A peak at the stride frequency → H3 is live.

**Decision after Phase 0:** rank H1–H4 by evidence and run arms only for the live ones,
in that order.

## 4. Phase 1 — arms (each a single change on the adopted recipe)

**Base for every arm:** `train_exp0_control.sh --ema_decay 0.999`, seed 42, 80k steps.
The reference is `arm_d_ema`'s already three-seed-scored 70–79k window (0.2554 ±
0.0071). Same decision rule (§2 amendment of increment_campaign). Replicate any
distinguishable winner at seed 43 against `arm_d_ema_s43` (0.2567 ± 0.0104).

| arm | single change | targets | code needed | risk |
|---|---|---|---|---|
| **E1** | Critic: 2 more stride-2 layers (→ 8×8×1024, RF ~255 px) | H1 | config only (`--disc_filters` length) | gradient penalty under a deeper critic; changes `features` shape |
| **E2** | Critic: minibatch-stddev layer + 1 nonlinear layer before the head | H1 | small (new layer) | UPGRADES #48: minibatch-stddev "collapsed by 2k" in the stacked-change era — untested alone |
| **E3** | Generator: 2× conv width (2048 … 128) | H2 | config only | VRAM; step time |
| **E4** | Generator: kernel 4, stride 2 (both nets) | H3 | config only | changes the parameter count of every layer — not truly single-change |
| **E5** | Generator: `gen_convs_per_resolution 2` | H2 (depth) | config only | UPGRADES #47: un-normalized it vanished to a flat frame; must not run without a norm decision |

Ordering is set by Phase 0, not by this table. Expected budget: 2–3 arms + 1–2
replicates ≈ 5 × 28 h ≈ 6 days of GPU.

**Every arm also reports:** step time and VRAM (cost of the capacity), the per-scale KID
from P0.1 (did it close the gap where predicted?), ‖∇D‖ (validity), diversity and the
memorization ratio (gates).

## 5. What this plan cannot tell us

- Whether a capacity gain transfers to the downstream task (AvAI uses the critic's
  features). Changing the critic changes those features. A KID win and a transfer win
  are separate questions.
- With 9 held-out groups, KID floors are noisy (F4). A model can only be ranked against
  another model, not against "real".
- Seeds also change the split (increment_campaign §2 correction), so replicates are
  paired within a seed.
