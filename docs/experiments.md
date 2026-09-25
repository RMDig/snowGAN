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
