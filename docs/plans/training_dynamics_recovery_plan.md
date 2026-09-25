# snowGAN training-dynamics recovery plan

**Status:** **Phase 0 implemented 2026-09-23** (suite 156 → 222 green). **1z run: both arms
PASS** — the implementation trains, and the λ_gp/SN hypothesis is *not* confirmed at small
scale, though the mechanism is now measured and its sign corrected (§1.2). Phase 1a
(`scripts/train_exp0_control.sh`) is the outstanding decision point.
**Revision 2** — revision 1 was substantially wrong; see §9.

## Implementation status

| Item | State | Where |
|---|---|---|
| 0.1 boolean/numeric CLI overrides | **done** | `utils.py`, `config.py`; `tests/unit/test_boolean_cli_flags.py` (37 cases) |
| 0.2 remove the silent λ_gp clamp | **done** | `Trainer._resolve_lambda_gp`; `--clamp_gp_under_sn` restores legacy |
| 0.3 critic instrumentation | **done** | `losses.critic_input_gradient_norm`, `Trainer._log_step_metrics` |
| 0.4 structured run log | **done** | `src/snowgan/metrics.py`, `scripts/check_gates.py` |
| 0.5a split-leakage fix | **done** | `DataManager.held_out_keys`; `honor_splits` default on |
| 0.5b FID `eigh` / modality / EMA guard | **done** (n=64 bias still open) | `trainer._compute_fid` |
| 0.5c NN memorization check | **done** (in 1z) | `scripts/overfit_sanity.py` |
| 0.6 scoreboard calibration | **done** | `scripts/kill_check.py`; numbers below |
| 0.7 recipe identity via save-path hash | **not done** | deferred, see below |
| 0.8 doc corrections | **done** | `experiments.md`, `UPGRADES.md` #1/#6/#7/#52-59 |

**0.7 deferred deliberately.** The review showed the drift-guard design was unworkable
(`main.py` overwrites both sidecars before `Trainer` is constructed, `atexit` flushes on
refusal, and the trainer's own mutations would make it refuse to start on its own output).
The replacement — hashing the recipe into the save path — is a checkpoint-layout change
that touches resume, the restart wrapper, and every existing `save_dir`, so it is its own
change with its own review, not a rider on this one. The immediate risk it addresses is
mitigated: 0.4's `run_start` record now pins what a run actually launched with.

**Measured calibration (0.6)** — `scripts/kill_check.py`, raw weights, n=6, seed 0:

| run | mean | std | saturated | latent diversity |
|---|---|---|---|---|
| `magnified_profiles` v0.1.0 (good) | −0.270 | 0.533 | 3.63% | **0.4898** |
| `exp0_magprof_control_256` (dead) | −0.540 | 0.841 | — | **0.000026** |
| `core_masked_256` (dead) | −0.003 | 1.000 | — | **0.000000** |

Four orders of magnitude of separation, so the diversity gate (`> 0.049`) is unambiguous.
Real-data reference: mean ≈ −0.22, std ≈ 0.63.

**Author:** drafted 2026-09-22 from a full-repo audit, a forensic diff of all thirteen run
directories across both artifact trees, and a four-lens adversarial review.
**Governs:** the next campaign. Supersedes the "Candidate increments" ordering in
[experiments.md](../experiments.md) until Phase 1 returns a verdict, and corrects two
factual claims in that file's "Ground truth" and "Retro takeaways" sections (§3, item 0.6).
**Rules it operates under:** [CLAUDE.md](../../CLAUDE.md) §2 (no bandaids), §3 (testing),
§9 (GAN experiment discipline).

---

## 0. Problem statement

Three runs in this repo's history produced real snow-crystal structure. None since has.
The working hypothesis going in was "the model learns too fast — slow it down." **The
evidence contradicts that specific hypothesis** (§1.5) while confirming the broader
instinct that the failure is in training dynamics, not architecture.

**Goal:** establish a baseline that provably trains from scratch to structure under the
*current* code, with instrumentation good enough that the next verdict is evidence rather
than inference. Then resume the increment queue.

**Non-goals:** image beauty; the CPU-RAM leak (worked around); the `tf.data` rewrite;
architecture search.

---

## 1. Diagnosis

### 1.1 The forensics had to be redone from snapshots, not terminal configs

Revision 1 of this plan read each run's *final* `discriminator_config.json` and treated it
as the recipe. That is wrong: the trainer mutates and persists training-dynamics fields
during a run (`adaptive_steps` rewrites `training_steps` at
[trainer.py:403](../../src/snowgan/trainer.py#L403); the SN clamp rewrites `lambda_gp` at
[trainer.py:223](../../src/snowgan/trainer.py#L223)). The terminal config describes the
*end* of a run, not the era that produced the model.

The per-snapshot configs in `magnified_profiles/batch_*/` carry `fade_step`, which *is*
`global_step` ([trainer.py:1113](../../src/snowgan/trainer.py#L1113)). Sorted by it, the
released run's real history is:

| global_step | disc_steps | lr_decay | spectral_norm | grad_clip | λ_gp | adaptive |
|---|---|---|---|---|---|---|
| ~146k–154k | **2** | absent | **absent** | **absent** | **10.0** | absent |
| 279,613 – 288,613 | **2** | absent | **absent** | **absent** | **10.0** | absent |
| 289,614 | 2 | **cosine on** | **SN on** | absent | 10.0 | absent |
| 295,614 | 2 | cosine | True | absent | **1.0** | absent |
| 296,614 | 2 | cosine | True | **1.0** | 1.0 | **True** |
| 298,623 → 425,644 | 1 → 2 → 4 → 8 → 16 → 32 → 47 → **61** | cosine | True | 1.0 | 1.0 | True |

**Two consequences kill revision 1's headline claims.**

1. **The "47 : 3 critic-heavy ratio" never trained anything.** It is the terminal value of
   the B3 ratchet (§1.4), reached at global_step ~422k. The run spent its first ~290k steps
   — the entire structure-acquisition era — at **disc 2 : gen 3**, the same ratio as the
   `core` release and as every legacy run.
2. **The late-era configuration was applied to a frozen model.** Cosine decay was switched
   on at global_step 289,614 under the pre-`7ab94a8` hard-coded 200k horizon, so
   `post_fade_step = 289,614 − 50,000 = 239,614 > 200,000` ⇒ progress clips to 1.0 ⇒
   **both LRs jump straight to `lr_min = 1e-7` on the first step decay is enabled.** Every
   subsequent change — SN on, clipping on, the ratchet to 61 — happened at 1e-7. They are
   decoration on weights that had stopped moving. The released checkpoint is, effectively,
   the step-~290k model.

### 1.2 The invariant that actually separates working from dead

Across all thirteen runs in both artifact trees, one split is perfect:

| | λ_gp | spectral_norm | disc:gen | gen_lr | outcome |
|---|---|---|---|---|---|
| `magnified_profiles` (structure era) | **10.0** | **off** | 2 : 3 | 1e-4 | ✅ crystal mesh |
| `core_v0.1.0_forensic` | **10.0** | **off** | 2 : 3 | 1e-4 | ✅ (blobby but structured) |
| `snowgan_slow_progressive` (legacy tree) | **10.0** | **off** | 2 : 4 | **1e-3** | ✅ **best image in the corpus, at 19k steps** |
| `exp0_magprof_control_256` | 1.0 | on | 1 : 3 | 1e-4 | ❌ |
| `core_masked_256` | 1.0 | on | 1 : 3 | 1e-4 | ❌ |
| `core_proven_v2` | 1.0 | on | **2 : 3** | 1e-4 | ❌ |
| `core_proven_recipe` | 1.0 | on | 2 : 3 | 1e-4 | ❌ |
| `core_sanity_256` | 1.0 | on | 5 : 3 | 1e-4 | ❌ |
| `core_sanity_256_snonly` | **0.0** | on | 5 : 3 | 1e-4 | ❌ |
| `core_sanity_256_v3` | **0.0** | on | 5 : 3 | 1e-4 | ❌ |
| `core/` (post-v0.2 run) | 1.0 | on | 39 : 1 | 1e-4 | ❌ |

**3/3 structured runs: λ_gp ≥ 10, spectral norm off. 8/8 dead runs: λ_gp ≤ 1, spectral norm on.**

The proximate cause is a silent clamp:
[trainer.py:223-226](../../src/snowgan/trainer.py#L223) rewrote `lambda_gp → 1.0` whenever
spectral norm was enabled, which also made `lambda_gp > 1` inexpressible under SN — so this
combination could never be tested. (Removed 2026-09-23; `--clamp_gp_under_sn` restores it.)

**What the wrong Lipschitz constant does is now measured rather than argued** — see the
1z-B box below. The effect is real and large, but it is the *opposite* of this plan's
first reading. The original argument ran: λ=1 is numerically negligible against an O(400)
Wasserstein term, so the penalty stops binding and the critic is left effectively
unconstrained by anything but Keras `SpectralNormalization` — which bounds a Conv3D's
*reshaped kernel matrix*, not the convolution's operator norm, and can be loose by up to
`∏√(k_h·k_w)` (Gouk et al. 2021). Measurement says the critic ends up **over**-constrained
instead: SN over-smooths it to ‖dD/dx‖ ≈ 0.35–0.42 and λ_gp=1 is too weak to pull it back
up toward 1. Same root cause — SN plus a weak GP put the critic at the wrong Lipschitz
constant — opposite sign, and a different predicted failure mode (weak, uninformative
gradients rather than a runaway critic). `core_sanity_256_snonly` and `_v3` ran SN with
λ_gp exactly 0 and died, so SN alone is insufficient here regardless of direction.

> **Measured 2026-09-23 (1z-B), and it corrects the sign of the mechanism above.** An 8-image
> overfit A/B at 64² ran this exact contrast. **Both arms fit** — so λ_gp/SN is not fatal at
> small scale, and the hypothesis is *not* confirmed. But the instrument shows a large,
> monotone difference in the opposite direction from "the penalty is too weak to bind":
> with λ_gp 10 / no SN the critic converges to **‖dD/dx‖ = 1.000** and its penalty decays to
> ~0; with λ_gp 1 + SN it settles at **0.35–0.42** with the penalty *stuck* at 0.33–0.47.
> The critic is **over-constrained**, not under-constrained — spectral normalization
> over-smooths it and λ_gp=1 is too weak to pull it back up to 1. Predicted failure mode is
> therefore a critic passing weak, uninformative gradients, not one running away. Whether
> that is what kills the long 1024² runs is exactly what experiment 0 still has to answer.

**Honest caveat, and it is the whole reason Phase 1 exists:** this split co-varies perfectly
with *code era* — every λ_gp-10/no-SN run predates the v0.2 audit, every λ_gp-1/SN run
postdates it. Config forensics cannot separate "the recipe changed" from "a regression
landed in the 51 commits since `fde5671`." **One run disambiguates them** (§4).

### 1.3 What the loss traces do and do not show

Revision 1 claimed the proven run had "balanced, O(1)" losses versus a "30× runaway" in
`exp0`. Both halves were artifacts of bad statistics:

- Over the proven run's first 40k steps the *medians* are gen 4.02 / disc 4.42, but the
  **median absolute values are 24.5 and 16.8**, with std 52.7 / 41.9 and p5–p95 spanning
  ±90. The medians sit near zero because the series oscillate symmetrically, not because
  the magnitudes are small. The run's critic level then drifted ~38× (4.0 → 151.6) across
  exactly the window in which it produced its best output.
- `exp0`'s −438 is the *run median*; at step 600 the rolling median is −266.
- Most importantly, −438 at 256² is **not anomalous**. `d = 196,608`, `√d = 443`. An exactly
  1-Lipschitz critic separating reals (std 0.63) from a tanh-railed generator should report
  |W| ≈ 500, with a hard ceiling of `2√d = 887`. The logged value is ~85% of the optimum.
  That is a critic doing its job against a bad generator.

**Correspondingly, revision 1's mechanism was wrong.** It argued the generator's effective
step is `lr × ‖∇_G‖` and that `‖∇_G‖` scales with critic output magnitude. Both halves fail:
`lr × ‖∇‖` is the *SGD* update rule, and this repo uses Adam for both models
([generator.py:154](../../src/snowgan/models/generator.py#L154),
[discriminator.py:63](../../src/snowgan/models/discriminator.py#L63)), whose update is
invariant to uniform gradient rescaling; and `∇_θ gen_loss` depends on `∇_x D`, not on D's
output *level*, which is free gauge here anyway since `SpectralNormalization` normalizes
`kernel` but leaves `bias` unconstrained.

The defensible mechanistic statement is narrower and lives in §1.4.

### 1.4 The defects that plausibly drive the failure

| # | Finding | Why it matters |
|---|---|---|
| **B0** | **The Lipschitz constraint does not bind.** The SN clamp forces λ_gp to 1.0 against an O(400) Wasserstein term, and SN-on-reshaped-Conv3D-kernel is a loose bound. | §1.2. The one variable that separates every working run from every dead one. |
| **B1** | **Boolean CLI flags cannot express "off."** `--spectral_norm`, `--augment`, `--multiscale_disc`, `--adaptive_steps`, `--rebuild` are all `action='store_true', default=None`; omitting one preserves the persisted JSON. Only `--mask_board` uses `BooleanOptionalAction`. | **The harness cannot run half the queue.** experiments.md's rank-1 increment is written `--no-adaptive_steps` — a flag that does not exist. Neither does `--seed`. |
| **B2** | **The critic is trained deep on a frozen 4-image batch.** `images` and `noise` are both fixed across all `disc_steps` inner iterations ([trainer.py:694](../../src/snowgan/trainer.py#L694), [706](../../src/snowgan/trainer.py#L706)). Gulrajani Algorithm 1 samples a fresh real minibatch *inside* the `n_critic` loop. | At `batch_size 4` the critic takes N Adam steps against a 4-point empirical distribution with the GP estimated on 4 interpolate segments, then the generator takes 3 Adam steps down that stale direction at full step size. This is the largest deviation from the reference implementation in the repo. (Augmentation *is* redrawn per iteration, so the critic sees N augmented views of the same 4 images — not N identical batches.) |
| **B3** | **`adaptive_steps` ratchets across restarts.** It persists `training_steps`, and `_base_disc_steps` re-reads it next launch while `max_steps = base × 2`. | The ratio is a random walk with memory: 2 → 61 in the released run, 1 in the dead ones. Its decision metric compares two uncalibrated WGAN magnitudes. |
| **B4** | **DiffAugment is not applied in the generator step.** Critic update augments ([trainer.py:711-712](../../src/snowgan/trainer.py#L711)); generator update feeds raw fakes at [trainer.py:786](../../src/snowgan/trainer.py#L786). Also `augment.py` has no **translation** — DiffAugment's strongest component for small data. | Critic trains on one distribution, generator optimizes against it on another. |
| **B5** | **Training ignores the persisted splits.** `batch()` filters on `datatype` only ([dataset.py:296](../../src/snowgan/data/dataset.py#L296)); nothing consults `trained_pool`. | **The GAN trains on its own `test_pool`.** Every downstream AvAI probe number against these backbones is contaminated — and CLAUDE.md §9 calls that probe "the real metric." |
| **B6** | **No instrument.** `‖∇_x D‖` is computed and discarded ([losses.py:31](../../src/snowgan/losses.py#L31)); `disc_loss` conflates W, λ·GP, and (when multiscale is on) `0.5 × W_lowres` from a critic that has **no gradient penalty at all** and never influences the generator. | No verdict in experiments.md was reached with a number that means what it was read to mean. |
| **B7** | **FID is broken four ways** and the control was about to gate checkpointing on it: n=64 against 2048-dim features (rank-deficient); `np.linalg.eigh(sigma_r @ sigma_f)` on a non-symmetric product ([trainer.py:492](../../src/snowgan/trainer.py#L492)); modality hardcoded to `magnified_profile` ([trainer.py:476](../../src/snowgan/trainer.py#L476)); reals drawn from the live training pointer. | Use KID (Bińkowski 2018), which is unbiased at small n, on a held-out pool. |
| **B8** | **No memorization check anywhere.** Latent diversity and saturation are properties of the generator alone. | With 2,350 magnified_profile images and ~34k critic updates per image, memorization is the most likely *success* mode and would pass every gate. |
| **B9** | Stale docs: UPGRADES #1 (double-GP) is already fixed. `load_gen_config`/`load_disc_config` ([config.py:353](../../src/snowgan/config.py#L353)) call `build(path, config)` against a one-arg `__init__`. `--latent_dim` is `type=float` and is reassigned *after* `configure()`'s `int()` cast, so passing it crashes model build. | — |

### 1.5 On "we're training too fast"

The instinct — that the failure is in training dynamics, not architecture — is right, and
experiments.md reached the same conclusion independently. **The specific remedy is not
supported.** `snowgan_slow_progressive` ran at `gen_lr 1e-3` — ten times the "proven"
rate — with no clipping, no LR decay, no EMA, and a *generator-heavy* 4:2 step ratio, and
produced the most structured image in either artifact tree **at 19,000 steps**. And the
step-ratio number revision 1 built its case on does not separate: `core_proven_v2` ran the
identical 2:3 ratio and LRs as both releases and died.

Two things that *do* make long runs worth less than they look, and both survive review:

- **The epoch loop never terminates for single-modality runs.** On exhaustion `train_ind`
  resets to 0 and `next_batch` immediately succeeds, so `trainable_data` never goes false
  ([trainer.py:547-554](../../src/snowgan/trainer.py#L547)). `current_epoch` never advances.
- **The stream is never shuffled.** `batch()` walks the manifest in order, so every batch is
  consecutive rows — same site, same column, same core — replayed in one fixed sequence
  forever.

---

## 2. Governing rules for this campaign

**Three classes of change, not two.**

1. **Validity fixes** — changes that make a measurement mean what it claims. These land
   *before* the control, accepting that they break comparability with runs that were
   already invalid. B5 (pool leakage) and B7 (FID) are here.
2. **Fixes** — change only what is *observed* or *persisted*. Batchable. B1, B6, B9.
3. **Increments** — change the gradient any model receives. One per campaign, §9 rules
   apply. B0, B2, B3, B4, and shuffling are here.

Revision 1 used a two-class rule and it misfiled the data-leakage fix behind five tuning
runs. It also misfiled item 0.3 below: running a kill-check inside the loop draws from the
global TF RNG and shifts every subsequent noise and augmentation decision, so it must use a
dedicated `tf.random.Generator` to stay observation-only.

---

## 3. Phase 0 — make the harness able to run the experiment

**0.1 — Boolean and numeric CLI overrides.** `BooleanOptionalAction` for the five flags in
B1 **keeping `default=None`** (the resume contract — switching to `default=False` would make
omission override a persisted `true`, and would trip
`assert_spectral_norm_consistency` on every existing save_dir). Normalize the two
truthiness-guarded application sites (`--rebuild` at [config.py:432](../../src/snowgan/config.py#L432),
`--fade` at [config.py:444](../../src/snowgan/config.py#L444)) to `is not None`, or `--no-`
will still be ignored for exactly those two. Add `--seed`. Fix `--latent_dim` to `type=int`
and cast at assignment. Sweep the other truthiness guards (`--gen_steps 0` is silently
dropped today). Tests: `--no-X` overrides a persisted `true`; omitting X leaves the
persisted value byte-identical.

**0.2 — Remove the λ_gp clamp.** Make [trainer.py:223-226](../../src/snowgan/trainer.py#L223)
an explicit opt-in flag rather than a silent rewrite. Today `--disc_lambda_gp 10` is
**inexpressible** under spectral norm, which makes B0 untestable and any λ_gp sweep
impossible. Log the resolved λ_gp at startup.

**0.3 — Instrument the critic.** Log, separately and as means over inner iterations (not
the last iteration, which is what `real_scores` currently retains): `E[D(real)]`,
`E[D(fake)]`, the **main critic's** Wasserstein term isolated from the multiscale
contribution, `λ·GP`, and `mean ‖∇_x D‖` measured at reals, at fakes, **and** at
interpolates. Make the norm an **unconditional probe with its own tape**, not a by-product
of `compute_gradient_penalty` — that call is skipped entirely when `lambda_gp == 0`
([trainer.py:729](../../src/snowgan/trainer.py#L729)), i.e. the instrument would vanish in
exactly the SN-only arm whose entire question is "is the critic Lipschitz?" Run it once per
`train_step` on a cadence, not per inner iteration. Accumulate as tensors and do one
`.numpy()` per step — `float(disc_loss)` is already one device sync per inner iteration.

**0.4 — Structured run log.** `metrics.jsonl`, append mode (`log.py` uses `"w"`; a writer
copied from it wipes the log on every restart). Stamp each record with a per-process
`launch_id` and a monotonic counter — `global_step` persists only every 50 steps and rewinds
on restart, and `batch` is recovered by globbing snapshot dirs and rewinds up to 999. Gates
read the last launch's rows, keyed on `global_step`, last occurrence. Rename note: UPGRADES
#16 calls this `loss.jsonl`; update it in the same commit (CLAUDE.md §7).

**0.5 — Validity fixes.** (a) `batch()` consults `trained_pool` (B5). (b) `--fid_interval 0`
until FID is replaced by KID on a held-out pool, and fix the `eigh`-on-non-symmetric-product
bug and the hardcoded modality regardless (B7). (c) Add a nearest-neighbour memorization
check — samples vs training set in pixel and feature space — to the scoreboard (B8).
(d) Wrap `_compute_fid`'s EMA swap in `try/finally`: it is the one swap site without it
([trainer.py:461-474](../../src/snowgan/trainer.py#L461)), and a failure there leaves the
generator permanently holding EMA weights while training silently continues.

**0.6 — Calibrate the scoreboard, read-only.** Copy the release to a scratch dir first —
any `snowgan` invocation pointed at `keras/snowgan/magnified_profiles/` rewrites both
sidecars ([main.py:37-47](../../src/snowgan/main.py#L37)) and again at exit via
`atexit.register(self.save_config)`, and those sidecars are the release artifacts AvAI
rebuilds from. Then run the kill-check on **raw** generator weights (not EMA) and record
latent diversity, saturation, output std, and `mean ‖∇_x D‖` as the reference row. Derive
the Phase 1 thresholds from these numbers rather than guessing them. Standing rule:
`magnified_profiles/` and `core_v0.1.0_forensic/` are read-only.

**0.7 — Recipe identity, not a drift guard.** Revision 1 proposed a guard that refuses to
start on config drift. Review killed it: `main.py` overwrites both sidecars *before*
`Trainer` is constructed, so a guard in the trainer has nothing left to compare; `atexit`
would flush the drifted config even on a refusal; and the trainer's own mutations (B0, B3)
would make it refuse to start on its own output, breaking the restart wrapper on relaunch
#1. The §2-compliant fix is to make co-location impossible: **hash the training-dynamics
fields into the save path**, so two recipes cannot share a directory. Persist the resolved
launch recipe once, as an immutable `recipe` block distinct from mutable state
(`training_steps` under adaptive, `lambda_gp` after any clamp, `fade_step`, `train_ind`).

**0.8 — Doc hygiene.** Mark UPGRADES #1 resolved; delete the dead config loaders; and
correct experiments.md's "Retro takeaways", which states the proven run was "disc-heavy
(disc 47 : gen 3)" — §1.1 shows that ratio postdates the structure era by ~130k steps and
occurred entirely at `lr = 1e-7`.

---

## 4. Phase 1 — one run that disambiguates recipe from regression

**Claim under test:** *does the current code, from scratch, train the recipe that actually
produced structure?* Because the λ_gp/SN split co-varies exactly with code era (§1.2), this
single run separates the two live hypotheses: **it trains** ⇒ the recipe was the cause and
the v0.2 audit's SN+λ_gp-1 default is the regression; **it does not** ⇒ a code regression
landed in the 51 commits since `fde5671` and Phase 2 is a bisect, not a campaign.

**Run 1a — the structure-era recipe, under current code.** Note how different this is from
revision 1's control, which pinned the *terminal* configuration:

```
--mode train --modality magnified_profile --save_dir keras/snowgan/exp0_control_1024/
--resolution "1024 1024" --batch_size 4 --seed 42
--gen_filters "1024 512 256 128 64" --gen_kernel "3 3" --gen_stride "2 2"
--gen_norm none --gen_upsampler transpose --gen_convs_per_resolution 1
--gen_lr 0.0001 --gen_steps 3
--disc_filters "64 128 256 512 1024" --disc_kernel "3 3" --disc_stride "2 2"
--disc_lr 0.00001 --disc_steps 2 --disc_lambda_gp 10.0
--no-spectral_norm --no-adaptive_steps --no-multiscale_disc --no-augment
--ada_target 0 --grad_clip_norm 0 --ema_decay 0 --lr_decay none --fid_interval 0
--epochs <explicit> --sample_batch_interval 50 --max_rss_mb <host>
```

Everything the structure era did not have is off. `--multiscale_disc` is dropped because
its low-res critic trains with no gradient penalty, never reaches the generator, and
contaminates the logged `disc_loss` — and at 256² its resize is a no-op, so run 1b would
otherwise silently get a second full-resolution critic. `--ada_target 0` because ADA's
signal is `mean(D(real) > 0)`, and a WGAN critic's zero is unconstrained gauge.
`--lr_decay none` because the structure era had no schedule; the plausible "slow anneal"
variant is an increment (§5 rank 4), not part of the control. `--epochs` must be explicit:
at 2,350 magnified_profile rows and `batch_size 4`, one pass is 587 steps, and the default
of 10 would end the run at 5,870 steps — below its own gate — the moment §5 rank 2 makes
the epoch loop terminate.

**Run 1z — the ten-minute control, run this first.** Can the code overfit a GAN to **8
fixed images** and reproduce them? This separates "the code is broken" from "the recipe is
wrong" far more sharply, and far more cheaply, than any 1024² run. If 1z fails, skip
straight to the bisect.

**Gates.** Expressed in **critic updates**, not train steps, so they stay comparable across
any later ratio sweep. Evaluated on **raw generator weights** — with `ema_decay 0.999` the
shadow retains 61% of random init at step 500 and 13% at step 2,000, so EMA-based early
gates would fail a healthy run and route the plan into a 51-commit bisect on an artifact.

| critic updates | gate | kill if |
|---|---|---|
| 1,000 | `mean ‖∇_x D‖` at reals and fakes within the 0.6-derived band | far outside ⇒ constraint not binding |
| 4,000 | `\|W_main\|` well under the `2√d` ceiling **and trending down** | growing ⇒ the critic is pulling away from the generator |
| 4,000 | latent diversity > **0.049** (0.6 reference × 0.1) | below ⇒ mode collapse |
| 20,000 | two latents give two visibly different, non-saturated images with low-frequency structure; NN check shows samples are not training images | no ⇒ dead or memorizing |

**Cost — measured 2026-09-23, and an order of magnitude better than estimated.**
`scripts/bench_train_step.py --device gpu --resolution 1024 --disc_steps 2` gives
**0.537 s/step** on the RTX 5080, so the 20,000-critic-update gate is ~10,000 train steps
≈ **1.5 hours**, and the 4,000-update gate ≈ 0.3 h.

This supersedes the earlier estimate of ~19 h, which was fitted from the released run's
checkpoint mtimes (`t ≈ 6.4 s + 0.40 s × disc_steps`, i.e. 6.8–7.5 s/batch at `disc_steps
2`). That fit is ~13× off against a working GPU toolchain, so it was measuring something
other than the arithmetic — CPU fallback, PTX thrashing, or the cuDNN problems in
UPGRADES #45. **Historical mtimes are not a cost model; benchmark the host.**

Two consequences: the "is 1024² affordable?" open question in §8 is largely answered, and
the CPU-RAM leak now arrives ~13× sooner in wall-clock (a 24 GB ceiling lands around batch
~6,000, so expect one or two `--max_rss_mb` restarts per run, each re-paying the ~10 min
manifest load).

**Free 27% win, available before any run:** because `noise` is hoisted above the inner loop,
`_generate_with_fade(noise)` recomputes an identical tensor every critic iteration, and the
generator is ~4.5× the critic per forward. Hoisting `synthetic_images` out of the loop is
behaviour-identical under current semantics.

**Run 1b — 256² reference**, same recipe, after 1a returns. Specify **both** filter lists
verbatim; naive truncation changes the `features` width that is AvAI's transfer contract
(`[256 512 1024]` → 1,048,576, matching the 1024² model; `[64 128 256]` → 262,144).

---

## 5. Phase 2A — increment queue (only if Phase 1 passes)

One piece per campaign, several tuning runs per piece, judged by best-tuned run, logged
before the next run starts.

| Rank | Increment | Cost | Why here |
|---|---|---|---|
| 1 | **Fresh real minibatch *and* fresh noise per critic inner step (B2).** | trivial | Reference-implementation parity (Gulrajani Alg. 1). At any `disc_steps > 1` the critic is currently a 4-image memorizer. Revision 1 ranked this 5th and called it "low expected effect"; review was right that this is likely the dominant defect. |
| 2 | **Shuffle the stream.** Permute *matching indices* in a separate `self._order`, seeded from `(seed, pass_index)`. **Do not permute `self.manifest`** — its row order is a cross-repo contract (`pair_index` values are HF row ids; snowGradient indexes `dataset["train"][row_idx]` with them), and permuting it yields wrong images with no exception. Single-modality path only; `batch_merged` finds partners by manifest adjacency. | small | §1.5. Fix the non-terminating epoch loop as a **separate row** — bundling them stacks two pieces (§9). |
| 3 | **DiffAugment in the generator step + add translation (B4).** | small | `--augment` is uninterpretable until this lands. |
| 4 | **LR schedule.** Sweep a real cosine horizon and `lr_min`; optionally a `gen_lr` warmup. Also sweep `gen_lr` **upward** toward 1e-3. | none | The stated hypothesis, tested honestly in both directions — `snowgan_slow_progressive` is direct evidence that 1e-3 can work (§1.5). Note `fade_steps` is an LR-schedule offset whenever `lr_decay` is on, even with `fade` off ([trainer.py:331](../../src/snowgan/trainer.py#L331)); gate that on `config.fade` or document it. |
| 5 | **Critic ratio sweep** 2 → 5 → 8, and `gen_steps` 3 → 1 (no WGAN-GP reference uses `n_gen > 1`). | none | Only meaningful after rank 1. |
| 6 | **λ_gp sweep** 10 → 20, and **re-test spectral norm** now that 0.2 makes SN+λ_gp-10 expressible. | none | Closes B0 properly instead of assuming it. |
| 7 | ADA, minibatch-stddev, R3GAN — unchanged from experiments.md, now runnable (0.1) and judgeable (0.3). | as before | |

---

## 6. Sequencing

1. Phase 0, one PR per item. No GPU time.
2. **Run 1z** (8-image overfit, ~10 min). Fail ⇒ bisect immediately.
3. Run 1a to the 20,000-critic-update gate (~19 h). **Run it twice at different seeds
   before concluding failure** — at `batch_size 4`, GAN run-to-run variance at fixed config
   is large, and a false negative costs a 51-commit bisect.
4. Fork: Phase 2A if it passed; bisect `fde5671..HEAD` (51 commits, 19 by `--first-parent`)
   using run 1z as the test if not. Prime suspects: `6549431` (resize-conv default),
   `b4e0677` (second conv + PixelNorm), `1a9a42b` (GP-disable + augment), `7ab94a8` (LR
   horizon), `275d398` (checkpoint rewrite).
5. Log every run in experiments.md before the next starts.

---

## 7. Deferred

- **The native CPU-RAM leak.** Worked around. Note `--max_rss_mb` is a **no-op off Linux**
  (`_process_rss_mb` reads `/proc/self/status` only) — if the host is the M4 mac mini the
  wrapper never sees exit 75 and the campaign stops on the first OOM.
- **`tf.data` (UPGRADES #10), pydantic config (#9), Trainer split (#22).**
- **Architecture changes.** Frozen until a baseline exists.

---

## 8. Open questions for Denny

1. **The critic head is a stop-and-ask, not a defer.** The 1,048,576-wide
   `Flatten → Dense(1)` has no pooling, so the critic can memorize absolute pixel positions
   — the worst inductive bias for 2,350 images, and exactly what DiffAugment/ADA exist to
   fight. Worse, a 1M-dim frozen tap makes AvAI's linear probe `p ≫ n` by three orders of
   magnitude, so the transfer contract that justifies protecting it is itself broken by it.
   This needs a decision before GPU-months, not after.
2. **The scoreboard's deciding tier does not exist.** experiments.md marks the downstream
   probe "Gated on labels (TBD)". Combined with B5, no valid probe number has ever been
   produced. What is the arbiter for Phase 2A?
3. **SimCLR gate.** experiments.md already recommends standing up a SimCLR arm on the same
   conv encoder *before* GPU-months, citing the modest GAN-disc transfer result (~+5%,
   Xiang & Li 2020). Phase 1 is a justified spend regardless. Should Phase 2A wait on a
   SimCLR probe number to beat?
4. **Phase 2B budget** if the control fails: bisect 51 commits, or check out `fde5671`,
   retrain there, and forward-port?

---

## 9. What revision 1 got wrong

Recorded because CLAUDE.md §9 exists to stop the same ground being re-derived.

| Claim | Status |
|---|---|
| Proven run was "disc 47 : gen 3" | **Wrong.** Terminal value of the B3 ratchet, reached at global_step ~422k under `lr = 1e-7`. The structure era was 2 : 3. |
| "gen÷disc movement 0.64× (good) vs 30× (dead)" is the cleanest separator | **Wrong.** Arithmetic error (`core` is 15×, not 1.5×) and the corrected ratio does not separate — `core_proven_v2` died at the same 15× as both releases. |
| Effective step size is `lr × ‖∇_G‖`, inflated by critic output scale | **Wrong.** Adam is invariant to uniform gradient rescaling; `∇_G` depends on `∇_x D`, not D's level, which is free gauge. |
| Proven run had "balanced, O(1)" losses; `exp0` showed a "30× runaway" | **Wrong.** Median of a sign-oscillating signal; median-\|disc\| is 16.8, not 4.26. −438 at 256² is ~85% of the 1-Lipschitz optimum, not a runaway. |
| `core/` is the release "continued with the recipe swapped in place" | **Wrong.** Different runs — zero identical loss lines over 127,616 rows, different sample intervals, different images, incompatible architectures. `core/` is a wipe-and-reuse of the directory. |
| `grad_clip_norm 1.0` separates working from dead | **Wrong.** The `magnified_profiles` structure era had clipping **off**. Under Adam it is a stability measure, not a step-size bound. |
| Recipe-drift guard | **Unworkable as specified** — see 0.7. |
| Gate on `0.3 ≤ \|gen_loss\|/\|W\| ≤ 3` | **Wrong, caught during implementation.** `gen_loss` is `−E[D(fake)]`, an absolute critic level, and that level is free gauge (SN normalizes the kernel, not the bias). Dividing it by a gauge-invariant difference is meaningless: a healthy 64² control read 19.2 precisely *because* `\|W\|` had converged toward 0. Replaced by "is `\|W\|` trending down". |
| Epoch loop never terminates; stream never shuffled; B1 boolean flags; B3 ratchet; B4 augment; B6 no instrument; UPGRADES #1 stale | **Verified, all.** |
