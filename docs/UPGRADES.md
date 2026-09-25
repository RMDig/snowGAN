# snowGAN Upgrade Roadmap

Prioritized by blast radius. Tiers labeled 🔴 (bugs that silently corrupt training or break
features), 🟠 (production blockers — correctness, reproducibility, observability), 🟡 (code
health / velocity), 🟢 (nice-to-have). Paired with [architecture.md](architecture.md).

## Resolved — v0.2 core-model audit (2026-06-12, branch `feat/snowgan-core-v0.2`)

A five-lens audit of the divergent core run produced these fixes. All ship with
focused regression tests; v0.2 is a from-scratch retrain (the generator changes
break the checkpoint format).

- ~~**Cosine LR horizon hard-coded to 200k.**~~ With `fade_steps=50k` both LRs
  floored at `lr_min` once `global_step` crossed 250k and froze learning for the
  rest of the run — the real cause of the post-250k "destabilization" read as
  disc/gen competition. Horizon is now config-driven (`lr_decay_steps`), with a
  long-horizon fallback + warning and live-LR logging each save.
- ~~**Checkerboard generator.**~~ Every upsample (incl. toRGB) was a stride-2
  `Conv3DTranspose` with kernel 3 (not divisible by stride) — textbook
  deconvolution checkerboard. Replaced with resize-convolution (`UpSampling3D`
  + stride-1 `Conv3D`); toRGB is now a stride-1 1×1 conv.
- ~~**Generator had no normalization.**~~ Added GP-safe `PixelNorm` (+ a second
  conv per resolution), preventing the activation drift that saturated the
  output tanh into the monochrome-blue collapse. New `gen_norm` config field
  (pixel|batch|none), derived from legacy `batch_norm` when absent.
- ~~**SN+GP double Lipschitz constraint.**~~ `lambda_gp=0` now genuinely
  disables the gradient penalty (the `float(x) or None` coercion had forced it
  back to 10.0), so the v0.2 critic relies on spectral norm alone.
- ~~**Augment manifold + entropy.**~~ GP is computed on the same augmented
  tensors the critic scores (not the raw manifold), and differentiable
  augmentation is now per-image instead of one scalar decision per batch.

## Tier 🔴 — correctness bugs to fix before the next training run

0. **Native CPU-RAM leak OOM-kills long runs (~+3.6 MiB/batch).** 🔴 OPEN — the
   single biggest blocker to a multi-day v0.2 run.

   **Symptom.** Process RSS climbs dead-linearly (R²=1.0) at ~+3.6 MiB/batch and
   the kernel OOM-kills training (observed: 31 GB RSS / `uord`=37 GB live arena,
   `Killed` at batch ~15k). It is a *live* glibc-arena allocation (`mallinfo2`
   `uordblks` tracks RSS; `malloc_trim` reclaims ~0) — **not** mmap/cuDNN
   (`hblkhd`/`maps_count` flat, disproving the earlier Blackwell-JIT theory) and
   **not** Python (`gc` object count + `tracemalloc` flat).

   **What's been ruled out** (2026-06-14/15 hunt, tooling on
   `diag/memtrace-native-v2`: `src/snowgan/memtrace.py`,
   `scripts/analyze_memtrace.py`, `scripts/bench_disc_loop.py`):
   every per-batch component is flat in isolation — disc forward/backward (SN on
   or off), generator forward, **generator training/backprop**, `diff_augment`,
   EMA (real-run A/B with `--ema_decay 0`), data load (`DataManager.next_batch`),
   NumPy-vs-tensor image feed, and `plot_history`. A faithful bench combining
   real HF images + full train step + gen training + plot stays **flat**, yet the
   assembled long-running real process leaks. In-situ per-phase `uord` marks are
   unreliable (glibc commits arena async between microsecond-apart marks — they
   reported +1.1 MiB in a bracket containing zero allocating code), so only the
   per-batch slope is trustworthy.

   **Working theory.** A native TF-eager / CUDA host-allocator accumulation tied
   to the assembled, long-running, trained-model process (possibly value- or
   process-age-dependent) — which is why a fresh 400-step bench never reproduces
   it.

   **Workaround (shipped, PR #26).** `--max_rss_mb` + `scripts/train_with_restarts.sh`:
   the trainer saves and exits (code 75) before OOM, the wrapper relaunches and
   resumes. Atomic checkpointing (#8) makes the kill/resume safe. This unblocks
   v0.2 but does not fix the leak.

   **Next diagnostic step.** LD_PRELOAD a tcmalloc/gperftools heap profiler on a
   short real run — it names the C++ allocation call stack (TF op / CUDA /
   eager-context) directly, the one tool that ends the hunt. (smaps-per-mapping
   diff is a cheaper-but-weaker fallback: identifies the growing region, not the
   caller.)

1. ~~**Double-applied gradient penalty.**~~
   **Resolved — verified 2026-09-23.** `compute_gradient_penalty` returns the *unscaled*
   `mean((‖∇‖−1)²)` and `Discriminator.get_loss` multiplies by λ exactly once. This entry
   had been sitting open as 🔴 against code that no longer had the defect. Regression
   coverage added:
   `tests/unit/test_lipschitz_instrument.py::test_gradient_penalty_is_applied_exactly_once`
   pins a known ‖∇‖=3 critic to penalty 4.0 and loss 40.0 at λ=10 (not 400).

2. ~~**`_ensure_depth_alignment` silently discards trained weights.**~~
   **Resolved 2026-05-09 (PR #12).** `DataManager.PAIR_DEPTH = 2` is now the
   single source of truth for the trainer's expected stack depth.
   `Trainer.__init__` syncs `config.depth = PAIR_DEPTH` and rebuilds gen/disc
   *before* loading weights — fresh models, no weights to lose. The per-batch
   `_ensure_depth_alignment` is now a hard assertion that surfaces upstream
   contract violations as `RuntimeError`. Co-resolves #40.

3. ~~**`inference.run_inference` is broken end-to-end.**~~
   **Resolved 2026-05-09 (PRs #7 + #11).** The Flatten layer was renamed to
   `name="features"` (PR #7) so AvAI's `prepare_backbone_for_transfer` resolves
   the tap by name. The broken `inference.py` itself was deleted (PR #11);
   `--mode infer` now redirects users to AvAI with `SystemExit(1)`. The
   transfer-learning pipeline lives in [AvAI](https://github.com/dennys246/AvAI)
   going forward.

4. **`configure_device` sets env vars after TF is already imported.**
   [main.py:1-8](../src/snowgan/main.py#L1-L8) triggers `import tensorflow` transitively
   before `configure_device` runs, so `CUDA_VISIBLE_DEVICES=-1` and `TF_XLA_FLAGS` are no-ops.
   Fix: create a `snowgan/bootstrap.py` that parses `--device`/`--xla` from `sys.argv`, sets
   env, and only then lets the rest of the package import TF. Invoke it first in the
   console-script entry.

5. ~~**`--resolution` flag is dead.**~~
   **Resolved 2026-07-17.** Was `type=set`, so `--resolution "256 256"` became the
   character set `{'2','5','6',' '}` and the requested size was silently discarded (the
   config kept its 1024 default → guaranteed real/fake size mismatch on any low-res run).
   Now `type=str`, split on whitespace into `[int, int]`, mirroring `--gen_kernel`.
   Regression test: `tests/unit/test_resolution_cli.py`.

6. ~~**Boolean CLI flags accept any truthy string.**~~
   **Resolved 2026-09-23** — and the entry understated the problem. Beyond `--fade`'s
   `type=bool`, five flags (`--spectral_norm`, `--augment`, `--multiscale_disc`,
   `--adaptive_steps`, `--rebuild`) were `action='store_true', default=None`, so they
   could only ever be turned **on**: once a value was persisted, no command line could
   turn it off. That made half of [experiments.md](experiments.md)'s increment queue
   unrunnable — its rank-1 entry is written `--no-adaptive_steps`, a flag that did not
   exist — and it is why several runs carried settings nobody intended. All six now use
   `argparse.BooleanOptionalAction` with `default=None` (the resume contract: omission
   preserves the persisted value, because the restart wrapper replays argv on every
   relaunch). `--rebuild` and `--fade` also had truthiness-guarded application sites in
   `configure_generic` that would have swallowed `False`; both are now `is not None`.
   `--lr_decay` gained an explicit `none` choice for the same reason. Regression coverage:
   `tests/unit/test_boolean_cli_flags.py` (37 cases — each flag's on/off/omitted
   behavior).

7. ~~**`latent_dim` is typed `float` but used as int.**~~
   **Resolved 2026-09-23** — and it was worse than "risk". `configure_generic` assigned
   the raw arg *after* `configure()`'s `int()` cast, so passing `--latent_dim` left
   `100.0` on the config and it reached `keras.Input(shape=(100.0,))`. It had never bitten
   only because no script in the repo passed the flag. Now `type=int` with an `int()` cast
   at the assignment. `--seed` was also added: it existed in the config schema and was
   read by `main.py`, but had **no CLI flag at all**, so the reproducibility story had no
   command-line surface.

32. **Stale trainer-owned optimizers are never used.**
    [trainer.py:75-81](../src/snowgan/trainer.py#L75-L81) constructs
    `self.gen_optimizer` / `self.disc_optimizer` from the configs, but `train_step` at
    [trainer.py:233](../src/snowgan/trainer.py#L233) and
    [trainer.py:259](../src/snowgan/trainer.py#L259) applies gradients via
    `self.disc.optimizer` / `self.gen.optimizer` — the optimizers attached to the Keras
    models at build time. Any learning-rate / β changes Denny thinks he's injecting through
    the trainer are silently ignored. Root cause: two optimizer instances per model, one of
    them dead. Fix: delete the trainer-level optimizer fields entirely and delegate to the
    model's own optimizer (single source of truth).

33. ~~**Logged loss is last-step-only, not averaged over `gen_steps` / `disc_steps`.**~~
    **Resolved 2026-05-09 (PR #13).** `train_step` now accumulates per-iteration
    losses into lists and appends the mean (via `Trainer._mean_loss`). Empty-list
    case returns 0.0 so the appended value is always a finite float, guarding
    `training_steps=0`. `_update_adaptive_steps` consumes the mean too, which
    stabilizes the EMA without changing the adaptive-steps decision logic.

34. **Config JSON is rewritten non-atomically every training step.**
    [config.py:139](../src/snowgan/config.py#L139) opens the config file with `open(..., 'w')`
    + `json.dump` — no temp-file + `os.replace()`. `Trainer._sync_fade_progress(persist=True)`
    at [trainer.py:268](../src/snowgan/trainer.py#L268) calls `save_config()` every single
    `train_step`, so a crash or SIGKILL mid-write leaves the config truncated. On resume,
    JSON parse fails and the whole run state is lost. Root cause: non-atomic writes in a
    high-frequency path. Fix: atomic writes (`tmp + os.replace()`), and demote the sync
    frequency from per-step to per-milestone. Pair with #8.

35. **`reset_seen_profiles` persists the cleared set before the epoch makes progress.**
    [trainer.py:132](../src/snowgan/trainer.py#L132) clears `seen_profiles` at epoch top;
    the very next `_sync_fade_progress(persist=True)` writes the empty set to disk. A
    SIGKILL anywhere later in the epoch means resume loses that epoch's profile bookkeeping
    and will re-pair profiles it already used. Root cause: mutation is eagerly persisted
    outside a transaction boundary. Fix: treat in-epoch `seen_profiles` as append-only;
    persist the reset only once the next epoch has committed at least one pair (or on
    graceful epoch-end only).

36. **Batch counter is recovered by globbing files that `_cleanup_saved_batches` deletes.**
    [trainer.py:137-141](../src/snowgan/trainer.py#L137-L141) derives the next `batch`
    number by scanning `synthetic_images/batch_*.png` for the max. `_cleanup_saved_batches`
    (called on milestone saves) trims those very files. After a milestone, resume reads a
    stale max → restarts at a lower batch number → overwrites existing
    `batch_N/generator.keras` snapshots. Silent data loss of trained checkpoints. Root
    cause: counter lives in an external filesystem artifact that another subsystem is
    authorized to delete. Fix: persist `global_batch` in config alongside `fade_step`,
    drive the loop from that, delete the glob path.

50. ~~**rmdig `datatype` int→string — defensive normalization (NOT a training-breaker).**~~
    **Resolved 2026-08-17. Reclassified 🔴→🟡 — see the correction below.** rmdig re-uploaded
    `rocky_mountain_snowpack` (3→7 sites, 1886→4040 rows); `metadata/preprocessed.jsonl` now
    stores `datatype` as strings (`'core'`/`'profile'`/`'magnified_profile'`) instead of
    ints.

    **Correction:** an earlier version of this entry (and PR #28) claimed the change broke
    `pair_index` / `batch()` and therefore snowGAN's own training. **That was wrong.** HF
    `datasets` declares `datatype` as
    `ClassLabel(['core','profile','magnified_profile','crystal_card'])`, so
    `load_dataset(...).to_pandas()` casts it to ints 0–3 — exactly `DATATYPE_TO_INT`.
    `DataManager` therefore sees ints and was never broken. The empty-`pair_index` symptom
    only appears when reading the JSONL directly with pandas (bypassing the cast), which is
    how it was mis-diagnosed.

    **What the fix does (defensive, still worth keeping):** `normalize_datatype()` maps the
    rmdig string, the modality name, or an already-int value to the canonical int, raising
    on anything unknown (never silently skipping); `DataManager.__init__` applies it to the
    manifest `datatype` column at load. This decouples the data layer from *whether* the
    ClassLabel cast happens (correct for either encoding), gives a single source of truth for
    the mapping so external readers of `dm.manifest` (snowGradient) don't re-derive it and
    drift, and turns an unknown datatype into a loud error. Idempotent. `batch()`'s
    `self.translator[datatype]` (which `KeyError`'d on an int arg) is now
    `normalize_datatype(datatype)`. Regression test `tests/unit/test_datatype_schema.py`
    (synthetic string manifest, no download).

    **Downstream contract:** `dm.manifest`'s `datatype` is an int (0/1/2/3) — already what
    `load_dataset` yields; this guarantees it regardless of encoding. snowGradient's `!= 0`
    filter is correct against it. (`wind_loading` int→string in the same upload was a genuine
    `load_dataset` `ValueError` — card declared `['none','low','moderate','high']` vs data
    `'medium'` — and was fixed dataset-side; separate from snowGAN.)
46. ~~**Adaptive `disc_steps` ratchets its own ceiling on every resume.**~~
    **Resolved 2026-07-16.** `_update_adaptive_steps` wrote the evolved step count back
    into `disc.config.training_steps`, which `Config.dump()` persists. On resume,
    `Trainer.__init__` read that evolved value as `_base_disc_steps` and set
    `max_steps = base * 2` — so the ceiling doubled per restart instead of staying
    anchored to the launch value. Observed on the core run: disc_steps 1 (batch 100k)
    → 6 → 10 → 16 (batch 332k) → 37, i.e. 37 critic updates per generator update.

    Root cause: `training_steps` served as both the launch parameter and the live
    adaptive counter, and persisting it conflated the two. Note `training_steps` is not
    in the resume-state contract (CLAUDE.md §5 lists `train_ind`, `seen_profiles`,
    `fade_step`, `current_epoch`, `global_batch`) — it was never meant to survive a
    process.

    Fix: the live counts (`Trainer._disc_steps` / `._gen_steps`) are runtime-only and
    die with the process; `config.training_steps` is the launch parameter and is never
    written during training. Regression test: `tests/unit/test_adaptive_steps.py`
    simulates three save/resume cycles and asserts the ceiling does not move.

    **Interacts with #0.** Each `--max_rss_mb` restart was a ratchet click, so the leak
    workaround accelerated the ratchet, and the extra disc steps in turn made the leak
    arrive sooner — a feedback loop. **Migration:** configs written before this fix hold
    a poisoned `training_steps` (the core run's reads 37). It is indistinguishable from
    a deliberate launch value, so it cannot be auto-detected — pass `--disc_steps N`
    explicitly once on the next launch to reset it.

    **Unverified lead for #0.** At disc_steps=37 the observed leak was ~46 MiB/batch
    (RSS 23914 → 23960 → 24007 on consecutive batches) against the ~3.6 MiB/batch
    measured when disc_steps was ~3. 37 × ~1.2 MiB ≈ 46 MiB fits, which would make the
    leak **per-train-step, not per-batch**. If so, the flat `bench_disc_loop.py` result
    may simply have run too few inner steps to show slope. Worth pinning the step count
    as an axis in the next heap-profiler run before trusting any per-batch figure.

## Tier 🟠 — production readiness (do before calling this a product)

49. **Blue measurement-board masking for the core modality** (`--mask_board`, added
    2026-07-25). Core photos are a snow sample on a blue ruler board; the board, ruler,
    and printed "Centimeters" text dominate every frame and are a confound for the
    downstream avalanche-risk transfer task. Every core GAN run collapsed onto the board
    (it is the most consistent, learnable structure), so the discriminator learned a
    blue-board detector — useless as a transfer backbone. Diagnosed by finally *looking*
    at a real core image (should have been step one).

    Fix: `mask_blue_board` (in `data/dataset.py`) zeroes the chromatic-blue board to
    neutral grey via an HSV rule. The board hue is lighting-stable — measured peak
    168-172 on PIL's 0-255 scale across a 10-image spread of the whole split, spread of
    4 — because lighting moves brightness (value), not hue; an RGB threshold would not
    generalize. Snow is achromatic (low saturation) and survives. Verified on the hard
    case (loose snow, no column, ruler present): board + ruler removed, snow structure
    kept. Runs identically in training and the on-device phone pipeline (per-pixel
    arithmetic, no accelerator needed), so both ends see the same input.

    **UNCERTAINTY — grey vs black fill (revisit if masked runs underperform).** Masked
    board pixels are filled with **neutral grey** (127.5 → 0.0 after the /127.5-1 rescale,
    the centre of tanh's linear region). This is a *reasoned default, not a validated
    one*: grey removes the confound WITHOUT handing the generator a large flat region at
    a tanh rail, which **black** (→ -1) would — and re-inviting a rail is precisely the
    saturation failure (#47/#48) we just escaped. But grey has not been shown superior
    for *feature transfer*; it is possible black (or per-image snow-mean) transfers
    better despite the rail risk, or that the fill choice barely matters. If masked
    training still degrades — collapse, or poor downstream probe accuracy — **try black
    as the documented fallback** before assuming masking itself failed. The fill value
    lives in one constant (`_BOARD_FILL_255`) to make the swap a one-line change.

### Found by the 2026-09-22 training-dynamics audit

Full write-up and the campaign that acts on them:
[plans/training_dynamics_recovery_plan.md](plans/training_dynamics_recovery_plan.md).

52. ~~**The silent λ_gp clamp.**~~ **Resolved 2026-09-23.** `Trainer.__init__` rewrote
    `lambda_gp → 1.0` whenever `spectral_norm` was on, then persisted it — so the config
    stopped describing the run, and `lambda_gp > 1` became **inexpressible** under SN.
    That matters because λ_gp is the one variable separating every run in this repo that
    produced structure (≥10, SN off) from every run that collapsed (≤1, SN on). Now
    opt-in via `clamp_gp_under_sn` / `--clamp_gp_under_sn`, with the decision extracted to
    the testable `Trainer._resolve_lambda_gp`.

53. ~~**No Lipschitz instrument.**~~ **Resolved 2026-09-23.** `compute_gradient_penalty`
    computed the critic's input-gradient norm — the quantity the penalty exists to drive
    to 1.0 — and discarded it, while `disc_loss` conflated the Wasserstein term with
    `λ·GP` (plus `0.5·W_lowres` when multiscale is on). So **no verdict in
    [experiments.md](experiments.md) was ever reached with a number that meant what it was
    read to mean.** Added `losses.critic_input_gradient_norm` as a standalone probe —
    deliberately *not* a by-product of the penalty, because the train step skips the
    penalty entirely at `lambda_gp == 0`, i.e. the instrument would vanish in exactly the
    SN-only arm whose question it answers. It runs on its own cadence
    (`--grad_probe_interval`), with its own `tf.random.Generator` so instrumentation
    cannot perturb the training stream, and with `training=False` so it cannot advance the
    critic's spectral-norm power iteration.

54. ~~**Training ignored the persisted splits.**~~ **Resolved 2026-09-23.**
    `derive_splits` partitioned groups 80/10/10 and `Trainer` mirrored the pools onto both
    configs specifically so AvAI could evaluate against `test_pool` — but `batch()`
    filtered on `datatype` alone and no batch path ever consulted them. **The GAN trained
    on its own test pool**, which invalidates the downstream transfer probe CLAUDE.md §9
    calls "the real metric", for both released backbones. Now gated on `honor_splits`
    (default **on**; `--no-honor_splits` reproduces the old stream). Excludes
    validation+test rather than filtering *to* `trained_pool`, because the pools derive
    from `pair_index` and filtering to them would silently discard every unpaired group.

55. **DiffAugment is not applied in the generator step.** 🔴 OPEN. The critic update
    augments real and fake ([trainer.py:711-712](../src/snowgan/trainer.py#L711)); the
    generator update feeds raw fakes at
    [trainer.py:786](../src/snowgan/trainer.py#L786). The critic therefore trains on one
    distribution and the generator optimizes against it on another, which defeats the
    point of *differentiable* augmentation. `--augment` was on in every run in the
    forensic table. Separately, `augment.py` has no **translation**, DiffAugment's
    strongest component for small data. Deferred to the increment queue rather than fixed
    here: it changes the gradient the generator receives, so under the plan's §2 rule it
    is an increment, not a fix.

56. ~~**`adaptive_steps` ratchets across restarts.**~~ **Already resolved on `main` by
    #46 (PR #30, merged 2026-08-18) — this entry was written against a stale branch.**
    The audit rediscovered the ratchet independently and proposed mitigating it via the
    CLI; #46's fix is better and is the one in effect: the live counts
    (`Trainer._disc_steps` / `._gen_steps`) are runtime-only and `config.training_steps`
    is never written during training, so the launch parameter and the live counter can no
    longer be conflated. `--no-adaptive_steps` (also from PR #30) makes a persisted
    `true` clearable.

    Retained here only for the forensic point it anchors: the released
    `magnified_profiles` config records `disc_steps 47` because of this ratchet, which is
    why that number must not be read as a deliberate recipe choice. See the correction in
    [experiments.md](experiments.md).
57. **The multiscale critic is nearly inert but contaminates the loss.** 🟠 OPEN. The
    low-res head trains with a bare `mean(fake) − mean(real)` and **no gradient penalty**
    ([trainer.py:747-767](../src/snowgan/trainer.py#L747)), its gradient reaches only its
    own variables, and the generator loss reads `self.disc.model` alone — so it never
    influences the generator. But `0.5 × disc_loss_lr` is folded into the logged
    `disc_loss`. And `_build_lowres_disc` hard-codes a 256×256 input, so at
    `--resolution "256 256"` its resize is a no-op and it becomes a second
    full-resolution critic.

58. ~~**FID: non-symmetric `eigh`, hardcoded modality, unguarded EMA swap.**~~
    **Partly resolved 2026-09-23.** `np.linalg.eigh(sigma_r @ sigma_f)` was computing
    eigenvalues of a different matrix — the product of two symmetric PSD matrices is not
    itself symmetric, and `eigh` reads only the lower triangle — so the result was not a
    monotone transform of FID. Replaced with the similarity-transform form
    `sqrt(Σr) Σf sqrt(Σr)`, which is symmetric PSD and shares the eigenvalues. The
    hardcoded `'magnified_profile'` (which scored every core/merged run against the wrong
    modality) now follows `config.modality`, and the EMA swap is wrapped in `try/finally`
    — it was the one swap site without it, so a failure left the generator permanently
    holding EMA weights while training silently continued. **Still open:** n=64 against
    2048-dim Inception features is rank-deficient and strongly biased (Chong & Forsyth
    2020); the reference reals are drawn from the live training pointer. Use KID
    (Bińkowski 2018) on a held-out pool. Until then, run with `--fid_interval 0`.

60. ~~**The dataset's `image` column is URL-backed: one HTTP GET per image, per epoch.**~~
    **Resolved 2026-09-23 (`--image_root`).** This was the single largest throughput
    defect in the repo and it masqueraded as "training is slow / the GPU is weak".

    `dataset['train'][i]['image']` does not read a local file — the HF manifest stores
    `https://huggingface.co/datasets/RMDig/.../preprocessed/magnified_profiles/image_N.png`,
    so **every access downloads a ~16 MB PNG**, once per image per epoch, and the source
    images are 3024×4032 (12 MP) that get immediately downsampled. Measured:

    | path | per image | 1024² train step |
    |---|---|---|
    | HF `image` column (HTTP) | **2.03 s** | **16.8 s/step** |
    | local mirror (512 px) | **0.007 s** | **0.87 s/step** (at 512²) |
    | GPU compute alone (synthetic data, 1024²) | — | **0.537 s/step** |

    So the data pipeline was **~97% of a train step** and the GPU sat idle. UPGRADES #51
    noted the HTTP cost in passing as "a distinct, larger issue not addressed here"; this
    is it. It also explains why the historical `batch_*` mtimes imply ~7 s/batch and why
    the recovery plan's first cost model was ~13× pessimistic — those runs were paying for
    network, not arithmetic.

    Fix: `config.image_root` / `--image_root` resolves `<root>/<manifest file_path>` and
    never touches the network. Rows missing locally fall back to the remote column, so a
    partial mirror costs speed, not correctness; a corrupt local file warns and falls back
    rather than killing a long run. `DataManager.load_image` is the single entry point for
    both batch paths. Regression coverage: `tests/unit/test_image_root.py`.

    Also removed in the same change: `batch_merged`'s log line called
    `self.dataset['train'][profile_ind]['segment']`, a **second full row fetch per pair**
    — i.e. a second HTTP round trip purely to print a field.

    **Still open:** this is a mirror, not a pipeline. UPGRADES #10 (`tf.data` with
    prefetch/parallel map/cache) remains the real fix, and no mirror exists at full
    resolution — `~/rmdig-cache-512` has all 2,350 magnified_profile images at 512 px,
    while the 53 GB `data/rmsnow` clone has only 1,355 at full resolution.

61. ~~**Config schema was not forward-compatible.**~~ **Resolved 2026-09-23.**
    `build.configure` is called as `configure(**config_json)` with a closed keyword
    list, so a sidecar written by a NEWER snowgan raised `TypeError` on an OLDER one.
    This is a cross-repo break, not a theoretical one: snowGradient pins
    `snowgan @ git+...@main` (a mutable ref) and calls `build()` on
    `discriminator_config.json` directly, so any environment installed before a schema
    change fails at backbone load. `configure` now accepts `**unknown_fields`, keeps
    them on the instance, and re-emits them from `dump()` so an old reader cannot
    silently strip a field it does not understand. Tests in
    `tests/unit/test_review_fixes.py`.

62. ~~**`save_config` was not atomic, and still ran at `atexit`.**~~
    **Resolved 2026-09-23 (partial — UPGRADES #34).** `generator_config.json` holds
    `fade_step`, which is the **only** source of `global_step` on resume, plus
    `train_ind` and the split pools. It was written with a plain `open(..., 'w')`, and
    `load_config` has no guard around `json.load` — so a signal during that write
    truncated the file and the next launch died with `JSONDecodeError`, the restart
    wrapper saw a non-75 exit, and a long run stopped permanently. It was the one
    failure mode in an unattended run that did not self-heal. Now `tmp + os.replace()`,
    matching `_atomic_save_weights`. **Still open:** the `atexit` registration itself
    (`config.py`), which is why a run's config is timestamped minutes after its weights.

63. ~~**`max_steps` was checked after the step, so re-entry was not idempotent.**~~
    **Resolved 2026-09-23.** Re-invoking a finished capped run performed one more real
    gradient update, overwrote the final checkpoint with an N+1-step model, and left
    `check_gates.py` evaluating a one-record window (`read_last_launch` filters to the
    final `launch_id`). The check now also runs at the top of the loop.

64. ~~**`critic_updates` was derived, not counted.**~~ **Resolved 2026-09-23.** It was
    `global_step * disc.config.training_steps`, which re-attributes the whole run
    history to whatever ratio is current — wrong after any resume with a different
    `--disc_steps`, and badly wrong under `adaptive_steps` (2 → 61 in the released run).
    Gates are expressed on this axis, so a rescaled axis silently moves every gate. Now
    accumulated per step and persisted as resume state.

65. ~~**The `clamp_gp_under_sn` path still destroyed `lambda_gp`.**~~
    **Resolved 2026-09-23.** Making the clamp opt-in did not make it non-destructive:
    it still assigned `1.0` onto `disc.config.lambda_gp`, `dump()` persisted it, and the
    user's `10.0` was unrecoverable. The resolved value now lives on the trainer and the
    config is left alone.

66. **`_cleanup_saved_batches` is a no-op at the default cadence.** 🟠 OPEN.
    `trainer.py` calls it with `keep_every=100` while snapshots are only written every
    `cleanup_milestone` (1000) steps — so every existing `batch_N` is a multiple of 100
    and nothing is ever removed. Because the preview trim is nested inside the removal
    branch, **no synthetic image is ever trimmed either**. Measured at 10k steps: a
    snapshot dir is 348 MB (including a 167 MB `generator_fade_endpoints.weights.h5`
    written even when `fade: false`) and `synthetic_images/` was 1,990 files / 1015 MB.
    At 100k steps that projects to ~45 GB and ~20k files in one run directory.
    Separately the trim logic is inverted — `indexed_images[-7:]` deletes the seven
    newest rather than keeping them.

67. **Preview PNGs are named by `batch`, which rewinds on restart.** 🟡 OPEN.
    `batch` is recovered by globbing snapshot dirs, so it rewinds to the last multiple
    of `cleanup_milestone`; previews named `batch_{batch}_synthetic` therefore replay
    and overwrite earlier images with ones from a LATER training state. CLAUDE.md §9's
    scoreboard item 2 is "open `synthetic_images/*.png`" — that timeline is non-monotone
    across a restart with nothing saying so. Name previews by `global_step`.

59. **The critic runs with `training=True` during the generator step.** 🟡 OPEN.
    [trainer.py:786](../src/snowgan/trainer.py#L786) advances the spectral-norm power
    iteration outside critic training; `losses.compute_gradient_penalty` does too. Low
    expected impact, listed so it is not rediscovered.

## Tier 🟠 — production readiness (do before calling this a product)

8. **Replace `atexit` with explicit, atomic, signal-safe checkpointing.**
   `atexit` does not fire on SIGKILL, OOM, or kernel death. Move saves to a
   `ModelCheckpoint`-style callback on a step interval. Write via `tmp + os.replace()` so a
   half-written file never replaces a good checkpoint. Persist config JSON the same way.
   **Manifested 2026-06-05 (now 🔴 in practice):** `save_model` calls
   `model.save_weights()` directly to the final path, so an OOM-kill mid-write
   truncated the live `generator.weights.h5` to 96 bytes and the next run failed
   to load it. Recovered by restoring the full consistent set from the newest
   intact `keras/snowgan/core/batch_<N>/` snapshot. Fix `save_weights` to
   `tmp + os.replace()` before the next long run. See
   [cpu-ram-leak-investigation.md](cpu-ram-leak-investigation.md#checkpoint-corruption--recovery).
   - ~~**atexit save + non-atomic weight writes**~~ **Resolved 2026-06-15
     (`feat/atomic-save-survivable-runs`).** `atexit.register(self.save_model)`
     is removed; every weight write in `save_model` goes through
     `_atomic_save_weights` (sibling `._tmp_*.weights.h5` + `os.replace`), and a
     rolling top-level checkpoint is written on the step interval during the run
     (not at exit). An OOM-kill / restart-wrapper kill mid-write can no longer
     truncate the live checkpoint, and resume always finds a current consistent
     one. This makes the `--max_rss_mb` restart wrapper (leak workaround) safe.
     **Still open:** atomic config-JSON writes (#34).

9. **Split the god-config.**
   [config.py](../src/snowgan/config.py) is a 350-line dict + mutable class with `atexit`
   save that mirrors itself across generator and discriminator. Replace with a
   `pydantic.BaseModel` (or `@dataclass(frozen=False)`) hierarchy:
   `RuntimeConfig`, `DataConfig`, `ModelConfig(generator=..., discriminator=...)`,
   `TrainingConfig(fade=..., checkpoint=...)`. Validate types, ranges, and invariants
   (`filter_counts` monotonicity, `resolution` compatible with `convolution_depth`).

10. **Introduce `tf.data` pipeline with prefetch, parallel map, and caching.**
    Current pipeline is synchronous, one-sample-at-a-time, recomputes resize every epoch.
    Goal: `tf.data.Dataset.from_generator(manifest_iter, ...).map(decode, num_parallel_calls=AUTOTUNE).cache().shuffle().batch().prefetch(AUTOTUNE)`.
    Precompute core→profile pairing index once at startup
    (`dict[(site, column, core), list[profile_idx]]`) to kill the linear rescan in
    `batch_merged`.

11. ~~**Honor `trained_pool / validation_pool / test_pool`.**~~
    **Resolved 2026-05-09 (PR #9).** `DataManager.derive_splits()` partitions
    `pair_index` keys 80/10/10 at the group level (`(site, column, core)` tuples)
    using `random.Random(config.seed)`. Trainer init populates the pools on first
    run and persists them; idempotent on resume. Pools persist as `list[list]`
    for JSON friendliness — consumers needing tuple-keyed `pair_index` lookups
    must `tuple(...)` each entry on read. AvAI's Phase 4 evaluation reads
    `config.test_pool` directly. Pools are not yet consumed by `batch_merged`;
    that path couples to the `tf.data` rebuild (#10) and is intentionally
    deferred.

12. **Seed everything, log the seeds.**
    `random.seed`, `np.random.seed`, `tf.random.set_seed`, `os.environ["PYTHONHASHSEED"]`
    and `os.environ["TF_DETERMINISTIC_OPS"]="1"`. Persist the seed in the config snapshot.
    Without this, training runs are not replayable even across same-host restarts.

    **Partially resolved 2026-05-09 (this PR).** `snowgan.utils.set_seed(seed)`
    pins `random`, `numpy`, and `tf.random` to a deterministic state. main.py
    calls it after configs are loaded, before any model construction, and
    logs the seed. `config.seed` (default 42) was added in PR #9. Still
    outstanding: full env-var coverage (`PYTHONHASHSEED`,
    `TF_DETERMINISTIC_OPS`) only takes effect when set before
    interpreter / TF startup — folds into UPGRADES #4 (bootstrap module
    that sets env vars before any TF import).

13. **Structured logging + TensorBoard / W&B.**
    Replace `print(...)` with the stdlib `logging` module (JSON handler for production, human
    handler for dev). Emit scalar metrics (disc_loss, gen_loss, grad_norm, fade_alpha,
    steps/sec, samples/sec) to TensorBoard. Wire an optional W&B or MLflow sink for
    cross-run comparison.

14. **GAN quality metrics, not just loss.**
    Add FID (Fréchet Inception Distance), KID, and per-modality equivalents (e.g., FID on
    core slice, FID on profile slice separately). Compute on a held-out validation pool
    every M batches. This is the only way to know the model is improving — WGAN-GP loss is
    not monotonic.

15. ~~**Mixed precision: audit and fix.**~~ **Resolved 2026-05-09 (PR #14).**
    Both breakages closed: (a) generator's `toRGB_curr` and both critics'
    `Dense(1)` pin to `dtype="float32"`, localizing the fp32 hotspot to where
    fp16 saturates / overflows; (b) `Generator`, `Discriminator`, and
    `disc_lowres` optimizers wrap in `tf.keras.mixed_precision.LossScaleOptimizer`
    conditional on `keras.mixed_precision.global_policy().name == "mixed_float16"`.
    Default-precision runs pay zero overhead.

16. **Loss history storage.**
    Append-only JSON Lines (`loss.jsonl`: `{"step", "gen_loss", "disc_loss", "epoch", "batch"}`)
    instead of rewriting two `.txt` files on every save. Atomic append; easy to resume; easy
    to plot externally.

17. ~~**Checkpoint format: weights-only + sidecar.**~~
    **Resolved 2026-05-09 (PR #10).** Save format is now `*.weights.h5`; the
    architecture sidecar is the existing per-model `*_config.json` (`config.dump()`
    already contained every architecture knob). New `snowgan.checkpoint.resolve_weights_path`
    prefers the new format and falls back to legacy `*.keras` if only that exists,
    so existing trained checkpoints in `keras/snowgan/` keep loading; on first
    save under new code the new format is written alongside. The bandaid
    `try/except` in `Trainer.__init__` that previously swallowed shape mismatches
    is gone — `load_weights` now raises loud per the spec.

18. **Hugging Face Hub integration.**
    You already host the dataset there; mirror the code. Add an `hf push-model` command that
    uploads `generator.keras`, `discriminator.keras`, the generator/discriminator config
    JSONs, and a model card (training curves, sample grids, dataset commit, seed, git SHA).
    That makes the pretrained backbone discoverable for AvAI.

19. **Dockerfile + CUDA pin.**
    CUDA/cuDNN version drift will be the #1 cause of failed re-runs. Ship a `Dockerfile`
    (or `uv` lockfile + devcontainer) that pins TF, CUDA runtime, and driver expectations.
    Today, onboarding a new host is a bespoke README incantation.

37. **Gradient-penalty α broadcasts uniformly across depth.**
    [losses.py:46-47](../src/snowgan/losses.py#L46-L47) reshapes `alpha` to `(B,1,1,1,1)`,
    so the same interpolation weight is applied to both the core slice and the profile
    slice of every paired sample. WGAN-GP theory assumes independent interpolation between
    real and fake; coupled α couples the two modality gradients and can bias the critic
    toward correlated features across depth. Root cause: rank-4 GP formula was lifted to
    rank-5 without rethinking the modality semantics. Fix: sample
    `alpha = tf.random.uniform([B, depth, 1, 1, 1])` so each modality gets its own
    interpolation. Alternative: document explicitly that coupled α is intentional and
    justify it; otherwise default to independent.

38. ~~**`generate()` runs every batch during training.**~~
    **Resolved 2026-05-10.** Originally the inner loop wrote `n_samples * depth` PNGs
    per `train_step` (debug tracing in the hot loop). PR #15 (May 9) removed the
    unconditional call entirely and gated emission on `sample_epoch_interval` at
    epoch boundaries — but at 1024×1024 an epoch never closes for typical mid-run
    crashes, so visibility went to zero. The follow-up adds a `sample_batch_interval`
    config field (default `0` = off; CLI: `--sample_batch_interval`) that re-enables
    per-batch emission at an explicit cadence, mirroring the EMA-wrapped epoch-end
    block. The async-queue / off-thread write idea is still open if I/O ever becomes
    a measured bottleneck, but at typical cadences (≥100 batches) it isn't worth
    the complexity.

39. **No lock on `save_dir` → concurrent runs silently corrupt each other.** 🟠 OPEN.
    Two `snowgan --mode train --save_dir ./models` processes race on
    `config.save_config()`, on weight writes, and on `batch_*`/`synthetic_images`
    naming. No lockfile, no PID file. Root cause: `save_dir` is shared mutable state
    with no concurrency discipline.

    **Observed in the wild, 2026-09-25.** A `pkill` silently failed during a relaunch
    and two trainers ran against `keras/snowgan/control_1024/` concurrently for about a
    minute. Symptom: an 81-step run whose sidecar read `critic_updates: 5800` — the
    other process's counter. Nothing was released from it, but it is no longer a
    theoretical entry.

    **What did and did not survive that incident**, which is the useful part for
    scoping the fix:

    | artifact | behavior under concurrency |
    |---|---|
    | `metrics.jsonl` | **tolerates it.** Append-only, records under `PIPE_BUF` so `O_APPEND` writes are atomic, and every record carries a `launch_id` — interleaved rows stay attributable. |
    | `*_config.json` | **corrupts.** Last writer wins and the two processes hold different in-memory state. This is what produced the 5800. |
    | `generator_loss.txt` / `discriminator_loss.txt` | **corrupts.** `save_history` rewrites the whole file from an in-memory list, so the loser's history is simply replaced. |
    | `*.weights.h5` | atomic per file (`_atomic_save_weights`), so never torn — but two trainers still interleave *whole* checkpoints from different models. |
    | `batch_N/`, `synthetic_images/` | collide on name; each write is internally fine. |

    **Design note from snowGradient (2026-09-25).** They hit the same class of bug in
    their `tf.data` cache — a deterministic cache key (deliberately so, to stop
    cross-arm contamination) meant two concurrent invocations shared cache files. They
    fixed it by scoping temp roots under a per-invocation `{pid}-{timestamp}` directory
    cleaned at exit, which also closed a stale-cache case, and observed: *"if each run
    owns its own directory, there's nothing to lock."*

    That does not replace the lock here, and they said as much: snowGAN's `save_dir` is
    **sequentially** shared by design — resume means a later run deliberately adopts an
    earlier run's weights and config. Per-invocation directories cannot express that.

    But it does usefully shrink the lock's scope. Split `save_dir` into:
      - **resume state** (weights, the two sidecars, and — once UPGRADES #36 lands — a
        persisted `global_batch`): genuinely shared, must be under the lock;
      - **derived output** (`batch_N/` snapshots, `synthetic_images/`, `history.png`,
        the legacy `*_loss.txt` pair): per-invocation, or append-only-with-`launch_id`
        like `metrics.jsonl` already is.

    Then a missing or failed lock degrades to duplicate previews rather than a corrupt
    checkpoint or a fabricated counter. Fix remains: `filelock.FileLock` in
    `Trainer.__init__` with PID+host in the payload, plus stale-lock detection — a
    lock left by an OOM-killed run must not block the restart wrapper, which is the
    normal path here, not the exception.

40. ~~**Mixed-depth datasets trigger a model rebuild every batch.**~~
    **Resolved 2026-05-09 (PR #12).** Co-resolved with #2: `DataManager.PAIR_DEPTH`
    is now constant, the per-batch rebuild is gone (replaced by a hard assertion
    in `_ensure_depth_alignment`), and the latent `config.depth = ...` mutations
    in `batch()` and `batch_merged()` are removed. Mixed-depth manifests would
    now trip the assertion immediately rather than triggering silent rebuilds.

45. **cuDNN-on-Blackwell native CPU-RAM leak in the train step (~+1 MiB/batch).**
    The GPU is an RTX 5080 (sm_120 / "CC 12.0a") but the installed CUDA/`ptxas`
    (12.0.140) predates Blackwell, so TF JIT-compiles kernels from PTX and cuDNN
    falls back to driver compilation every step (`None of the algorithms ...
    trying fallback algorithms`), holding native workspace `malloc_trim` cannot
    reclaim. Environmental, not a code bug; it predated the OOM problem and was
    deferred. `TF_CUDNN_USE_FRONTEND=0` / `TF_CUDNN_USE_AUTOTUNE=0` did not help.
    Durable fix: a CUDA 12.8+ / Blackwell-capable TF+cuDNN build. Stopgap:
    periodic checkpoint+restart, or fewer disc/gen steps. Full context in
    [cpu-ram-leak-investigation.md](cpu-ram-leak-investigation.md#still-open).

## Tier 🟡 — code health & velocity

51. ~~**Per-image debug print in `preprocess_image` forced a GPU sync every image.**~~
    **Resolved 2026-08-17.** `preprocess_image` did
    `print(f"Max - {tf.reduce_max(image).numpy()} | Min {tf.reduce_min(image).numpy()}")`
    after the resize — two `.numpy()` calls on GPU tensors in the hottest data-pipeline
    loop. Each forces a device synchronization, so the GPU could not pipeline across
    images; it also flooded stdout (a calibration run logged 1,682 `Max - …` lines and
    nothing else). Measured cost, isolated on an already-local tensor (no HTTP): ~0.50
    ms/image of pure sync (0.60 ms with the two syncs vs 0.10 ms pipelined, 6×) — ~71 s of
    GPU stall per snowGradient epoch (~143k accesses) from this line alone, separate from
    the URL-backed-image HTTP fetch cost (a distinct, larger issue not addressed here).
    Fix: deleted the print (CLAUDE.md §6 — print is legacy). Regression test
    `tests/unit/test_preprocess_no_sync_print.py` asserts `preprocess_image` emits no
    per-image stdout and still normalizes to [-1, 1].

20. **Tests.**
    - `tests/unit/test_losses.py`: `compute_gradient_penalty` on a frozen conv, assert the
      double-λ fix.
    - `tests/unit/test_config.py`: round-trip JSON, CLI overrides, type validation.
    - `tests/unit/test_dataset.py`: pair index built from a tiny synthetic manifest; asserts
      pairing correctness and no profile reuse within an epoch.
    - `tests/integration/test_train_step.py`: 1 step at 64×64 with `depth=2` completes and
      updates weights (compare `sum(weights)` before/after).
    - GitHub Actions CI: `pytest`, `ruff check`, `mypy src/snowgan`.

21. **Type hints + `mypy --strict` on `src/snowgan/`.**
    Most functions are untyped. `config.build.configure` takes 30+ positional args — a
    `TypedDict` or dataclass erases a class of bugs.

22. **Extract `Trainer` into focused collaborators.**
    Current trainer owns loading, saving, fade scheduling, sample generation, cleanup, plot.
    Aim for:
    - `Trainer` — loop + train_step.
    - `Checkpointer` — save/load, atomic writes, rotation.
    - `FadeScheduler` — alpha, step counter, completion.
    - `SampleReporter` — synthetic grid generation + TensorBoard image writer.
    - `ArtifactCleaner` — disk hygiene.
    Each ~100 lines, each unit-testable.

23. **Fix `load_discriminator` / `load_generator` path mangling.**
    [models/discriminator.py:65-71](../src/snowgan/models/discriminator.py#L65-L71) does
    `split.pop()` and then `"/".join(split) + "/"` — order-dependent, mutates the list
    mid-use, drops the filename. Use `pathlib.Path` and `parent / name`.

24. ~~**Stop committing `__pycache__` and `*.egg-info`.**~~ **Resolved 2026-04-18:**
    `src/**/__pycache__/` and `src/**/*.egg-info/` patterns added to `.gitignore`. No
    tracked files needed removal (check ran clean at resolution time).

25. ~~**Document the modality-blending contract.**~~
    **Resolved 2026-05-09 (PR #8).** `snowgan.modality.Modality` IntEnum
    (`PROFILE=0`, `CORE=1`) is the single source of truth. Re-exported from the
    package so AvAI can `from snowgan import Modality`. `merge_images` stacks
    via explicit `Modality.{PROFILE,CORE}` lookups; `generate.py` derives view
    filename suffixes from the enum. Output filename suffixes (`_profile.png`,
    `_core.png`) preserved.

26. **Replace hand-rolled batch-counter recovery with a persistent step counter.**
    [trainer.py:137-141](../src/snowgan/trainer.py#L137-L141) globs `synthetic_images/batch_*.png`
    to resume the `batch` number. Persist it in config (`global_batch`), just like
    `fade_step`. See #36 for the silent-data-loss angle when `_cleanup_saved_batches`
    trims the files this glob depends on — both fold into the same fix.

27. ~~**Fix the README CLI example.**~~ **Resolved 2026-04-18:** space inserted between
    `keras/` and `--gen_steps` in the README. A `Documentation` section was also added
    to the README pointing at `CLAUDE.md` and all files under `docs/`.

41. **LeakyReLU `negative_slope=0.25` is non-standard.**
    [config.py:85](../src/snowgan/config.py#L85) (and the layer constructors in
    [generator.py:46](../src/snowgan/models/generator.py#L46) /
    [discriminator.py:39](../src/snowgan/models/discriminator.py#L39)) default to 0.25; the
    near-universal GAN default is 0.2. Larger leak softens the critic's feature gating and
    may weaken the Lipschitz signal. Root cause: default value chosen without documented
    justification. Fix: either (a) revert to 0.2 and treat 0.25 as a tuning knob per-run,
    or (b) document the empirical reason 0.25 won and pin it.

42. **Fade path blends RGB, not feature-map resolution.**
    [generator.py:49-74](../src/snowgan/models/generator.py#L49-L74) implements "fade" as a
    1×1×1 Conv3D on the second-to-last feature map, resized to match the current RGB
    output, then linearly blended with the final `toRGB_curr` output. That is *not* the
    ProgGAN / StyleGAN pattern (grow spatial resolution, then blend old + new block
    outputs) — it's a color-only blend at a fixed spatial resolution. Training doesn't get
    the smoothing benefit of true progressive growth. Root cause: naming vs. implementation
    drift. Fix: either rename to avoid the "progressive fade" connotation (e.g.,
    `output_blend`) and document the actual semantics, or refactor to genuine progressive
    growth where spatial resolution ramps up. Do not ship under the current name.

43. ~~**Matplotlib plot uses implicit global state each batch.**~~
    **Resolved 2026-06-04 (PR #22).** Isolation showed that even an explicit
    `Figure` + `plt.close(fig)` leaks per *created* figure on the Agg backend
    (RSS 61→1867 MiB over 300 calls). The fix went further: `plot_history` now
    creates **one persistent figure once** and reuses it via `ax.cla()` each call
    (flat at 189 MiB). See
    [cpu-ram-leak-investigation.md](cpu-ram-leak-investigation.md).

44. **Manifest duplicated in memory alongside the HF dataset.**
    [data/dataset.py:17-20](../src/snowgan/data/dataset.py#L17-L20) materializes the HF
    split to pandas, drops `image`/`audio`, then stores the remaining rows as a list. The
    full HF `DatasetDict` also stays live via `self.dataset`. For a few-thousand-row
    snowpack split it's fine; for a production-scale dataset it's two copies of the
    metadata. Root cause: eager materialization used as a lookup layer instead of an
    index. Fix: at init, precompute the pair index
    `dict[(site, column, core), list[profile_idx]]`, then drop `self.manifest`
    entirely — iterate via the index and `self.dataset[i]`.

    **Partially resolved 2026-05-06:** `DataManager.pair_index` cached property added
    ([data/dataset.py](../src/snowgan/data/dataset.py)) returning
    `dict[(site, column, core), list[(core_idx, profile_idx)]]` — the full Cartesian
    product of cores × magnified profiles per group, computed in one linear pass over
    `self.manifest`. Shape diverges from the original ask (`list[profile_idx]` →
    `list[(core_idx, profile_idx)]`) to support AvAI's group-level transfer-learning
    splits, which need every cross-pair per group as a deliberate combinatorial
    augmentation. The accessor is non-mutating and is *not* used by the GAN trainer —
    `batch_merged`'s seen-profiles / per-epoch semantics are unchanged. Regression
    coverage at [tests/unit/test_dataset.py](../tests/unit/test_dataset.py).
    Still outstanding: drop `self.manifest` in favor of iterating the index plus
    `self.dataset[i]`, which requires `batch_merged` to consult the index instead
    of linear-scanning the manifest. That is a separate cross-lens refactor (it
    couples to #10's `tf.data` pipeline rebuild).

## Tier 🟢 — nice-to-have

28. **Submodule cleanup.** `.gitmodules` references `external/snowMaker` but the directory is
    missing. Either restore or drop.

29. **Remove dead imports / dead code.**
    ~~`generate.make_movie` imports `cv2`, which is only needed for that path.~~
    **cv2 sub-issue resolved 2026-04-18:** `import cv2` moved from `generate.py`
    module scope into `make_movie()` body. `import snowgan` no longer requires
    opencv-python; downstream consumers (AvAI) can depend on `snowgan` without
    pulling the 100 MB native dependency. The video-writing path still requires
    `opencv-python` and raises `ImportError` with a clear message if invoked
    without it. Still outstanding: `click` is in deps but unused; `emd_loss`
    in `losses.py` duplicates `Discriminator.get_loss`.

30. **Pre-commit hooks.** `ruff`, `ruff-format`, `end-of-file-fixer`, `check-added-large-files`
    (helpful given `.keras` weights lurk).

31. **Model card + dataset card cross-links.** Describe licensing, intended-use, limitations,
    evaluation results, bias considerations.

41. **Release multiple model sizes for transfer-learning consumers.**
    The depth-axis architecture's compute scales with `resolution²·depth`, and the
    backbone-resolution choice is a one-time commitment per release (AvAI builds
    classification heads on top of a frozen Conv3D backbone whose feature dim is
    `depth·H·W·filter_counts[-1]`). A consumer running on a 16-GB card cannot
    transfer from a 1024×1024 backbone but can from 512×512 or 256×256. Plan:
    after the primary AvAI run-1 (1024×1024 paired-modality, this cycle), train
    smaller depth=2 backbones at 512×512 and 256×256 and publish them with
    matching `discriminator_config.json` + `discriminator.weights.h5` so AvAI's
    `load_backbone` resolves whichever fits the deploy target. Dependency
    note: filter_counts length must shrink by one per halving of resolution
    (16 × 2^(N+1) coupling, see [docs/architecture.md](architecture.md)).

---

## Cross-lens observation — unify persistence as a single subsystem

A striking pattern across items #8, #16, #17, #26, #34, #35, #36, #39, and part of #40: all
of them are persistence bugs with the same shape — **some piece of state is written in the
middle of an operation, without atomicity, and without a transaction boundary that defines
when the state is "committed."** Each finding is individually fixable, but every
independent fix risks drifting out of sync with the others (a new atomic write that doesn't
know about the lockfile; a lockfile that doesn't cover the JSONL loss log; a persisted
`global_batch` that updates before or after `fade_step` depending on the code path).

Proposed root-cause fix: a single `Checkpointer` module owns all persistence:

- Atomic writes (tmp + `os.replace()`) for configs, weights, histories, and sample indexes.
- A file-lock (`{save_dir}/.snowgan.lock`) acquired at Trainer init.
- A single "commit point" at the end of each `train_step` (or every N steps for
  throughput). Before the commit, state changes are staged in memory; on commit, all
  persisted artifacts — config JSON, loss JSONL, `global_batch`, `seen_profiles`,
  `fade_step` — land together. A crash between commits rewinds cleanly to the previous
  committed state.
- Resume reads the last committed snapshot only; no globbing the filesystem.

Shipping this retires #8, #16, #26, #34, #35, #36, #39 in one coherent change and
eliminates a whole class of future bugs. (#17 is already resolved independently —
PR #10 — but the pattern still applies to the rest.) Until this exists, the
individual fixes are bandaids (in the sense of CLAUDE.md §2 — the symptom goes
away, the design defect persists).

## Suggested sequencing

**Status as of 2026-05-09:** the AvAI Phase 4 unblocker cycle landed every Tier A
and Tier B item from `/tmp/avai_phase_4_unblockers.md`. Resolved this cycle:
#2, #3, #11, #15, #17, #25, #33, #40 (and #12 partial). What remains below is
backlog beyond that cycle; the next gate is the fresh training run that
produces the AvAI backbone.

Remaining 🔴 (correctness):
- #4 device env ordering, #5 `--resolution` dead, #6 boolean CLI flags,
  #7 `latent_dim` float, #34 non-atomic config writes (partial),
  #35 `reset_seen_profiles` persist order, #36 batch-counter recovery.

Remaining 🟠 (production):
- #8 atexit → Checkpointer, #9 pydantic config, #10 tf.data pipeline,
  #12 seed env-var portion (folds into #4), #13 structured logging,
  #14 FID + KID metrics, #16 loss JSONL, #18 HF Hub push, #19 Docker pin,
  #37 per-modality α, #39 save_dir lockfile.

Remaining 🟡 (code health):
- #20 tests + CI, #21 mypy, #22 Trainer split, #23 `load_*` path mangling,
  #26 batch counter (folds into #36), #29 dead imports remainder,
  #30 pre-commit hooks, #41 LeakyReLU 0.25, #42 fade rename / refactor,
  #43 matplotlib lifecycle, #44 drop `self.manifest` (partial), #45 port
  rank-4 features (`diff_augment`, multi-scale disc) to rank-5.

Remaining 🟢 (nice-to-have):
- #28 submodule cleanup, #31 model card.

A reasonable next slice once the fresh training run is underway:
the persistence-subsystem cross-lens fix (#8 + #34 + #35 + #36 + #39 + #16,
folded into the proposed `Checkpointer` above) is the highest-leverage piece
remaining — it retires several 🔴 / 🟠 items in one coherent design rather
than playing whack-a-mole on the symptoms.
