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

1. **Double-applied gradient penalty.**
   [losses.py:55](../src/snowgan/losses.py#L55) returns `mean((‖∇‖−1)²) · λ`. Then
   [models/discriminator.py:63](../src/snowgan/models/discriminator.py#L63) adds
   `λ · gp` a second time. Effective penalty = `λ² · mean((‖∇‖−1)²)` (100× at λ=10).
   Fix: either return the raw `mean((‖∇‖−1)²)` from `compute_gradient_penalty` and multiply
   once in the loss, or stop multiplying in the loss. Add a unit test that asserts the
   penalty magnitude for a known input.

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

5. **`--resolution` flag is dead.**
   [utils.py:97](../src/snowgan/utils.py#L97) declares `type=set` — argparse cannot build a
   set from a single string. The flag accepts input but silently discards it. Replace with
   `nargs=2, type=int` or `type=parse_resolution` where the parser splits `"1024x1024"`.

6. **Boolean CLI flags accept any truthy string.**
   `--fade`, `--xla`, `--mixed_precision`, `--rebuild`, `--gen_norm` all use `type=bool`.
   `--fade False` evaluates to `True`. Swap to `argparse.BooleanOptionalAction` (py ≥ 3.9)
   for `--fade / --no-fade` semantics.

7. **`latent_dim` is typed `float` but used as int.**
   [utils.py:101](../src/snowgan/utils.py#L101) → `type=float`. `config.latent_dim = int(...)`
   downstream works but `args.latent_dim` is a float. Risk: `tf.random.normal([B, 100.0])`
   is not accepted in strict TF builds. Type as `int`.

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

39. **No lock on `save_dir` → concurrent runs silently corrupt each other.**
    Two `snowgan --mode train --save_dir ./models` processes will race on
    `config.save_config()`, on `keras.Model.save()` (not atomic — writes several temp files
    internally), and on `synthetic_images/` counter recovery. No lockfile, no PID file.
    Root cause: `save_dir` is shared mutable state with no concurrency discipline. Fix:
    acquire `filelock.FileLock("{save_dir}/.snowgan.lock")` in `Trainer.__init__`;
    release on graceful shutdown; include the current PID and host in the lock payload
    so a stuck lock can be diagnosed.

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
