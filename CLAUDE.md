# CLAUDE.md — snowGAN systems engineering guide

Operating instructions for Claude when working in this repo (and its sibling AvAI repo).
These rules are enforced; deviations require an explicit override from Denny in the current
session.

## 1. Session workflow

Every non-trivial session follows the same four phases. Skipping a phase is a workflow
violation — call it out rather than quietly skipping.

1. **Plan.** Restate the goal in one or two sentences. Surface the smallest set of files that
   must change. Identify which architectural invariants the change touches (data shape,
   model rank, checkpoint format, config schema, CLI surface, HF dataset schema). If the
   change is > ~30 lines across > ~2 files, write the plan as an ordered list of commits
   before touching code.
2. **Implement with focused tests.** Write the change and the *narrow* test(s) that prove
   that change is correct. "Focused test" = a unit or targeted integration test that
   exercises only the modified code path. Do **not** run the full test suite during this
   phase — it's slow, it wastes cycles, and it muddies cause/effect if multiple unrelated
   failures appear.
3. **Two-lens review.** Before running the full suite or asking to push, run both lenses in
   order. Document findings briefly in the session (not in files unless asked):
   - **Architecture lens.** What contract did we change? Who else depends on the old
     contract? What shape / rank / schema / CLI surface moved? What invariant weakened? Does
     the change survive a rename, a dataset column addition, or a resolution change? If
     anything downstream (including AvAI) breaks under the new contract, that's a blocker.
   - **Execution lens.** Walk the change through the repo's runtime flow: CLI parse →
     device config → data load → model build → train step → checkpoint → generate. Does
     any step now fire in the wrong order, double-fire, not fire, or pay a new cost on every
     batch? Does the change interact badly with `atexit` / resume / fade / mixed precision /
     tf.data prefetch? If the change runs clean in isolation but wedges when spliced into
     the real runtime, the execution lens catches it.
4. **Full suite + push.** Only after both lenses pass, run the full test suite locally.
   Suite must be green before a PR is opened. Pushing directly to `main` is reserved for
   Denny's explicit request per-push; never default to it.

## 2. No-bandaids rule (hard enforcement)

When an issue appears — a test failure, a crash, a shape error, a silent numerical drift —
**the first move is to diagnose the root cause, not to make the symptom go away.**

- "Try/except around the failing line" is a bandaid. Only legitimate when the exception is
  genuinely recoverable and the recovery path is documented.
- "Cast to the expected type" without asking *why* the type was wrong upstream is a bandaid.
- "Add a conditional to skip the failing case" is a bandaid unless that case is a known
  invalid input and the skip is named and logged.
- "Bump a tolerance until the test passes" is a bandaid.
- "Delete the failing test" is the worst bandaid.
- "Suppress the warning" is a bandaid.

The test of a real fix: could you write a one-line explanation that names the upstream
cause, and does the fix make that cause impossible rather than invisible? If not, it's a
bandaid — keep digging.

When a bandaid is genuinely the right call for *right now* (e.g., to unblock a training run
over a weekend), mark it explicitly: a `# BANDAID(<owner>, <date>, <issue-link>):` comment
in code and a follow-up item in the session summary. It is then a debt, not a fix.

This rule applies upward, too: if the root cause is a design-level issue (shape contract,
config schema, module ownership), surface it rather than patching symptoms at the leaves.
The upgrade tiers in [docs/UPGRADES.md](docs/UPGRADES.md) are the canonical backlog for
those cases.

## 3. Testing discipline

- **Focused tests** run continuously during implementation. Aim for sub-second runtimes:
  small tensors, stubbed datasets, no real HF downloads, no 1024×1024 images (use 64×64 for
  shape checks).
- **Integration tests** cover a train_step on a tiny model, checkpoint round-trip, config
  round-trip, pair-index construction.
- **Full test suite** runs once per phase-4 gate, not during implementation.
- **Regression tests follow bugs.** Every 🔴 bug fixed in [docs/UPGRADES.md](docs/UPGRADES.md)
  ships with a test that fails under the old behavior and passes under the new.
- **No new bandaid tests.** Tests that assert the buggy behavior "because that's what it
  does" are forbidden. Tests encode intended behavior.
- **Determinism.** Tests set `tf.random.set_seed`, `np.random.seed`, `random.seed`, and
  `PYTHONHASHSEED`. Any flake is a correctness bug; don't retry, diagnose.

## 4. Code & architecture standards

- **Respect the shape contract.** This branch operates on rank-5 tensors
  `(B, depth, H, W, C)` with `depth=2 = [profile, core]`. Any change that assumes rank-4
  must document the reconciliation in the architecture lens.
- **Typed configs.** New config fields land in a typed schema (dataclass / pydantic), not
  as another loose key in `config_template`. Old loose keys are migrated, not duplicated.
- **Names over indices.** Tap layers by name (`model.get_layer("features").output`), not by
  positional index (`base.layers[-2].output`). Positional taps break the moment someone
  inserts a Dropout.
- **Atomic writes.** All persistence writes to `tmp + os.replace()`. A crash mid-save
  never corrupts the last-good file.
- **No `atexit` for correctness-critical state.** `atexit` is fine for telemetry flushes,
  not for model checkpoints.
- **Environment vars set before TF import.** TF caches env config at import time; any flag
  that affects device placement, XLA, or determinism is set in a bootstrap module that
  runs before `import tensorflow` anywhere in the process.
- **One job per module.** `Trainer` does not also own cleanup, plotting, and sample
  generation. Split when a file passes ~300 lines or 3+ responsibilities.
- **Public surfaces documented inline.** Every CLI flag has a one-line `help=` that says
  what the flag *does*, not what it *is*.

## 5. Data & model contracts

- `dataset.py` owns the HF manifest. Downstream code asks `DataManager` for indices and
  pairs, never re-opens the HF split directly.
- The pair index (`(site, column, core) → [profile_idx]`) is computed once at startup and
  persisted. No O(N²) rescans per batch.
- `train_ind`, `seen_profiles`, `fade_step`, `current_epoch`, and (new) `global_batch` are
  the resume state. Any change that touches resume semantics updates all of them together
  or documents why not.
- Checkpoint format changes are breaking changes. They require a migration path for
  existing `keras/snowgan/*.keras` artifacts (or an explicit "start from scratch" gate).

## 6. Observability expectations

- `print` is legacy. New code uses `logging` with a module-level logger.
- Metrics land in structured form (JSON Lines or TensorBoard), not plain text.
- Every training run logs at start: git SHA, seed, config snapshot, dataset commit hash,
  host (GPU model + driver), TF version.
- Every training run logs at end: total steps, wall time, final metrics, checkpoint path.

## 7. Review etiquette

- If a change touches a 🔴 item in [docs/UPGRADES.md](docs/UPGRADES.md), update that doc
  to mark the item resolved (or reclassify it) in the same commit.
- If the two-lens review surfaces a new invariant worth remembering, update
  [docs/architecture.md](docs/architecture.md) alongside the code.
- Commit messages explain *why*, not *what*. The diff already shows the what.
- Never `git push --force` to `main`. Never `git push --no-verify`.

## 8. When to stop and ask

- The change requires deleting or overwriting trained weights in `keras/snowgan/`.
- The change modifies the HF dataset contract or config schema in a backwards-incompatible
  way.
- The change touches CUDA / driver setup in a way that could brick the training host.
- The two-lens review surfaces a contract break that wasn't in the stated goal — stop,
  summarize, let Denny decide whether the scope widens or the plan shrinks.

## 9. GAN experiment discipline (hard enforcement)

Adopted 2026-07-26 after a run of investigations that changed several things at once
(generator depth, normalization, output gain, minibatch-stddev, board masking, critic
ratios) and could attribute none of the outcomes. Training a GAN is an experiment, not
a build; it is governed by controls, not vibes.

- **Ground truth first.** Every experiment campaign starts from a baseline that
  *provably trains* — not one that merely loads old weights. "This architecture produced
  images once" is a static claim; "this code trains it from scratch to structure" is the
  claim that matters. Establish the latter (a short control run) before building on it.
- **One architectural piece at a time — but budget multiple runs to tune it.** Each
  campaign introduces *exactly one* new architecture/runtime element on top of the
  established baseline. No stacking two new pieces. But introducing a piece is **not** a
  single run: assume the piece needs tuning before it can be fairly judged, so a single
  piece may span several runs that vary *only that piece's own knobs* (its learning rate,
  weight, init, ratio, etc.) — never a second new piece. Judge a piece by its
  **best-tuned run**, not its first: a piece that failed once may just have been
  mis-tuned. Declare a verdict on the piece (keep / drop) only after a genuine tuning
  effort, and record how many runs it took. A run that changed more than the piece under
  test — or that changed a *different* piece — is uninterpretable and does not count as
  evidence.
- **The scoreboard is fixed and applied identically every run** — no judging by loss
  curves (WGAN loss magnitude is not quality). In order:
  1. **Kill-check (cheap, minutes):** latent diversity — mean pairwise
     `|G(zᵢ) − G(zⱼ)|` must be well above 0 (≈0 ⇒ mode collapse); saturation —
     `frac(|output| > 0.99)` low and output std near the data's (~0.5). A run failing
     this is dead; do not let it run for hours.
  2. **Sample inspection:** open `synthetic_images/*.png`; two latents must yield two
     *different*, non-saturated images. (Loss can look fine while samples are garbage —
     the samples are the arbiter.)
  3. **Downstream probe (the real metric):** a linear probe on avalanche-risk labels.
     Feature transferability, not image beauty, is the objective. Generation-not-
     collapsing is necessary, not sufficient.
- **Log before the next run.** Every run gets a row in
  [docs/experiments.md](docs/experiments.md) — base commit, the single change, scoreboard
  result, verdict — *before* the next run starts. The log is the memory; re-deriving what
  was already tried is the waste this section exists to prevent.
- **Research justifies changes.** A proposed increment cites a reason (repo evidence or a
  source), not a guess. "Might help" is not a plan.
