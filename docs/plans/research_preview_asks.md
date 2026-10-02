# snowGradient research-preview asks (2026-10-02)

snowGradient's research-preview model (modality recognition for the AvApp) needs
six things from snowGAN. Every claim in their brief was re-derived from raw
artifacts before this plan was written; corrections are in §2.

Constraint during build: another training job holds this machine. Code and CPU
unit tests at 64 px only; anything that loads the HF dataset or the GPU (item 6
scoring, full-res export timing) waits.

## 1. Verified facts this plan rests on

- **Export is exact for every released model.** Every Conv3D / Conv3DTranspose
  has kernel `(1,kH,kW)` and depth stride 1 (discriminator.py:36-37,
  generator.py:164-167). A generic layer-walk converter matched the Conv3D
  model to max |diff| 0.0 for: discriminator with/without SN (after
  power-iteration updates), and generator across all 12 combinations of
  upsampler {transpose, resize} x convs {1,2} x norm {none, pixel, batch}.
- **It is NOT exact at depth=2 (`modality="merged"`)**: the critic's
  `Flatten -> Dense(1)` head mixes depth slices (measured: changing slice 1
  moved the score by 0.083). No released sidecar has depth != 1. The export
  refuses depth != 1 rather than emitting a silently different model.
- **The fade path is not in `g.model`.** Exporting a mid-fade checkpoint would
  return only the current-resolution endpoint. The export refuses `fade=True`
  with `fade_step < fade_steps`.
- **`preprocess_image` cannot handle grayscale or RGBA.** `tf.image.resize`
  rejects rank 2, so the `expand_dims` after it is dead code; RGBA passes
  through as 4 channels. Not reachable in training today (the dataset is RGB),
  but the phone will see such inputs.
- **The released sidecars predate the gen_* fields; v1.0.0 already writes
  them.** Fields entered in: `gen_norm` b4e0677 (06-13), `gen_upsampler`
  1a925d4 (06-17), `gen_convs_per_resolution` 6ceda9e (07-23). Releases were cut
  at fde5671 (06-02). Architecture by era:

  | sidecar has | era | correct architecture | v1.0.0 resolves to |
  |---|---|---|---|
  | no `gen_norm` | <= fde5671 | transpose, 1 conv | resize, 2 (**wrong**) |
  | `gen_norm`, no `gen_upsampler` | b4e0677 | resize, 2 | resize, 2 |
  | `gen_upsampler`, no `gen_convs_per_resolution` | 1a925d4..6ceda9e | as written, 2 | as written, 2 |

  So "always write them" is already true; the bug is in *reading* era-1
  sidecars. `gen_norm` absent is already derived correctly from `batch_norm`.
- **`datatype` is a dead field.** Nothing sets it (no CLI flag, no assignment);
  it is always the template's `"magnified_profile"`. Data selection uses
  `modality`. Since 10cba23 every sidecar's `datatype` is meaningless.
- **v0.1.0 is an HF tag only** (no git tag, no GitHub release). In both HF repos
  it was moved from a first "Release v0.1.0" commit to a second one ~50 min
  later. Weights and generator sidecars are byte-identical (same LFS sha256)
  across both; only README/MANIFEST and (magnified-profile) the discriminator
  sidecar's pool lists changed. `release_weights.py` will happily re-tag.
- **The local `keras/snowgan/core/generator_config.json` is not the release
  copy** — later training rewrote it (`gen_norm: pixel`, fade_step 369267).
  Anything about the core release must read the HF snapshot (8a270f7).

## 2. Corrections to the brief (to send back to snowGradient)

1. **47:3 is the magnified-profile release, not core**, and it is the terminal
   value of the adaptive-steps ratchet (1 -> 61 over steps 298k-425k, after SN
   and cosine decay to 1e-7 had already frozen learning). The model learned at
   disc=2:gen=3. Core ran 2:3 with adaptive off. Already retracted in
   docs/experiments.md:76-95.
2. **Ceiling crossing:** the 3,547 ceiling is `2*sqrt(1*1024*1024*3)`. The
   sustained crossing is ~14-15k (1k-rolling mean last <= 3547 at 14,098;
   5k-window median 3,599 over 13.5-18.5k). **4,357 is a single raw step**
   (index 13,191), not a level. The GP/lowres caveat does not rescue it: GP >= 0
   and lowres adds at most 0.5*887 = 443, so any value below -3,991 requires
   the Wasserstein term itself past the ceiling — 23.5% of steps in 13-18k.
3. **"Six colour statistics beat the backbone"** is a breakability result
   (RESEARCH_PREVIEW_PLAN.md:23). On modality, colour stats (0.699) score
   *below* the backbone (0.868). NEGATIVE_RESULTS.md is internally consistent;
   the brief conflated the two tasks.
4. Item 3's fix is on the read side (above), not the write side.

## 3. Commits (this branch: feat/research-preview-asks)

1. **Standalone preprocess.** New `snowgan/data/preprocess.py` (tf + numpy
   only) holding `mask_blue_board` and
   `preprocess_image(image, resolution, *, mask_board=False)`; dataset.py
   re-exports both and `DataManager.preprocess_image` delegates. Channel
   normalisation moves *before* resize: rank 2 -> expand, 1 channel -> RGB,
   4 -> drop alpha, anything else raises. RGB output is unchanged bit-for-bit
   (tested against the old method).
2. **Era-1 sidecar resolution.** In `build.__init__`, a loaded sidecar with no
   `gen_norm` key gets `gen_upsampler="transpose"`,
   `gen_convs_per_resolution=1`. Keyed on key *presence*, never on values, and
   never applied to the template path. `datatype` follows `modality` on
   configure (deprecated alias, kept in dump for older readers). Tests use the
   actual released sidecar contents.
3. **`snowgan.export`.** `to_conv2d(model)` for a Discriminator / Generator /
   their keras.Model: a layer walk that rebuilds a rank-4 functional model with
   identical layer order and copied weights; raises on any layer type it does
   not know (no silent drops), on depth != 1, on non-unit depth stride or
   dilation, and on a mid-fade generator. Keeps `Flatten(name="features")`.
   `export_tflite(model, path, *, int8_representative=None)` via
   `keras.export.ExportArchive -> SavedModel -> from_saved_model` (the path
   that does not produce NaN). `python -m snowgan.export --sidecar ... --weights
   ... --out ...`. Tests: equivalence <= 1e-5 over the variant matrix, the
   refusals, and a tiny TFLite round trip.
4. **No tag mutation.** `release_weights.py` refuses a tag that already exists
   on the HF repo (checked before upload, so no orphan commit).
5. **Model-card drafts** (local only; HF push needs Denny's sign-off). Drafted
   in `docs/release_cards/`, applying snowGradient's list with §2's
   corrections, plus one they missed: both READMEs say downstream consumers
   "evaluate on the same held-out cores the GAN never saw", which UPGRADES #54
   refutes. Item 4 resolution: document the re-tag in the card rather than cut
   v0.1.1 — the weights did not change, so a new version would imply a model
   difference that does not exist.
6. **Generator quality evidence** (tooling now, scoring after the other job):
   `kid_check.py --real_vs_real` (two disjoint real sets -> the floor and its
   SE), score both released generators against sites 3-6, plus the existing
   pixel-space memorisation check against everything the release trained on
   (13 groups, including its "held-out" pools). Pre-registered threshold to be
   stated once, as a number derived from the measured floor, before scoring.

   Built (21c4887): `kid_check.py --sites 3,4,5,6 --real_vs_real`, which
   refuses any site the sidecar's pools contain, and a corrected
   `memorization_check.py` corpus (it read `honor_splits` from a built config,
   which defaults a missing key to True, so for the releases it dropped the
   pools they trained on and compared against sites they never saw).

   Before scoring (after the arm A2 run frees the GPU):
   - **Core needs an image cache first.** `~/rmdig-cache-1024` holds only the
     2,350 magnified profiles; 845 core images at ~16 MB each is ~13 GB of
     downloads (`scripts/build_image_cache.py`). Magnified-profile can be
     scored as soon as the GPU is free.
   - **The memorisation reference includes same-group neighbours.** Images
     within a group are near-duplicates, so "real -> nearest other real" is
     mostly a sibling distance, which sets a low bar. **Decided 2026-10-02,
     before any release was scored:** the verdict uses the cross-group
     reference (nearest real image from a *different* group); the
     any-neighbour reference is printed alongside for comparability.
   - **Pre-registered KID criterion (decided 2026-10-02, before scoring):** a
     release passes if its KID is within 2 standard errors of the real-vs-real
     floor from the same invocation, where the standard error is
     sqrt(SE_release² + SE_floor²). Stated once, as this formula; no second
     form.
   - **The real loader resizes with PIL bilinear**, not the canonical
     `preprocess_image`. That is consistent with every campaign KID so far, so
     it stays as-is for comparability; the release numbers inherit it.
   - Sample sizes: 200 generated per release, 10 subsets of 100, and floor
     halves of the same subset size, so floor and score use one estimator at
     one n.

Commits 1-4 touch the sidecar contract (2) and add a public surface (1, 3):
snowGradient and AvAI both read sidecars via `build()` — commit 2 only changes
what an era-1 sidecar resolves to, which is a fix for both.
