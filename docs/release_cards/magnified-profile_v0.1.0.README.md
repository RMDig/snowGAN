---
license: apache-2.0
language:
  - en
library_name: snowgan
pipeline_tag: image-to-image
tags:
  - gan
  - wgan-gp
  - image-generation
  - snowpack
  - modality-magnified_profile
datasets:
  - rmdig/rocky_mountain_snowpack
---

# snowGAN — magnified_profile backbone (v0.1.0)

WGAN-GP trained on the
[Rocky Mountain Snowpack dataset](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack),
single-modality (`magnified_profile`), depth=1, 1024x1024 resolution.
Published from the [snowGAN](https://github.com/dennys246/snowGAN) project as a
research artifact.

> **Not for safety decisions.** Nothing here predicts avalanche danger, slope
> stability or snowpack breakability. Do not use it for any backcountry decision.

## How to use

The canonical consumer-side path uses [`snowgan.weights.fetch`](https://github.com/dennys246/snowGAN/blob/main/src/snowgan/weights.py)
to download + cache, then `snowgan.models.Discriminator` to rebuild and load:

```python
from snowgan.weights import fetch
from snowgan.config import build
from snowgan.models.discriminator import Discriminator

# Fetch the release (cached locally after first call).
path = fetch("RMDig/snowGAN-magnified-profile", "v0.1.0")

# Reconstruct the model from its sidecar config, then load weights.
cfg = build(str(path / "discriminator_config.json"))
disc = Discriminator(cfg)
disc.model.build((None, cfg.depth, cfg.resolution[0], cfg.resolution[1], cfg.channels))
disc.model.load_weights(str(path / "discriminator.weights.h5"))

# Tap the named features layer — the cross-repo contract with downstream consumers.
features = disc.model.get_layer("features")
print("backbone features:", features.output.shape)  # (None, 1048576)
```

Requires `pip install snowgan[hub]` (pulls in `huggingface_hub`). Without the
`[hub]` extra, `fetch()` raises a clean ImportError naming the missing dep.

To rebuild the **generator** from `generator_config.json`, use a snowgan
version that includes the legacy-sidecar fix (feat/research-preview-asks). On
older versions pass `gen_upsampler="transpose"` and
`gen_convs_per_resolution=1` yourself. This release's sidecar predates those
fields, and the newer defaults describe a different generator.

## Intended use

Research artifact from snowGAN's GAN-training experiments. The discriminator's
`features` layer and the generator are published so the results can be
reproduced and tested. Publishing them does not mean they are known to be useful
downstream.

**Transfer value is unmeasured here and contradicted downstream.** snowGAN has
not measured whether these features help any task. snowGradient's measurements
(see [Quality evidence](#quality-evidence)) found that they do not beat a
randomly initialised copy of the same network. Compare against that baseline
before relying on these features.

## Architecture

| Field | Value |
| --- | --- |
| Modality | `magnified_profile` (depth=1) |
| Resolution | 1024x1024 |
| Channels | 3 |
| Latent dim | 100 |
| Generator filter counts | `[1024, 512, 256, 128, 64]` |
| Discriminator filter counts | `[64, 128, 256, 512, 1024]` |
| Conv kernel / stride | `[3, 3]` / `[2, 2]` |
| Backbone (Flatten "features") dim | 1048576 |
| Final activation | `tanh` |

The discriminator's `Conv3D` layers use `ksize=(1, kH, kW)`, so the depth axis is
broadcast — kernels themselves are depth-agnostic. This is the contract that lets
downstream consumers compose multiple single-modality backbones into a depth=N model
(e.g. profile + core stacked at depth=2 for paired-modality transfer).

## Training

Trained with WGAN-GP loss (Wasserstein + gradient penalty, λ_gp=1.0)
on the magnified_profile samples of [`rmdig/rocky_mountain_snowpack`](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack).
At release: `fade_step=428139`.

**The settings below are the run's terminal configuration, not the
configuration it learned under.** Per-snapshot configs, sorted by step, show:

| Steps | disc:gen | λ_gp | Spectral norm | LR schedule |
| --- | --- | --- | --- | --- |
| ≤ ~288.6k | 2:3 | 10.0 | off | constant |
| 289.6k | 2:3 | 10.0 | on | cosine on |
| 295.6k-296.6k | 2:3 | 1.0 | on | cosine |
| 298.6k → 425.6k | 1:3 → … → 47:3 → 61:3 (adaptive) | 1.0 | on | cosine |

Cosine decay was enabled past its then hard-coded 200k horizon, so both
learning rates dropped straight to `lr_min = 1e-7`. The changes after ~289k were
applied to a model that had largely stopped moving. The model acquired its
structure at **disc:gen 2:3, λ_gp 10, without spectral norm**. The 47:3 in the
MANIFEST is the adaptive ratchet's value at release.

| Stabilizer | Setting |
| --- | --- |
| Spectral norm | `True` |
| Differentiable augmentation | `True` |
| Adaptive augmentation (ADA) target | `0.6` |
| Adaptive disc/gen step ratio | `True` |
| EMA decay (generator shadow) | `0.999` |
| Multi-scale discriminator | `True` |
| Gradient clip (global norm) | `1.0` |
| LR decay schedule | `cosine` (lr_min=`1e-07`) |
| FID eval interval | `5000` steps (no valid FID; see Quality evidence) |

### Training data and splits

- **13 `(site, column, core)` groups, all from sites 0-2, one operator.** The
  dataset's sites 3-6 were added on 2026-08-17, after this release, and are
  unseen by it.
- The sidecars persist a `trained_pool` / `validation_pool` / `test_pool` split
  (10 / 1 / 2 groups). **These pools are not held out from this model.** The
  training loop at release ignored them, so the GAN trained on all 13 groups
  (snowGAN [UPGRADES #54](https://github.com/dennys246/snowGAN/blob/main/docs/UPGRADES.md)).
  Images from sites 3-6 are the only data this release has not seen.

## Quality evidence

- **No valid FID.** The in-training FID used 64 samples against
  2048-dimensional Inception features. That covariance estimate is
  rank-deficient and biased by sample size, so the `fid_interval` setting above
  produced no usable number. No KID or memorisation check has been run on this
  release yet.
- **The critic stayed within its Lipschitz bound.** Max |logged loss| was 783
  over 427,220 steps, against a ceiling of 2*sqrt(1024*1024*3) ≈ 3,547.
- The generator's samples are varied and not collapsed (snowGradient measured
  EMA diversity 0.539). That is the only positive evidence for this release, and
  it is generative, not discriminative.
- **Downstream measurements (snowGradient, 2026-10):**
  - In a pre-registered modality-recognition probe (core vs profile vs magnified
    profile, leave-one-site-out over sites 3-6), the pretrained magnified-profile
    backbone scored 0.868 balanced accuracy. A randomly initialised copy of the
    same network scored 0.987 / 0.982 (2 seeds) and was better on every held-out
    site. Almost all pretrained errors were core <-> magnified-profile confusions.
  - On snowpack breakability, no feature set beat a constant predictor, and six
    colour statistics beat a pretrained snowGAN backbone.
  - Pretrained features beat random init only at clustering images by site,
    which reflects lighting and camera, not snow.
  - Write-up: snowGradient `docs/NEGATIVE_RESULTS.md`.

## Limitations

- Trained exclusively on the magnified-profile modality of the rmdig/rocky_mountain_snowpack dataset; generalization to other distributions untested.
- Single-modality backbone (depth=1); paired-modality features must be composed on the consumer side. See AvAIs load_backbones for the canonical two-backbone composition pattern.
- 13 groups from sites 0-2, one operator: a very small, narrow training set. There is no held-out data from those sites (see Training data and splits).
- Its features appear specialised to magnified-profile texture. In snowGradient's modality probe, almost every error confused cores with magnified profiles.

## Release history

The `v0.1.0` tag was moved once. On 2026-06-03 a first "Release v0.1.0" commit
was followed about 50 minutes later by a second one that corrected metadata
only (MANIFEST/README provenance, and on magnified-profile the persisted split
pools). The tag now points at the second commit. **All weight files are
byte-identical across both commits** (same LFS sha256), so anything loaded
from either snapshot computes the same function. snowGAN's release script now
refuses to move an existing tag.

Card corrected 2026-10: removed unmeasured claims about feature quality and
transfer learning, and added the measurements above.

## Files in this release

- `discriminator.weights.h5` — main discriminator weights.
- `discriminator_config.json` — architecture sidecar; pass to `snowgan.models.Discriminator(cfg)`.
- `generator.weights.h5` + `generator_config.json` — generator weights and sidecar.
- `generator_ema.weights.h5` — EMA shadow weights (only if EMA was enabled during training).
- `generator_fade_endpoints.weights.h5` — progressive-fade toRGB endpoints (only if fade was used).
- `discriminator_lowres.weights.h5` — multi-scale 256×256 critic (only if multiscale_disc was on).
- `MANIFEST.md` — full provenance dump (git SHA, every training flag, every artifact). Read this for debugging.
- `README.md` — this file.

## License

Apache 2.0 — see the [snowGAN repository](https://github.com/dennys246/snowGAN) for the full license text.

## Cross-references

- **Source code**: [github.com/dennys246/snowGAN](https://github.com/dennys246/snowGAN)
- **Downstream project**: [github.com/dennys246/AvAI](https://github.com/dennys246/AvAI)
- **Companion modality backbones**:
  [`RMDig/snowGAN-magnified-profile`](https://huggingface.co/RMDig/snowGAN-magnified-profile),
  [`RMDig/snowGAN-core`](https://huggingface.co/RMDig/snowGAN-core)
- **Training dataset**: [`rmdig/rocky_mountain_snowpack`](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack)

## Release notes

First public release. Trained ~428k steps (427,220 logged). Spectral norm,
gradient clipping, cosine LR and adaptive steps were switched on only after
~289k steps, by which point learning had effectively stopped (see Training).
The discriminator loss stayed far below the Lipschitz ceiling throughout.

*Corrected 2026-10.* Earlier text said the model was trained "with full
production stabilizers", which describes only the last ~130k steps. It also
said this release "supersedes an earlier v0.1.0". See Release history: the
earlier snapshot differed only in metadata.
