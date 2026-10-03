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
  - modality-core
datasets:
  - rmdig/rocky_mountain_snowpack
---

# snowGAN — core backbone (v0.1.0)

WGAN-GP trained on the
[Rocky Mountain Snowpack dataset](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack),
single-modality (`core`), depth=1, 1024x1024 resolution.
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
path = fetch("RMDig/snowGAN-core", "v0.1.0")

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

This release's `generator_config.json` predates the generator's architecture
fields. With snowgan 1.0.0, rebuild the generator with
`gen_upsampler="transpose"` and `gen_convs_per_resolution=1`; the 1.0.0
defaults describe a different generator and the weights will not load.

## Intended use

Research artifact from snowGAN's GAN-training experiments. The discriminator's
`features` layer and the generator are published so the results can be
reproduced and tested. Publishing them does not mean they are known to be useful
downstream.

**Transfer value is unmeasured.** Nobody has measured whether this core
backbone's features help any task, or whether they beat a randomly initialised
copy of the same network. The one such comparison so far, on the companion
magnified-profile backbone, favoured random init (see
[Quality evidence](#quality-evidence)). Compare against that baseline before
relying on these features.

## Architecture

| Field | Value |
| --- | --- |
| Modality | `core` (depth=1) |
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

Trained with WGAN-GP loss (Wasserstein + gradient penalty, λ_gp=10.0)
on the core samples of [`rmdig/rocky_mountain_snowpack`](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack).
At release: `fade_step=130000`.

| Stabilizer | Setting |
| --- | --- |
| Spectral norm | `False` |
| Differentiable augmentation | `True` |
| Adaptive augmentation (ADA) target | `0.6` |
| Adaptive disc/gen step ratio | `False` |
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
- The GAN streamed the dataset's row-level `train` split as it stood at release:
  391 of the 495 images of this modality, covering all 13 groups. The
  remaining images come from the same groups, so they are not independent
  held-out data. Sites 3-6 are the only data unseen at the group level.

## Quality evidence

- **No valid FID.** The in-training FID used 64 samples against
  2048-dimensional Inception features, a rank-deficient and sample-size-biased
  estimate. In this run it also compared core samples against
  magnified-profile reals. The `fid_interval` setting above produced no usable
  number. No KID or memorisation check has been run on this release yet.
- **The logged critic loss left the 1-Lipschitz range early.** A 1-Lipschitz
  critic on inputs in [-1, 1] at 1024x1024x3 has |Wasserstein estimate| at most
  2*sqrt(1024*1024*3) ≈ 3,547. Single steps exceeded that from step ~120; the
  logged |loss| stayed above it persistently from about step 14-15k (median
  3,599 over steps 13.5k-18.5k). From 80k to the end the mean |loss| was 28,903
  and the peak 156,321 (230,377 over the whole run), as a roughly symmetric
  ± oscillation rather than a critic that was winning.
  - Caveat: the logged value is main-critic Wasserstein + λ·GP + 0.5·(low-res
    critic Wasserstein). The low-res critic had no gradient penalty and no
    spectral norm, so nothing bounded it. Values past the ceiling show that at
    least one of the two critics exceeded its bound; the log cannot say which.
- disc:gen update ratio 2:3 throughout (adaptive steps off).
- **Downstream measurement (snowGradient, 2026-10).** In a pre-registered
  modality-recognition probe (core vs profile vs magnified profile,
  leave-one-site-out over sites 3-6, at 256x256 with global average pooling),
  the pretrained **magnified-profile** backbone scored 0.868 balanced accuracy.
  A randomly initialised copy of the same network scored 0.987 / 0.982 (2
  seeds) and was better on every held-out site. 24 of the pretrained
  backbone's 26 errors were core <-> magnified-profile confusions. This is one
  task, at a quarter of training resolution.
- The core backbone itself has not been compared against random init.

## Limitations

- Trained on the core modality of rmdig/rocky_mountain_snowpack; generalization untested.
- The logged critic loss was outside the 1-Lipschitz range for most of training (see Quality evidence), so the score is not a usable Wasserstein estimate. Its features have not been shown to be informative.
- Single-modality backbone (depth=1); paired-modality features must be composed on the consumer side. See AvAIs load_backbones for the canonical two-backbone composition pattern.
- 13 groups from sites 0-2, one operator: a very small, narrow training set. There is no held-out data from those sites (see Training data and splits).

## Release history

Two commits titled "Release v0.1.0" were made about 50 minutes apart on
2026-06-03. The first carried no model card (a 3-line README); the second added
this card and corrected the MANIFEST's provenance. The `v0.1.0` tag was
created on the second commit, one second after it; there is no record of a tag
on the first. **All weight files are byte-identical across both commits**
(same LFS sha256), so either snapshot computes the same function.

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

First core release. Trained ~127k steps without spectral_norm. The fade-step
counter reads 130,000 but the loss log has 127,616 steps. The logged
discriminator loss left the 1-Lipschitz range persistently from about step
14-15k (see Quality evidence).

*Corrected 2026-10.* Earlier text said the backbone "has learned meaningful
features" and is "usable for transfer learning". Neither claim was measured,
and the one downstream comparison against random init (on the magnified-profile
backbone) went the other way. It also dated the divergence to step 80k.
