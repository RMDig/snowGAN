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
  - transfer-learning
  - modality-magnified_profile
datasets:
  - rmdig/rocky_mountain_snowpack
---

# snowGAN — magnified_profile backbone (v0.1.0)

WGAN-GP trained on the
[Rocky Mountain Snowpack dataset](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack),
single-modality (`magnified_profile`), depth=1, 1024x1024 resolution.
Published from the [snowGAN](https://github.com/dennys246/snowGAN) project for
downstream transfer-learning consumers.

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

## Intended use

Primary use case is **transfer learning** — downstream classifiers (e.g. [AvAI](https://github.com/dennys246/AvAI)) attach task heads to the discriminator's Conv3D backbone via `model.get_layer("features").output`. Secondary use is generating synthetic magnified_profile samples via the generator.

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
| FID eval interval | `5000` steps |

### Dataset splits

Splits are deterministic at the `(site, column, core)` group level (seed=42),
persisted in both sidecar configs so downstream consumers (e.g. AvAI) evaluate
on the same held-out cores the GAN never saw:

- `trained_pool`: 10 groups
- `validation_pool`: 1 groups
- `test_pool`: 2 groups

## Limitations

- Trained exclusively on the magnified-profile modality of the rmdig/rocky_mountain_snowpack dataset; generalization to other distributions untested.
- Single-modality backbone (depth=1); paired-modality features must be composed on the consumer side. See AvAIs load_backbones for the canonical two-backbone composition pattern.
- Held-out test_pool is only 2 (site, column, core) groups due to the small underlying dataset (~13 unique cores total). Downstream evaluation metrics will be noisy.

## Files in this release

- `discriminator.weights.h5` — main discriminator weights (the transfer backbone).
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
- **Downstream transfer-learning project**: [github.com/dennys246/AvAI](https://github.com/dennys246/AvAI)
- **Companion modality backbones**:
  [`RMDig/snowGAN-magnified-profile`](https://huggingface.co/RMDig/snowGAN-magnified-profile),
  [`RMDig/snowGAN-core`](https://huggingface.co/RMDig/snowGAN-core)
- **Training dataset**: [`rmdig/rocky_mountain_snowpack`](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack)

## Release notes

First public release. Trained ~428k steps with full production stabilizers (spectral_norm + augment + EMA + multiscale_disc + ADA + adaptive_steps + grad_clip + cosine LR). Disc loss stayed bounded throughout. This release supersedes an earlier v0.1.0 that had null trained/val/test pools in discriminator_config.json (the original PR #9 split-persistence bug, since fixed in trainer.py).
