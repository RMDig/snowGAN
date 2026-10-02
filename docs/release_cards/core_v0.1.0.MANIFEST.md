# snowGAN release v0.1.0

Snapshot of a snowGAN training run, packaged for downstream consumers
(AvAI etc.). Consume with `snowgan.weights.fetch("RMDig/snowGAN-core", "v0.1.0")`.

## Provenance
- snowGAN package version: `unknown`
- snowGAN git SHA at release: `fde5671fed1b746962b6ca381e3a9b20ccf62e1c`
- Training dataset: `rmdig/rocky_mountain_snowpack`
- Source save_dir: `/mnt/d/GitSpot/snowGAN/keras/snowgan/core`

## Model architecture
- depth: 1
- resolution: [1024, 1024]
- modality: core
- channels: 3
- latent_dim: 100
- filter_counts (gen): [1024, 512, 256, 128, 64]
- filter_counts (disc): [64, 128, 256, 512, 1024]
- kernel_size / kernel_stride: [3, 3] / [2, 2]

## Training state at release
- fade_step: 130000 (gen) / 130000 (disc)
- fade_steps target: 50000
- current_epoch: 0
- training_steps: gen=3, disc=2
- lambda_gp: 10.0

## Advanced training options
- spectral_norm: False
- augment: True
- multiscale_disc: True
- ema_decay: 0.999
- lr_decay: cosine (lr_min=1e-07)
- ada_target: 0.6
- adaptive_steps: False
- grad_clip_norm: 1.0
- fid_interval: 5000

## Persisted dataset splits
- trained_pool: 10 groups
- validation_pool: 1 groups
- test_pool: 2 groups

## Artifacts
- `MANIFEST.md`
- `discriminator.weights.h5`
- `discriminator_config.json`
- `discriminator_lowres.weights.h5`
- `generator.weights.h5`
- `generator_config.json`
- `generator_ema.weights.h5`
- `generator_fade_endpoints.weights.h5`

## Notes
First core release. Trained ~127k steps (127,616 logged) without spectral_norm; disc:gen 2:3 throughout. Discriminator |loss| crossed the 1-Lipschitz ceiling (~3,547 at 1024x1024x3) persistently from about step 14-15k, not 80k; the logged loss includes λ·GP and 0.5·lowres terms, but those cannot account for values below -3,991 (23.5% of steps 13k-18k). Training data: 13 groups from sites 0-2, one operator, all trained on (split pools were not honoured; snowGAN UPGRADES #54). No valid FID. Feature quality and transfer value are unmeasured here and contradicted by snowGradient's 2026-10 probes. Not for any safety or backcountry decision. Corrected 2026-10; see README Release history.
