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
First core release. Trained ~127k steps without spectral_norm (the v0.1.0 ms2 fade-step counter shows higher numbers but real training is ~127k). Disc loss diverged post-step-80k due to small dataset (~13 unique cores) + no Lipschitz constraint + unconstrained multiscale_disc lowres critic. Generator produces structured outputs (vertical snow-like patterns, blue/white palette) so the Conv3D backbone has learned meaningful features even if the final Dense head is noisy. Backbone usable for transfer learning; v0.2 planned with spectral_norm enabled for a stable retrain.
