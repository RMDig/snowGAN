# snowGAN release v0.1.0

Snapshot of a snowGAN training run, packaged for downstream consumers
(AvAI etc.). Consume with `snowgan.weights.fetch("RMDig/snowGAN-magnified-profile", "v0.1.0")`.

## Provenance
- snowGAN package version: `unknown`
- snowGAN git SHA at release: `fde5671fed1b746962b6ca381e3a9b20ccf62e1c`
- Training dataset: `rmdig/rocky_mountain_snowpack`
- Source save_dir: `/mnt/d/GitSpot/snowGAN/keras/snowgan/magnified_profiles`

## Model architecture
- depth: 1
- resolution: [1024, 1024]
- modality: magnified_profile
- channels: 3
- latent_dim: 100
- filter_counts (gen): [1024, 512, 256, 128, 64]
- filter_counts (disc): [64, 128, 256, 512, 1024]
- kernel_size / kernel_stride: [3, 3] / [2, 2]

## Training state at release
- fade_step: 428139 (gen) / 428139 (disc)
- fade_steps target: 50000
- current_epoch: 0
- training_steps: gen=3, disc=47
- lambda_gp: 1.0

## Advanced training options
- spectral_norm: True
- augment: True
- multiscale_disc: True
- ema_decay: 0.999
- lr_decay: cosine (lr_min=1e-07)
- ada_target: 0.6
- adaptive_steps: True
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
First public release. Trained ~428k steps (427,220 logged). The training_steps / lambda_gp / spectral_norm values above are terminal: the model learned at disc:gen 2:3 and configured λ_gp 10 (an effective 100 before the 2026-03-13 GP fix), without spectral norm. SN, augmentation, EMA, cosine LR (which dropped straight to 1e-7), the multiscale critic, ADA, clipping and the adaptive ratchet (up to 61 at 425.6k, 47 at release) came on after ~289k. Max |logged disc loss| 783, inside the ~3,547 ceiling. Training data: 13 groups from sites 0-2, one operator; the GAN streamed the dataset's row-level train split (1,101 of 1,355 images) across all 13 groups, ignoring the split pools (snowGAN UPGRADES #54). No valid FID. KID vs real sites 3-6 (2026-10): 0.386 (EMA 0.338) against a real-vs-real floor of 0.0025 and a site-shift-only 0.069 -- fails the pre-registered criterion; KID vs its own training sites is the same (0.384), so the gap is the generator. Memorisation check: samples are not copies of training images. In snowGradient's 2026-10 modality probe a random-init copy beat this backbone (0.987 vs 0.868). Not for any safety or backcountry decision. Corrected 2026-10; see README.
