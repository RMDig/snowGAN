# snowGAN Architecture

Technical reference for snowGAN's model architectures, training pipeline, and internal design.

---

## Table of Contents

- [Overview](#overview)
- [Generator](#generator)
- [Discriminator](#discriminator)
- [WGAN-GP Loss](#wgan-gp-loss)
- [Progressive Fade-In](#progressive-fade-in)
- [Training Pipeline](#training-pipeline)
- [Data Pipeline](#data-pipeline)
- [Configuration System](#configuration-system)
- [Checkpoint & Persistence](#checkpoint--persistence)
- [Inference Export (rank 4)](#inference-export-rank-4)

---

## Overview

snowGAN is a Wasserstein GAN with Gradient Penalty (WGAN-GP) that generates synthetic 1024x1024 snow crystal images from the [Rocky Mountain Snowpack dataset](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack). It features progressive fade-in training and post-progressive stabilization techniques.

**Framework:** TensorFlow 2.15+ / Keras 3
**Architecture:** WGAN-GP with optional progressive growing
**Resolution:** 1024x1024 RGB
**Latent space:** 100-dimensional Gaussian noise

---

## Generator

**File:** `src/snowgan/models/generator.py`

The generator maps a latent vector `z ~ N(0, I)` of dimension `latent_dim` (default 100) to a 1024x1024x3 RGB image in [-1, 1] range.

### Architecture

```
z ∈ R^100
    │
    ▼
Dense(16 * 16 * 1024)  →  Reshape(16, 16, 1024)
    │
    ▼  Conv2DTranspose(1024, 3x3, stride 2)  →  32x32x1024
    │  [BatchNorm]  →  LeakyReLU(0.25)
    ▼
    │  Conv2DTranspose(512, 3x3, stride 2)   →  64x64x512
    │  [BatchNorm]  →  LeakyReLU(0.25)
    ▼
    │  Conv2DTranspose(256, 3x3, stride 2)   →  128x128x256
    │  [BatchNorm]  →  LeakyReLU(0.25)
    ▼
    │  Conv2DTranspose(128, 3x3, stride 2)   →  256x256x128
    │  [BatchNorm]  →  LeakyReLU(0.25)
    ▼
    │  Conv2DTranspose(64, 3x3, stride 2)    →  512x512x64
    │  [BatchNorm]  →  LeakyReLU(0.25)
    ▼
    │  Conv2DTranspose(3, 3x3, stride 2)     →  1024x1024x3
    │  activation=tanh
    ▼
output ∈ [-1, 1]^(1024×1024×3)
```

### Fade Endpoints

When progressive training is enabled (`fade=True`), the generator builds a secondary `fade_endpoints` model that outputs two images:

- **`toRGB_prev`**: A 1x1 Conv2D applied to the second-to-last feature map, upsampled to match the final output dimensions
- **`toRGB_curr`**: The standard final output

During fade-in, these are blended: `output = (1 - alpha) * prev_up + alpha * curr_img`

### Loss Function

Standard Wasserstein generator loss:
```
L_G = -E[D(G(z))]
```

The generator wants the discriminator to output large (positive) values for its fakes.

---

## Discriminator

**File:** `src/snowgan/models/discriminator.py`

The discriminator (critic) maps a 1024x1024x3 image to a single unbounded real value (no sigmoid — this is WGAN).

### Architecture

```
input ∈ R^(1024×1024×3)
    │
    ▼  [SpectralNorm] Conv2D(64, 3x3, stride 2)   →  512x512x64
    │  LeakyReLU(0.25)
    ▼
    │  [SpectralNorm] Conv2D(128, 3x3, stride 2)  →  256x256x128
    │  LeakyReLU(0.25)
    ▼
    │  [SpectralNorm] Conv2D(256, 3x3, stride 2)  →  128x128x256
    │  LeakyReLU(0.25)
    ▼
    │  [SpectralNorm] Conv2D(512, 3x3, stride 2)  →  64x64x512
    │  LeakyReLU(0.25)
    ▼
    │  [SpectralNorm] Conv2D(1024, 3x3, stride 2) →  32x32x1024
    │  LeakyReLU(0.25)
    ▼
    Flatten  →  R^(32*32*1024)
    │
    ▼  [SpectralNorm] Dense(1)  →  R^1
    │  (no activation)
    ▼
critic score ∈ R
```

SpectralNormalization wrapping is conditional on `config.spectral_norm`. Filter counts auto-invert if passed in generator order.

### Loss Function

WGAN-GP discriminator loss:
```
L_D = E[D(G(z))] - E[D(x)] + λ_gp * GP
```

Where GP is the gradient penalty (see below).

---

## WGAN-GP Loss

**File:** `src/snowgan/losses.py`

### Wasserstein Distance

The discriminator minimizes the negative Wasserstein distance:
```
L_D = mean(D(fake)) - mean(D(real)) + λ * GP
```

The generator maximizes:
```
L_G = -mean(D(fake))
```

### Gradient Penalty

Enforces the 1-Lipschitz constraint on the discriminator:

```python
alpha = Uniform(0, 1)  # per-sample
interpolated = alpha * real + (1 - alpha) * fake
gradients = ∇_interpolated D(interpolated)
GP = E[(||gradients||_2 - 1)^2] * lambda_gp
```

Default `lambda_gp = 10.0`. Gradient penalty is computed on *unaugmented* images for stable gradients, even when differentiable augmentation is enabled.

---

## Progressive Fade-In

**Files:** `src/snowgan/utils.py`, `src/snowgan/trainer.py`, `src/snowgan/models/generator.py`

Progressive training blends two generator output paths (previous and current resolution) over a configurable number of steps.

### How It Works

This is an **intra-model** progressive scheme (not multi-resolution ProGAN). The generator always outputs at full resolution, but during the fade phase, the output blends between:

1. **Previous path**: Second-to-last feature map → 1x1 Conv2D (toRGB_prev) → nearest-neighbor upsample
2. **Current path**: Full generator output through all layers

### Alpha Schedule

```
alpha = clip(current_step / fade_steps, 0, 1)
output = (1 - alpha) * prev_path + alpha * current_path
```

When `alpha = 1.0` (after `fade_steps`), training switches to standard mode using only the current path.

### Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `fade` | `false` | Enable progressive fade-in |
| `fade_steps` | `50000` | Steps to ramp alpha from 0 to 1 |
| `fade_step` | `0` | Current position (persisted for resume) |

### Resume Support

- `fade_step` is persisted in both generator and discriminator configs after every training step
- `generator_fade_endpoints.weights.h5` stores `toRGB_prev` weights for mid-fade resume
- On resume, the trainer reads `fade_step` from config to restore the exact alpha position

---

## Training Pipeline

**File:** `src/snowgan/trainer.py`

### Initialization (`Trainer.__init__`)

1. Load generator and discriminator weights from checkpoints
2. Load fade endpoint weights if they exist
3. Initialize Adam optimizers for both models
4. Load training history (loss curves)
5. Restore global step from `fade_step` config
6. Initialize post-progressive features (augment, LR decay, EMA, FID)

### Training Loop (`Trainer.train`)

```
for epoch in range(epochs):
    batch = (resume from last batch number)
    while data available:
        x = dataset.batch(batch_size, 'magnified_profile')
        if x is None: reset train_ind to 390, try again

        train_step(x)

        log loss, plot history
        generate samples (using EMA weights if enabled)

        if fid_interval > 0 and global_step % fid_interval == 0:
            compute FID, save best model

        if batch % save_interval == 0:
            save checkpoint
            cleanup old checkpoints
```

### Training Step (`Trainer.train_step`)

**Discriminator (N steps, default 2):**
```
noise = Normal(batch_size, latent_dim)
synthetic = stop_gradient(generate_with_fade(noise))  # no gen gradients
disc_real = augment(real_images)    # outside tape
disc_fake = augment(synthetic)     # outside tape

with GradientTape:
    real_output = D(disc_real)
    fake_output = D(disc_fake)
    gp = gradient_penalty(real_images, synthetic)  # on unaugmented
    loss = mean(fake) - mean(real) + lambda_gp * gp

apply_gradients(loss, D.variables)
```

**Generator (M steps, default 3):**
```
noise = Normal(batch_size, latent_dim)

with GradientTape:
    synthetic = generate_with_fade(noise)
    disc_fake = augment(synthetic)  # inside tape (differentiable)
    fake_output = D(disc_fake)
    loss = -mean(fake_output)

variables = G.variables + fade_endpoint_variables (if fading)
apply_gradients(loss, variables)
```

**Post-step:** update EMA weights, apply LR decay

---

## Data Pipeline

**File:** `src/snowgan/data/dataset.py`

### DataManager

Loads data from the HuggingFace dataset `rmdig/rocky_mountain_snowpack`.

**Dataset datatypes:**
| Code | Name | Description |
|------|------|-------------|
| 0 | `core` | Snow core samples |
| 1 | `profile` | Snow profiles |
| 2 | `magnified_profile` | Magnified crystal images (used for training) |
| 3 | `crystal_card` | Crystal card images |

### Batch Loading

```python
manager = DataManager(config)
batch = manager.batch(batch_size=8, datatype='magnified_profile')
# Returns: numpy array (8, 1024, 1024, 3) in [-1, 1] range, or None
```

Processing per image:
1. Filter by `datatype == magnified_profile`
2. Resize to `config.resolution` (default 1024x1024)
3. Convert to float32
4. Normalize: `(pixel / 127.5) - 1.0` → range [-1, 1]

The `train_ind` config field tracks the current position in the dataset and persists across restarts.

---

## Configuration System

**File:** `src/snowgan/config.py`

### Config Class (`build`)

The `build` class manages configuration persistence:

```python
config = build("keras/snowgan/generator_config.json")
# Loads JSON if exists, otherwise uses default template

config.learning_rate = 1e-4
config.save_config()  # Persists to disk
# Also auto-saves on process exit via atexit
```

### Separate Configs

Generator and discriminator each have their own config file:
- `{save_dir}/generator_config.json`
- `{save_dir}/discriminator_config.json`

Both share the same schema but differ in defaults (e.g., filter order, learning rate, training steps).

### Key Config Fields

**Model Architecture:**
| Field | Generator Default | Discriminator Default |
|-------|-------------------|----------------------|
| `filter_counts` | [1024, 512, 256, 128, 64] | [64, 128, 256, 512, 1024] |
| `kernel_size` | [3, 3] | [3, 3] |
| `kernel_stride` | [2, 2] | [2, 2] |
| `latent_dim` | 100 | 100 |
| `batch_norm` | false | false |
| `spectral_norm` | varies | varies |

**Training:**
| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | 1e-5 / 1e-4 | Adam learning rate |
| `beta_1` | 0.5 | Adam beta_1 |
| `beta_2` | 0.9 | Adam beta_2 |
| `training_steps` | 3 (gen) / 2 (disc) | Steps per batch per model |
| `lambda_gp` | 10.0 | Gradient penalty weight |
| `batch_size` | 8 | Images per training batch |

**Progressive & Post-Progressive:**
| Field | Default | Description |
|-------|---------|-------------|
| `fade` | false | Enable progressive fade-in |
| `fade_steps` | 50000 | Steps for alpha ramp |
| `fade_step` | 0 | Current fade progress |
| `spectral_norm` | false | SpectralNorm on discriminator |
| `augment` | false | Differentiable augmentation |
| `lr_decay` | null | "cosine" for cosine annealing |
| `lr_min` | 1e-7 | LR floor for decay |
| `ema_decay` | 0.0 | Generator EMA (0 = disabled) |
| `fid_interval` | 0 | FID eval frequency (0 = disabled) |

### CLI Override Flow

```
CLI args  →  parse_args()  →  configure_gen(config, args)
                            →  configure_disc(config, args)
                                 └→ configure_generic(config, args)
```

CLI args use `None` defaults so that omitted flags don't overwrite persisted config values. Only explicitly set flags override.

---

## Checkpoint & Persistence

### Saved Files

Each checkpoint directory contains:

| File | Description |
|------|-------------|
| `generator.keras` | Full generator model (Keras 3 format) |
| `discriminator.keras` | Full discriminator model |
| `generator_config.json` | Generator configuration snapshot |
| `discriminator_config.json` | Discriminator configuration snapshot |
| `generator_fade_endpoints.weights.h5` | Fade endpoint weights (if progressive) |
| `generator_ema.weights.h5` | EMA shadow weights (if `ema_decay > 0`) |

### Loss History

Stored in the main save directory:
- `generator_loss.txt` - One loss value per line
- `discriminator_loss.txt` - One loss value per line
- `trained.txt` - Training metadata
- `history.png` - Loss plot

### Checkpoint Lifecycle

1. **Auto-save on exit** via `atexit.register(self.save_model)`
2. **Periodic saves** every `cleanup_milestone` batches to `{save_dir}/batch_{N}/`
3. **Cleanup** removes intermediate checkpoints, keeping only those at milestone intervals
4. **Best FID** saves to `{save_dir}/best_fid/` when FID improves

### Synthetic Image Tracking

Generated samples are saved to `{save_dir}/synthetic_images/` as:
```
batch_{N}_synthetic_{1..n_samples}.png
```

Cleanup trims to the 7 most recent images per batch for non-milestone batches.

---

## Image Generation

**File:** `src/snowgan/generate.py`

### Generating Samples

```bash
snowgan --mode generate --n_samples 10 --save_dir keras/snowgan/
```

Generates `n_samples` images by:
1. Sampling noise `z ~ N(0, I)` of shape `(count, latent_dim)`
2. Forward pass through generator: `G(z)` → images in [-1, 1]
3. Denormalize: `(image + 1) * 127.5` → [0, 255] uint8
4. Save as PNG files

### Creating Progress Videos

```python
from snowgan.generate import make_movie
make_movie(save_dir="keras/snowgan/synthetic_images/",
           videoname="progress.mp4",
           framerate=15,
           filepath_pattern="batch_*_synthetic_1.png")
```

Compiles synthetic images into an MP4 video using OpenCV, useful for visualizing training progression.

---

## Inference Export (rank 4)

Training is rank 5, `(B, depth, H, W, C)`. Phones are not: TFLite's CONV_3D is
a CPU-only kernel with no int8 and no GPU / NNAPI / Core ML delegate.
`snowgan.export.to_conv2d(model)` rebuilds a **depth-1** model with Conv2D /
Conv2DTranspose layers and the same weights. Inputs and outputs become
`(B, H, W, C)`, and the discriminator keeps `Flatten(name="features")`.
`export_tflite` writes it as TFLite via ExportArchive -> SavedModel. A TFLite
file has one output: for the full discriminator that is the critic score, so
a backbone for a downstream head is exported with `features_only=True`
(optionally at another `input_hw`, e.g. 256x256; conv weights do not depend on
resolution, the critic's Dense head does). Every rebuilt layer is float32. CLI:
`python -m snowgan.export --kind ... --sidecar ... --weights ... --out x.tflite
[--features [--resolution H W]]`; it checks the export against the Conv3D
model at batch 1 and runs the written `.tflite`, with gates relative to the
output's scale.

The export is exact only because of invariants this repo currently holds. A
change that breaks one of them breaks the export, and
`tests/unit/test_export_conv2d.py` will say so:

- **Every conv kernel is `(1, kH, kW)` with depth stride and dilation 1.** A
  kernel that spans depth has no rank-4 equivalent.
- **Nothing mixes depth slices except the critic's `Flatten -> Dense` head.**
  That head is why `depth == 2` (`modality="merged"`) is refused.
- **Every layer type has a registered rank-4 counterpart.** An unknown layer
  raises; the export never skips one.
- **`Generator.model` is the whole generator only outside a fade.** A mid-fade
  generator is refused.

The rank-4 model is an inference artifact, never a training or checkpoint
format.
