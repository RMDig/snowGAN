# snowGAN experiment log

The running record of training experiments. One row per run. Fill a row **before**
starting the next run — this log is the memory that stops us re-deriving what we already
tried. Governed by [CLAUDE.md §9](../CLAUDE.md).

## The scoreboard (applied identically to every run)

Never judge by loss curves — WGAN loss magnitude is not quality. In order:

1. **Kill-check** (minutes, before letting a run continue):
   - **Latent diversity:** mean pairwise `|G(zᵢ) − G(zⱼ)|` over ~8 samples. Must be well
     above 0. `≈ 0.000` ⇒ full mode collapse (generator ignoring z) ⇒ dead run.
   - **Saturation:** `frac(|output| > 0.99)` low, output std near the data's (~0.5).
     `std ≈ 1.0 / 100% pinned` ⇒ tanh-rail collapse. `std ≈ 0` ⇒ vanished (grey frame).
   - Real core data reference: mean ≈ −0.22, std ≈ 0.63, ~0% saturated (in [−1,1]).
2. **Sample inspection:** open `synthetic_images/*.png`; two latents must give two
   *different*, non-saturated images.
3. **Downstream probe (the objective):** linear probe on avalanche-risk labels. Feature
   transferability, not image beauty. Gated on labels (TBD).

Snippet for the kill-check (adjust `$D`):
```
python -c "
import numpy as np,tensorflow as tf
from snowgan.config import build; from snowgan.models.generator import Generator
g=Generator(build('$D/generator_config.json')); g.model.load_weights('$D/generator.weights.h5')
z=tf.random.normal([8,100]); img=g.model(z,training=False).numpy()
d=np.mean([np.mean(np.abs(img[i]-img[j])) for i in range(8) for j in range(i+1,8)])
print('std=%.4f frac|x|>0.99=%.1f%% pairwise_div=%.4f'%(img.std(),100*np.mean(np.abs(img)>0.99),d))"
```

## Protocol

- One architectural piece per campaign; multiple tuning runs on that piece are expected
  (vary only its own knobs). Judge a piece by its best-tuned run. See CLAUDE.md §9.
- Ground truth = a baseline that **provably trains from scratch**, not one that only loads
  old weights.

## Ground truth

- **Proven architecture** (commit `7b487a0`, pre-2026-06-13): Conv3DTranspose per
  resolution, **one** conv per block, **no** normalization, kernel 3 / stride 2, tanh
  head. This trained the magnified_profiles v0.1.0 backbone (157k batches, genuine
  crystal-mesh output). Reproduced on `feat/shallow-generator-baseline` via
  `--gen_convs_per_resolution 1 --gen_norm none --gen_upsampler transpose`.
- **Static validation done:** loading `magnified_profiles/generator.weights.h5` into the
  reproduced architecture regenerates the mesh image (std 0.506, 3.95% saturated). Shapes
  and behavior match.
- **Dynamic validation OUTSTANDING:** we have NOT confirmed the current code *trains* this
  architecture from scratch. That is experiment 0.

## Log

Verdict key: ✅ trains/structure · ⚠️ partial · ❌ collapsed · ⏳ pending

| # | Date | Base | Single change under test | Data / res | Scoreboard result | Verdict |
|---|------|------|--------------------------|-----------|-------------------|---------|
| retro-A | ~06→07 | deep stack + PixelNorm + transpose | (many at once — uninterpretable) | core 1024 | 332k batches, tanh-saturated throughout (std 2.26 pre-tanh, 37% pinned) | ❌ |
| retro-B | 07-20 | + OutputGain (#47), SN+GP | (stacked) | core 256 | gain settled ~2.0, still saturated | ❌ |
| retro-C | 07-20 | + SN-only (GP off) | (stacked) | core 256 | gain settled 1.09 (healthy!) but collapsed to constant | ❌ |
| retro-D | 07-20 | + minibatch-stddev (#48) | (stacked) | core 256 | collapsed by 2k, saturated | ❌ |
| retro-E | 07-22 | shallow stack, no norm, no OutputGain | (arch swap) | core 256 | current weights std 1.0, 100% pinned | ❌ |
| retro-F | 07-25 | + board masking (#49, grey fill) | (masking) | core 256 masked | std 1.0, 100% pinned, pairwise_div **0.0000** | ❌ |
| **0** | pending | proven arch (`7b487a0` repro) | **control: does current code train it?** | **magnified_profiles 256** | ⏳ | ⏳ |

**Retro takeaways (what NOT to repeat):**
- Saturation/collapse survived *every* generator-side change and the data-confound removal
  (masking). ⇒ the driver is **training dynamics**, not generator architecture or the
  board confound.
- retro-C is the tell: a *healthy gain* (1.09, un-saturated head) still collapsed to a
  constant ⇒ the failure is the generator ignoring z (mode collapse), separable from tanh
  saturation, and not fixed by head tuning.
- retro-F (masked) reached pairwise diversity **exactly 0** with `gen_steps 3 : disc_steps
  1` — an *inverted* WGAN ratio (generator 3× the critic). The proven mag_profiles run was
  disc-heavy (disc 47 : gen 3). Critic/generator step ratio is a prime suspect and has
  never been isolated.
- Every run changed multiple variables — none is clean evidence. Hence experiment 0 and
  §9.
