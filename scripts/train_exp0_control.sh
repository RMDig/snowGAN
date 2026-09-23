#!/usr/bin/env bash
#
# Long-horizon baseline — docs/plans/training_dynamics_recovery_plan.md §4/§5.
#
# This runs the SAME recipe that passed experiment 0, at 1024px, for 100,000
# steps. Its job is to become the REFERENCE CURVE every increment in the queue
# is measured against. We do not have one of those yet.
#
# Experiment 0 (2026-09-23, 512px, 10,000 steps) already answered the code-vs-
# recipe question: ||dD/dx|| 1.046, |W| 0% of ceiling, latent diversity 0.609
# (released model 0.490), crystal-mesh samples. The current code trains this
# recipe from scratch. What is still unknown is whether it KEEPS improving over
# a long horizon, and whether anything degrades there -- which is exactly what
# the 8-image A/B at 64px could not see.
#
# THE RECIPE: disc 2 : gen 3, lambda_gp 10, no SN, no clipping, no LR decay, no
# EMA, no ADA, no multiscale. Reconstructed from the batch_*/ config snapshots
# for the era that actually trained the released backbone -- NOT from the
# released config, which describes disc_steps 47 / SN on / clip 1.0, a state
# reached ~130k steps AFTER that model acquired its structure and entirely at
# lr 1e-7.
#
# DO NOT fold increments into this run (LR anneal, the DiffAugment generator-step
# fix, fresh reals per critic step). Each is queued as its own campaign; stacking
# one here produces a result that cannot be attributed, which is the trap
# CLAUDE.md §9 was written after.
#
# Cost: ~0.6 s/step measured on the RTX 5080 with the local 1024 mirror
# => ~17-22 h for 100,000 steps. Expect ~1 restart from the CPU-RAM leak
# (~434 KiB/batch; a 24 GB ceiling lands near step 51,000).
#
# Usage:
#   scripts/train_exp0_control.sh                   # defaults below
#   MAX_STEPS=500 scripts/train_exp0_control.sh     # short smoke
#   SEED=43 SAVE_DIR=keras/snowgan/control_1024_s43/ scripts/train_exp0_control.sh
#
# Gate it from another shell (check_gates needs no TF and no GPU):
#   python scripts/check_gates.py keras/snowgan/control_1024/
#   python scripts/kill_check.py  keras/snowgan/control_1024/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v snowgan >/dev/null 2>&1; then
  echo "ERROR: 'snowgan' not on PATH. Activate the venv first." >&2
  exit 1
fi

# --- CUDA toolchain ----------------------------------------------------------
# This host has two ptxas: /usr/bin/ptxas is 12.0.140 and wins on the default
# PATH, and /usr/local/cuda-12.8/bin/ptxas is 12.8.93. TF warns loudly about the
# former: "CUDA versions 12.x.y up to and including 12.6.2 miscompile certain
# edge cases around clamping." This architecture clamps on every step (the tanh
# head, clip_by_value in augment, gradient clipping), so a miscompile there
# would be an un-diagnosable correctness bug in a run we intend to interpret.
# Measured cost of using 12.8 instead: none (identical s/step).
#
# The GPU is an RTX 5080 (sm_120a) and TF 2.21 ships no cubins for it, so
# kernels JIT from PTX on first use. The driver caches them in ~/.nv/ComputeCache
# (already ~424 MB warm here); raise the cap so a long run cannot evict and
# re-JIT mid-training. See docs/UPGRADES.md #45.
if [ -x /usr/local/cuda-12.8/bin/ptxas ]; then
  export PATH="/usr/local/cuda-12.8/bin:${PATH}"
fi
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"

ptxas_version="$(ptxas --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
echo "=== ptxas ${ptxas_version:-unknown} | $(command -v ptxas) ==="
case "${ptxas_version}" in
  12.0|12.1|12.2|12.3|12.4|12.5|12.6)
    echo "WARNING: ptxas ${ptxas_version} has the clamping miscompile. Install CUDA >= 12.6.3" >&2
    echo "         or fix PATH before trusting this run's results." >&2 ;;
esac

# RSS ceiling for the native-leak workaround. Set a few GB under physical RAM.
# NOTE: this is a no-op off Linux — _process_rss_mb reads /proc/self/status
# only, so on macOS the trainer never exits with code 75 and the wrapper cannot
# pre-empt the OOM-kill.
MAX_RSS_MB="${MAX_RSS_MB:-24000}"
SAVE_DIR="${SAVE_DIR:-keras/snowgan/control_1024/}"

# Local image mirror. WITHOUT THIS THE RUN IS ~19x SLOWER AND NOT GPU-BOUND.
# The HF `image` column is URL-backed, so every image access is an HTTP GET of a
# ~16 MB PNG — 2.03 s each, once per image per epoch. Measured end to end:
# 16.8 s/step remote vs 0.87 s/step local, against 0.54 s of actual compute.
IMAGE_ROOT="${IMAGE_ROOT:-$HOME/rmdig-cache-1024}"
if [ ! -d "${IMAGE_ROOT}" ]; then
  echo "WARNING: image_root ${IMAGE_ROOT} not found; falling back to the" >&2
  echo "         URL-backed column, which is ~19x slower." >&2
fi

# 200,000 critic updates at disc_steps 2 = 100,000 train steps: the long-horizon
# baseline. A hard cap is required because a single-modality run otherwise never
# terminates: train_ind resets to 0 on manifest exhaustion and the next fetch
# succeeds, so `trainable_data` never goes false and the epoch budget is inert.
MAX_STEPS="${MAX_STEPS:-100000}"

exec scripts/train_with_restarts.sh \
  --mode train --modality magnified_profile \
  --save_dir "${SAVE_DIR}" \
  --max_rss_mb "${MAX_RSS_MB}" \
  --max_steps "${MAX_STEPS}" \
  --image_root "${IMAGE_ROOT}" \
  --resolution "1024 1024" --batch_size 4 --latent_dim 100 --seed "${SEED:-42}" \
  --gen_filters "1024 512 256 128 64" --gen_kernel "3 3" --gen_stride "2 2" \
  --gen_norm none --gen_upsampler transpose --gen_convs_per_resolution 1 \
  --gen_lr 0.0001 --gen_steps 3 \
  --disc_filters "64 128 256 512 1024" --disc_kernel "3 3" --disc_stride "2 2" \
  --disc_lr 0.00001 --disc_steps 2 --disc_lambda_gp 10.0 \
  --no-spectral_norm \
  --no-adaptive_steps \
  --no-multiscale_disc \
  --no-augment \
  --ada_target 0 --grad_clip_norm 0 --ema_decay 0 \
  --lr_decay none --fid_interval 0 \
  --grad_probe_interval 50 \
  --cleanup_milestone 1000 --sample_batch_interval 50 \
  --epochs 1000 \
  "$@"

# Why each "off" is off, so nobody re-adds one as a tidy-up:
#
#   --no-spectral_norm   The hypothesis. SN + the lambda_gp clamp is the
#                        candidate regression.
#   --no-adaptive_steps  Not reproducible: it persists its adjustment into
#                        training_steps, which is re-read as the next launch's
#                        base, so the ratio ratchets across restarts (2 -> 61
#                        in the released run). A control cannot use it.
#   --no-multiscale_disc Its low-res head trains with NO gradient penalty and
#                        never reaches the generator, but 0.5*its loss is
#                        folded into the logged disc_loss.
#   --no-augment         DiffAugment is currently applied in the critic update
#                        but not the generator update, so the two optimize
#                        against different distributions. Increment 3 fixes it.
#   --ada_target 0       ADA steers on mean(D(real) > 0), but a WGAN critic has
#                        no calibrated zero — SpectralNormalization normalizes
#                        the kernel and leaves the bias free.
#   --grad_clip_norm 0   The structure era had no clipping. Under Adam it is a
#                        stability measure, not a step-size bound.
#   --ema_decay 0        The EMA shadow is ~13% random init at step 2,000, so
#                        early gates read on it fail healthy runs.
#   --lr_decay none      The structure era had no schedule. Testing a real
#                        anneal is increment 4, not part of the control.
