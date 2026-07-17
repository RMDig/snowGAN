#!/usr/bin/env bash
#
# Launch the magnified_profile v0.2 training run (transpose-upsampler config)
# through the checkpoint-and-restart wrapper.
#
# Exists so this long invocation is rerun-safe and paste-safe: pasting it into
# a terminal once glued flags together with non-breaking spaces (U+00A0) and
# argparse rejected them as one giant token. Run the file instead:
#
#   source .venv/bin/activate
#   scripts/train_magnified_profiles_v0.2.sh
#
# Extra flags are forwarded, e.g.:
#
#   scripts/train_magnified_profiles_v0.2.sh --epochs 1000

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v snowgan >/dev/null 2>&1; then
  echo "ERROR: 'snowgan' not on PATH. Activate the venv first: source .venv/bin/activate" >&2
  exit 1
fi

# RSS ceiling for the leak workaround (see train_with_restarts.sh header).
# Set a few GB under the host's physical RAM; override via env if needed.
# Default sized for big-mac-mini (M4 Pro, 48 GB unified memory).
MAX_RSS_MB="${MAX_RSS_MB:-40000}"

exec scripts/train_with_restarts.sh \
  --mode train --modality magnified_profile \
  --save_dir keras/snowgan/magnified_profiles_v0.2/ \
  --max_rss_mb "${MAX_RSS_MB}" \
  --batch_size 4 \
  --gen_lr 0.0001 --gen_steps 1 \
  --gen_filters "1024 512 256 128 64" --gen_kernel "4 4" --gen_stride "2 2" \
  --gen_norm pixel --gen_upsampler transpose \
  --disc_lr 0.00001 --disc_steps 5 \
  --disc_filters "64 128 256 512 1024" --disc_kernel "3 3" --disc_stride "2 2" \
  --disc_lambda_gp 1.0 \
  --spectral_norm --augment --adaptive_steps --ada_target 0.6 \
  --lr_decay cosine --lr_min 0.00001 --lr_decay_steps 400000 \
  --ema_decay 0.999 --fid_interval 5000 --grad_clip_norm 1.0 \
  --cleanup_milestone 1000 --sample_epoch_interval 1 --sample_batch_interval 50 \
  "$@"
