#!/usr/bin/env bash
#
# Checkpoint-and-restart wrapper for snowGAN training.
#
# There is a native CPU-RAM leak (~+3.6 MiB/batch, glibc live arena) that the
# leak hunt could not root-cause -- it isn't any single train_step op and does
# not reproduce in isolation; it OOM-kills long runs (observed: 31 GB RSS, killed
# at batch ~15k). Until it's fixed, this wrapper sidesteps it: pass --max_rss_mb
# so the trainer saves a fresh checkpoint and exits with code 75 *before* the OS
# OOM-kills it, and this loop relaunches a clean process that resumes from the
# rolling top-level checkpoint. Atomic saves (UPGRADES #8) make the kill/resume
# safe -- no truncated checkpoints.
#
# Usage (forward the normal training args; include --max_rss_mb):
#
#   scripts/train_with_restarts.sh \
#     --mode train --save_dir keras/snowgan/core/ --modality core \
#     --max_rss_mb 24000 \
#     --epochs 1000 --batch_size 4 ... (rest of your training flags)
#
# Pick --max_rss_mb a few GB under your physical RAM (e.g. 24000 on a 32 GB host)
# so the trainer exits cleanly with headroom before the kernel intervenes.

set -u

RESTART_CODE=75

case " $* " in
  *" --max_rss_mb "*) : ;;
  *)
    echo "WARNING: --max_rss_mb not in args. Without it the trainer never exits" >&2
    echo "         with code ${RESTART_CODE}, so this wrapper can't pre-empt the OOM-kill." >&2
    ;;
esac

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== [restart-wrapper] launch #${attempt} $(date) ==="
  snowgan "$@"
  code=$?
  if [ "${code}" -eq "${RESTART_CODE}" ]; then
    echo "=== [restart-wrapper] hit RSS ceiling (exit ${code}); relaunching to resume ==="
    continue
  fi
  echo "=== [restart-wrapper] training exited with code ${code}; stopping ==="
  exit "${code}"
done
