#!/usr/bin/env bash
#
# Capacity campaign launcher (docs/plans/capacity_plan.md §4).
#
# The adopted recipe plus the campaign base flags, so every run (E0 and each
# arm) differs only by the single change passed in "$@":
#   --ema_decay 0.999          adopted after Arm D (replicated at two seeds)
#   --kid_antialias            campaign primary KID (capacity_plan F6)
#   --cleanup_milestone 0      keep every snapshot: the 50-59k confirmation
#                              window must survive for three-seed scoring
#   MAX_STEPS 60000            EMA plateaus by 50-60k (horizon run); the
#                              control script would otherwise default to 100k
#
# Usage:
#   SAVE_DIR=keras/snowgan/cap_e0/ scripts/train_capacity_arm.sh
#   SAVE_DIR=keras/snowgan/cap_c1/ scripts/train_capacity_arm.sh --augment
#   SEED=43 SAVE_DIR=keras/snowgan/cap_e0_s43/ scripts/train_capacity_arm.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SAVE_DIR:?set SAVE_DIR to a fresh run directory}"
export SAVE_DIR
export MAX_STEPS="${MAX_STEPS:-60000}"
export SEED="${SEED:-42}"
exec "${REPO_ROOT}/scripts/train_exp0_control.sh" \
  --ema_decay 0.999 --kid_antialias --cleanup_milestone 0 \
  "$@"
