#!/bin/bash
# Queue 21, step P1: the energy-matched non-key deletions.
set -u
cd "${ROOT}" || exit 1
mkdir -p runs_nseed_queue21
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export NSEED_DETERMINISTIC=1
export NSEED_DETERMINISTIC_GPU=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
PY=/root/venv/bin/python
echo "START $(date -u +%FT%TZ)"
echo "free disk: $(df -BG / | tail -1 | awk '{print $4}')"
$PY nseed_backbone/p21_energy.py --procs 4 \
    --out runs_nseed_queue21/m21_energy.csv \
    --check runs_nseed_queue21/p21_source_check.csv
echo "END $(date -u +%FT%TZ) rc=$?"
echo "free disk: $(df -BG / | tail -1 | awk '{print $4}')"
