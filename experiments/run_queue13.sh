#!/usr/bin/env bash
# Thirteenth overnight queue on the rented Linux GPU instance (Ubuntu, /venv/main).
#
# Steps, in order. Every step appends its rows to one CSV under
# runs_nseed_queue13/, so a partial run survives and the finished rows are
# skipped. There is no self-stop here: the gated stop is issued by hand from the
# laptop once results_queue13.zip has been verified there.
#
#   1  B1  the task-family matrix: the source tasks add, sub and mult against
#           the target tasks affine_add, affine_sub and affine_mult, seeds 5-9.
#   2  B2  the hyperparameter robustness: one change at a time to the weight
#           decay, the learning rate or the coverage, seeds 5-8.
#   3  B3  the data efficiency of the Pre-LN sources: 5%, 10%, 20% and 40% of
#           the target pairs.
#   4  pack the queue13 CSVs and the report into results_queue13.zip.
#
# Flags:
#   --smoke            run every step with a tiny size so that the wiring can
#                      be tested
#
# Environment:
#   PROCS   worker processes (default 4)
#   PYTHON  interpreter to use (default: /venv/main/bin/python, else python3)
#
# Usage:  bash nseed_backbone/run_queue13.sh
#         bash nseed_backbone/run_queue13.sh --smoke
set -u

SMOKE=0
for argument in "$@"; do
    case "${argument}" in
        --smoke) SMOKE=1 ;;
        --no-stop|--stop-dry-run) ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown flag: ${argument}" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

PROCS="${PROCS:-4}"
if [ -z "${PYTHON:-}" ]; then
    if [ -x /venv/main/bin/python ]; then
        PYTHON=/venv/main/bin/python
    else
        PYTHON=python3
    fi
fi
CAP_MINUTES="${CAP_MINUTES:-210}"
DEADLINE=$(( $(date +%s) + CAP_MINUTES * 60 ))

export NSEED_DETERMINISTIC=1
export NSEED_DETERMINISTIC_GPU=1
export PYTHONUNBUFFERED=1
export PYTHON

LOG="runs_nseed/batch.log"
mkdir -p runs_nseed

if [ "${SMOKE}" = "1" ]; then
    OUT_DIR="runs_nseed/_queue13_smoke"
    B1_SEEDS="5"
    B2_SEEDS="5"
    B3_SEEDS="5"
    CAP=60
    STEPS=20
    EVAL_EVERY=10
    B3_STEPS=20
    B3_EVAL=10
    B3_FRACTIONS="0.05 0.40"
else
    OUT_DIR="runs_nseed_queue13"
    B1_SEEDS="5 6 7 8 9"
    B2_SEEDS="5 6 7 8"
    CAP=50000
    STEPS=2000
    EVAL_EVERY=100
    B3_STEPS=5000
    B3_EVAL=250
    B3_FRACTIONS="0.05 0.10 0.20 0.40"
fi

echo "Queue 13 - project root ${ROOT}"
echo "  python   : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "  procs    : ${PROCS}"
echo "  smoke    : ${SMOKE}"
echo "  out      : ${OUT_DIR}"
echo "  log      : ${LOG}"
echo

TOTAL_STEPS=4
step () {
    local number="$1"; shift
    local description="$1"; shift
    "${PYTHON}" - "${number}" "${description}" "start" "${TOTAL_STEPS}" <<'PY'
import sys
sys.path.insert(0, 'nseed_backbone')
import common
common.set_batch_log('runs_nseed/batch.log')
common.prompt_step(sys.argv[1], int(sys.argv[4]), sys.argv[2], phase=sys.argv[3])
PY
    if ! "$@"; then
        "${PYTHON}" - "${number}" "${description}" "end" "${TOTAL_STEPS}" <<'PY'
import sys
sys.path.insert(0, 'nseed_backbone')
import common
common.set_batch_log('runs_nseed/batch.log')
common.prompt_step(sys.argv[1], int(sys.argv[4]), sys.argv[2] + ' [FAILED]', phase='end')
PY
        echo "Queue 13: step ${number} failed. Re-run this script to continue;" \
             "the finished rows are skipped." >&2
        exit 1
    fi
    "${PYTHON}" - "${number}" "${description}" "end" "${TOTAL_STEPS}" <<'PY'
import sys
sys.path.insert(0, 'nseed_backbone')
import common
common.set_batch_log('runs_nseed/batch.log')
common.prompt_step(sys.argv[1], int(sys.argv[4]), sys.argv[2], phase='end')
PY
}

b1_step () {
    local source_task="$1"; shift
    local split_env=""
    if [ "${source_task}" = "sub" ]; then split_env="NSEED_SPLIT_TASK=sub"; fi
    env ${split_env} "${PYTHON}" nseed_backbone/queue13_steps.py b1 \
        --source-tasks "${source_task}" --seeds ${B1_SEEDS} --out "${OUT_DIR}" \
        --cap "${CAP}" --steps "${STEPS}" --eval-every "${EVAL_EVERY}" \
        --procs "${PROCS}" --device cuda
}

b2_step () {
    local setting="$1"; shift
    local value="$1"; shift
    local wd="" lr="" coverage=""
    case "${setting}" in
        wd) wd="${value}";;
        lr) lr="${value}";;
        coverage) coverage="${value}";;
    esac
    env NSEED_WD="${wd:-1.0}" NSEED_LR="${lr:-0.001}" NSEED_COVERAGE="${coverage:-0.4}" \
        "${PYTHON}" nseed_backbone/queue13_steps.py b2 --setting "${setting}" \
        --value "${value}" --seeds ${B2_SEEDS} --out "${OUT_DIR}" --cap "${CAP}" \
        --steps "${STEPS}" --eval-every "${EVAL_EVERY}" --procs "${PROCS}" --device cuda
}

b3_step () {
    local runs_root="$1"; shift
    local seeds="$1"; shift
    NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" nseed_backbone/queue13_steps.py b3 \
        --runs-root "${runs_root}" --seeds ${seeds} --out "${OUT_DIR}" \
        --fractions ${B3_FRACTIONS} --steps "${B3_STEPS}" --eval-every "${B3_EVAL}" \
        --procs "${PROCS}" --device cuda
}

pack_step () {
    "${PYTHON}" nseed_backbone/queue13_steps.py pack --out results_queue13.zip \
        --root "${ROOT}"
}

skip_step () {
    local number="$1"; shift
    local description="$1"; shift
    "${PYTHON}" - "${number}" "${description}" "end" "${TOTAL_STEPS}" <<'PY'
import sys
sys.path.insert(0, 'nseed_backbone')
import common
common.set_batch_log('runs_nseed/batch.log')
common.prompt_step(sys.argv[1], int(sys.argv[4]), sys.argv[2] + ' [skipped]', phase='end')
PY
    echo "Queue 13: step ${number} skipped, only $(remaining_minutes) min of the cap are left."
}

export -f b1_step b2_step b3_step pack_step step 2>/dev/null || true
export B1_SEEDS B2_SEEDS B3_SEEDS CAP STEPS EVAL_EVERY B3_STEPS B3_EVAL B3_FRACTIONS OUT_DIR PROCS PYTHON

remaining_minutes () {
    echo $(( (DEADLINE - $(date +%s)) / 60 ))
}

if [ "$(remaining_minutes)" -gt 80 ]; then
    step 1 "B1 the task-family matrix: add, sub, mult against affine_add, affine_sub, affine_mult" \
        bash -c 'b1_step add && b1_step mult && b1_step sub'
else
    skip_step 1 "B1 the task-family matrix: add, sub, mult against affine_add, affine_sub, affine_mult"
fi

if [ "$(remaining_minutes)" -gt 70 ]; then
    step 2 "B2 the hyperparameter robustness: weight decay, learning rate and coverage" \
        bash -c 'b2_step wd 0.5 && b2_step wd 2.0 && b2_step lr 5e-4 && b2_step lr 2e-3 && b2_step coverage 0.3 && b2_step coverage 0.5'
else
    skip_step 2 "B2 the hyperparameter robustness: weight decay, learning rate and coverage"
fi

if [ "$(remaining_minutes)" -gt 55 ]; then
    step 3 "B3 the data efficiency of the Pre-LN sources" \
        bash -c 'b3_step runs_nseed_p1 5 6 7 8 && b3_step runs_nseed_k2 10 11 12 13 14'
else
    skip_step 3 "B3 the data efficiency of the Pre-LN sources"
fi

step 4 "Pack the queue13 CSVs and the report into results_queue13.zip" \
    pack_step

echo
echo "Queue 13 finished. The archive is results_queue13.zip. The instance is left running."
