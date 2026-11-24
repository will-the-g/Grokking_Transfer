#!/usr/bin/env bash
# Tenth overnight queue on the rented Linux GPU instance (Ubuntu, /venv/main).
#
# Steps, in order. Every step writes into its own new folder, so nothing
# existing is overwritten. There is no self-stop: the instance is left running.
#
#   1  K1 scratch Pre-LN   seeds 5-9: a random-init model trained on the
#                           affine task with the Pre-LN block and batch 512, the
#                           dev mean at 2,000 and 10,000 steps.
#   2  K2 p=59 Pre-LN      seeds 10-14: the source trained to grok (40,000-step
#                           cap) and the memorized, transition, grokked and
#                           norm_control transfers.
#   3  K3 Jia probe        the Jia setting with Pre-LN, batch 512 and the random
#                           split, seeds 5-7, up to 40,000 steps.
#   4  K4 Jia stages       the same configuration, seeds 5-13, the stage
#                           transfers, only when at least two of the three probe
#                           seeds grokked.
#   5  K5 Jia cold         the cold-recipient factorial with the Jia grokked
#                           source as the donor, only when K4 ran.
#   6  pack the new folders into results_queue10.zip and the report into
#           REPORT_job10.md.
#
# Flags:
#   --smoke            run every step with a tiny size so that the wiring can
#                      be tested
#
# Environment:
#   PROCS   worker processes (default 4)
#   PYTHON  interpreter to use (default: /venv/main/bin/python, else python3)
#   CAP_MINUTES  wall-clock cap for the queue (default 300 = 5 hours)
#
# Usage:  bash nseed_backbone/run_queue10.sh
#         bash nseed_backbone/run_queue10.sh --smoke
set -u

SMOKE=0
for argument in "$@"; do
    case "${argument}" in
        --smoke) SMOKE=1 ;;
        --no-stop|--stop-dry-run) ;;
        -h|--help) sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
CAP_MINUTES="${CAP_MINUTES:-300}"
DEADLINE=$(( $(date +%s) + CAP_MINUTES * 60 ))

export NSEED_DETERMINISTIC=1
export NSEED_DETERMINISTIC_GPU=1
export PYTHONUNBUFFERED=1
export PYTHON

LOG="runs_nseed/batch.log"
mkdir -p runs_nseed

if [ "${SMOKE}" = "1" ]; then
    ROOT_DIR="runs_nseed/_queue10_smoke"
    K1_SEEDS="5"
    K2_SEEDS="10"
    K3_SEEDS="5"
    K4_SEEDS="5"
    STEPS=200
    SCRATCH_STEPS=40
    EVAL_AT="20 40"
    SOURCE_CAP=60
    EVAL_EVERY=20
    CKPT_EVERY=30
    OUT_DIR="${ROOT_DIR}/out"
else
    ROOT_DIR=""
    K1_SEEDS="5 6 7 8 9"
    K2_SEEDS="10 11 12 13 14"
    K3_SEEDS="5 6 7"
    K4_SEEDS="5 6 7 8 9 10 11 12 13"
    STEPS=2000
    SCRATCH_STEPS=10000
    EVAL_AT="2000 10000"
    SOURCE_CAP=40000
    EVAL_EVERY=250
    CKPT_EVERY=500
    OUT_DIR="results_queue10.zip"
fi

K1_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_k1"
K2_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_k2"
K3_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_k3"
K4_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_k4"
K5_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_k5"

echo "Queue 10 - project root ${ROOT}"
echo "  python   : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "  procs    : ${PROCS}"
echo "  smoke    : ${SMOKE}"
echo "  cap      : ${CAP_MINUTES} min"
echo "  folders  : ${K1_DIR}, ${K2_DIR}, ${K3_DIR}, ${K4_DIR}, ${K5_DIR}"
echo "  log      : ${LOG}"
echo

TOTAL_STEPS=6
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
        echo "Queue 10: step ${number} failed. Re-run this script to continue;" \
             "the finished jobs are skipped." >&2
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

remaining_minutes () {
    echo $(( (DEADLINE - $(date +%s)) / 60 ))
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
    echo "Queue 10: step ${number} skipped, only $(remaining_minutes) min of the cap are left."
}

k1_step () {
    NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" nseed_backbone/queue10_steps.py scratch \
        --seeds ${K1_SEEDS} --out "${K1_DIR}" --steps "${SCRATCH_STEPS}" \
        --eval-at ${EVAL_AT} --device cuda
}

k2_step () {
    local seed
    for seed in ${K2_SEEDS}; do
        NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" nseed_backbone/run_seed.py \
            --seed "${seed}" --out "${K2_DIR}/seed_${seed}" --tiers A --resume \
            --source-cap "${SOURCE_CAP}" --eval-every "${EVAL_EVERY}" \
            --ckpt-every "${CKPT_EVERY}" --transfer-steps "${STEPS}"
    done
    NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" nseed_backbone/queue10_steps.py stage \
        --name K2 --runs-root "${K2_DIR}" --seeds ${K2_SEEDS} --out "${K2_DIR}" \
        --csv k2_preln.csv
}

k3_step () {
    local seed
    for seed in ${K3_SEEDS}; do
        NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 NSEED_SPLIT=random "${PYTHON}" \
            nseed_backbone/run_seed.py --seed "${seed}" --out "${K3_DIR}/seed_${seed}" \
            --tiers A --resume --source-cap "${SOURCE_CAP}" \
            --eval-every "${EVAL_EVERY}" --ckpt-every "${CKPT_EVERY}" \
            --transfer-steps "${STEPS}"
    done
    NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 NSEED_SPLIT=random "${PYTHON}" \
        nseed_backbone/queue10_steps.py probe --runs-root "${K3_DIR}" \
        --seeds ${K3_SEEDS} --out "${K3_DIR}" --csv k3_probe.csv
}

k4_step () {
    local seed
    for seed in ${K4_SEEDS}; do
        NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 NSEED_SPLIT=random "${PYTHON}" \
            nseed_backbone/run_seed.py --seed "${seed}" --out "${K4_DIR}/seed_${seed}" \
            --tiers A --resume --source-cap "${SOURCE_CAP}" \
            --eval-every "${EVAL_EVERY}" --ckpt-every "${CKPT_EVERY}" \
            --transfer-steps "${STEPS}"
    done
    NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 NSEED_SPLIT=random "${PYTHON}" \
        nseed_backbone/queue10_steps.py stage --name K4 --runs-root "${K4_DIR}" \
        --seeds ${K4_SEEDS} --out "${K4_DIR}" --csv k4_stage.csv
}

k5_step () {
    NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 NSEED_SPLIT=random \
        NSEED_SOURCE_ROOT="${K4_DIR}" "${PYTHON}" nseed_backbone/run_cold_factorial.py \
        --seeds ${K4_SEEDS} --out "${K5_DIR}" --steps "${STEPS}" --procs "${PROCS}" \
        --device cuda
    NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 NSEED_SPLIT=random "${PYTHON}" \
        nseed_backbone/queue10_steps.py cold --source-csv "${K5_DIR}/embed_ablation.csv" \
        --out "${K5_DIR}"
}

pack_step () {
    "${PYTHON}" nseed_backbone/queue10_steps.py pack --out "${OUT_DIR}" --root "${ROOT}"
}

probe_grokked () {
    "${PYTHON}" - "${K3_DIR}/k3_probe.csv" <<'PY'
import csv
import sys
path = sys.argv[1]
try:
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
except OSError:
    print(0)
    raise SystemExit(0)
print(sum(1 for row in rows if row.get('grokked') == 'yes'))
PY
}

if [ "$(remaining_minutes)" -gt 30 ]; then
    step 1 "K1 scratch Pre-LN: a random-init model on the affine task, seeds 5-9" \
        k1_step
else
    skip_step 1 "K1 scratch Pre-LN: a random-init model on the affine task, seeds 5-9"
fi

if [ "$(remaining_minutes)" -gt 60 ]; then
    step 2 "K2 p=59 Pre-LN sources and stage transfers, seeds 10-14" \
        k2_step
else
    skip_step 2 "K2 p=59 Pre-LN sources and stage transfers, seeds 10-14"
fi

if [ "$(remaining_minutes)" -gt 60 ]; then
    step 3 "K3 Jia probe with the random split, seeds 5-7" \
        k3_step
else
    skip_step 3 "K3 Jia probe with the random split, seeds 5-7"
fi

if [ "$(probe_grokked)" -ge 2 ] && [ "$(remaining_minutes)" -gt 90 ]; then
    step 4 "K4 Jia Pre-LN random-split sources and stage transfers, seeds 5-13" \
        k4_step
else
    skip_step 4 "K4 Jia Pre-LN random-split sources and stage transfers, seeds 5-13"
fi

if [ -f "${K4_DIR}/k4_stage.csv" ] && [ "$(remaining_minutes)" -gt 40 ]; then
    step 5 "K5 the cold-recipient factorial with the Jia grokked source as the donor" \
        k5_step
else
    skip_step 5 "K5 the cold-recipient factorial with the Jia grokked source as the donor"
fi

step 6 "Pack the new folders into ${OUT_DIR}" \
    pack_step

echo
echo "Queue 10 finished. The archive is ${OUT_DIR}. The instance is left running."
