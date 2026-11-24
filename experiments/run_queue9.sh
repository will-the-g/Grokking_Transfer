#!/usr/bin/env bash
# Ninth overnight queue on the rented Linux GPU instance (Ubuntu, /venv/main).
#
# Steps, in order. Every step writes into its own new folder, so nothing
# existing is overwritten. There is no self-stop: the instance is left running.
#
#   1  P0 forced recompute  three p=97 grokked-source affine transfers
#                           (seeds 5, 6, 7) from the stored checkpoints into a
#                           fresh output directory.
#   2  J0 grok probe        the Jia setting with Pre-LN and batch 512, seeds
#                           5-7, up to 40,000 steps.
#   3  P1 our p=59 setting with Pre-LN and batch 512, seeds 5-9, trained until
#                           it groks, then the memorized, transition, grokked and
#                           norm_control transfers.
#   4  J1 the same for the Jia setting, seeds 5-13, only when at least two of
#                           the three probe seeds grokked.
#   5  J2 the cold-recipient factorial with the Jia grokked source as the
#                           donor, only when J1 ran.
#   6  pack the new folders into results_queue9.zip and the report into
#           REPORT_job9.md.
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
# Usage:  bash nseed_backbone/run_queue9.sh
#         bash nseed_backbone/run_queue9.sh --smoke
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
    ROOT_DIR="runs_nseed/_queue9_smoke"
    P0_SEEDS="5"
    J0_SEEDS="5"
    P1_SEEDS="5"
    J1_SEEDS="5"
    STEPS=20
    SOURCE_CAP=60
    EVAL_EVERY=20
    CKPT_EVERY=30
    OUT_DIR="${ROOT_DIR}/out"
else
    ROOT_DIR=""
    P0_SEEDS="5 6 7"
    J0_SEEDS="5 6 7"
    P1_SEEDS="5 6 7 8 9"
    J1_SEEDS="5 6 7 8 9 10 11 12 13"
    STEPS=2000
    SOURCE_CAP=40000
    EVAL_EVERY=250
    CKPT_EVERY=500
    OUT_DIR="results_queue9.zip"
fi

P0_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_p0"
J0_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_j0"
P1_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_p1"
J1_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_j1"
J2_DIR="${ROOT_DIR:+${ROOT_DIR}/}runs_nseed_j2"

echo "Queue 9 - project root ${ROOT}"
echo "  python   : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "  procs    : ${PROCS}"
echo "  smoke    : ${SMOKE}"
echo "  cap      : ${CAP_MINUTES} min"
echo "  folders  : ${P0_DIR}, ${J0_DIR}, ${P1_DIR}, ${J1_DIR}, ${J2_DIR}"
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
        echo "Queue 9: step ${number} failed. Re-run this script to continue;" \
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
    echo "Queue 9: step ${number} skipped, only $(remaining_minutes) min of the cap are left."
}

p0_step () {
    NSEED_P=97 "${PYTHON}" nseed_backbone/queue9_steps.py p0 --runs-root runs_nseed_p97 \
        --seeds ${P0_SEEDS} --out "${P0_DIR}" --steps "${STEPS}" --device cuda
}

j0_step () {
    local seed
    for seed in ${J0_SEEDS}; do
        NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" \
            nseed_backbone/run_seed.py --seed "${seed}" --out "${J0_DIR}/seed_${seed}" \
            --tiers A --resume --source-cap "${SOURCE_CAP}" \
            --eval-every "${EVAL_EVERY}" --ckpt-every "${CKPT_EVERY}" \
            --transfer-steps "${STEPS}"
    done
    NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" \
        nseed_backbone/queue9_steps.py probe --runs-root "${J0_DIR}" \
        --seeds ${J0_SEEDS} --out "${J0_DIR}"
}

p1_step () {
    local seed
    for seed in ${P1_SEEDS}; do
        NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" nseed_backbone/run_seed.py \
            --seed "${seed}" --out "${P1_DIR}/seed_${seed}" --tiers A --resume \
            --source-cap "${SOURCE_CAP}" --eval-every "${EVAL_EVERY}" \
            --ckpt-every "${CKPT_EVERY}" --transfer-steps "${STEPS}"
    done
    NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" nseed_backbone/queue9_steps.py stage \
        --name P1 --runs-root "${P1_DIR}" --seeds ${P1_SEEDS} --out "${P1_DIR}" \
        --csv p1_preln.csv
}

j1_step () {
    local seed
    for seed in ${J1_SEEDS}; do
        NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" \
            nseed_backbone/run_seed.py --seed "${seed}" --out "${J1_DIR}/seed_${seed}" \
            --tiers A --resume --source-cap "${SOURCE_CAP}" \
            --eval-every "${EVAL_EVERY}" --ckpt-every "${CKPT_EVERY}" \
            --transfer-steps "${STEPS}"
    done
    NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" \
        nseed_backbone/queue9_steps.py stage --name J1 --runs-root "${J1_DIR}" \
        --seeds ${J1_SEEDS} --out "${J1_DIR}" --csv j1_stage.csv
}

j2_step () {
    NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 NSEED_SOURCE_ROOT="${J1_DIR}" \
        "${PYTHON}" nseed_backbone/run_cold_factorial.py --seeds ${J1_SEEDS} \
        --out "${J2_DIR}" --steps "${STEPS}" --procs "${PROCS}" --device cuda
    NSEED_JIA=1 NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" \
        nseed_backbone/queue9_steps.py cold --source-csv "${J2_DIR}/embed_ablation.csv" \
        --out "${J2_DIR}"
}

pack_step () {
    "${PYTHON}" nseed_backbone/queue9_steps.py pack --out "${OUT_DIR}" --root "${ROOT}"
}

probe_grokked () {
    "${PYTHON}" - "${J0_DIR}/j0_probe.csv" <<'PY'
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

if [ "$(remaining_minutes)" -gt 10 ]; then
    step 1 "P0 forced recompute: three p=97 grokked-source affine transfers" \
        p0_step
else
    skip_step 1 "P0 forced recompute: three p=97 grokked-source affine transfers"
fi

if [ "$(remaining_minutes)" -gt 60 ]; then
    step 2 "J0 grok probe: the Jia setting with Pre-LN and batch 512, seeds 5-7" \
        j0_step
else
    skip_step 2 "J0 grok probe: the Jia setting with Pre-LN and batch 512, seeds 5-7"
fi

if [ "$(remaining_minutes)" -gt 90 ]; then
    step 3 "P1 our p=59 setting with Pre-LN and batch 512, seeds 5-9" \
        p1_step
else
    skip_step 3 "P1 our p=59 setting with Pre-LN and batch 512, seeds 5-9"
fi

if [ "$(probe_grokked)" -ge 2 ] && [ "$(remaining_minutes)" -gt 120 ]; then
    step 4 "J1 the Jia setting with Pre-LN and batch 512, seeds 5-13" \
        j1_step
else
    skip_step 4 "J1 the Jia setting with Pre-LN and batch 512, seeds 5-13"
fi

if [ -f "${J1_DIR}/j1_stage.csv" ] && [ "$(remaining_minutes)" -gt 40 ]; then
    step 5 "J2 the cold-recipient factorial with the Jia grokked source as the donor" \
        j2_step
else
    skip_step 5 "J2 the cold-recipient factorial with the Jia grokked source as the donor"
fi

step 6 "Pack the new folders into ${OUT_DIR}" \
    pack_step

echo
echo "Queue 9 finished. The archive is ${OUT_DIR}. The instance is left running."
