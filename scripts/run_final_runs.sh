#!/bin/bash
# Final runs: {DDPG, TD3} x {no LN, LN} x seeds 15-34, in parallel. Resumable.
# Usage: N_JOBS=4 THREADS=1 caffeinate -is ./scripts/run_final_runs.sh

set -u
cd "$(dirname "$0")/.."
mkdir -p logs results/raw

N_JOBS=${N_JOBS:-4}
THREADS=${THREADS:-2}
SEEDS=${SEEDS:-$(seq 15 34)}
MAX_STEPS=${MAX_STEPS:-350000}

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "No virtual environment found: run 'uv sync' first."
    exit 1
fi

jobs_file=$(mktemp)
for seed in $SEEDS; do
    for algo in ddpg td3; do
        echo "--algo $algo --seed $seed --max-steps $MAX_STEPS" >> "$jobs_file"
        echo "--algo $algo --seed $seed --max-steps $MAX_STEPS --layernorm" >> "$jobs_file"
    done
done

echo "$(date '+%H:%M') - $(wc -l < "$jobs_file" | tr -d ' ') runs, $N_JOBS in parallel, $THREADS threads each"

export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export TQDM_MININTERVAL=60
export PYTHONPATH=.

xargs -P "$N_JOBS" -L 1 sh -c '
    log="logs/$(echo "$*" | sed -e "s/--max-steps [0-9]*//" -e "s/--//g" -e "s/  */_/g" -e "s/_$//").log"
    if "$0" scripts/train.py "$@" >> "$log" 2>&1; then
        echo "$(date +%H:%M) done:   $*"
    else
        echo "$(date +%H:%M) FAILED: $*  (see $log)"
    fi
' "$PY" < "$jobs_file"

rm -f "$jobs_file"
echo "$(date '+%H:%M') - finished, $(ls results/raw/*.json 2>/dev/null | wc -l | tr -d ' ') result files in results/raw"