#!/bin/bash

cd "$(dirname "$0")/.."
mkdir -p results/hyperparameter_search logs

for algo in ddpg td3; do
    for seed in 0 1 2; do
        for hidden in 64 256; do
            for tau in 0.05 0.005 0.001; do
                tag="tau${tau}-h${hidden}"
                log="logs/${algo}-s${seed}-${tag}.log"
                echo "Launching $algo seed=$seed hidden=$hidden tau=$tau"
                OMP_NUM_THREADS=4 uv run python scripts/train.py \
                    --algo $algo --seed $seed \
                    --hidden $hidden --tau $tau \
                    --max-steps 350000 --tag "$tag" > "$log" 2>&1
            done
        done
    done
done

echo "All runs done."

mv results/raw/*-tau*.json results/hyperparameter_search/ 2>/dev/null || true