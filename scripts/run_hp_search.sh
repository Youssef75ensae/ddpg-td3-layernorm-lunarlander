#!/bin/bash

cd "$(dirname "$0")/.."
mkdir -p results/hyperparameter_search

# Launch runs in batches of 6 (safe for T4 VRAM)
batch=""
count=0

for algo in ddpg td3; do
    for seed in 0 1 2; do
        for hidden in 64 128 256; do
            for tau in 0.05 0.01 0.005; do
                tag="tau${tau}-h${hidden}"
                echo "Launching $algo seed=$seed hidden=$hidden tau=$tau"
                OMP_NUM_THREADS=2 python scripts/train.py \
                    --algo $algo --seed $seed \
                    --hidden $hidden --tau $tau \
                    --max-steps 350000 --tag "$tag" &

                count=$((count + 1))
                if [ $((count % 6)) -eq 0 ]; then
                    wait
                    echo "--- Batch done ---"
                fi
            done
        done
    done
done

wait
echo "All runs done."

mv results/raw/*-tau*.json results/hyperparameter_search/ 2>/dev/null