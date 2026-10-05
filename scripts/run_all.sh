#!/bin/bash

set -e

SEEDS="0 1 2 3 4"

for seed in $SEEDS; do
    echo "=== DDPG, no LN, seed $seed ==="
    uv run python scripts/train.py --algo ddpg --seed $seed --max-steps 300000

    echo "=== DDPG, LN, seed $seed ==="
    uv run python scripts/train.py --algo ddpg --seed $seed --layernorm --max-steps 300000

    echo "=== TD3, no LN, seed $seed ==="
    uv run python scripts/train.py --algo td3 --seed $seed --max-steps 300000

    echo "=== TD3, LN, seed $seed ==="
    uv run python scripts/train.py --algo td3 --seed $seed --layernorm --max-steps 300000
done

echo "All done."