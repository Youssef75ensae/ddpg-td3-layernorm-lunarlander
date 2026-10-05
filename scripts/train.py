import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.agents import run_ddpg, run_td3
from src.config import DDPGConfig, TD3Config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ddpg", "td3"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--layernorm", action="store_true")
    p.add_argument("--max-steps", type=int, default=300_000)
    return p.parse_args()


def main():
    args = parse_args()

    if args.algo == "ddpg":
        cfg = DDPGConfig(
            seed=args.seed,
            use_layernorm=args.layernorm,
            max_steps=args.max_steps,
        )
    else:
        cfg = TD3Config(
            seed=args.seed,
            use_layernorm=args.layernorm,
            max_steps=args.max_steps,
        )

    ln_tag = "ln" if args.layernorm else "noln"
    run_name = f"{args.algo}-{ln_tag}-s{args.seed}"

    print(f"Running {run_name}...")
    cfg_dict = asdict(cfg)

    if args.algo == "ddpg":
        evaluator = run_ddpg(cfg, run_name)
    else:
        evaluator = run_td3(cfg, run_name)

    results = {
        "config": cfg_dict,
        "history": [
            {"step": r.step, "mean": r.mean, "rewards": r.rewards.tolist()}
            for r in evaluator.history
        ],
        "best_reward": evaluator.best_reward,
    }

    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {out_path}")
    print(f"Best reward: {evaluator.best_reward:.1f}")


if __name__ == "__main__":
    main()