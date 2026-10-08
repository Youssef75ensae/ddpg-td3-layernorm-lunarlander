import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import src.agents as agents
from src.agents import run_ddpg, run_td3
from src.config import DDPGConfig, TD3Config

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ddpg", "td3"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--layernorm", action="store_true")
    p.add_argument("--max-steps", type=int, default=350_000)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "raw")
    p.add_argument("--overwrite", action="store_true",
                   help="Rerun even if the result file already exists")
    return p.parse_args()


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = parse_args()

    overrides = {}
    if args.tau is not None:
        overrides["tau"] = args.tau
    if args.hidden is not None:
        overrides["actor_hidden"] = (args.hidden, args.hidden)
        overrides["critic_hidden"] = (args.hidden, args.hidden)

    if args.algo == "ddpg":
        base_cfg = DDPGConfig(
            seed=args.seed,
            use_layernorm=args.layernorm,
            max_steps=args.max_steps,
        )
    else:
        base_cfg = TD3Config(
            seed=args.seed,
            use_layernorm=args.layernorm,
            max_steps=args.max_steps,
        )
    cfg = replace(base_cfg, **overrides)

    ln_tag = "ln" if args.layernorm else "noln"
    base_name = f"{args.algo}-{ln_tag}-s{args.seed}"
    run_name = f"{base_name}-{args.tag}" if args.tag else base_name

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.json"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} already exists, skipping (use --overwrite to rerun)")
        return

    print(f"Running {run_name}...")
    cfg_dict = asdict(cfg)

    start = time.time()
    if args.algo == "ddpg":
        evaluator = run_ddpg(cfg, run_name)
    else:
        evaluator = run_td3(cfg, run_name)
    duration = time.time() - start

    results = {
        "config": cfg_dict,
        "history": [
            {"step": r.step, "mean": r.mean, "rewards": r.rewards.tolist()}
            for r in evaluator.history
        ],
        "best_reward": evaluator.best_reward,
        "bias_history": getattr(evaluator, "bias_history", []),
        "meta": {
            "algo": args.algo,
            "run_name": run_name,
            "bias_interval": agents.BIAS_INTERVAL,
            "duration_s": round(duration, 1),
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "gymnasium": package_version("gymnasium"),
            "rl_mind": package_version("rl-mind"),
            "device": str(agents.device),
        },
    }

    # Atomic write
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, out_path)

    print(f"Saved to {out_path}")
    print(f"Best reward: {evaluator.best_reward:.1f} ({duration / 60:.1f} min)")


if __name__ == "__main__":
    main()