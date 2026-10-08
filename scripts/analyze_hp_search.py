import json
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

# Load all JSONs
results_dir = Path("results/hyperparameter_search")
data = []
for path in sorted(results_dir.glob("*.json")):
    with open(path) as f:
        d = json.load(f)
    cfg = d["config"]
    final_rewards = [x for h in d["history"][-10:] for x in h["rewards"]]
    data.append({
        "algo": path.name.split("-")[0],
        "tau": cfg["tau"],
        "hidden": cfg["actor_hidden"][0],
        "seed": cfg["seed"],
        "mean_reward": float(np.mean(final_rewards)),
        "std_reward": float(np.std(final_rewards)),
        "best_reward": d["best_reward"],
    })

print(f"Total runs: {len(data)}\n")

# === Summary: mean over seeds for each (algo, tau, hidden) ===
groups = defaultdict(list)
for row in data:
    key = (row["algo"], row["tau"], row["hidden"])
    groups[key].append(row["mean_reward"])

print("=== Summary (mean over 3 seeds) ===")
print(f"{'algo':<6} {'tau':<8} {'hidden':<8} {'mean':<10} {'std':<10} {'n'}")
print("-" * 55)
for (algo, tau, hidden), rewards in sorted(groups.items()):
    m = np.mean(rewards)
    s = np.std(rewards)
    n = len(rewards)
    print(f"{algo:<6} {tau:<8} {hidden:<8} {m:<10.1f} {s:<10.1f} {n}")
print()

# === Kruskal-Wallis on tau (per algorithm) ===
print("=== Kruskal-Wallis on tau ===")
for algo in ["ddpg", "td3"]:
    by_tau = defaultdict(list)
    for row in data:
        if row["algo"] == algo:
            by_tau[row["tau"]].append(row["mean_reward"])
    groups_kw = [by_tau[t] for t in sorted(by_tau.keys())]
    stat, p = stats.kruskal(*groups_kw)
    print(f"  {algo}: H={stat:.2f}, p={p:.4f}")
print()

# === Mann-Whitney on hidden (per algo, per tau) ===
print("=== Mann-Whitney on hidden (64 vs 256) ===")
for algo in ["ddpg", "td3"]:
    for tau in sorted(set(r["tau"] for r in data)):
        h64 = [r["mean_reward"] for r in data if r["algo"] == algo and r["tau"] == tau and r["hidden"] == 64]
        h256 = [r["mean_reward"] for r in data if r["algo"] == algo and r["tau"] == tau and r["hidden"] == 256]
        if h64 and h256:
            stat, p = stats.mannwhitneyu(h64, h256, alternative="two-sided")
            print(f"  {algo}, tau={tau}: U={stat:.1f}, p={p:.4f}")