"""Figures and statistical tests for the final runs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = [("ddpg", False), ("ddpg", True), ("td3", False), ("td3", True)]
ALGO = {"ddpg": "DDPG", "td3": "TD3"}
LN = {False: "no LN", True: "LN"}
COLOR = {False: "#2a78d6", True: "#eb6834"}
LINESTYLE = {"ddpg": "-", "td3": "--"}
MARKER = {"ddpg": "o", "td3": "^"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SOLVED = 200.0


def name(algo, ln):
    return f"{ALGO[algo]}, {LN[ln]}"


def summarize(d, algo, last_evals, min_bias_step):
    history = d["history"]
    bias = list({b["step"]: b for b in d["bias_history"]}.values())  # drop duplicated steps
    late = [b for b in bias if b["step"] >= min_bias_step]
    return {
        "algo": algo,
        "ln": bool(d["config"]["use_layernorm"]),
        "seed": int(d["config"]["seed"]),
        "eval_steps": np.array([h["step"] for h in history]),
        "eval_means": np.array([h["mean"] for h in history]),
        "perf": float(np.mean([h["mean"] for h in history[-last_evals:]])),
        "bias_steps": np.array([b["step"] for b in bias]),
        "bias_curve": np.array([b["bias_valid"] for b in bias]),
        "bias": float(np.mean([b["bias_valid"] for b in late])),
        "bias_raw": float(np.mean([b["bias"] for b in late])),
    }


def load(raw_dir, seeds, tau, hidden, last_evals, min_bias_step):
    g = defaultdict(list)
    for path in sorted(Path(raw_dir).glob("*.json")):
        d = json.load(open(path))
        cfg = d["config"]
        if cfg["seed"] in seeds and cfg["tau"] == tau and list(cfg["actor_hidden"]) == [hidden, hidden]:
            r = summarize(d, d["meta"]["algo"], last_evals, min_bias_step)
            g[(r["algo"], r["ln"])].append(r)
    return g


def random_baseline(env_name, out_dir, n_episodes=100):
    cache = out_dir / "random_baseline.json"
    if cache.exists():
        return json.load(open(cache))["mean"]
    import gymnasium as gym
    env = gym.make(env_name)
    returns = []
    for k in range(n_episodes):
        env.reset(seed=10_000 + k)
        env.action_space.seed(10_000 + k)
        done, total = False, 0.0
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total += r
            done = term or trunc
        returns.append(total)
    env.close()
    mean = float(np.mean(returns))
    json.dump({"mean": mean, "n_episodes": n_episodes}, open(cache, "w"))
    return mean


def welch(comparison, measure, a, b, m, alpha, n_boot, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {
        "comparison": comparison, "measure": measure,
        "mean_a": a.mean(), "mean_b": b.mean(), "diff": a.mean() - b.mean(), "ci_low": lo, "ci_high": hi,
        "prob_improvement": float((a[:, None] > b[None, :]).mean()),
        "p": p, "p_bonf": min(1.0, p * m),
        # smallest difference detected with 80% power at the Bonferroni level (normal approximation)
        "mde_80": (stats.norm.ppf(1 - alpha / (2 * m)) + stats.norm.ppf(0.8)) * se,
    }


def tests(g, alpha, n_boot, rng):
    v = lambda algo, ln, k: [r[k] for r in g[(algo, ln)]]
    ln_ddpg = ("DDPG: LN vs no LN", ("ddpg", True), ("ddpg", False))
    ln_td3 = ("TD3: LN vs no LN", ("td3", True), ("td3", False))
    algos = ("no LN: TD3 vs DDPG", ("td3", False), ("ddpg", False))
    order = [(ln_ddpg, "bias"), (ln_td3, "bias"), (ln_ddpg, "perf"), (ln_td3, "perf"), (algos, "bias"), (algos, "perf")]
    return [welch(c, k, v(*a, k), v(*b, k), 6, alpha, n_boot, rng) for (c, a, b), k in order]


def correlations(g):
    groups = [(name(*c), g[c]) for c in CONFIGS] + [("pooled", [r for c in CONFIGS for r in g[c]])]
    rows = []
    for label, rs in groups:
        rho, p = stats.spearmanr([r["bias"] for r in rs], [r["perf"] for r in rs])
        rows.append({"group": label, "n": len(rs), "rho": rho, "p": p, "p_bonf": min(1.0, p * len(groups))})
    return rows


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK_2,
        "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def find(rows, comparison, measure):
    return next(t for t in rows if t["comparison"] == comparison and t["measure"] == measure)


def curves_figure(g, xkey, ykey, ylabel, rows, measure, alpha, baselines=()):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, algo in zip(axes, ["ddpg", "td3"]):
        curves = {}
        for ln in (False, True):
            rs = g[(algo, ln)]
            x = rs[0][xkey]
            ys = np.vstack([r[ykey] for r in rs])
            curves[ln] = ys
            lo, hi = np.percentile(ys, [10, 90], axis=0)
            ax.plot(x, ys.mean(axis=0), color=COLOR[ln], ls=LINESTYLE[algo], label=f"{LN[ln]} (n = {len(rs)})")
            ax.fill_between(x, lo, hi, color=COLOR[ln], alpha=0.15, lw=0)
        for value, text in baselines:
            ax.axhline(value, color=INK_2, lw=0.8, ls=":")
            at_end = value < 0
            ax.annotate(text, (x[-1] if at_end else x[0], value), xytext=(-2 if at_end else 2, 2),
                        textcoords="offset points", ha="right" if at_end else "left", va="bottom",
                        fontsize=6.5, color=INK_2)
        # black dots: steps where LN and no LN differ (Welch, uncorrected)
        p = np.array([stats.ttest_ind(curves[True][:, i], curves[False][:, i], equal_var=False).pvalue
                      for i in range(len(x))])
        sig = p < alpha
        ax.plot(x[sig], np.full(sig.sum(), ax.get_ylim()[1]), ls="none", marker="o", ms=2.2, color=INK, clip_on=False)
        t = find(rows, f"{ALGO[algo]}: LN vs no LN", measure)
        ax.set_title(ALGO[algo], color=INK, pad=8)
        ax.set_xlabel("environment steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                  title=f"LN vs no LN: Welch {fmt_p(t['p'])}, Bonferroni {fmt_p(t['p_bonf'])}", title_fontsize=7)
    axes[0].set_ylabel(ylabel)
    return fig


def profiles_and_scatter(g, corr, n_boot, rng):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    perfs = np.array([r["perf"] for c in CONFIGS for r in g[c]])
    taus = np.linspace(perfs.min() - 10, perfs.max() + 10, 300)
    for algo, ln in CONFIGS:
        x = np.array([r["perf"] for r in g[(algo, ln)]])
        profile = lambda s: (s[None, :] > taus[:, None]).mean(axis=1)
        lo, hi = np.percentile([profile(rng.choice(x, len(x))) for _ in range(n_boot)], [2.5, 97.5], axis=0)
        ax1.step(taus, profile(x), where="post", color=COLOR[ln], ls=LINESTYLE[algo], label=name(algo, ln))
        ax1.fill_between(taus, lo, hi, step="post", color=COLOR[ln], alpha=0.12, lw=0)
    ax1.axhline(0.5, color=INK_2, lw=0.7, ls=":")
    ax1.axvline(SOLVED, color=INK_2, lw=0.8, ls=":")
    ax1.annotate("solved", (SOLVED, 1.0), xytext=(3, -2), textcoords="offset points", va="top", fontsize=6.5, color=INK_2)
    ax1.set_xlabel("final performance threshold τ (return)")
    ax1.set_ylabel("fraction of runs with performance > τ")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("(a) Performance profiles", color=INK)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    rho = {c["group"]: c for c in corr}
    for algo, ln in CONFIGS:
        rs = g[(algo, ln)]
        ax2.scatter([r["bias"] for r in rs], [r["perf"] for r in rs], s=24, marker=MARKER[algo],
                    facecolor=COLOR[ln], edgecolor="white", linewidth=0.6,
                    label=f"{name(algo, ln)}: ρ = {rho[name(algo, ln)]['rho']:.2f}")
    pooled = rho["pooled"]
    ax2.axvline(0, color=INK_2, lw=0.8, ls=":")
    ax2.set_xlabel("overestimation bias Q − G (mean over steps ≥ 100k)")
    ax2.set_ylabel("final performance (return)")
    ax2.set_title("(b) Bias vs performance", color=INK)
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.3,
               columnspacing=0.8, title_fontsize=7,
               title=f"Spearman ρ (pooled: ρ = {pooled['rho']:.2f}, Bonferroni {fmt_p(pooled['p_bonf'])})")
    fig.tight_layout()
    return fig


def save(fig, out_dir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_range(text):
    a, b = text.split("-")
    return set(range(int(a), int(b) + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    p.add_argument("--seeds", default="15-34")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--last-evals", type=int, default=10)
    p.add_argument("--min-bias-step", type=int, default=100_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=5_000)
    p.add_argument("--env", default="LunarLanderContinuous-v3")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    g = load(args.raw_dir, parse_range(args.seeds), args.tau, args.hidden, args.last_evals, args.min_bias_step)
    for c in CONFIGS:
        print(f"{name(*c):14s} n = {len(g[c])}")
    if any(not g[c] for c in CONFIGS):
        raise SystemExit("A configuration has no run.")

    random_mean = random_baseline(args.env, args.out_dir)
    print(f"Random policy: {random_mean:.1f}")

    print("\n=== Per configuration ===")
    summary = []
    for c in CONFIGS:
        perf = np.array([r["perf"] for r in g[c]])
        row = {
            "config": name(*c), "n": len(perf), "perf_mean": perf.mean(),
            "bias_mean": np.mean([r["bias"] for r in g[c]]),
            "truncation_effect": np.mean([r["bias_raw"] - r["bias"] for r in g[c]]),
        }
        summary.append(row)
        print(f"{row['config']:14s} perf {row['perf_mean']:7.1f} | bias {row['bias_mean']:6.2f} "
              f"(truncation effect {row['truncation_effect']:+.2f})")

    rows = tests(g, args.alpha, args.n_boot, rng)
    print("\n=== Welch's t-tests, Bonferroni over 6 ===")
    for t in rows:
        print(f"{t['comparison']:20s} {t['measure']:5s} diff {t['diff']:8.1f} [{t['ci_low']:7.1f}, {t['ci_high']:7.1f}]  "
              f"P(A>B) = {t['prob_improvement']:.2f}  p = {t['p']:.4f}  p_Bonf = {t['p_bonf']:.4f}  MDE = {t['mde_80']:.1f}")

    corr = correlations(g)
    print("\n=== Spearman bias vs performance, Bonferroni over 5 ===")
    for c in corr:
        print(f"{c['group']:14s} n = {c['n']:2d}  rho = {c['rho']:+.2f}  p_Bonf = {c['p_bonf']:.4f}")

    write_csv(args.out_dir / "summary.csv", summary)
    write_csv(args.out_dir / "tests.csv", rows)
    write_csv(args.out_dir / "correlations.csv", corr)
    write_csv(args.out_dir / "per_run.csv", [{k: r[k] for k in ("algo", "ln", "seed", "perf", "bias")} for c in CONFIGS for r in g[c]])

    save(curves_figure(g, "eval_steps", "eval_means", "evaluation return", rows, "perf", args.alpha,
                       baselines=[(SOLVED, "solved"), (random_mean, "random policy")]),
         args.out_dir, "fig1_learning_curves")
    save(curves_figure(g, "bias_steps", "bias_curve", "overestimation bias Q − G", rows, "bias", args.alpha,
                       baselines=[(0.0, "")]),
         args.out_dir, "fig2_bias")
    save(profiles_and_scatter(g, corr, 2_000, rng), args.out_dir, "fig3_profiles_bias_perf")
    print(f"\nFigures and CSV files written to {args.out_dir}")


if __name__ == "__main__":
    main()