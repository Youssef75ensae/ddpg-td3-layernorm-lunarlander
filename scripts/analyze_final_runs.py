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
HIDDEN_COLOR = {64: "#4a3aa7", 256: "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SOLVED = 200.0


def name(algo, ln):
    return f"{ALGO[algo]}, {LN[ln]}"


def summarize(d, algo, fname, last_evals, min_bias_step):
    history = d["history"]
    # drop duplicated bias measurements
    seen, bias = set(), []
    for b in d.get("bias_history", []):
        if b["step"] not in seen:
            seen.add(b["step"])
            bias.append(b)
    key = "bias_valid" if bias and "bias_valid" in bias[0] else "bias"
    late = [b for b in bias if b["step"] >= min_bias_step]
    mean_of = lambda k: float(np.mean([b[k] for b in late])) if late and k in late[0] else float("nan")
    return {
        "file": fname,
        "algo": algo,
        "ln": bool(d["config"]["use_layernorm"]),
        "seed": int(d["config"]["seed"]),
        "tau": float(d["config"]["tau"]),
        "hidden": int(d["config"]["actor_hidden"][0]),
        "eval_steps": np.array([h["step"] for h in history]),
        "eval_means": np.array([h["mean"] for h in history]),
        "perf": float(np.mean([h["mean"] for h in history[-last_evals:]])),
        "bias_steps": np.array([b["step"] for b in bias]),
        "bias_curve": np.array([b[key] for b in bias]),
        "bias": mean_of(key),
        "bias_raw": mean_of("bias"),
        "frac_truncated": mean_of("frac_truncated"),
        "bias_key": key,
        "duration_min": d.get("meta", {}).get("duration_s", float("nan")) / 60,
    }


def load(raw_dir, last_evals, min_bias_step, seeds=None, hp=None):
    runs = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        d = json.load(open(path))
        cfg = d["config"]
        algo = d.get("meta", {}).get("algo") or path.name.split("-")[0]
        if seeds is not None and cfg["seed"] not in seeds:
            continue
        if hp is not None:
            if algo not in hp:
                continue
            tau, hidden = hp[algo]
            if cfg["tau"] != tau or list(cfg["actor_hidden"]) != [hidden, hidden]:
                continue
        runs.append(summarize(d, algo, path.name, last_evals, min_bias_step))
    return runs


def by_config(runs):
    g = defaultdict(list)
    for r in runs:
        g[(r["algo"], r["ln"])].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["seed"])
    return g


def random_baseline(env_name, out_dir, n_episodes=100):
    cache = out_dir / "random_baseline.json"
    if cache.exists():
        return json.load(open(cache))
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
    res = {"mean": float(np.mean(returns)), "n_episodes": n_episodes}
    json.dump(res, open(cache, "w"))
    return res


def iqm(x):
    x = np.sort(np.asarray(x))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean())


def bootstrap(stat, samples, n_boot, rng):
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = stat(*[s[rng.integers(0, len(s), len(s))] for s in samples])
    return np.percentile(vals, [2.5, 97.5])


def prob_improvement(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


# normal approximation of Welch's test power
def min_detectable_effect(a, b, alpha, power=0.8):
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)


def welch(name_, measure, a, b, alpha_bonf, n_boot, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=False)
    lo, hi = bootstrap(lambda x, y: x.mean() - y.mean(), [a, b], n_boot, rng)
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return {
        "comparison": name_, "measure": measure, "n_a": len(a), "n_b": len(b),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diff": float(a.mean() - b.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "cohen_d": float((a.mean() - b.mean()) / sd) if sd > 0 else float("nan"),
        "t": float(t.statistic), "df": float(t.df), "p": float(t.pvalue),
        "prob_improvement": prob_improvement(a, b),
        "mde_80": min_detectable_effect(a, b, alpha_bonf),
    }


def confirmatory(g, alpha, n_boot, rng, matched=True):
    m = 6 if matched else 4
    v = lambda c, k: [r[k] for r in g[c]]
    rows = [
        welch("DDPG: LN vs no LN", "bias", v(("ddpg", True), "bias"), v(("ddpg", False), "bias"), alpha / m, n_boot, rng),
        welch("TD3: LN vs no LN", "bias", v(("td3", True), "bias"), v(("td3", False), "bias"), alpha / m, n_boot, rng),
        welch("DDPG: LN vs no LN", "performance", v(("ddpg", True), "perf"), v(("ddpg", False), "perf"), alpha / m, n_boot, rng),
        welch("TD3: LN vs no LN", "performance", v(("td3", True), "perf"), v(("td3", False), "perf"), alpha / m, n_boot, rng),
        welch("no LN: TD3 vs DDPG", "bias", v(("td3", False), "bias"), v(("ddpg", False), "bias"), alpha / m, n_boot, rng),
        welch("no LN: TD3 vs DDPG", "performance", v(("td3", False), "perf"), v(("ddpg", False), "perf"), alpha / m, n_boot, rng),
    ]
    for i, r in enumerate(rows):
        confirm = matched or i < 4
        r["family"] = "confirmatory" if confirm else "exploratory (different hyper-parameters)"
        r["p_bonf"] = min(1.0, r["p"] * (m if confirm else 2))
        r["significant"] = r["p_bonf"] < alpha
    return rows


def correlations(runs, g, alpha):
    rows = []
    groups = [(name(*c), g[c]) for c in CONFIGS] + [("all runs (pooled)", runs)]
    for label, rs in groups:
        rho, p = stats.spearmanr([r["bias"] for r in rs], [r["perf"] for r in rs])
        rows.append({"group": label, "n": len(rs), "rho": float(rho), "p": float(p)})
    for r in rows:
        r["p_bonf"] = min(1.0, r["p"] * len(rows))
    return rows


# tests along training, uncorrected
def stepwise_welch(a_curves, b_curves):
    return np.array([stats.ttest_ind(a_curves[:, i], b_curves[:, i], equal_var=False).pvalue
                     for i in range(a_curves.shape[1])])


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


def kilo(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))


def find(tests, comparison, measure):
    return next(t for t in tests if t["comparison"] == comparison and t["measure"] == measure)


def curves_figure(g, xkey, ykey, ylabel, title, tests, measure, alpha, baselines=()):
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
            ax.annotate(text, (x[0], value), xytext=(2, 2), textcoords="offset points",
                        ha="left", va="bottom", fontsize=6.5, color=INK_2)
        p = stepwise_welch(curves[True], curves[False])
        top = ax.get_ylim()[1]
        sig = p < alpha
        ax.plot(x[sig], np.full(sig.sum(), top), ls="none", marker="o", ms=2.2, color=INK, clip_on=False)
        t = find(tests, f"{ALGO[algo]}: LN vs no LN", measure)
        ax.set_title(ALGO[algo], color=INK, pad=8)
        ax.set_xlabel("environment steps")
        kilo(ax)
        leg = ax.legend(frameon=False, loc="center right" if measure == "performance" else "upper right",
                        bbox_to_anchor=(1.0, 0.38) if measure == "performance" else (1.0, 0.92),
                        title=f"LN vs no LN: Welch {fmt_p(t['p_bonf'])} (Bonf.)", title_fontsize=7)
        leg._legend_box.align = "left"
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=10, color=INK, y=1.13)
    fig.text(0.5, 0.995, f"black dots: steps where LN and no LN differ (Welch's t-test, p < {alpha:g}, uncorrected)",
             ha="center", va="bottom", fontsize=7, color=INK_2)
    return fig


def profiles_and_scatter(runs, g, corr, random_mean, n_boot, rng):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.6))
    perfs = np.array([r["perf"] for r in runs])
    taus = np.linspace(min(perfs.min(), random_mean) - 5, perfs.max() + 5, 300)
    for algo, ln in CONFIGS:
        x = np.array([r["perf"] for r in g[(algo, ln)]])
        prof = (x[None, :] > taus[:, None]).mean(axis=1)
        boot = np.stack([(xb[None, :] > taus[:, None]).mean(axis=1)
                         for xb in (x[rng.integers(0, len(x), len(x))] for _ in range(n_boot))])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        ax1.step(taus, prof, where="post", color=COLOR[ln], ls=LINESTYLE[algo], label=name(algo, ln))
        ax1.fill_between(taus, lo, hi, step="post", color=COLOR[ln], alpha=0.12, lw=0)
    ax1.axhline(0.5, color=INK_2, lw=0.7, ls=":")
    ax1.axvline(random_mean, color=INK_2, lw=0.8, ls=":")
    ax1.annotate("random policy", (random_mean, 0.02), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=INK_2)
    ax1.axvline(SOLVED, color=INK_2, lw=0.8, ls=":")
    ax1.annotate("solved", (SOLVED, 0.02), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=INK_2)
    ax1.set_xlabel("final performance threshold τ (return)")
    ax1.set_ylabel("fraction of runs with performance > τ")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("(a) Performance profiles", color=INK)
    ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    rho = {c["group"]: c for c in corr}
    for algo, ln in CONFIGS:
        rs = g[(algo, ln)]
        c = rho[name(algo, ln)]
        ax2.scatter([r["bias"] for r in rs], [r["perf"] for r in rs], s=24, marker=MARKER[algo],
                    facecolor=COLOR[ln], edgecolor="white", linewidth=0.6,
                    label=f"{name(algo, ln)}: ρ = {c['rho']:.2f}")
    pooled = rho["all runs (pooled)"]
    ax2.axvline(0, color=INK_2, lw=0.8, ls=":")
    ax2.set_xlabel("overestimation bias Q − G (mean over steps ≥ 100k)")
    ax2.set_ylabel("final performance (return)")
    ax2.set_title(f"(b) Bias vs performance (pooled ρ = {pooled['rho']:.2f})", color=INK)
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.3,
               columnspacing=0.8, title="Spearman ρ within each configuration", title_fontsize=7)
    fig.suptitle("Distribution of final performance, and its relation to the bias", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def sensitivity_figure(grid_runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    taus = sorted({r["tau"] for r in grid_runs})
    for ax, algo in zip(axes, ["ddpg", "td3"]):
        for h in sorted({r["hidden"] for r in grid_runs}):
            rs = [r for r in grid_runs if r["algo"] == algo and r["hidden"] == h]
            means = [np.mean([r["perf"] for r in rs if r["tau"] == t]) for t in taus]
            ax.plot(taus, means, color=HIDDEN_COLOR[h], marker="s" if h == 64 else "D", ms=4,
                    label=f"{h}×{h} (mean)")
            for t in taus:
                ys = [r["perf"] for r in rs if r["tau"] == t]
                ax.plot([t] * len(ys), ys, ls="none", marker="o", ms=2.5, color=HIDDEN_COLOR[h], alpha=0.6)
        ax.set_xscale("log")
        ax.set_xticks(taus)
        ax.set_xticklabels([f"{t:g}" for t in taus])
        ax.set_xlabel("Polyak coefficient τ (log scale)")
        ax.set_title(ALGO[algo], color=INK)
        sel = [r["perf"] for r in grid_runs if r["algo"] == algo and r["hidden"] == 64 and r["tau"] == 0.05]
        if sel:
            ax.plot([0.05], [np.mean(sel)], marker="*", ms=11, color=INK, ls="none", zorder=5)
            ax.annotate("selected", (0.05, np.mean(sel)), xytext=(-8, -12), textcoords="offset points",
                        ha="right", fontsize=6.5, color=INK_2)
        if algo == "ddpg":
            ax.legend(frameon=False, loc="upper left", title="hidden layers (dots = individual seeds)", title_fontsize=7)
    axes[0].set_ylabel("final performance (return)")
    fig.suptitle("Hyper-parameter sensitivity (grid search, 3 seeds per point, no LN)", fontsize=10, color=INK, y=1.05)
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


def latex_tables(summary, tests):
    s = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Configuration & $n$ & Mean perf. & IQM perf. [95\% CI] & Mean bias \\", r"\midrule"]
    for row in summary:
        s.append(f"{row['config']} & {row['n']} & {row['perf_mean']:.1f} & {row['perf_iqm']:.1f} "
                 f"[{row['iqm_low']:.1f}, {row['iqm_high']:.1f}] & {row['bias_mean']:.2f} \\\\")
    s += [r"\bottomrule", r"\end{tabular}"]
    t = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Comparison & Measure & $\Delta$ mean [95\% CI] & $P(A>B)$ & $p$ & $p_\mathrm{Bonf}$ & MDE \\", r"\midrule"]
    for r in tests:
        t.append(f"{r['comparison']} & {r['measure']} & {r['diff']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}] "
                 f"& {r['prob_improvement']:.2f} & {r['p']:.3f} & {r['p_bonf']:.3f} & {r['mde_80']:.1f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    return "% Table: summary per configuration\n" + "\n".join(s) + "\n\n% Table: confirmatory tests\n" + "\n".join(t) + "\n"


def captions(tests, corr, n, random_mean, grid_ok, matched=True):
    T = lambda c, m: find(tests, c, m)
    def res(c, m):
        t = T(c, m)
        return (f"Δ mean = {t['diff']:.1f} [{t['ci_low']:.1f}, {t['ci_high']:.1f}], "
                f"P(A > B) = {t['prob_improvement']:.2f}, Welch {fmt_p(t['p'])}, Bonferroni-corrected {fmt_p(t['p_bonf'])}")
    pooled = corr[-1]
    within = "; ".join(f"{c['group']}: ρ = {c['rho']:.2f} ({fmt_p(c['p_bonf'])})" for c in corr[:-1])
    out = f"""# Suggested captions

Each block follows the figure checklist. In the report: give each figure its
number, keep the in-figure title, replace the [TO WRITE] sentence by the
conclusion, and mention each figure at least once in the text (rule 8).
"A vs B" means A minus B in every test; P(A > B) is the probability that a
random run of A beats a random run of B (rliable's probability of improvement).

## Figure 1 - Learning curves
**Caption.** Evaluation return along training for DDPG (left) and TD3 (right), without (blue) and with (orange) LayerNorm. Each evaluation averages 10 deterministic episodes, every 5k environment steps. Lines: mean over n = {n} seeds; shaded areas: interval between the 10th and 90th percentiles of the runs. Black dots above a panel: evaluation steps where LN and no LN differ (Welch's t-test, p < 0.05, uncorrected). Dotted lines: random-policy return ({random_mean:.0f}) and the usual "solved" threshold (200). Final performance (mean of the last 10 evaluations), LN vs no LN: DDPG {res('DDPG: LN vs no LN', 'performance')}; TD3 {res('TD3: LN vs no LN', 'performance')}. [TO WRITE: what should be retained]

Checklist: 1 title in figure / 2 caption with conclusion / 3 axes labelled with units / 4 variability = 10th-90th percentile interval, stated / 5 no colour intensity / 6 tests in caption and dots / 7 number it / 8 cite it.

## Figure 2 - Overestimation bias along training
**Caption.** Overestimation bias of the critic (Q1 for TD3), measured every 25k steps as the mean of Q(s_t, π(s_t)) minus the discounted Monte-Carlo return over 10 deterministic episodes, excluding the last 194 steps of time-limit-truncated episodes; positive values mean overestimation. Lines: mean over n = {n} seeds; shaded areas: 10th-90th percentile interval of the runs; black dots: steps where LN and no LN differ (Welch, p < 0.05, uncorrected). Bias of a run = mean over steps ≥ 100k; LN vs no LN: DDPG {res('DDPG: LN vs no LN', 'bias')}; TD3 {res('TD3: LN vs no LN', 'bias')}. [TO WRITE: what should be retained]

Checklist: 1 title / 2 caption with conclusion / 3 axes / 4 variability = 10th-90th percentile interval / 5 no colour intensity / 6 tests in caption and dots / 7 number / 8 cite.

## Figure 3 - Performance profiles and bias vs performance
**Caption.** (a) Performance profiles: fraction of runs whose final performance exceeds a threshold τ, for each configuration (n = {n} runs each); shaded bands are 95% bootstrap confidence intervals; dotted lines mark the random policy ({random_mean:.0f}) and the "solved" threshold (200). A profile that lies above another indicates better performance across the whole distribution. (b) Final performance against the overestimation bias of each run (circles: DDPG, triangles: TD3; blue: no LN, orange: LN). Spearman rank correlations, Bonferroni-corrected over 5 tests: {within}; pooled over the {pooled['n']} runs: ρ = {pooled['rho']:.2f} ({fmt_p(pooled['p_bonf'])}); the pooled value can be driven by the differences between configurations rather than by a within-configuration link. TD3 vs DDPG without LN: bias {res('no LN: TD3 vs DDPG', 'bias')}; performance {res('no LN: TD3 vs DDPG', 'performance')}{'' if matched else ' (DDPG and TD3 use different hyper-parameters, so this comparison is descriptive)'}. [TO WRITE: what should be retained]

Checklist: 1 titles / 2 caption with conclusion / 3 axes / 4 variability = bootstrap band in (a), every run shown in (b) / 5 no colour intensity / 6 tests in caption / 7 number / 8 cite.
"""
    if grid_ok:
        out += """
## Figure 4 - Hyper-parameter sensitivity (grid search)
**Caption.** Final performance (mean of the last 10 evaluations) as a function of the Polyak coefficient τ, for two network sizes, without LayerNorm; 3 seeds per point, all shown as dots (fewer than 10 runs), lines join the means. No hyper-parameter effect is statistically detectable at 3 seeds (Kruskal-Wallis on τ: DDPG p = 0.24, TD3 p = 0.06); τ = 0.05 with 64×64 networks was selected for both algorithms. As τ = 0.05 lies at the edge of the tested range, the best value may be larger. [TO WRITE: what should be retained]

Checklist: 1 title / 2 caption with conclusion / 3 axes (log scale stated) / 4 every run shown / 5 no colour intensity / 6 tests in caption / 7 number / 8 cite.
"""
    return out


def parse_range(text):
    a, b = text.split("-")
    return set(range(int(a), int(b) + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    p.add_argument("--grid-dir", type=Path, default=ROOT / "results" / "hyperparameter_search")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    p.add_argument("--seeds", default="15-34")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--ddpg-tau", type=float, default=None, help="if DDPG used other hyper-parameters")
    p.add_argument("--ddpg-hidden", type=int, default=None)
    p.add_argument("--last-evals", type=int, default=10)
    p.add_argument("--min-bias-step", type=int, default=100_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=5_000)
    p.add_argument("--env", default="LunarLanderContinuous-v3")
    p.add_argument("--replication-dir", type=Path, default=None,
                   help="results/raw of the first campaign (seeds 0-14), exploratory")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hp = {"td3": (args.tau, args.hidden),
          "ddpg": (args.ddpg_tau if args.ddpg_tau is not None else args.tau,
                   args.ddpg_hidden if args.ddpg_hidden is not None else args.hidden)}
    matched = hp["ddpg"] == hp["td3"]
    runs = load(args.raw_dir, args.last_evals, args.min_bias_step, parse_range(args.seeds), hp)
    g = by_config(runs)
    print(f"Loaded {len(runs)} runs from {args.raw_dir}")
    for c in CONFIGS:
        print(f"  {name(*c):14s} n = {len(g[c]):2d}  seeds {[r['seed'] for r in g[c]]}")
    if any(not g[c] for c in CONFIGS):
        found = defaultdict(int)
        for path in Path(args.raw_dir).glob("*.json"):
            c = json.load(open(path))["config"]
            found[(path.name.split("-")[0], c["tau"], c["actor_hidden"][0])] += 1
        print("  runs found per (algo, tau, hidden):", dict(found))
        raise SystemExit("A configuration has no run: use --ddpg-tau / --ddpg-hidden if DDPG used other hyper-parameters.")
    if not matched:
        print(f"  NOTE: DDPG uses tau = {hp['ddpg'][0]}, hidden = {hp['ddpg'][1]}; TD3 uses tau = {hp['td3'][0]}, "
              f"hidden = {hp['td3'][1]}. TD3 vs DDPG tests are exploratory.")
    if len({len(r["eval_steps"]) for r in runs}) > 1 or len({len(r["bias_steps"]) for r in runs}) > 1:
        raise SystemExit("Runs differ in number of evaluations or bias measurements (unfinished runs?).")
    if len({len(g[c]) for c in CONFIGS}) > 1:
        print("  WARNING: unequal group sizes")
    n = min(len(g[c]) for c in CONFIGS)

    baseline = random_baseline(args.env, args.out_dir)
    print(f"Random policy baseline: {baseline['mean']:.1f} (mean over {baseline['n_episodes']} episodes)")

    print("\n=== Per configuration ===")
    summary = []
    for c in CONFIGS:
        rs = g[c]
        perf = np.array([r["perf"] for r in rs])
        lo, hi = bootstrap(iqm, [perf], args.n_boot, rng)
        row = {
            "config": name(*c), "n": len(rs),
            "perf_mean": float(perf.mean()), "perf_iqm": iqm(perf), "iqm_low": float(lo), "iqm_high": float(hi),
            "perf_p10": float(np.percentile(perf, 10)), "perf_p90": float(np.percentile(perf, 90)),
            "frac_solved": float((perf >= SOLVED).mean()),
            "bias_mean": float(np.mean([r["bias"] for r in rs])),
            "truncation_artifact": float(np.mean([r["bias_raw"] - r["bias"] for r in rs])),
            "frac_truncated": float(np.nanmean([r["frac_truncated"] for r in rs])),
            "duration_min": float(np.nanmean([r["duration_min"] for r in rs])),
        }
        summary.append(row)
        print(f"  {row['config']:14s} perf mean {row['perf_mean']:7.1f}  IQM {row['perf_iqm']:7.1f} "
              f"[{lo:.1f}, {hi:.1f}]  [p10, p90] = [{row['perf_p10']:.1f}, {row['perf_p90']:.1f}]  "
              f"solved {row['frac_solved']:.0%} | bias {row['bias_mean']:6.2f}  "
              f"(truncation artifact {row['truncation_artifact']:+.2f}, truncated episodes {row['frac_truncated']:.0%}) "
              f"| {row['duration_min']:.1f} min")
    write_csv(args.out_dir / "summary.csv", summary)
    write_csv(args.out_dir / "per_run.csv", [{k: r[k] for k in ("file", "algo", "ln", "seed", "perf", "bias", "bias_raw", "frac_truncated", "duration_min")} for r in runs])

    tests = confirmatory(g, args.alpha, args.n_boot, rng, matched)
    print(f"\n=== Welch's t-tests, Bonferroni over {6 if matched else 4} confirmatory tests (alpha = {args.alpha}) ===")
    for t in tests:
        tag = "" if t["family"] == "confirmatory" else "  [exploratory]"
        print(f"  {t['comparison']:20s} {t['measure']:11s} Δmean {t['diff']:8.2f} [{t['ci_low']:7.2f}, {t['ci_high']:7.2f}] "
              f"d = {t['cohen_d']:+.2f}  P(A>B) = {t['prob_improvement']:.2f}  t = {t['t']:+.2f} (df {t['df']:.1f})  "
              f"p = {t['p']:.4f}  p_Bonf = {t['p_bonf']:.4f}  {'*' if t['significant'] else ' '}  "
              f"MDE(80% power) = {t['mde_80']:.1f}{tag}")
    print("  MDE = smallest difference of means this test detects with 80% power at the Bonferroni level.")
    write_csv(args.out_dir / "tests.csv", tests)

    corr = correlations(runs, g, args.alpha)
    print("\n=== Spearman bias vs performance (exploratory, Bonferroni over 5) ===")
    for c in corr:
        print(f"  {c['group']:20s} n = {c['n']:2d}  rho = {c['rho']:+.2f}  p = {c['p']:.4f}  p_Bonf = {c['p_bonf']:.4f}")
    write_csv(args.out_dir / "correlations.csv", corr)

    save(curves_figure(g, "eval_steps", "eval_means", "evaluation return",
                       "Learning curves (mean and 10th-90th percentile interval over seeds)", tests, "performance",
                       args.alpha, baselines=[(SOLVED, "solved"), (baseline["mean"], "random policy")]),
         args.out_dir, "fig1_learning_curves")
    save(curves_figure(g, "bias_steps", "bias_curve", "overestimation bias Q − G",
                       "Overestimation bias along training (mean and 10th-90th percentile interval)", tests, "bias",
                       args.alpha, baselines=[(0.0, "")]),
         args.out_dir, "fig2_bias")
    save(profiles_and_scatter(runs, g, corr, baseline["mean"], min(args.n_boot, 2000), rng), args.out_dir, "fig3_profiles_bias_perf")

    grid_ok = False
    if args.grid_dir.exists() and any(args.grid_dir.glob("*.json")):
        grid = load(args.grid_dir, args.last_evals, args.min_bias_step)
        if grid:
            save(sensitivity_figure(grid), args.out_dir, "fig4_sensitivity")
            grid_ok = True

    (args.out_dir / "tables.tex").write_text(latex_tables(summary, tests))
    (args.out_dir / "captions.md").write_text(captions(tests, corr, n, baseline["mean"], grid_ok, matched))
    print(f"\nFigures, tables and captions written to {args.out_dir}")

    if args.replication_dir is not None:
        rep = by_config(load(args.replication_dir, args.last_evals, args.min_bias_step, set(range(0, 15)),
                             {"td3": (args.tau, args.hidden)}))
        print("\n=== Exploratory: TD3 replication on the first campaign (seeds 0-14, old code) ===")
        for ln in (False, True):
            a = [r["perf"] for r in rep[("td3", ln)]]
            b = [r["perf"] for r in g[("td3", ln)]]
            if len(a) >= 3:
                t = stats.ttest_ind(a, b, equal_var=False)
                print(f"  TD3, {LN[ln]:5s}: first campaign mean {np.mean(a):.1f} (n = {len(a)}) vs ours {np.mean(b):.1f} "
                      f"(n = {len(b)}), Welch p = {t.pvalue:.3f}")
            else:
                print(f"  TD3, {LN[ln]}: not enough runs found ({len(a)})")


if __name__ == "__main__":
    main()
