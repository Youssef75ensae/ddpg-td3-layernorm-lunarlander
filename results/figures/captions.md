# Suggested captions

Each block follows the figure checklist. In the report: give each figure its
number, keep the in-figure title, replace the [TO WRITE] sentence by the
conclusion, and mention each figure at least once in the text (rule 8).
"A vs B" means A minus B in every test; P(A > B) is the probability that a
random run of A beats a random run of B (rliable's probability of improvement).

## Figure 1 - Learning curves
**Caption.** Evaluation return along training for DDPG (left) and TD3 (right), without (blue) and with (orange) LayerNorm. Each evaluation averages 10 deterministic episodes, every 5k environment steps. Lines: mean over n = 20 seeds; shaded areas: interval between the 10th and 90th percentiles of the runs. Black dots above a panel: evaluation steps where LN and no LN differ (Welch's t-test, p < 0.05, uncorrected). Dotted lines: random-policy return (-219) and the usual "solved" threshold (200). Final performance (mean of the last 10 evaluations), LN vs no LN: DDPG Δ mean = 5.5 [-16.1, 27.7], P(A > B) = 0.54, Welch p = 0.629, Bonferroni-corrected p = 1.000; TD3 Δ mean = 21.9 [-20.5, 66.1], P(A > B) = 0.58, Welch p = 0.348, Bonferroni-corrected p = 1.000. [TO WRITE: what should be retained]

Checklist: 1 title in figure / 2 caption with conclusion / 3 axes labelled with units / 4 variability = 10th-90th percentile interval, stated / 5 no colour intensity / 6 tests in caption and dots / 7 number it / 8 cite it.

## Figure 2 - Overestimation bias along training
**Caption.** Overestimation bias of the critic (Q1 for TD3), measured every 25k steps as the mean of Q(s_t, π(s_t)) minus the discounted Monte-Carlo return over 10 deterministic episodes, excluding the last 194 steps of time-limit-truncated episodes; positive values mean overestimation. Lines: mean over n = 20 seeds; shaded areas: 10th-90th percentile interval of the runs; black dots: steps where LN and no LN differ (Welch, p < 0.05, uncorrected). Bias of a run = mean over steps ≥ 100k; LN vs no LN: DDPG Δ mean = -4.9 [-7.4, -2.3], P(A > B) = 0.20, Welch p < 0.001, Bonferroni-corrected p = 0.006; TD3 Δ mean = -2.8 [-4.0, -1.5], P(A > B) = 0.12, Welch p < 0.001, Bonferroni-corrected p = 0.001. [TO WRITE: what should be retained]

Checklist: 1 title / 2 caption with conclusion / 3 axes / 4 variability = 10th-90th percentile interval / 5 no colour intensity / 6 tests in caption and dots / 7 number / 8 cite.

## Figure 3 - Performance profiles and bias vs performance
**Caption.** (a) Performance profiles: fraction of runs whose final performance exceeds a threshold τ, for each configuration (n = 20 runs each); shaded bands are 95% bootstrap confidence intervals; the dotted line marks the "solved" threshold (200); the random policy scores -219 (Fig. 1). A profile that lies above another indicates better performance across the whole distribution. (b) Final performance against the overestimation bias of each run (circles: DDPG, triangles: TD3; blue: no LN, orange: LN). Spearman rank correlations, Bonferroni-corrected over 5 tests: DDPG, no LN: ρ = 0.40 (p = 0.409); DDPG, LN: ρ = -0.44 (p = 0.264); TD3, no LN: ρ = -0.41 (p = 0.361); TD3, LN: ρ = -0.22 (p = 1.000); pooled over the 80 runs: ρ = 0.57 (p < 0.001); the pooled value can be driven by the differences between configurations rather than by a within-configuration link. TD3 vs DDPG without LN: bias Δ mean = -29.9 [-32.0, -27.7], P(A > B) = 0.00, Welch p < 0.001, Bonferroni-corrected p < 0.001; performance Δ mean = -129.0 [-159.6, -96.7], P(A > B) = 0.04, Welch p < 0.001, Bonferroni-corrected p < 0.001. [TO WRITE: what should be retained]

Checklist: 1 titles / 2 caption with conclusion / 3 axes / 4 variability = bootstrap band in (a), every run shown in (b) / 5 no colour intensity / 6 tests in caption / 7 number / 8 cite.

## Figure 4 - Hyper-parameter sensitivity (grid search)
**Caption.** Final performance (mean of the last 10 evaluations) as a function of the Polyak coefficient τ, for two network sizes, without LayerNorm; 3 seeds per point, all shown as dots (fewer than 10 runs), lines join the means. No hyper-parameter effect is statistically detectable at 3 seeds (Kruskal-Wallis on τ: DDPG p = 0.24, TD3 p = 0.06); τ = 0.05 with 64×64 networks (star) was selected for both algorithms. As τ = 0.05 lies at the edge of the tested range, the best value may be larger. [TO WRITE: what should be retained]

Checklist: 1 title / 2 caption with conclusion / 3 axes (log scale stated) / 4 every run shown / 5 no colour intensity / 6 tests in caption / 7 number / 8 cite.
