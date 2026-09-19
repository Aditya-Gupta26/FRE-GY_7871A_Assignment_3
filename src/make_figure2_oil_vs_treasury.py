"""Figure 2: three panels showing why Brent oil identifies the model far better than
the 2-year Treasury yield for this episode.

(a) Var_H vs Var_L for each candidate normalizing variable itself, which is the root
    cause: a good normalizing variable's own variance should jump sharply on H-days.
    Brent oil's does (~11x); the 2yr Treasury yield's barely moves (~1.3x).
(b) |t-stat| for every other variable's coefficient, treasury_2y-normalized vs
    brent_oil-normalized, with a dashed reference line at |t|=2 (conventional
    significance). Shows the oil specification clears significance almost
    everywhere the Treasury specification doesn't.
(c) H/L-threshold sensitivity: how much each normalizing variable's estimated
    coefficients swing (including sign flips) across three alternative H-day cutoffs.
    Direct visual evidence of the weak-instrument instability.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, PROJECT_ROOT

NV_LABELS = {"treasury_2y": "2yr Treasury yield\n(primary, matches original paper)",
             "brent_oil": "Brent crude\n(secondary experiment)"}
NV_COLORS = {"treasury_2y": "#2980b9", "brent_oil": "#c0392b"}


def panel_a(ax):
    var_h_l = {}
    for nv in NV_LABELS:
        with open(DATA_PROCESSED / f"event_study_omegas_{nv}.json") as f:
            d = json.load(f)
        k = next(iter(d["results"]))
        var_h_l[nv] = (d["results"][k]["omega_H"][0][0], d["results"][k]["omega_L"][0][0])

    x = np.arange(len(NV_LABELS))
    width = 0.35
    l_vals = [var_h_l[nv][1] for nv in NV_LABELS]
    h_vals = [var_h_l[nv][0] for nv in NV_LABELS]
    # Normalize each variable's own L-day variance to 1.0 so both series are visually
    # comparable on one axis (raw units differ enormously: yield pp^2 vs $^2).
    l_norm = [1.0, 1.0]
    h_norm = [h / l for h, l in zip(h_vals, l_vals)]

    ax.bar(x - width / 2, l_norm, width, label="Var on L (calm) days", color="#95a5a6")
    ax.bar(x + width / 2, h_norm, width, label="Var on H (war-news) days", color="#e67e22")
    for i, nv in enumerate(NV_LABELS):
        ax.annotate(f"{h_norm[i]:.1f}x", (x[i] + width / 2, h_norm[i]), ha="center",
                     va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([NV_LABELS[nv] for nv in NV_LABELS], fontsize=9)
    ax.set_ylabel("Variance, normalized to L-day variance = 1.0")
    ax.set_title("(a) Does the normalizing variable's OWN variance\njump on war-news days? (the identification condition)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def panel_b(ax):
    tables = {nv: pd.read_parquet(DATA_PROCESSED / f"table2_estimates_{nv}.parquet").set_index("variable")
              for nv in NV_LABELS}
    common_vars = [v for v in tables["brent_oil"].index if v in tables["treasury_2y"].index]
    common_vars = sorted(common_vars, key=lambda v: abs(tables["brent_oil"].loc[v, "iv_w1_tstat"]), reverse=True)

    x = np.arange(len(common_vars))
    width = 0.35
    t_treasury = [abs(tables["treasury_2y"].loc[v, "iv_w1_tstat"]) for v in common_vars]
    t_oil = [abs(tables["brent_oil"].loc[v, "iv_w1_tstat"]) for v in common_vars]

    ax.bar(x - width / 2, t_treasury, width, label="2yr Treasury-normalized", color=NV_COLORS["treasury_2y"])
    ax.bar(x + width / 2, t_oil, width, label="Brent oil-normalized", color=NV_COLORS["brent_oil"])
    ax.axhline(2.0, color="#333333", linestyle="--", linewidth=0.8, label="|t| = 2 (conventional significance)")
    ax.set_xticks(x)
    ax.set_xticklabels(common_vars, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|t-statistic| (IV ω1 estimator)")
    ax.set_title("(b) Statistical significance of each variable's estimated\nsensitivity, by normalizing-variable choice")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def panel_c(ax):
    """Aggregate instability score across ALL variables (not just one example):
    for each variable, (max - min) coefficient across the three H-day thresholds,
    divided by the median |coefficient|, giving a scale-free measure of how much the
    estimate swings under an alternative, equally-defensible threshold choice.
    A value near/above 2 means the estimate swings by more than its own typical
    magnitude, i.e. including a sign flip is plausible."""
    sens = {nv: pd.read_parquet(DATA_PROCESSED / f"robustness_threshold_sensitivity_{nv}.parquet")
            for nv in NV_LABELS}
    instability = {nv: [] for nv in NV_LABELS}
    for nv in NV_LABELS:
        for var, grp in sens[nv].groupby("variable"):
            coefs = grp["coef"].values
            med_abs = np.median(np.abs(coefs))
            if med_abs > 0:
                instability[nv].append((coefs.max() - coefs.min()) / med_abs)

    data = [instability["treasury_2y"], instability["brent_oil"]]
    bp = ax.boxplot(data, tick_labels=[NV_LABELS[nv].split("\n")[0] for nv in NV_LABELS],
                     patch_artist=True, widths=0.5)
    for patch, nv in zip(bp["boxes"], NV_LABELS):
        patch.set_facecolor(NV_COLORS[nv])
        patch.set_alpha(0.5)
    for nv, ys in zip(NV_LABELS, data):
        xs = np.full(len(ys), 1 + list(NV_LABELS).index(nv)) + np.random.default_rng(0).uniform(-0.08, 0.08, len(ys))
        ax.scatter(xs, ys, color=NV_COLORS[nv], zorder=5, s=18)
    ax.axhline(2.0, color="#333333", linestyle="--", linewidth=0.8,
               label="swing exceeds own typical magnitude\n(plausible sign flip)")
    ax.set_ylabel("(max − min coef across 0.85/0.90/0.95 thresholds)\n/ median |coef|, per variable")
    ax.set_title("(c) How much do ALL variables' coefficients swing\nacross alternative H-day cutoffs? (lower = more stable)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, axis="y")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    fig.suptitle("Why Brent oil identifies the model far better than the 2-year Treasury yield here", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = PROJECT_ROOT / "notebooks" / "figure2_oil_vs_treasury.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
