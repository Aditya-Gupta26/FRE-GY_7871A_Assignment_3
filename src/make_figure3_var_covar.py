"""Figure 3: raw variance and covariance values on H (war-news) vs L (calm) days.

This is the actual Omega_H and Omega_L building block that everything else (Eq 6, 7,
10, the IV estimators, Table 3) gets computed from. Earlier figures show derived
diagnostics (ratios, t-stats, instability scores); this one just shows the raw numbers
themselves so a reader can see what is actually driving those diagnostics.

Top row: Var(x1) on H vs L, for each normalizing variable (x1's own variance, same
number for every variable pair since it does not depend on xj).
Bottom row: Cov(x1, xj) on H vs L, for every other variable xj, one panel per
normalizing variable.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, NORMALIZING_VARIABLES_TO_RUN, PROJECT_ROOT

NV_LABELS = {"treasury_2y": "2yr Treasury yield", "brent_oil": "Brent crude"}


def load_omegas(nv: str) -> dict:
    with open(DATA_PROCESSED / f"event_study_omegas_{nv}.json") as f:
        return json.load(f)["results"]


def panel_var(ax, nv: str):
    omegas = load_omegas(nv)
    xj0 = next(iter(omegas))
    var_h = omegas[xj0]["omega_H"][0][0]
    var_l = omegas[xj0]["omega_L"][0][0]
    ax.bar(["L (calm)", "H (war news)"], [var_l, var_h], color=["#95a5a6", "#e67e22"])
    ax.set_title(f"Var({NV_LABELS[nv]})")
    ax.set_ylabel("Variance of daily change")
    ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate([var_l, var_h]):
        ax.annotate(f"{v:.4g}", (i, v), ha="center", va="bottom", fontsize=9)


def panel_cov(ax, nv: str):
    omegas = load_omegas(nv)
    variables = list(omegas.keys())
    cov_h = [omegas[v]["omega_H"][0][1] for v in variables]
    cov_l = [omegas[v]["omega_L"][0][1] for v in variables]
    order = np.argsort(-np.abs(np.array(cov_h) - np.array(cov_l)))
    variables = [variables[i] for i in order]
    cov_h = [cov_h[i] for i in order]
    cov_l = [cov_l[i] for i in order]

    x = np.arange(len(variables))
    width = 0.35
    ax.bar(x - width / 2, cov_l, width, label="Cov on L days", color="#95a5a6")
    ax.bar(x + width / 2, cov_h, width, label="Cov on H days", color="#e67e22")
    ax.axhline(0, color="#333333", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Cov({NV_LABELS[nv]}, other variable)")
    ax.set_ylabel("Covariance of daily changes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1.6]})
    for col, nv in enumerate(NORMALIZING_VARIABLES_TO_RUN):
        panel_var(axes[0, col], nv)
        panel_cov(axes[1, col], nv)
    fig.suptitle("Variance and covariance of daily changes, H (war-news) vs L (calm) days", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = PROJECT_ROOT / "notebooks" / "figure3_var_covar.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
