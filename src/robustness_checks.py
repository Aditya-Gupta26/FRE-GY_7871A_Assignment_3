"""Robustness diagnostics the source papers treat as central but a naive replication
would skip:
  (a) first-half vs second-half coefficient-stability check (mirrors the general
      paper's Table-5-style F-tests across sub-samples of very different war phases);
  (b) H/L threshold sensitivity (mirrors Rigobon & Sack's Table 6/7 alternate-window
      check, and empirically tests the general paper's Proposition-3 misspecification-
      robustness claim);
  (c) the Sargan/J overidentification summary from run_regressions.py's ω3 IV, pulled
      together here for the report.

Runs per normalizing variable (see config.NORMALIZING_VARIABLES_TO_RUN).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_PROCESSED, FRED_SERIES, NORMALIZING_VARIABLE, NORMALIZING_VARIABLES_TO_RUN, YFINANCE_TICKERS
from run_regressions import build_instruments, iv_estimate

ALL_VARIABLES = list(YFINANCE_TICKERS.keys()) + list(FRED_SERIES.keys())


def half_split_stability(panel: pd.DataFrame, normalizing_variable: str, other_variables: list[str]) -> pd.DataFrame:
    midpoint = panel["date"].min() + (panel["date"].max() - panel["date"].min()) / 2
    first_half = panel[panel["date"] < midpoint]
    second_half = panel[panel["date"] >= midpoint]

    rows = []
    for xj in other_variables:
        try:
            sub1 = build_instruments(first_half, normalizing_variable, xj)
            sub2 = build_instruments(second_half, normalizing_variable, xj)
            if sub1["is_H"].sum() < 3 or sub1["is_L"].sum() < 3 or sub2["is_H"].sum() < 3 or sub2["is_L"].sum() < 3:
                rows.append({"variable": xj, "status": "insufficient_obs_in_a_half"})
                continue
            est1 = iv_estimate(sub1, normalizing_variable, xj, ["omega1"])
            est2 = iv_estimate(sub2, normalizing_variable, xj, ["omega1"])
            diff = est1["coef"] - est2["coef"]
            se_diff = np.sqrt(est1["se"] ** 2 + est2["se"] ** 2)
            z = diff / se_diff if se_diff > 0 else np.nan
            pval = 2 * (1 - stats.norm.cdf(abs(z))) if pd.notna(z) else np.nan
            rows.append({
                "variable": xj, "status": "ok",
                "coef_first_half": est1["coef"], "coef_second_half": est2["coef"],
                "stability_z": z, "stability_pvalue": pval,
            })
        except Exception as e:
            rows.append({"variable": xj, "status": f"error: {e}"})
    return pd.DataFrame(rows)


def threshold_sensitivity(normalizing_variable: str, other_variables: list[str], quantiles=(0.85, 0.90, 0.95)) -> pd.DataFrame:
    intensity_panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")
    rows = []
    for q in quantiles:
        threshold = intensity_panel["composite_intensity"].quantile(q)
        h = intensity_panel["composite_intensity"] >= threshold
        alt_panel = intensity_panel.copy()
        alt_panel["is_H"] = h
        for xj in other_variables:
            try:
                sub = build_instruments(alt_panel, normalizing_variable, xj)
                if sub["is_H"].sum() < 3 or sub["is_L"].sum() < 3:
                    continue
                est = iv_estimate(sub, normalizing_variable, xj, ["omega1"])
                rows.append({"variable": xj, "H_quantile": q, "n_H": int(h.sum()), "coef": est["coef"]})
            except Exception:
                continue
    return pd.DataFrame(rows)


def sargan_summary(normalizing_variable: str) -> pd.DataFrame:
    table2 = pd.read_parquet(DATA_PROCESSED / f"table2_estimates_{normalizing_variable}.parquet")
    return table2[["variable", "iv_w3_sargan_stat", "iv_w3_sargan_pvalue"]]


def main(normalizing_variable: str = NORMALIZING_VARIABLE):
    other_variables = [v for v in ALL_VARIABLES if v != normalizing_variable]
    panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")

    stability = half_split_stability(panel, normalizing_variable, other_variables)
    print(f"[{normalizing_variable}] === First-half vs second-half coefficient stability ===")
    print(stability.to_string(index=False))

    sensitivity = threshold_sensitivity(normalizing_variable, other_variables)
    print("\n=== H/L threshold sensitivity ===")
    if not sensitivity.empty:
        print(sensitivity.pivot(index="variable", columns="H_quantile", values="coef").to_string())

    sargan = sargan_summary(normalizing_variable)
    print("\n=== Sargan/J overidentification test (ω3 = [ω1, ω2]) ===")
    print(sargan.to_string(index=False))

    stability.to_parquet(DATA_PROCESSED / f"robustness_stability_{normalizing_variable}.parquet", index=False)
    sensitivity.to_parquet(DATA_PROCESSED / f"robustness_threshold_sensitivity_{normalizing_variable}.parquet", index=False)
    summary = {
        "n_coef_stability_rejections_at_5pct": int((stability.get("stability_pvalue", pd.Series(dtype=float)) < 0.05).sum()),
        "n_sargan_rejections_at_5pct": int((sargan["iv_w3_sargan_pvalue"] < 0.05).sum()),
    }
    with open(DATA_PROCESSED / f"robustness_summary_{normalizing_variable}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        main(nv)
