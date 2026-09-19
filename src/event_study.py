"""Build Ω_H, Ω_L, ΔΩ for every (x1, xj) pair, after checking for and optionally
correcting serial correlation. The general paper's own empirical section runs a VAR
"to remove the effects of serial correlation" before computing regime covariance
matrices; here a simpler per-series AR(1) filter is used and documented per-variable
rather than applied blindly.

The AR(1) filtering pass is independent of which variable is chosen as x1 (it's applied
per-series), so it runs once and is shared across every normalizing-variable experiment
(see run_all_normalizations.py); only the Ω_H/Ω_L computation is x1-specific.
"""
import json

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg

from config import DATA_PROCESSED, FRED_SERIES, NORMALIZING_VARIABLE, YFINANCE_TICKERS

ALL_VARIABLES = list(YFINANCE_TICKERS.keys()) + list(FRED_SERIES.keys())
LJUNG_BOX_ALPHA = 0.05


def check_and_filter_serial_correlation(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """For each Δx series, test for serial correlation (Ljung-Box, lag 5). If
    significant, replace the series with its AR(1) residual (documented per variable,
    not silently applied) so the covariance estimates aren't biased by autocorrelation."""
    filtered = panel.copy()
    report = {}
    for var in ALL_VARIABLES:
        col = f"d_{var}"
        series = panel[col].dropna()
        if len(series) < 20:
            report[var] = {"tested": False, "reason": "insufficient observations"}
            continue
        lb = acorr_ljungbox(series, lags=[5], return_df=True)
        pval = float(lb["lb_pvalue"].iloc[0])
        significant = pval < LJUNG_BOX_ALPHA
        report[var] = {"tested": True, "ljung_box_pvalue": pval, "ar1_filtered": significant}
        if significant:
            model = AutoReg(series.values, lags=1).fit()
            resid = pd.Series(model.resid, index=series.index[1:])
            filtered.loc[resid.index, col] = resid.values
            filtered.loc[series.index[0], col] = np.nan  # first obs has no AR(1) residual
    return filtered, report


def compute_omega(panel: pd.DataFrame, mask: pd.Series, x1: str, xj: str) -> np.ndarray:
    sub = panel.loc[mask, [f"d_{x1}", f"d_{xj}"]].dropna()
    return np.cov(sub[f"d_{x1}"], sub[f"d_{xj}"], ddof=1)


def run_filtering() -> tuple[pd.DataFrame, dict]:
    panel = pd.read_parquet(DATA_PROCESSED / "master_panel.parquet")
    filtered_panel, ar_report = check_and_filter_serial_correlation(panel)
    ar1_vars = [v for v, r in ar_report.items() if r.get("ar1_filtered")]
    print(f"Serial correlation: AR(1)-filtered {len(ar1_vars)}/{len(ALL_VARIABLES)} variables: {ar1_vars}")
    filtered_panel.to_parquet(DATA_PROCESSED / "master_panel_filtered.parquet", index=False)
    with open(DATA_PROCESSED / "ar1_report.json", "w") as f:
        json.dump(ar_report, f, indent=2, default=float)
    return filtered_panel, ar_report


def compute_omegas(normalizing_variable: str = NORMALIZING_VARIABLE) -> dict:
    filtered_panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")
    other_variables = [v for v in ALL_VARIABLES if v != normalizing_variable]

    h_mask = filtered_panel["is_H"]
    l_mask = filtered_panel["is_L"]
    print(f"[{normalizing_variable}] H days available for covariance: {h_mask.sum()}, L days: {l_mask.sum()}")

    results = {}
    for xj in other_variables:
        omega_H = compute_omega(filtered_panel, h_mask, normalizing_variable, xj)
        omega_L = compute_omega(filtered_panel, l_mask, normalizing_variable, xj)
        delta_omega = omega_H - omega_L
        results[xj] = {
            "omega_H": omega_H.tolist(), "omega_L": omega_L.tolist(), "delta_omega": delta_omega.tolist(),
        }

    out_path = DATA_PROCESSED / f"event_study_omegas_{normalizing_variable}.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "normalizing_variable": normalizing_variable}, f, indent=2, default=float)
    print(f"Saved -> {out_path}")
    return results


def main(normalizing_variable: str = NORMALIZING_VARIABLE):
    if not (DATA_PROCESSED / "master_panel_filtered.parquet").exists():
        run_filtering()
    return compute_omegas(normalizing_variable)


if __name__ == "__main__":
    run_filtering()
    compute_omegas(NORMALIZING_VARIABLE)
