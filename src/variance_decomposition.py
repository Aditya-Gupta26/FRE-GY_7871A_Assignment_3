"""Table-3-equivalent: how much of each variable's variance is attributable to Iran
war risk, on H-days and over the full analysis window, following Rigobon & Sack's
lower-bound formula d_j1^2 * ΔVar(x1) / Var(xj). Runs per normalizing variable (see
config.NORMALIZING_VARIABLES_TO_RUN)."""
import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, FRED_SERIES, NORMALIZING_VARIABLE, NORMALIZING_VARIABLES_TO_RUN, YFINANCE_TICKERS

ALL_VARIABLES = list(YFINANCE_TICKERS.keys()) + list(FRED_SERIES.keys())


def main(normalizing_variable: str = NORMALIZING_VARIABLE) -> pd.DataFrame:
    other_variables = [v for v in ALL_VARIABLES if v != normalizing_variable]
    panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")
    table2 = pd.read_parquet(DATA_PROCESSED / f"table2_estimates_{normalizing_variable}.parquet").set_index("variable")

    h_mask, l_mask = panel["is_H"], panel["is_L"]
    var_x1_H = panel.loc[h_mask, f"d_{normalizing_variable}"].var(ddof=1)
    var_x1_L = panel.loc[l_mask, f"d_{normalizing_variable}"].var(ddof=1)
    delta_var_x1 = var_x1_H - var_x1_L

    rows = []
    for xj in other_variables:
        d_hat = table2.loc[xj, "iv_w1_coef"]
        col = f"d_{xj}"
        var_L_j = panel.loc[l_mask, col].var(ddof=1)
        var_H_j = panel.loc[h_mask, col].var(ddof=1)
        var_all_j = panel[col].var(ddof=1)

        predicted_delta_var = (d_hat ** 2) * delta_var_x1 if pd.notna(d_hat) else np.nan
        pct_explained_H = predicted_delta_var / var_H_j if var_H_j else np.nan
        pct_explained_all = predicted_delta_var / var_all_j if var_all_j else np.nan

        rows.append({
            "variable": xj, "var_L": var_L_j, "var_H": var_H_j,
            "predicted_delta_var": predicted_delta_var,
            "pct_explained_H_days": pct_explained_H, "pct_explained_all_days": pct_explained_all,
        })

    table3 = pd.DataFrame(rows)
    print(f"[{normalizing_variable}]")
    print(table3.to_string(index=False))
    # Sanity flag: this is a lower-bound formula (Rigobon & Sack's own framing), so a
    # value above 100% signals estimation noise (small H/L samples, a near-zero
    # denominator, or -- as observed for the treasury_2y run on the GDELT-timeline-only
    # partial signal -- a weak-instrument symptom) rather than a literal reading.
    over_100 = table3[(table3["pct_explained_H_days"].abs() > 1) | (table3["pct_explained_all_days"].abs() > 1)]
    if not over_100.empty:
        print(f"NOTE: {len(over_100)} variable(s) show |>100%| variance explained "
              f"(estimation noise / weak identification, not a literal reading): {over_100['variable'].tolist()}")
    out_path = DATA_PROCESSED / f"table3_variance_decomp_{normalizing_variable}.parquet"
    table3.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")
    return table3


if __name__ == "__main__":
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        main(nv)
