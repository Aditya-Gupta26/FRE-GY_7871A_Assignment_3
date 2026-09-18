"""Core heteroskedasticity-based estimator: eq.(6)/(7) direct computation from ΔΩ, plus
the equivalent IV/2SLS implementation (eq. 8-12) with ω1, ω2, and a genuine two-
instrument ω3=[ω1,ω2] specification (Sargan/J overidentification test included).
Analytic (IV2SLS) and bootstrap (H/L-stratified resampling) standard errors are both
reported.

Runs per normalizing variable (see config.NORMALIZING_VARIABLES_TO_RUN) -- the primary
choice (2-year Treasury yield, matching Rigobon & Sack 2003) and a parallel Brent-crude
experiment, motivated by the war's oil-supply-shock transmission channel. Neither run
overwrites the other; outputs are saved with a per-variable suffix.

Display scale: chosen empirically from the sign/magnitude of ΔΩ11 (the normalizing
variable's own variance shift), not presumed in advance.
"""
import json

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

from config import DATA_PROCESSED, FRED_SERIES, NORMALIZING_VARIABLE, NORMALIZING_VARIABLES_TO_RUN, YFINANCE_TICKERS

ALL_VARIABLES = list(YFINANCE_TICKERS.keys()) + list(FRED_SERIES.keys())
N_BOOTSTRAP = 500
RNG_SEED = 42


def direct_estimators(omega_H: np.ndarray, omega_L: np.ndarray) -> dict:
    delta_omega = omega_H - omega_L
    d11, d12, d22 = delta_omega[0, 0], delta_omega[0, 1], delta_omega[1, 1]
    eq6 = d22 / d12 if d12 != 0 else np.nan   # ΔΩ22 / ΔΩ21
    eq7 = d12 / d11 if d11 != 0 else np.nan   # ΔΩ21 / ΔΩ11
    return {"eq6": eq6, "eq7": eq7, "delta_omega_11": d11, "delta_omega_12": d12, "delta_omega_22": d22}


def build_instruments(panel: pd.DataFrame, x1: str, xj: str) -> pd.DataFrame:
    mask = panel["is_H"] | panel["is_L"]
    sub = panel.loc[mask, ["is_H", "is_L", f"d_{x1}", f"d_{xj}"]].dropna().copy()
    sign = np.where(sub["is_H"], 1.0, -1.0)
    sub["omega1"] = sign * sub[f"d_{x1}"]
    sub["omega2"] = sign * sub[f"d_{xj}"]
    sub["const"] = 1.0
    return sub


def iv_estimate(sub: pd.DataFrame, x1: str, xj: str, instruments: list[str]) -> dict:
    endog = sub[f"d_{xj}"]
    exog = sub[["const"]]
    endog_reg = sub[[f"d_{x1}"]]
    instr = sub[instruments]
    model = IV2SLS(dependent=endog, exog=exog, endog=endog_reg, instruments=instr)
    fit = model.fit(cov_type="robust")
    result = {
        "coef": float(fit.params[f"d_{x1}"]),
        "se": float(fit.std_errors[f"d_{x1}"]),
        "tstat": float(fit.tstats[f"d_{x1}"]),
        "pvalue": float(fit.pvalues[f"d_{x1}"]),
        "n_obs": int(fit.nobs),
    }
    if len(instruments) > 1:  # overidentified -- Sargan/J test available
        try:
            sargan = fit.sargan
            result["sargan_stat"] = float(sargan.stat)
            result["sargan_pvalue"] = float(sargan.pval)
        except Exception as e:
            result["sargan_error"] = str(e)
    return result


def bootstrap_ci(panel: pd.DataFrame, x1: str, xj: str, n_boot: int = N_BOOTSTRAP) -> dict:
    """Resample H-days and L-days separately (stratified), recompute eq(7)-equivalent
    (ω1 IV) each time -- matches Rigobon (2003)'s bootstrap approach for GMM SEs."""
    rng = np.random.default_rng(RNG_SEED)
    h_df = panel[panel["is_H"]][[f"d_{x1}", f"d_{xj}"]].dropna()
    l_df = panel[panel["is_L"]][[f"d_{x1}", f"d_{xj}"]].dropna()
    if len(h_df) < 3 or len(l_df) < 3:
        return {"status": "insufficient_data"}

    estimates = []
    for _ in range(n_boot):
        h_samp = h_df.sample(n=len(h_df), replace=True, random_state=rng.integers(1e9))
        l_samp = l_df.sample(n=len(l_df), replace=True, random_state=rng.integers(1e9))
        cov_h = np.cov(h_samp[f"d_{x1}"], h_samp[f"d_{xj}"], ddof=1)
        cov_l = np.cov(l_samp[f"d_{x1}"], l_samp[f"d_{xj}"], ddof=1)
        denom = cov_h[0, 0] - cov_l[0, 0]
        if denom == 0:
            continue
        estimates.append((cov_h[0, 1] - cov_l[0, 1]) / denom)

    estimates = np.array(estimates)
    if len(estimates) == 0:
        return {"status": "degenerate_bootstrap"}
    return {
        "status": "ok", "n_boot": len(estimates),
        "mean": float(np.mean(estimates)), "se": float(np.std(estimates, ddof=1)),
        "ci_2.5": float(np.percentile(estimates, 2.5)), "ci_97.5": float(np.percentile(estimates, 97.5)),
    }


def main(normalizing_variable: str = NORMALIZING_VARIABLE) -> pd.DataFrame:
    panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")
    other_variables = [v for v in ALL_VARIABLES if v != normalizing_variable]
    with open(DATA_PROCESSED / f"event_study_omegas_{normalizing_variable}.json") as f:
        omegas = json.load(f)["results"]

    first_delta11 = omegas[other_variables[0]]["delta_omega"][0][0]
    shock_note = ("positive" if first_delta11 > 0 else "negative") + f" ΔΩ11 observed for {normalizing_variable}"

    rows = []
    for xj in other_variables:
        omega_H = np.array(omegas[xj]["omega_H"])
        omega_L = np.array(omegas[xj]["omega_L"])
        direct = direct_estimators(omega_H, omega_L)

        sub = build_instruments(panel, normalizing_variable, xj)
        try:
            iv_w1 = iv_estimate(sub, normalizing_variable, xj, ["omega1"])
        except Exception as e:
            iv_w1 = {"error": str(e)}
        try:
            iv_w2 = iv_estimate(sub, normalizing_variable, xj, ["omega2"])
        except Exception as e:
            iv_w2 = {"error": str(e)}
        try:
            iv_w3 = iv_estimate(sub, normalizing_variable, xj, ["omega1", "omega2"])
        except Exception as e:
            iv_w3 = {"error": str(e)}

        boot = bootstrap_ci(panel, normalizing_variable, xj)

        rows.append({
            "variable": xj,
            "eq6_direct": direct["eq6"], "eq7_direct": direct["eq7"],
            "iv_w1_coef": iv_w1.get("coef"), "iv_w1_tstat": iv_w1.get("tstat"), "iv_w1_pvalue": iv_w1.get("pvalue"),
            "iv_w2_coef": iv_w2.get("coef"), "iv_w2_tstat": iv_w2.get("tstat"), "iv_w2_pvalue": iv_w2.get("pvalue"),
            "iv_w3_coef": iv_w3.get("coef"), "iv_w3_tstat": iv_w3.get("tstat"), "iv_w3_pvalue": iv_w3.get("pvalue"),
            "iv_w3_sargan_stat": iv_w3.get("sargan_stat"), "iv_w3_sargan_pvalue": iv_w3.get("sargan_pvalue"),
            "bootstrap_mean": boot.get("mean"), "bootstrap_se": boot.get("se"),
            "bootstrap_ci_low": boot.get("ci_2.5"), "bootstrap_ci_high": boot.get("ci_97.5"),
            "n_obs": iv_w1.get("n_obs"),
        })

    table2 = pd.DataFrame(rows)
    print(f"[{normalizing_variable}] Shock-sign note: {shock_note}")
    print(table2[["variable", "eq7_direct", "iv_w1_coef", "iv_w1_tstat", "iv_w3_sargan_pvalue"]].to_string(index=False))

    out_path = DATA_PROCESSED / f"table2_estimates_{normalizing_variable}.parquet"
    table2.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")
    return table2


if __name__ == "__main__":
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        main(nv)
