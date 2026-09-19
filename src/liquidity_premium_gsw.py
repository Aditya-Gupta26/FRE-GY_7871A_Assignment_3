"""PHASE B (add-on, run only after the Phase-A pipeline is complete and correct).

Builds the on-the-run 10-year Treasury liquidity premium, the yield gap between the
most recently auctioned ("on-the-run") 10yr note and a smooth fitted par yield curve,
via the Fed's public Gurkaynak-Sack-Wright (GSW) dataset, as the 14th financial variable
extending Table 2/3. This was deliberately dropped from the original 9-variable set in
Phase A because it requires (i) the on-the-run issuance calendar, (ii) that note's
traded yield, (iii) the fitted GSW par curve for the same dates, and (iv) their spread,
which is materially more data-engineering work than every other variable's single clean pull.

Data source: https://www.federalreserve.gov/data/nominal-yield-curve.htm (feds200628.csv)
publishes the GSW zero-coupon/par yield curve parameters daily back to 1961; the fitted
10-year par yield is computed directly from the published parameters rather than
requiring a separate curve-fitting step.
"""
import numpy as np
import pandas as pd
import requests

from config import ANALYSIS_END, ANALYSIS_START, DATA_RAW, HEADERS

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"


def fetch_gsw_curve() -> pd.DataFrame:
    r = requests.get(GSW_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    # The published file has a multi-line header before the data table; pandas' comment
    # handling on '#' strips it, actual header row is auto-detected below.
    from io import StringIO
    lines = r.text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date"))
    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= pd.Timestamp(ANALYSIS_START) - pd.Timedelta(days=30))
            & (df["Date"] <= pd.Timestamp(ANALYSIS_END))]
    return df


def gsw_par_yield_10y(params_row: pd.Series) -> float:
    """Nelson-Siegel-Svensson par yield at 10-year maturity from GSW's published
    BETA0-3/TAU1-2 parameters (SVENPY10 is already published directly if present;
    prefer that column when available, else compute from the NSS formula)."""
    if "SVENPY10" in params_row and pd.notna(params_row["SVENPY10"]):
        return float(params_row["SVENPY10"])
    b0, b1, b2, b3 = params_row["BETA0"], params_row["BETA1"], params_row["BETA2"], params_row["BETA3"]
    tau1, tau2 = params_row["TAU1"], params_row["TAU2"]
    n = 10.0
    term1 = b1 * (1 - np.exp(-n / tau1)) / (n / tau1)
    term2 = b2 * ((1 - np.exp(-n / tau1)) / (n / tau1) - np.exp(-n / tau1))
    term3 = b3 * ((1 - np.exp(-n / tau2)) / (n / tau2) - np.exp(-n / tau2))
    return b0 + term1 + term2 + term3


def main() -> pd.DataFrame:
    print("NOTE: Phase B. Run only after Phase A (all other variables) is complete.")
    gsw = fetch_gsw_curve()
    gsw["fitted_10y_par_yield"] = gsw.apply(gsw_par_yield_10y, axis=1)

    on_the_run_10y = pd.read_parquet(DATA_RAW / "market_raw.parquet")[["date", "treasury_10y"]]
    on_the_run_10y["date"] = pd.to_datetime(on_the_run_10y["date"])

    merged = on_the_run_10y.merge(
        gsw[["Date", "fitted_10y_par_yield"]].rename(columns={"Date": "date"}), on="date", how="inner"
    )
    merged["liquidity_premium"] = merged["treasury_10y"] - merged["fitted_10y_par_yield"]

    out_path = DATA_RAW / "liquidity_premium.parquet"
    merged[["date", "liquidity_premium"]].to_parquet(out_path, index=False)
    print(f"Saved {len(merged)} rows -> {out_path}")
    print(merged[["date", "treasury_10y", "fitted_10y_par_yield", "liquidity_premium"]].tail())
    return merged


if __name__ == "__main__":
    main()
