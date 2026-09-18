"""Merge market data + composite intensity + regime labels into one analysis-ready panel.

Trading-calendar alignment (documented convention, not silently ignored): GDELT
timestamps are UTC and news runs 7 days/week; financial data only exists on trading
days. News on a weekend or holiday is attributed to the *next* trading day, on the
assumption that markets price in accumulated news at their next open/close. This is an
approximation -- perfect alignment across US/Gulf/Israeli market hours isn't achievable
with daily data -- and is flagged explicitly in the report's limitations section.
"""
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, FRED_SERIES, YFINANCE_TICKERS

ALL_VARIABLES = list(YFINANCE_TICKERS.keys()) + list(FRED_SERIES.keys())


def main() -> pd.DataFrame:
    market = pd.read_parquet(DATA_RAW / "market_raw.parquet")
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").set_index("date")

    intensity = pd.read_parquet(DATA_PROCESSED / "daily_intensity.parquet")
    intensity["date"] = pd.to_datetime(intensity["date"])

    regimes = pd.read_parquet(DATA_PROCESSED / "regime_labels.parquet")
    regimes["date"] = pd.to_datetime(regimes["date"])

    # Trading-calendar alignment: roll news dates that fall on a non-trading day forward
    # to the next date present in the market index.
    trading_dates = pd.Series(market.index.unique()).sort_values().reset_index(drop=True)

    def roll_forward(d: pd.Timestamp) -> pd.Timestamp:
        idx = trading_dates.searchsorted(d)
        if idx >= len(trading_dates):
            return pd.NaT
        return trading_dates.iloc[idx]

    intensity["trading_date"] = intensity["date"].apply(roll_forward)
    regimes["trading_date"] = regimes["date"].apply(roll_forward)

    # Where multiple calendar days roll onto the same trading day (weekend pileup),
    # aggregate: intensity sums (news accumulates), H/L flags via "any" (a trading day
    # that absorbs ANY H-flagged calendar day is treated as H).
    intensity_agg = intensity.groupby("trading_date").agg(
        composite_intensity=("composite_intensity", "sum"),
        avg_tone=("avg_tone", "mean"),
    ).reset_index().rename(columns={"trading_date": "date"})

    regimes_agg = regimes.groupby("trading_date").agg(
        is_H=("is_H", "any"),
        is_L=("is_L", "any"),
        confound_flag=("confound_flag", "any"),
        direction=("direction", lambda s: s.mode().iat[0] if not s.mode().empty else "Unclear"),
    ).reset_index().rename(columns={"trading_date": "date"})

    panel = market.reset_index().merge(intensity_agg, on="date", how="left").merge(
        regimes_agg, on="date", how="left"
    )
    panel["is_H"] = panel["is_H"].fillna(False)
    panel["is_L"] = panel["is_L"].fillna(False)
    panel["confound_flag"] = panel["confound_flag"].fillna(False)

    # First differences (Δx) for every financial variable -- the quantity the
    # heteroskedasticity estimator actually operates on.
    for var in ALL_VARIABLES:
        panel[f"d_{var}"] = panel[var].diff()

    out_path = DATA_PROCESSED / "master_panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Saved master panel: {panel.shape[0]} rows x {panel.shape[1]} cols -> {out_path}")
    print(f"H days in panel: {int(panel['is_H'].sum())}, L days: {int(panel['is_L'].sum())}")
    return panel


if __name__ == "__main__":
    main()
