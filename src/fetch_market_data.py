"""Pull Phase-A financial variables: yfinance tickers + FRED series (public CSV endpoint,
no API key required) over the analysis window, plus a short lead so first-differencing
doesn't lose the first observation.
"""
import time

import pandas as pd
import yfinance as yf
import requests
from io import StringIO

from config import ANALYSIS_START, ANALYSIS_END, DATA_RAW, FRED_SERIES, HEADERS, YFINANCE_TICKERS

# Extra lead days before ANALYSIS_START so the first real day's Δx isn't dropped.
LEAD_START = "2026-02-20"


def fetch_fred_series(series_id: str) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={LEAD_START}&coed={ANALYSIS_END}"
    )
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")  # FRED uses "." for missing
    return df.set_index("date")[series_id]


def fetch_yfinance_ticker(ticker: str) -> pd.Series:
    df = yf.download(ticker, start=LEAD_START, end=ANALYSIS_END, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index.name = "date"
    return close.rename(ticker)


def main() -> pd.DataFrame:
    series = {}

    for name, ticker in YFINANCE_TICKERS.items():
        print(f"Fetching yfinance {name} ({ticker})...")
        series[name] = fetch_yfinance_ticker(ticker)

    for name, fred_id in FRED_SERIES.items():
        print(f"Fetching FRED {name} ({fred_id})...")
        series[name] = fetch_fred_series(fred_id)
        time.sleep(0.5)

    out = pd.concat(series.values(), axis=1)
    out.columns = list(series.keys())
    out.index.name = "date"
    out = out.sort_index()
    out = out[out.index >= pd.Timestamp(LEAD_START)]

    out_path = DATA_RAW / "market_raw.parquet"
    out.reset_index().to_parquet(out_path, index=False)
    print(f"Saved {out.shape[0]} rows x {out.shape[1]} variables -> {out_path}")
    print(out.tail())
    return out


if __name__ == "__main__":
    main()
