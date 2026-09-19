"""Pull the primary, uncapped Iran-war-risk news intensity signal from GDELT.

Uses GDELT DOC 2.0 API timeline modes (timelinevol, timelinevolraw, timelinetone),
which return a full daily series in one call for the whole analysis window and are
NOT subject to the 250-record/artlist cap. That's why they're used as the primary
intensity signal (see config.py comments and the plan's methodological-rigor notes).
"""
import json
import time

import pandas as pd
import requests

from config import ANALYSIS_START, ANALYSIS_END, DATA_RAW, GDELT_BASE, GDELT_QUERY, HEADERS, REQUEST_DELAY_SECONDS

TIMELINE_MODES = ["timelinevol", "timelinevolraw", "timelinetone"]


def _to_gdelt_dt(date_str: str, end_of_day: bool = False) -> str:
    suffix = "235959" if end_of_day else "000000"
    return date_str.replace("-", "") + suffix


def fetch_timeline(mode: str, max_retries: int = 6) -> list[dict]:
    params = {
        "query": GDELT_QUERY,
        "mode": mode,
        "format": "json",
        "startdatetime": _to_gdelt_dt(ANALYSIS_START),
        "enddatetime": _to_gdelt_dt(ANALYSIS_END, end_of_day=True),
    }
    backoff = 20.0
    for attempt in range(1, max_retries + 1):
        r = requests.get(GDELT_BASE, params=params, timeout=60, headers=HEADERS)
        if r.status_code == 200:
            try:
                data = r.json()
            except json.JSONDecodeError:
                print(f"  [{mode}] attempt {attempt}: 200 but non-JSON body, retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 1.5
                continue
            return data["timeline"][0]["data"]
        print(f"  [{mode}] attempt {attempt}: status {r.status_code}, retrying in {backoff:.0f}s")
        time.sleep(backoff)
        backoff *= 1.5
    raise RuntimeError(f"GDELT {mode} query failed after {max_retries} retries")


def main() -> pd.DataFrame:
    frames = {}
    for i, mode in enumerate(TIMELINE_MODES):
        print(f"Fetching GDELT {mode} ({ANALYSIS_START} to {ANALYSIS_END})...")
        series = fetch_timeline(mode)
        df = pd.DataFrame(series)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ").dt.date
        frames[mode] = df.set_index("date")["value"].rename(mode)
        print(f"  -> {len(df)} days, max={df['value'].max():.3f} on "
              f"{df.loc[df['value'].idxmax(), 'date'] if 'date' in df else '?'}")
        if i < len(TIMELINE_MODES) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    out = pd.concat(frames.values(), axis=1).reset_index().rename(
        columns={"timelinevol": "volume_pct", "timelinevolraw": "volume_raw", "timelinetone": "avg_tone"}
    )
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    out_path = DATA_RAW / "gdelt_daily_timeline.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved {len(out)} rows -> {out_path}")
    return out


if __name__ == "__main__":
    main()
