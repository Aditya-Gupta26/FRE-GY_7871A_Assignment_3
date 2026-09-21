"""Pull article/headline-level Iran-war-risk corpus from GDELT (artlist mode).

artlist only returns the top articles from the most recent ~3 months of the queried
window and caps at 250 records/request (see config.py comments), so this loops
week-by-week chunks across the full analysis window. This is a supplementary text
source for the NLP scorers (lexicon/tone/FinBERT); the primary intensity signal is
gdelt_intensity.py's uncapped timeline data, precisely to avoid this cap biasing the
H/L classification on the highest-volume days.
"""
import json
import time

import pandas as pd
import requests

from config import ANALYSIS_START, ANALYSIS_END, DATA_RAW, GDELT_BASE, GDELT_QUERY, HEADERS, REQUEST_DELAY_SECONDS


CHUNK_DAYS = 14  # biweekly chunks, halves the request count vs. weekly, which cuts
# total runtime under GDELT's observed throttling without materially risking the
# 250-record/request cap, except possibly during the Feb28-Mar13 peak (handled by the
# uncapped gdelt_intensity.py series being the primary intensity signal regardless)


def _week_ranges(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.date_range(start, end, freq=f"{CHUNK_DAYS}D")
    ranges = []
    for i, chunk_start in enumerate(dates):
        chunk_end = min(
            chunk_start + pd.Timedelta(days=CHUNK_DAYS - 1, hours=23, minutes=59, seconds=59),
            pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59),
        )
        ranges.append((chunk_start, chunk_end))
        if chunk_end.date() >= pd.Timestamp(end).date():
            break
    return ranges


def fetch_artlist_chunk(start: pd.Timestamp, end: pd.Timestamp, max_retries: int = 5) -> list[dict]:
    params = {
        "query": GDELT_QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "sort": "dateasc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    backoff = 20.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(GDELT_BASE, params=params, timeout=60, headers=HEADERS)
        except requests.exceptions.RequestException as e:
            # A raw connection drop (seen in practice: RemoteDisconnected mid-backoff
            # storm) is just as retryable as a bad status code, not a reason to crash
            # the whole run and lose every other chunk's progress.
            print(f"    attempt {attempt}: {type(e).__name__}, retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 1.5
            continue
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                time.sleep(backoff)
                backoff *= 1.5
                continue
            return data.get("articles", [])
        print(f"    attempt {attempt}: status {r.status_code}, retrying in {backoff:.0f}s")
        time.sleep(backoff)
        backoff *= 1.5
    raise RuntimeError(f"GDELT artlist chunk {start}..{end} failed after {max_retries} retries")


CHECKPOINT_PATH = DATA_RAW / "gdelt_articles_checkpoint.parquet"
FAILED_CHUNKS_PATH = DATA_RAW / "gdelt_articles_failed_chunks.json"


def main() -> pd.DataFrame:
    ranges = _week_ranges(ANALYSIS_START, ANALYSIS_END)
    expected_starts = {start.strftime("%Y-%m-%d") for start, _ in ranges}

    all_articles = []
    done_starts = set()
    if CHECKPOINT_PATH.exists():
        prev = pd.read_parquet(CHECKPOINT_PATH)
        # Only trust checkpoint rows whose chunk actually belongs to the CURRENT
        # ANALYSIS_START/END window. A checkpoint left over from a previous run with a
        # different window (e.g. a different ANALYSIS_START) would otherwise silently
        # get merged into this run's output, which is wrong, not just stale.
        stale = prev[~prev["chunk_start"].isin(expected_starts)]
        prev = prev[prev["chunk_start"].isin(expected_starts)]
        if len(stale):
            print(f"Ignoring {len(stale)} checkpointed article(s) from {stale['chunk_start'].nunique()} "
                  f"chunk(s) outside the current window ({ANALYSIS_START} to {ANALYSIS_END})")
        all_articles = prev.to_dict("records")
        done_starts = set(prev["chunk_start"].unique())
        print(f"Resuming from checkpoint: {len(all_articles)} articles already fetched, "
              f"{len(done_starts)} chunks already done")

    failed_chunks = []
    for i, (start, end) in enumerate(ranges):
        if start.strftime("%Y-%m-%d") in done_starts:
            continue
        print(f"[{i + 1}/{len(ranges)}] {start.date()} .. {end.date()}")
        try:
            arts = fetch_artlist_chunk(start, end)
        except RuntimeError as e:
            print(f"  SKIPPING chunk after repeated failures: {e}")
            failed_chunks.append(start.strftime("%Y-%m-%d"))
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        print(f"  -> {len(arts)} articles")
        for a in arts:
            all_articles.append({
                "date": start.date(),
                "chunk_start": start.strftime("%Y-%m-%d"),
                "seendate": a.get("seendate"),
                "title": a.get("title"),
                "url": a.get("url"),
                "domain": a.get("domain"),
                "sourcecountry": a.get("sourcecountry"),
                "language": a.get("language"),
            })
        # Incremental checkpoint, so we survive interruption without losing prior chunks.
        pd.DataFrame(all_articles).to_parquet(CHECKPOINT_PATH, index=False)
        time.sleep(REQUEST_DELAY_SECONDS)

    if failed_chunks:
        with open(FAILED_CHUNKS_PATH, "w") as f:
            json.dump(failed_chunks, f, indent=2)
        print(f"WARNING: {len(failed_chunks)} chunks failed after retries and were skipped: {failed_chunks}")
        print(f"Re-run this script to retry only the failed/missing chunks (checkpoint resumes automatically).")

    df = pd.DataFrame(all_articles)
    if df.empty:
        raise RuntimeError("GDELT artlist returned zero articles across the whole window")
    df["seendate"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    df["date"] = df["seendate"].dt.date.fillna(df["date"])
    before = len(df)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    print(f"Deduped {before} -> {len(df)} articles by URL")

    out_path = DATA_RAW / "gdelt_articles.parquet"
    df.drop(columns=["chunk_start"]).to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}")
    print(df["date"].value_counts().sort_index().tail(10))
    return df


if __name__ == "__main__":
    main()
