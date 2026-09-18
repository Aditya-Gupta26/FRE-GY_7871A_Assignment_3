"""Best-effort full-text fetch for a capped subsample of GDELT articles.

High expected failure rate (paywalls, anti-bot, dead links) is normal and handled
gracefully -- coverage % is logged and reported rather than treated as an error.
Respects robots.txt. This is a secondary signal (tone_finbert.py falls back to
title-only scoring where text isn't available); do not block the pipeline on low
coverage here.
"""
import time
import urllib.robotparser as robotparser
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import DATA_RAW, HEADERS, REQUEST_DELAY_SECONDS

MAX_ARTICLES_PER_DAY = 15  # cap: prioritize breadth across days over exhaustive per-day coverage
FETCH_TIMEOUT = 12
_robots_cache: dict[str, robotparser.RobotFileParser] = {}


def _allowed_by_robots(url: str) -> bool:
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in _robots_cache:
            rp = robotparser.RobotFileParser()
            try:
                # RobotFileParser.read() uses urllib with no timeout and can hang
                # indefinitely on a slow/unresponsive server -- fetch via requests
                # (which has a real timeout) and feed the text in via parse() instead.
                r = requests.get(base + "/robots.txt", headers=HEADERS, timeout=5)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp = None  # if robots.txt itself can't be fetched, don't block on it
            _robots_cache[base] = rp
        rp = _robots_cache[base]
        if rp is None:
            return True
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True


def fetch_one(url: str) -> dict:
    if not _allowed_by_robots(url):
        return {"url": url, "text": None, "fetch_status": "robots_disallowed"}
    try:
        r = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT)
        if r.status_code != 200:
            return {"url": url, "text": None, "fetch_status": f"http_{r.status_code}"}
        soup = BeautifulSoup(r.content, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        if len(text) < 200:
            return {"url": url, "text": text or None, "fetch_status": "too_short"}
        return {"url": url, "text": text[:5000], "fetch_status": "ok"}
    except requests.exceptions.RequestException as e:
        return {"url": url, "text": None, "fetch_status": f"error_{type(e).__name__}"}


def main() -> pd.DataFrame:
    articles = pd.read_parquet(DATA_RAW / "gdelt_articles.parquet")
    subsample = (
        articles.groupby("date", group_keys=False)
        .apply(lambda g: g.head(MAX_ARTICLES_PER_DAY))
        .reset_index(drop=True)
    )
    print(f"Scraping {len(subsample)}/{len(articles)} articles "
          f"(capped at {MAX_ARTICLES_PER_DAY}/day)...")

    results = []
    for i, url in enumerate(subsample["url"]):
        results.append(fetch_one(url))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(subsample)} done")
        time.sleep(0.3)  # gentler than GDELT's limit; per-domain politeness

    result_df = pd.DataFrame(results)
    coverage = (result_df["fetch_status"] == "ok").mean()
    print(f"Full-text fetch coverage: {coverage:.1%} ({(result_df['fetch_status'] == 'ok').sum()}/{len(result_df)})")
    print(result_df["fetch_status"].value_counts())

    out_path = DATA_RAW / "article_text.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")
    return result_df


if __name__ == "__main__":
    main()
