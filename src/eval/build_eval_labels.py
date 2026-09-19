"""Sample a reference set of articles for the NLP relevance-filter evaluation.

This just does the sampling. The actual relevance judgments are produced separately
(see apply_labels.py) by Claude reading each sampled headline, disclosed in AI_USE.md
as such rather than presented as independent human annotation. This two-step split
keeps the sampling procedure itself fully reproducible/scriptable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import DATA_PROCESSED, DATA_RAW

SAMPLE_SIZE = 180
RANDOM_SEED = 42


def main() -> pd.DataFrame:
    articles = pd.read_parquet(DATA_RAW / "gdelt_articles.parquet")
    # Stratify by date-week so the eval set spans the whole war timeline, not just the
    # highest-volume weeks.
    articles["date"] = pd.to_datetime(articles["date"])
    articles["week"] = articles["date"].dt.to_period("W")
    n_weeks = articles["week"].nunique()
    per_week = max(1, SAMPLE_SIZE // n_weeks)

    sample = (
        articles.groupby("week", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_week, len(g)), random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )
    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    out_path = DATA_PROCESSED / "eval_sample_unlabeled.parquet"
    sample[["date", "title", "url", "domain"]].to_parquet(out_path, index=False)
    print(f"Sampled {len(sample)} articles across {n_weeks} weeks -> {out_path}")
    return sample


if __name__ == "__main__":
    main()
