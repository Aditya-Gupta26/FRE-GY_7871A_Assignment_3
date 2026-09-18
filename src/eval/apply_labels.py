"""Apply Claude-judged relevance labels to the sampled evaluation set.

`LABELS` below is filled in by reading `data/processed/eval_sample_unlabeled.parquet`
(produced by build_eval_labels.py) and judging each headline for Iran-war-risk
relevance -- disclosed in AI_USE.md as an LLM-as-judge evaluation, not independent
human annotation. Keyed by URL for robustness to row-order changes.

Note: the actual `data/processed/eval_labels.parquet` used in this run was produced by
a one-off inline script that applied the same 176 judgments in date-sorted order (see
the conversation/AI_USE.md) rather than by filling in this file's LABELS dict by hand --
both produce the identical output schema (url, title, relevant_gold). This module is
kept as the reusable, documented entry point for re-labeling a future sample.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import DATA_PROCESSED

# url -> True/False, filled in after reviewing eval_sample_unlabeled.parquet.
LABELS: dict[str, bool] = {}


def main() -> pd.DataFrame:
    sample = pd.read_parquet(DATA_PROCESSED / "eval_sample_unlabeled.parquet")
    if not LABELS:
        raise RuntimeError("LABELS dict is empty -- fill it in from the unlabeled sample first")
    sample["relevant_gold"] = sample["url"].map(LABELS)
    missing = sample["relevant_gold"].isna().sum()
    if missing:
        print(f"WARNING: {missing}/{len(sample)} rows have no label -- dropping them")
        sample = sample.dropna(subset=["relevant_gold"])
    sample["relevant_gold"] = sample["relevant_gold"].astype(bool)

    out_path = DATA_PROCESSED / "eval_labels.parquet"
    sample[["url", "title", "relevant_gold"]].to_parquet(out_path, index=False)
    print(f"Saved {len(sample)} labeled rows -> {out_path}")
    return sample


if __name__ == "__main__":
    main()
