"""Novel-phrasing analysis: look at the lexicon relevance filter's false negatives
(articles Claude judged genuinely relevant, that the filter said were not relevant) and
find what vocabulary is actually causing the misses.

Two buckets, kept separate on purpose so we don't just re-discover the already-known
"missing the literal word iran" limitation and call it new:
  - Bucket 1: titles with no "iran" token at all. This is the already-documented
    structural gate failure (see Limitations, e.g. the Beirut-strikes headline example).
  - Bucket 2: titles that DO contain "iran" but still have no RELEVANCE_TERMS hit. This
    is where genuinely missed vocabulary lives, and is what this script is really for.

The term-frequency counting here is fully automatic. Tagging each top term as "novel
2026-war-specific" vs. "generic term the narrow list happened to skip" is a manual
Claude-judged pass done afterward on this script's output, the same disclosed
LLM-as-judge pattern already used for the eval relevance labels themselves (see
AI_USE.md), not something this script tries to automate.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import DATA_PROCESSED
from lexicon import DEESCALATION_TERMS, ESCALATION_TERMS, RELEVANCE_TERMS, _tokenize

TOP_N = 20
KNOWN_TERMS = set(RELEVANCE_TERMS) | set(ESCALATION_TERMS) | set(DEESCALATION_TERMS) | {"iran"}


def main() -> pd.DataFrame:
    labels = pd.read_parquet(DATA_PROCESSED / "eval_labels.parquet")
    labels["tokens"] = labels["title"].fillna("").apply(_tokenize)
    labels["has_iran"] = labels["tokens"].apply(lambda t: "iran" in t)
    labels["has_relevance_term"] = labels["tokens"].apply(lambda t: any(w in RELEVANCE_TERMS for w in t))
    labels["predicted_relevant"] = labels["has_iran"] & labels["has_relevance_term"]

    false_negatives = labels[(~labels["predicted_relevant"]) & (labels["relevant_gold"])]
    bucket1 = false_negatives[~false_negatives["has_iran"]]
    bucket2 = false_negatives[false_negatives["has_iran"]]

    print(f"False negatives: {len(false_negatives)} total "
          f"({len(bucket1)} missing 'iran' entirely, {len(bucket2)} have 'iran' but no relevance term)")

    counts = Counter()
    examples: dict[str, str] = {}
    for _, row in bucket2.iterrows():
        for tok in row["tokens"]:
            if tok in KNOWN_TERMS or tok in ENGLISH_STOP_WORDS or len(tok) < 3:
                continue
            counts[tok] += 1
            examples.setdefault(tok, row["title"])

    top_terms = pd.DataFrame(
        [{"term": term, "count": n, "example_headline": examples[term]} for term, n in counts.most_common(TOP_N)]
    )
    print(f"\nTop {len(top_terms)} missed terms (bucket 2, 'iran' present but no relevance-term hit):")
    print(top_terms.to_string(index=False))

    out_path = DATA_PROCESSED / "false_negative_terms.parquet"
    top_terms.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    summary = {
        "n_false_negatives": len(false_negatives),
        "n_bucket1_no_iran_token": len(bucket1),
        "n_bucket2_has_iran_no_relevance_term": len(bucket2),
    }
    return top_terms, summary


if __name__ == "__main__":
    main()
