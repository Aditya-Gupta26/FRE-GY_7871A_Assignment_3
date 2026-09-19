"""NLP scoring method 1: hand-built Iran-war-risk escalation lexicon.

Scores each article title (+ text where available) for:
  - relevance: mentions Iran AND at least one war-risk topic term
  - intensity: weighted count of escalation-related terms (magnitude only)
  - direction: net polarity (escalation-positive minus de-escalation-negative terms),
    used only for the descriptive "direction" column in the Table-1 equivalent.
    The econometric method itself only uses variance/intensity, never direction.
"""
import re

import pandas as pd

from config import DATA_PROCESSED, DATA_RAW

# Broad relevance gate: title must contain "iran" plus at least one of these.
RELEVANCE_TERMS = [
    "war", "strike", "missile", "hormuz", "nuclear", "iaea", "ceasefire",
    "blockade", "sanction", "irgc", "khamenei", "tanker", "natanz",
    "military", "attack", "bomb", "invasion",
]

# Escalation terms (positive weight = higher perceived war-risk intensity).
ESCALATION_TERMS = {
    "strike": 2.0, "strikes": 2.0, "airstrike": 2.5, "missile": 2.0, "missiles": 2.0,
    "killed": 2.0, "casualties": 2.0, "attack": 1.5, "attacks": 1.5, "bombed": 2.0,
    "bombing": 2.0, "invasion": 2.5, "blockade": 2.0, "closure": 1.5, "closed": 1.0,
    "mobilization": 1.5, "mobilize": 1.5, "retaliate": 2.0, "retaliation": 2.0,
    "escalate": 1.5, "escalation": 1.5, "collapse": 1.5, "collapsed": 1.5,
    "sanctions": 1.0, "warship": 1.5, "warships": 1.5, "mines": 2.0, "mine-laying": 2.0,
    "casualty": 2.0, "wounded": 1.5, "dead": 1.5, "explosion": 2.0, "shelling": 2.0,
}

# De-escalation terms (negative weight, used for direction, not intensity magnitude).
DEESCALATION_TERMS = {
    "ceasefire": -1.5, "truce": -1.5, "peace": -1.5, "deal": -1.0, "agreement": -1.5,
    "talks": -0.5, "negotiation": -1.0, "negotiations": -1.0, "de-escalate": -2.0,
    "de-escalation": -2.0, "reopen": -1.5, "reopened": -1.5, "withdraw": -1.0,
    "withdrawal": -1.0, "diplomatic": -0.5, "resolution": -1.0,
}

_word_re = re.compile(r"[a-z]+(?:-[a-z]+)?")


def _tokenize(text: str) -> list[str]:
    return _word_re.findall(text.lower())


def score_text(text: str) -> dict:
    tokens = _tokenize(text or "")
    token_set = set(tokens)
    relevant = "iran" in token_set and any(t in token_set for t in RELEVANCE_TERMS)
    intensity = sum(ESCALATION_TERMS.get(tok, 0.0) for tok in tokens)
    direction = intensity + sum(DEESCALATION_TERMS.get(tok, 0.0) for tok in tokens)
    return {"relevant": relevant, "intensity": intensity, "direction": direction}


def score_articles(articles: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    scored = articles[text_col].fillna("").apply(score_text).apply(pd.Series)
    return pd.concat([articles.reset_index(drop=True), scored], axis=1)


def build_daily_lexicon_score() -> pd.DataFrame:
    articles = pd.read_parquet(DATA_RAW / "gdelt_articles.parquet")
    scored = score_articles(articles, text_col="title")
    relevant = scored[scored["relevant"]]
    daily = relevant.groupby("date").agg(
        lexicon_article_count=("intensity", "size"),
        lexicon_intensity_sum=("intensity", "sum"),
        lexicon_intensity_mean=("intensity", "mean"),
        lexicon_direction_mean=("direction", "mean"),
    ).reset_index()
    out_path = DATA_PROCESSED / "lexicon_daily_score.parquet"
    daily.to_parquet(out_path, index=False)
    print(f"Lexicon relevance: {len(relevant)}/{len(scored)} articles flagged relevant "
          f"({len(relevant) / max(len(scored), 1):.1%})")
    print(f"Saved {len(daily)} days -> {out_path}")
    return daily


if __name__ == "__main__":
    build_daily_lexicon_score()
