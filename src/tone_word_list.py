"""NLP scoring method 2: TF-IDF cosine similarity to hand-written war-risk seed
sentences. Deliberately a different mechanism from lexicon.py's weighted keyword
counting: this captures phrase-level/contextual similarity rather than raw term
hits, giving a genuinely independent second signal for the inter-method agreement
diagnostic in build_intensity_index.py.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_PROCESSED, DATA_RAW

HIGH_RISK_SEEDS = [
    "Iran launches missile strikes against Israel and US military bases in the Gulf",
    "Israel and the United States carry out airstrikes on Iranian nuclear facilities",
    "Iran closes the Strait of Hormuz to oil tankers, threatening global energy supply",
    "Iranian forces attack commercial shipping vessels in the Persian Gulf",
    "War breaks out between Iran and Israel with casualties on both sides",
    "Iran's Revolutionary Guard mobilizes forces amid fears of imminent war",
    "Ceasefire between Iran and Israel collapses after renewed attacks",
]

LOW_RISK_SEEDS = [
    "Iran and the United States hold diplomatic talks over the nuclear program",
    "Iran's economy grows amid easing international sanctions",
    "Iranian officials discuss cultural exchange with European counterparts",
    "Oil markets remain calm as Iran tensions ease",
    "Iran and world powers reach a peaceful agreement on nuclear inspections",
]


def _fit_vectorizer(corpus_titles: list[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    vec.fit(corpus_titles + HIGH_RISK_SEEDS + LOW_RISK_SEEDS)
    return vec


def score_articles(articles: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    titles = articles[text_col].fillna("").tolist()
    vec = _fit_vectorizer(titles)
    title_vecs = vec.transform(titles)
    high_vecs = vec.transform(HIGH_RISK_SEEDS)
    low_vecs = vec.transform(LOW_RISK_SEEDS)

    high_sim = cosine_similarity(title_vecs, high_vecs).max(axis=1)
    low_sim = cosine_similarity(title_vecs, low_vecs).max(axis=1)

    out = articles.reset_index(drop=True).copy()
    out["tone_high_sim"] = high_sim
    out["tone_low_sim"] = low_sim
    out["tone_relevant"] = out["tone_high_sim"] > 0.08  # low bar: any real overlap with the escalation seeds
    out["tone_intensity"] = out["tone_high_sim"]
    out["tone_direction"] = out["tone_high_sim"] - out["tone_low_sim"]
    return out


def build_daily_tone_score() -> pd.DataFrame:
    articles = pd.read_parquet(DATA_RAW / "gdelt_articles.parquet")
    scored = score_articles(articles, text_col="title")
    relevant = scored[scored["tone_relevant"]]
    daily = relevant.groupby("date").agg(
        tone_article_count=("tone_intensity", "size"),
        tone_intensity_sum=("tone_intensity", "sum"),
        tone_intensity_mean=("tone_intensity", "mean"),
        tone_direction_mean=("tone_direction", "mean"),
    ).reset_index()
    out_path = DATA_PROCESSED / "tone_word_list_daily_score.parquet"
    daily.to_parquet(out_path, index=False)
    print(f"TF-IDF relevance: {len(relevant)}/{len(scored)} articles flagged relevant "
          f"({len(relevant) / max(len(scored), 1):.1%})")
    print(f"Saved {len(daily)} days -> {out_path}")
    return daily


if __name__ == "__main__":
    build_daily_tone_score()
