"""NLP scoring method 3: transformer sentiment (FinBERT) over article titles.

Runs on titles only by default, since GDELT's artlist mode returns titles/URLs, not full
body text, and full-text scraping (scrape_article_text.py) has a documented,
meaningfully high failure rate (paywalls/anti-bot). Where scraped text IS available
for an article, it's used in place of the title; this fallback rate is logged so the
report can state what fraction of the corpus was title-only vs. full-text.
"""
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DATA_PROCESSED, DATA_RAW

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
# FinBERT's label order for ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
LABELS = ["positive", "negative", "neutral"]


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def score_texts(texts: list[str], tokenizer, model) -> pd.DataFrame:
    all_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT batches"):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()
            all_scores.append(probs)
    import numpy as np
    probs = np.concatenate(all_scores, axis=0)
    df = pd.DataFrame(probs, columns=LABELS)
    # Negativity score: higher = more negative/bearish tone. War-risk escalation news
    # is expected to skew negative; this is used as the daily FinBERT intensity signal.
    df["finbert_negativity"] = df["negative"] - df["positive"]
    return df


def build_daily_finbert_score() -> pd.DataFrame:
    articles = pd.read_parquet(DATA_RAW / "gdelt_articles.parquet")
    try:
        text_df = pd.read_parquet(DATA_RAW / "article_text.parquet")
        merged = articles.merge(text_df[["url", "text", "fetch_status"]], on="url", how="left")
        merged["scoring_text"] = merged["text"].where(merged["fetch_status"] == "ok", merged["title"])
        coverage = (merged["fetch_status"] == "ok").mean()
    except FileNotFoundError:
        merged = articles.copy()
        merged["scoring_text"] = merged["title"]
        coverage = 0.0
    print(f"Full-text coverage: {coverage:.1%} (remainder scored on title only)")

    tokenizer, model = _load_model()
    scores = score_texts(merged["scoring_text"].fillna("").tolist(), tokenizer, model)
    scored = pd.concat([merged.reset_index(drop=True), scores], axis=1)

    daily = scored.groupby("date").agg(
        finbert_article_count=("finbert_negativity", "size"),
        finbert_negativity_mean=("finbert_negativity", "mean"),
        finbert_negativity_sum=("finbert_negativity", lambda s: s.clip(lower=0).sum()),
    ).reset_index()
    out_path = DATA_PROCESSED / "finbert_daily_score.parquet"
    daily.to_parquet(out_path, index=False)
    print(f"Saved {len(daily)} days -> {out_path}")
    return daily


if __name__ == "__main__":
    build_daily_finbert_score()
