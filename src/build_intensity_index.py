"""Merge the GDELT aggregate signal (primary, uncapped) with the three article-level
NLP scores (lexicon, TF-IDF tone, FinBERT) into one composite daily war-risk news
intensity index, and evaluate the NLP relevance filter against a labeled reference set.

Combination rule, spelled out here so it's not a black box: each component is
z-scored over the full window, then combined as a weighted sum. GDELT volume_pct/
volume_raw get the largest weight since they're the only uncapped signal (see
config.py / gdelt_corpus.py comments on the 250-record artlist cap). The three
article-level scores get smaller, equal weights, acting as a validating/refining
layer on top.

NLP evaluation: precision/recall of the lexicon relevance filter against a reference
label set, plus pairwise correlation between the three daily article-level scores as
an inter-method agreement diagnostic. The reference labels for this evaluation set
were produced by Claude reading each sampled headline and judging Iran-war-risk
relevance (see eval/build_eval_labels.py and AI_USE.md), so this is not blind or
independent human annotation. Disclosing that explicitly rather than presenting it
as manual ground truth.
"""
import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW

WEIGHTS = {
    "volume_pct_z": 0.30,
    "volume_raw_z": 0.20,
    "lexicon_z": 0.15,
    "tone_z": 0.15,
    "finbert_z": 0.20,
}


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not std or np.isnan(std):
        # Zero-variance column (e.g. an upstream scoring file wasn't available yet, so
        # the column is all zeros). Return all-zero z-scores rather than NaN (0/0),
        # since that would otherwise poison the composite sum for every day.
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def evaluate_nlp_component() -> dict:
    eval_path = DATA_PROCESSED / "eval_labels.parquet"
    if not eval_path.exists():
        print("No eval_labels.parquet found, skipping NLP evaluation (run eval/build_eval_labels.py first)")
        return {"status": "skipped_no_labels"}

    from lexicon import score_text
    labels = pd.read_parquet(eval_path)  # columns: url, title, relevant_gold
    preds = labels["title"].fillna("").apply(lambda t: score_text(t)["relevant"])
    tp = int(((preds) & (labels["relevant_gold"])).sum())
    fp = int(((preds) & (~labels["relevant_gold"])).sum())
    fn = int(((~preds) & (labels["relevant_gold"])).sum())
    tn = int(((~preds) & (~labels["relevant_gold"])).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    result = {
        "status": "ok", "n_labeled": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
    }
    print(f"Lexicon relevance filter vs. eval set (n={len(labels)}): "
          f"precision={precision:.2f}, recall={recall:.2f}")
    return result


def main() -> pd.DataFrame:
    timeline = pd.read_parquet(DATA_RAW / "gdelt_daily_timeline.parquet")
    timeline["date"] = pd.to_datetime(timeline["date"])

    frames = [timeline.set_index("date")[["volume_pct", "volume_raw", "avg_tone"]]]
    frame_names = []
    for name, path, col in [
        ("lexicon", DATA_PROCESSED / "lexicon_daily_score.parquet", "lexicon_intensity_sum"),
        ("tone", DATA_PROCESSED / "tone_word_list_daily_score.parquet", "tone_intensity_sum"),
        ("finbert", DATA_PROCESSED / "finbert_daily_score.parquet", "finbert_negativity_sum"),
    ]:
        if path.exists():
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df.set_index("date")[[col]].rename(columns={col: name}))
            frame_names.append(name)
        else:
            print(f"WARNING: {path.name} not found, composite index will exclude '{name}'")

    merged = pd.concat(frames, axis=1).sort_index()
    for c in ["lexicon", "tone", "finbert"]:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = merged[c].fillna(0.0)  # no relevant articles that day -> zero article-level intensity

    merged["volume_pct_z"] = _zscore(merged["volume_pct"])
    merged["volume_raw_z"] = _zscore(merged["volume_raw"])
    merged["lexicon_z"] = _zscore(merged["lexicon"])
    merged["tone_z"] = _zscore(merged["tone"])
    merged["finbert_z"] = _zscore(merged["finbert"])

    merged["composite_intensity"] = sum(merged[k] * w for k, w in WEIGHTS.items())

    # Inter-method agreement diagnostic among the three article-level (non-GDELT) scores.
    corr = merged[["lexicon", "tone", "finbert"]].corr().round(3)
    print("Inter-method correlation (article-level scores):")
    print(corr)

    nlp_eval = evaluate_nlp_component()
    nlp_eval["inter_method_correlation"] = corr.to_dict()

    out = merged.reset_index().rename(columns={"index": "date"})
    out_path = DATA_PROCESSED / "daily_intensity.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved {len(out)} days -> {out_path}")

    eval_path = DATA_PROCESSED / "nlp_eval_report.json"
    with open(eval_path, "w") as f:
        json.dump(nlp_eval, f, indent=2, default=float)
    print(f"Saved NLP evaluation report -> {eval_path}")
    return out


if __name__ == "__main__":
    main()
