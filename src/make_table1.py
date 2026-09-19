"""Render the Table-1-equivalent (H/L regime dates + representative headlines +
direction + confound flag) as an HTML fragment for the report."""
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW


def df_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x), border=0, na_rep="-", escape=True)


def build() -> str:
    table1 = pd.read_csv(DATA_PROCESSED / "table1_regime_dates.csv")
    table1["date"] = pd.to_datetime(table1["date"])

    articles = pd.read_parquet(DATA_RAW / "gdelt_articles.parquet")
    articles["date"] = pd.to_datetime(articles["date"])
    # One representative (highest-relevance-adjacent) headline per date, via lexicon score.
    from lexicon import score_text
    articles["intensity"] = articles["title"].fillna("").apply(lambda t: score_text(t)["intensity"])
    top_headline = (
        articles.sort_values("intensity", ascending=False)
        .drop_duplicates(subset="date", keep="first")[["date", "title"]]
        .rename(columns={"title": "representative_headline"})
    )

    merged = table1.merge(top_headline, on="date", how="left")
    # GDELT's artlist mode (sort=dateasc, capped at 250 records/chunk) ends up
    # returning only the earliest-dated articles of each 14-day chunk, so in
    # practice the whole 250-record quota for a chunk lands on that chunk's first
    # day. Only the chunk-start dates actually have article-level text, every other
    # date in the window has zero rows in gdelt_articles.parquet, hence no headline
    # to show here. Rather than show a blank/placeholder row, those dates are
    # dropped from this display table entirely, see the Limitations note on this
    # same issue for the full H/L date list (table1_regime_dates.csv still has all
    # of them, this filtering only affects the headline-annotated display table).
    n_total = len(merged)
    merged = merged.dropna(subset=["representative_headline"]).copy()
    n_shown = len(merged)
    merged["regime"] = merged.apply(lambda r: "H" if r["is_H"] else ("L" if r["is_L"] else ""), axis=1)
    merged["War Risk"] = merged["direction"]
    display = merged[["date", "representative_headline", "War Risk", "regime", "confound_flag"]].rename(
        columns={"date": "Date", "representative_headline": "Representative headline",
                 "regime": "Regime", "confound_flag": "Confound flag"}
    )
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")

    html = df_to_html_table(display)
    out_path = DATA_PROCESSED / "table1.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved Table 1 HTML ({n_shown}/{n_total} H/L days have a headline and are shown) -> {out_path}")
    return html


if __name__ == "__main__":
    build()
