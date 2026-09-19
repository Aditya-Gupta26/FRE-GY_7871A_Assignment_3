# The Effects of Iran War Risk on Global Financial Markets (2026)

## Aim

Estimate the sensitivity of global financial market variables to the risk of the 2026
Iran war, replicating the methodology of Rigobon & Sack (2003), *"The Effects of War
Risk on U.S. Financial Markets"* (NBER WP 9609), which itself applies the
heteroskedasticity-based identification method of Rigobon (2003), *"Identification
through Heteroskedasticity"* (*Review of Economics and Statistics*).

The original paper hand-picked 17 "war news" dates from reading financial press
coverage of the Iraq war buildup (Jan–Mar 2003). This project replaces that manual step
with a data-driven NLP pipeline that scores actual dated news coverage of Iran war risk
(GDELT-indexed, Jan 1 – Sep 17, 2026) to systematically classify high-variance ("H") vs.
calm ("L") war-news days, then applies the identical heteroskedasticity-based estimator
to a set of global market financial variables.

## Methods applied

1. **NLP war-risk news intensity index.** Three parallel scoring methods over a GDELT
   DOC 2.0 API news corpus (title-level, with best-effort full-text supplementation):
   a hand-built escalation-term lexicon, TF-IDF cosine similarity to hand-written
   high/low war-risk seed sentences, and FinBERT transformer sentiment. Combined with
   GDELT's own (uncapped) daily volume/tone timeline into one composite intensity index.
2. **Data-driven H/L classification.** Top-decile composite intensity = H; a
   matched-size set of the nearest lower-intensity days = L, mirroring the original
   paper's "choose L days close to H days" logic. Cross-validated against a
   independently-verified timeline of major 2026 Iran war events. Days coinciding with
   known confounds (FOMC meetings) are flagged, not silently dropped.
3. **Heteroskedasticity-based estimation.** For each financial variable paired against
   the US 2-year Treasury yield (the normalizing variable, matching the original
   paper), Ω_H − Ω_L is computed and the war-risk sensitivity coefficient is estimated
   three ways: the direct ΔΩ-ratio formulas (eq. 6/7), and the equivalent
   instrumental-variables regressions (ω1, ω2, and an overidentified two-instrument
   ω3 = [ω1, ω2] specification with a Sargan/J test), with both analytic and
   H/L-stratified bootstrap standard errors.
4. **Variance decomposition.** Following the original paper, the lower-bound share of
   each variable's variance attributable to the Iran war-risk factor, on H-days and
   over the full analysis window.
5. **Robustness checks:** serial-correlation diagnostics (Ljung-Box, AR(1)-filtered
   where significant) before computing covariances; first-half vs. second-half
   coefficient-stability testing across the war's very different phases; H/L threshold
   sensitivity re-estimation; the Sargan/J overidentification test.

## Key results

Full corpus: 4,500 GDELT-indexed articles across 18 of 19 biweekly windows (Jan 1–Sep 17,
2026; one window, Aug 27–Sep 9, could not be fetched after repeated GDELT rate-limiting
and is documented as a gap, not near any Table 1 milestone date). H/L classification:
26 top-decile "war news" days, 26 matched calm days. NLP relevance filter (lexicon)
evaluated against a 176-article Claude-judged reference set: precision 0.83, recall 0.29
(picky but accurate, it misses many relevant articles that don't use its specific
escalation vocabulary). Inter-method correlation: TF-IDF tone and FinBERT strongly agree
(0.885); both correlate more loosely with the pure-keyword lexicon (0.57–0.66).

**Primary specification (2-year Treasury yield, matching the original paper):**
weakly identified. The yield's own variance is *not* clearly elevated on H-days in this
sample (unlike Iraq 2003, where it was ~6x higher), so coefficients are large and
statistically insignificant (e.g. S&P 500: coefficient -961, t=-1.26). Reported in full
in the report/notebook rather than suppressed, because the instability is itself an
informative finding, see below.

**Secondary specification (Brent crude, motivated by the war's Strait-of-Hormuz/oil-supply
character):** stable and statistically strong across nearly every variable (most
|t-stats| > 2–8). Per $1 move in Brent, driven by war risk:

| Variable | Coefficient | t-stat |
|---|---|---|
| S&P 500 | −10.12 | −3.26 |
| Israel equities | −0.23 | −2.21 |
| EM equities | −0.20 | −3.63 |
| VIX | +0.33 | +3.48 |
| Broad dollar index | +0.074 | +6.14 |
| 2yr Treasury yield | +0.0042 | +2.02 |
| 10yr Treasury yield | +0.0032 | +1.89 |
| 10yr breakeven inflation | +0.0027 | +3.97 |
| BBB spread | +0.0023 | +4.58 |
| High-yield spread | +0.0123 | +8.14 |
| Gold | −10.81 | −3.42 |

**Interpretation:** Treasury yields and inflation breakevens *rise* (not fall) with
war-risk-driven oil moves, the opposite mechanism from Rigobon & Sack's Iraq 2003
result, where war risk was a flight-to-safety shock (yields fell). The 2026 Iran war
appears to transmit primarily as an **oil supply shock** (inflationary, hawkish for
rates) rather than a classic flight-to-safety shock. This is consistent with the war's actual
character (Strait of Hormuz disruption) and with Brent oil, not the 2yr Treasury yield,
being the variable whose own variance is genuinely elevated on war-news days. Gold's
negative coefficient is a notable, robust-but-counterintuitive result (a safe haven
"falling" alongside a war-risk shock) flagged for discussion rather than smoothed over.

**Figure 2** (`notebooks/figure2_oil_vs_treasury.png`) makes the case for oil over
Treasury directly: the 2yr yield's own H/L variance ratio is only 1.3x, against oil's
11.2x (the identification condition itself); the oil-normalized specification clears
conventional significance (|t|>2) for almost every variable where the Treasury-normalized
one doesn't; and an aggregate, scale-free instability score across *all* variables (not
one cherry-picked example) shows the Treasury specification's typical coefficient swing
across alternative H-day thresholds sits right at the boundary where a sign flip becomes
plausible, while oil's sits mostly below it.

See `report/report.pdf` and `notebooks/analysis.ipynb` for the full Table 1/2/3, the
explicit Iraq-2003-vs-Iran-2026 comparison, and all robustness checks.

## How this was built, step by step

1. `src/config.py`: paths, verified constants (analysis window, tickers/FRED codes,
   confound calendar, reference war timeline).
2. `src/gdelt_intensity.py` + `src/fetch_market_data.py`: primary (uncapped) GDELT
   intensity timeline and the 13 Phase-A financial variables.
3. `src/gdelt_corpus.py` → `src/scrape_article_text.py` → `src/lexicon.py` /
   `src/tone_word_list.py` / `src/tone_finbert.py` → `src/build_intensity_index.py`:
   article-level NLP corpus, three parallel scoring methods, composite index, and NLP
   evaluation (`src/eval/`).
4. `src/classify_regimes.py`: H/L day classification with confound flagging.
5. `src/build_master_dataset.py`: merges everything into one trading-calendar-aligned
   analysis panel.
6. `src/event_study.py` → `src/run_regressions.py` → `src/variance_decomposition.py` →
   `src/robustness_checks.py`: the econometric core.
7. `src/make_figure1.py`, `src/make_table1.py` / `make_table2.py` / `make_table3.py`,
   `src/generate_report.py`: figures, tables, and the assembled PDF report.
8. `notebooks/analysis.ipynb`: executed end-to-end solution notebook.

## Repository structure

```
src/                    pipeline scripts (see "How this was built" above for order)
src/eval/               NLP relevance-filter evaluation sampling/labeling
notebooks/               analysis.ipynb (solution notebook) + figure1 PNG
data/raw/, data/processed/   generated by the pipeline scripts (gitignored)
report/                  generated PDF report (gitignored)
AI_USE.md                AI-use disclosure
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd src
python3 gdelt_intensity.py
python3 fetch_market_data.py
python3 gdelt_corpus.py          # resumable; re-run if it reports skipped chunks
python3 scrape_article_text.py
python3 lexicon.py && python3 tone_word_list.py && python3 tone_finbert.py
python3 eval/build_eval_labels.py   # then label eval/eval_sample_unlabeled.parquet -> data/processed/eval_labels.parquet
python3 build_intensity_index.py
python3 classify_regimes.py
python3 build_master_dataset.py
python3 event_study.py && python3 run_regressions.py && python3 variance_decomposition.py && python3 robustness_checks.py
python3 make_figure1.py
python3 generate_report.py
```

GDELT's DOC 2.0 API enforces informal rate limiting (~1 request/5s, observed to be
stricter under burst load); the data-pull scripts retry with exponential backoff and
`gdelt_corpus.py` checkpoints incrementally so an interrupted run can be resumed.

## Data sources

- **News**: [GDELT DOC 2.0 API](https://api.gdeltproject.org/api/v2/doc/doc), the only
  free, date-range-capable news source checked that could reach back to Jan 2026
  (NewsAPI's free tier and RSS feeds cannot).
- **Financial data**: [FRED](https://fred.stlouisfed.org) (public `fredgraph.csv`
  endpoint, no API key) for Treasury yields, breakeven inflation, and credit spreads;
  [Yahoo Finance](https://finance.yahoo.com) (`yfinance`) for oil, gold, dollar index,
  VIX, S&P 500, and regional equity ETFs.

## Further reading

- Rigobon, R. (2003), "Identification through Heteroskedasticity," *The Review of
  Economics and Statistics*, 85(4): 777–792.
- Rigobon, R. & Sack, B. (2003), "The Effects of War Risk on U.S. Financial Markets,"
  NBER Working Paper No. 9609 (published version: *Journal of Banking & Finance*,
  2005, 29(7): 1769–1789).
