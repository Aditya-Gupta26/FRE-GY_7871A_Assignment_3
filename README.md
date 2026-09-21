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
(GDELT-indexed, Feb 28 to Sep 20 2026, the actual war window, per the assignment's
deliverables text; an earlier version of this project used Jan 1 as the start date,
following the assignment's looser "beginning of 2026" phrasing, since superseded) to
systematically classify high-variance ("H") vs. calm ("L") war-news days, then applies
the identical heteroskedasticity-based estimator to a set of global market financial
variables. On top of the core H/L replication, this project also splits H-days into
"bad" (escalatory) vs. "good" (de-escalatory) war news as an extension, and includes a
written comparison of heteroskedasticity-based identification against other approaches
(narrative event-study, text-based shock index, a market-priced instrument, GARCH/
regime-switching, local projections), see the report/notebook for both.

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
6. **Three-regime extension.** H-days are split further into "bad" (war risk
   escalating) and "good" (war risk de-escalating), using the tone-based `direction`
   signal, and the same heteroskedasticity estimator is re-run on bad-vs-calm,
   good-vs-calm, and bad-vs-good. Small-sample by construction, reported with an
   explicit N-count caveat rather than hidden.
7. **Novel-phrasing check.** The lexicon relevance filter's false negatives are split
   into "missing the word iran entirely" (already known) vs. "has iran but no
   relevance-term hit" (genuinely missed vocabulary), and the top missed terms are
   pulled out and categorized as novel-to-this-war vs. a generic gap in the hand-built
   list.
8. **Alternatives to heteroskedasticity ID.** A written comparison against narrative
   event-study, a text-based shock index, a market-priced instrument (Polymarket
   odds, checked for real hits in our own corpus), GARCH/regime-switching, and local
   projections/SVAR, with a ranked recommendation.

## Key results

Corpus: 3,750 GDELT-indexed articles across all 15 biweekly windows in the Feb 28 - Sep
20, 2026 war window (some windows needed a few retries against GDELT's rate limiting,
see AI_USE.md, but all eventually came through). H/L classification: 20 top-decile
"war news" days, 20 matched calm days (18 H / 16 L after trading-calendar alignment).
NLP relevance filter (lexicon) evaluated against a 180-article Claude-judged reference
set: precision 1.00, recall 0.31 (very picky, misses close to 70% of genuinely relevant
articles that don't use its specific escalation vocabulary, see the false-negative
check in the report for concrete examples and what vocabulary it's actually missing).
Inter-method correlation: TF-IDF tone and FinBERT agree reasonably well (0.79), both
correlate more loosely with the pure-keyword lexicon (0.56-0.59).

**Primary specification (2-year Treasury yield, matching the original paper):** still
weakly identified. The yield's own variance is only about 1.2x higher on H-days than
L-days in this window (vs. Iraq 2003's roughly 6x), so most coefficients don't clear
conventional significance.

**Secondary specification (Brent crude):** stable and statistically strong across
nearly every variable, own H/L variance ratio about 5.3x. Per $1 move in Brent, driven
by war risk:

| Variable | Coefficient | t-stat |
|---|---|---|
| S&P 500 | -8.60 | -2.57 |
| Israel equities | -0.18 | -1.79 |
| EM equities | -0.14 | -2.51 |
| VIX | +0.30 | +3.01 |
| Broad dollar index | +0.077 | +5.23 |
| 2yr Treasury yield | +0.0033 | +1.68 |
| 10yr Treasury yield | +0.0024 | +1.48 |
| 10yr breakeven inflation | +0.0018 | +2.43 |
| BBB spread | +0.0020 | +4.94 |
| High-yield spread | +0.013 | +9.18 |
| Gold | -7.70 | -2.61 |

**Interpretation:** same qualitative story as before the window rebase, equities down,
volatility/dollar/spreads up, Treasury yields and breakevens *rising* (not falling)
with war-risk-driven oil moves, the opposite of Iraq 2003's flight-to-safety pattern.
Consistent with an oil-supply-shock transmission mechanism. Gold still falls, a
counterintuitive but consistently-signed result across both this run and the earlier
Jan-1-anchored one.

**Three-regime extension** (bad war news vs. good war news vs. calm, see report for
full tables): bad-news H-days = 5, good-news H-days = 10, both small subsets of an
already-small H set. Checked the brief's exact stated hypothesis directly (bad news:
yields and oil up, equities down, good news the reverse) using raw average daily
changes by regime, not the IV coefficients, and it holds up on 7 of the 8 variables
the brief names directly: oil and both yields rise on bad news and fall on good news,
credit spreads widen then narrow, and the S&P 500 falls on bad news and jumps back up
on good news. The full IV-based estimator on the same split is noisier and coefficients
often flip sign between the two, which the raw-average check above is not, so that
part is reported with an N-count caveat front and center rather than smoothed over.

**Identification-alternatives writeup:** heteroskedasticity ID stays the primary
method (matches both reference papers, avoids needing to sign every headline), but a
Polymarket-odds market instrument is flagged as the strongest, cheapest robustness
cross-check to add. Checked, not assumed: an earlier corpus sample (before the window
was rebased to Feb 28) did have real Polymarket-mentioning articles, this final
corpus's particular 14-day chunk sample happens not to, disclosed honestly in the
report rather than reusing the old quote (see AI_USE.md).

See `report/report.pdf` and `notebooks/analysis.ipynb` for the full Table 1/2/3, the
three-regime extension, the Eq(10) check, the identification-alternatives writeup, the
novel-phrasing/false-negative analysis, and all robustness checks.

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
7. `src/three_regime_analysis.py` → `src/make_table_three_regime.py`: the bad/good
   war-news extension, reuses the econometric core's own functions on sub-masks of
   the H set rather than duplicating any estimator logic.
8. `src/eval/analyze_false_negatives.py`: the novel-phrasing check, run after a fresh
   eval-label pass.
9. `src/make_figure1.py` / `make_figure2_oil_vs_treasury.py` / `make_figure3_var_covar.py`,
   `src/make_table1.py` / `make_table2.py` / `make_table3.py` / `make_table_three_regime.py`,
   `src/generate_report.py`: figures, tables, and the assembled PDF report.
10. `notebooks/analysis.ipynb`: executed end-to-end solution notebook.

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
python3 three_regime_analysis.py
python3 eval/analyze_false_negatives.py
python3 make_figure1.py && python3 make_figure2_oil_vs_treasury.py && python3 make_figure3_var_covar.py
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
