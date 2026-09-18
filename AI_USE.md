# AI Use Disclosure

## How AI was used

This project was built end-to-end with Claude (Claude Code), from methodology design
through implementation:

- Reading and synthesizing both reference papers (Rigobon 2003, "Identification through
  Heteroskedasticity"; Rigobon & Sack 2003, "The Effects of War Risk on U.S. Financial
  Markets") into an implementation plan.
- Live web research to verify facts that could not be taken from model memory: the
  actual 2026 Iran war timeline, GDELT DOC 2.0 API mechanics (endpoint, modes, rate
  limits — confirmed via a real test call before committing to it as the data source),
  and real, currently-active tickers/FRED series codes.
- All pipeline code (`src/*.py`), the composite NLP intensity index design, the
  heteroskedasticity-based IV estimator implementation, robustness checks, report
  generation, and this documentation.
- **NLP evaluation reference labels**: the relevance labels used to compute the lexicon
  filter's precision/recall (`src/eval/`) were produced by Claude reading each sampled
  headline and judging Iran-war-risk relevance directly — this is disclosed explicitly
  because it is *not* independent human annotation, and should be read as an
  LLM-as-judge evaluation rather than a manually curated gold standard.

## What was not AI-generated, or was a joint decision

- **Scope and design decisions were made by the user, not defaulted by AI**: the
  analysis window (full Jan–Sep 2026, single H/L split, rather than restricting to a
  pre-war anticipation phase or splitting into sub-regimes), the normalizing variable
  (US 2-year Treasury yield, matching the original paper exactly, over the alternative
  of normalizing to Brent crude), the H/L threshold rule (top-decile with nearby-matched
  L days), and the decision to build the on-the-run Treasury liquidity premium as a
  deliberate Phase B add-on rather than attempting it inline with the rest of the
  Phase-A pipeline.
- The plan itself went through an explicit user-requested critique pass ("go over the
  plan... see if something's off... make this more robust") before implementation began,
  which surfaced and added: the confound-calendar mitigation for the core identifying
  assumption, serial-correlation pre-whitening, first-half/second-half stability testing,
  H/L threshold sensitivity testing, the Sargan/J overidentification test, the NLP
  evaluation step, GDELT query precision (boolean/theme-based instead of a bare keyword),
  trading-calendar alignment, the Table 1 direction column, and the explicit
  Iraq-2003-vs-Iran-2026 comparison table — none of these were in the first draft.

## Mistakes made, and how they were caught and fixed

### Mistake 1: GDELT rate-limit handling was initially too aggressive
Early interactive testing sent GDELT DOC 2.0 API requests faster than its informal
~1-request/5-seconds limit, triggering repeated 429 responses; a burst of rapid retries
during debugging then appears to have triggered a longer soft-throttling window than the
API's own stated 5-second limit. Fixed by: widening `REQUEST_DELAY_SECONDS` to 10s,
adding exponential backoff with a higher retry ceiling in the data-pull scripts, and
switching `gdelt_corpus.py` from weekly to biweekly chunking (halving the request count)
with incremental checkpointing so a long, retry-heavy run doesn't lose progress if
interrupted.

### Mistake 2: a broken column-merge in `build_intensity_index.py`
An early draft of the composite-index merge used a convoluted
`DataFrame.get(...).combine_first(...)` pattern to backfill missing `lexicon`/`tone`/
`finbert` columns when an upstream scoring file hadn't been generated yet; this was
logically confused and would have raised or produced wrong results. Caught on review
before running against real data, and replaced with a straightforward per-column
existence check + `fillna(0.0)`.

### Mistake 3: a duplicated, contradictory `direction` computation in `classify_regimes.py`
The first draft computed the descriptive `direction` column two different ways in
sequence (a `np.select` on lexicon/tone scores, immediately overwritten by a
tone-change-based rule), leaving dead code that would confuse a future reader. Caught on
review and removed the unused first computation.

### Mistake 4: `statsmodels.tsa.ar_model.AutoReg` called with a removed keyword argument
`event_study.py`'s serial-correlation filter called `AutoReg(..., old_names=False)`,
which was valid in older statsmodels but has been removed in the installed version
(0.15.0), raising `TypeError` on the very first real run. Caught immediately (the error
is explicit) and fixed by dropping the obsolete argument.

### Mistake 5: duplicate `const` column passed to `linearmodels.IV2SLS`, breaking every regression
`run_regressions.py`'s `iv_estimate()` originally passed `instruments=["const", "omega1"]`
while also passing `exog=["const"]` — `linearmodels.IV2SLS` combines `exog` and
`instruments` internally, so `const` appeared twice and every regression failed with
"instruments do not have full column rank," silently caught by a broad `except Exception`
and surfacing only as an entire table of `None`s with no visible traceback. This was not
caught by reading the code — it only showed up by deliberately running the full pipeline
against synthetic data before trusting it on real data, then debugging the silent
failure directly. Fixed by removing `const` from the `instruments` list in all three
call sites (`run_regressions.py` and `robustness_checks.py`); verified afterward with a
synthetic ground-truth IV test (known coefficient recovered correctly) before re-running
against real data.

### Mistake 6: z-scoring an all-zero column produced `NaN`, silently poisoning the composite intensity index
When `build_intensity_index.py` runs before the article-level NLP scoring files exist
(exactly the situation created by GDELT's rate limiting -- see below), the
`lexicon`/`tone`/`finbert` columns default to a constant 0.0. Z-scoring a zero-variance
column divides by a zero standard deviation, producing `NaN` for the entire column, which
then poisoned the weighted-sum composite for every single day (`composite_intensity` was
`NaN` everywhere, silently, with no error). Caught by inspecting `classify_regimes.py`'s
output ("H threshold ... >= nan -> 0/252 days") rather than trusting a clean-looking run.
Fixed by making `_zscore` return all-zero output for a zero-variance input instead of
`NaN`.

### Mistake 7 (not a bug, a real external constraint worth recording): GDELT's rate limit is far stricter under sustained/burst load than its own error message states
GDELT's 429 response text says "one every 5 seconds," but sustained use (many requests
across interactive debugging plus the scripted retries) triggered a much longer-lived
throttle -- even single, isolated requests kept returning 429 for an extended period
after a burst, and recovered only after a genuine cooldown with no further requests. The
fix was behavioral, not code: stop hammering the endpoint, wait, and resume once a
single diagnostic call confirmed the block had cleared, rather than assuming a fixed
backoff schedule would always succeed.

### Mistake 8: `include_groups=True` no longer accepted by the installed pandas version
`scrape_article_text.py` and `eval/build_eval_labels.py` both used
`groupby(...).apply(fn, include_groups=True)` to cap/sample rows per group; the installed
pandas version (2.x with the new groupby-apply deprecation cycle) raises
`ValueError: include_groups=True is no longer allowed.` immediately on the first call.
Neither function actually referenced the grouping column inside the applied lambda, so
the parameter was unnecessary in both cases. Fixed by removing the argument entirely
(same one-line fix in both files) rather than switching to `include_groups=False`, which
would have been a no-op change with extra surface area.

### Mistake 9: `scrape_article_text.py`'s robots.txt check could hang indefinitely
`_allowed_by_robots()` used `urllib.robotparser.RobotFileParser.read()`, which has no
timeout parameter and hung the whole scraping run on a single slow/unresponsive site
(observed directly: progress stalled at 225/240 articles for several minutes with the
process still alive but making no forward progress). Fixed by fetching `robots.txt` via
`requests` (which has a real timeout) and feeding the text into `RobotFileParser.parse()`
instead of letting it fetch the file itself.

### Note (not a mistake): FinBERT's domain mismatch shows up in spot checks
FinBERT (`ProsusAI/finbert`) is trained on financial-news sentiment (earnings, deals,
stock moves), not geopolitical/military news. A spot check scored "UK temporarily closes
embassy in Tehran" as strongly *positive* (+0.898 on the negativity scale, i.e. read as
bullish), which is a counterintuitive result for a war-escalation headline. The model's
label ordering was verified directly against its Hugging Face config
(`{0: positive, 1: negative, 2: neutral}`) to rule out a code bug -- it is not one; this
is a genuine domain-mismatch limitation of using a finance-tuned sentiment model on
geopolitical text, which is exactly why the composite index treats FinBERT as one of
three independent signals (20% weight) rather than the sole tone measure, and why the
inter-method correlation diagnostic exists.

## Tools

- Claude Code (Sonnet 5), for all code authorship, research, and this documentation.
- GDELT DOC 2.0 API (news corpus), FRED `fredgraph.csv` (Treasury/spread data),
  Yahoo Finance via `yfinance` (oil/gold/dollar/VIX/equity data).
- `transformers` (ProsusAI/finbert), `scikit-learn` (TF-IDF), `linearmodels` (IV2SLS +
  Sargan/J test), `statsmodels` (Ljung-Box, AR(1) filtering), `weasyprint` (PDF report
  rendering).
