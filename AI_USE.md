# AI Use Disclosure

## How AI was used

This project was built end-to-end with Claude (Claude Code), from methodology design
through implementation:

- Reading and synthesizing both reference papers (Rigobon 2003, "Identification through
  Heteroskedasticity"; Rigobon & Sack 2003, "The Effects of War Risk on U.S. Financial
  Markets") into an implementation plan.
- Live web research to verify facts that could not be taken from model memory: the
  actual 2026 Iran war timeline, GDELT DOC 2.0 API mechanics (endpoint, modes, rate
  limits, confirmed via a real test call before committing to it as the data source),
  real, currently-active tickers/FRED series codes, and (added when the analysis
  window was rebased to Feb 28 onward) current and 2003 US crude oil production
  figures straight from EIA, used in the Discussion (13.8 million bbl/day now vs. 5.70
  million in 2003, see `config.py`'s `US_OIL_PRODUCTION_CONTEXT` for both sources).
- A one-line check that a Polymarket-odds instrument, floated as an alternative to
  heteroskedasticity ID, is not purely hypothetical: grepped our own article corpus
  for the word. An earlier corpus sample (before the Feb 28 window rebase) had real
  hits, one headline was directly quoted as evidence at that point. The final
  Feb 28-Sep 20 corpus, resampled from different 14-day chunk boundaries, happens to
  have zero Polymarket-mentioning articles, this is disclosed honestly in the report's
  identification-alternatives section rather than reusing the old, no-longer-true
  quote.
- All pipeline code (`src/*.py`), the composite NLP intensity index design, the
  heteroskedasticity-based IV estimator implementation, robustness checks, report
  generation, and this documentation.
- **NLP evaluation reference labels**: the relevance labels used to compute the lexicon
  filter's precision/recall (`src/eval/`) were produced by Claude reading each sampled
  headline and judging Iran-war-risk relevance directly. This is disclosed explicitly
  because it is *not* independent human annotation, and should be read as an
  LLM-as-judge evaluation rather than a manually curated gold standard.

## What was not AI-generated, or was a joint decision

- **Scope and design decisions were made by the user, not defaulted by AI**: the
  analysis window (originally Jan-Sep 2026, single H/L split, rather than restricting
  to a pre-war anticipation phase or splitting into sub-regimes), the normalizing
  variable (US 2-year Treasury yield, matching the original paper exactly, over the
  alternative of normalizing to Brent crude), the H/L threshold rule (top-decile with
  nearby-matched L days), and the decision to build the on-the-run Treasury liquidity
  premium as a deliberate Phase B add-on rather than attempting it inline with the
  rest of the Phase-A pipeline.
- **The window and the sub-regime decision above were both revisited later.** The
  assignment's deliverables text specifically asked for "the Iran war (Feb 28 to
  present)", not "beginning of 2026", so the window was rebased to 2026-02-28 through
  2026-09-20. The same update also asked for a bad-news/good-news/no-news split, so the
  earlier "single H/L split, not sub-regimes" decision was explicitly reopened and a
  three-regime extension was added on top of (not instead of) the original H/L Table
  1/2/3. Both changes were user decisions, not something defaulted quietly.
- **GDELT chunk-cap fix, considered and declined.** `gdelt_corpus.py`'s artlist pull
  is known to only get real article text on 1 day out of every 14 (see the Mistake 1/7
  entries below and the report's Limitations), because the sort order plus the
  250-record cap means one chunk's entire quota lands on its first day. Fixing this
  properly (smaller chunks) was considered when the window was rebased anyway, and
  declined on purpose, it would add real re-fetch time and re-throttling risk for a
  problem the three-regime extension does not actually depend on (it uses the dense,
  uncapped tone-based direction signal instead).
- The plan itself went through an explicit user-requested critique pass ("go over the
  plan... see if something's off... make this more robust") before implementation began,
  which surfaced and added: the confound-calendar mitigation for the core identifying
  assumption, serial-correlation pre-whitening, first-half/second-half stability testing,
  H/L threshold sensitivity testing, the Sargan/J overidentification test, the NLP
  evaluation step, GDELT query precision (boolean/theme-based instead of a bare keyword),
  trading-calendar alignment, the Table 1 direction column, and the explicit
  Iraq-2003-vs-Iran-2026 comparison table, none of which were in the first draft.

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
while also passing `exog=["const"]`. `linearmodels.IV2SLS` combines `exog` and
`instruments` internally, so `const` appeared twice and every regression failed with
"instruments do not have full column rank," silently caught by a broad `except Exception`
and surfacing only as an entire table of `None`s with no visible traceback. This was not
caught by reading the code. It only showed up by deliberately running the full pipeline
against synthetic data before trusting it on real data, then debugging the silent
failure directly. Fixed by removing `const` from the `instruments` list in all three
call sites (`run_regressions.py` and `robustness_checks.py`); verified afterward with a
synthetic ground-truth IV test (known coefficient recovered correctly) before re-running
against real data.

### Mistake 6: z-scoring an all-zero column produced `NaN`, silently poisoning the composite intensity index
When `build_intensity_index.py` runs before the article-level NLP scoring files exist
(exactly the situation created by GDELT's rate limiting, see below), the
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
throttle. Even single, isolated requests kept returning 429 for an extended period
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

### Mistake 10: re-fetching the article corpus for the Feb 28 window silently reused a stale checkpoint from the old Jan 1 window
`gdelt_corpus.py`'s checkpoint/resume logic (`gdelt_articles_checkpoint.parquet`) was
built to survive an interrupted run of the *same* window, it was never designed to
notice that `ANALYSIS_START`/`ANALYSIS_END` had changed entirely. When the window was
rebased to Feb 28 - Sep 20 and the corpus re-fetch was kicked off, it happily resumed
from the leftover checkpoint of the earlier Jan 1 - Sep 17 run, meaning the final
corpus would have silently included pre-Feb-28 articles that no longer belong in this
analysis window at all. Caught by inspecting the checkpoint's `chunk_start` values
mid-run and noticing dates from January and early February sitting next to the new
window's dates. Fixed two ways: deleted the stale checkpoint/output files once, and
more importantly, added a real guard in `gdelt_corpus.py` so any future run filters
the checkpoint to only chunks whose `chunk_start` falls inside the *current*
`ANALYSIS_START`/`ANALYSIS_END`, discarding (and printing a warning about) anything
else, so a stale checkpoint from a different window can no longer be silently reused.

### Mistake 11: `gdelt_corpus.py` crashed outright on a raw connection drop instead of retrying
`fetch_artlist_chunk`'s retry loop only handled bad HTTP status codes and JSON decode
errors, a `RemoteDisconnected`/`ReadTimeout` raised directly by `requests.get()` (seen
in practice during a GDELT throttling episode) wasn't caught at all, so it propagated
up and killed the entire script, losing whatever chunks hadn't been checkpointed yet
in that run. Caught immediately from the traceback (an uncaught
`requests.exceptions.ConnectionError`). Fixed by wrapping the request itself in a
`try/except requests.exceptions.RequestException` inside the same retry loop, so a raw
connection drop is now treated exactly like a bad status code, log it, back off, retry,
not a reason to crash the whole run.

### Note (not a mistake): rebasing the analysis window meant re-verifying every specific number written into the report's prose, not just re-running the pipeline
Several report/notebook paragraphs (the ceasefire-day narrative, the Beirut-headline
example in Limitations, several t-stats and ratios) were originally written quoting
specific numbers and even specific headlines from the Jan 1 - Sep 17 run. Re-running
the pipeline on the new Feb 28 window produced a different corpus sample (different
GDELT chunk boundaries land on different calendar dates), so some of those exact
headlines no longer exist in the new corpus, and this makes stale narrative text a real
risk on any full re-run, not a one-off oversight the first time it happened. Handled by
grepping the rendered report text for every previously-known hardcoded number after
each full re-run and either verifying it against fresh output or rewriting it, and by
converting as many of these callouts as possible (recall/precision, variance ratios,
gold/broad-dollar t-stats, zero-inflation %, max z-score) into values computed live
from the current run's data inside `generate_report.py`, rather than typed prose, so
future re-runs can't silently go stale the same way again.

### Note (not a mistake): FinBERT's domain mismatch shows up in spot checks
FinBERT (`ProsusAI/finbert`) is trained on financial-news sentiment (earnings, deals,
stock moves), not geopolitical/military news. A spot check scored "UK temporarily closes
embassy in Tehran" as strongly *positive* (+0.898 on the negativity scale, i.e. read as
bullish), which is a counterintuitive result for a war-escalation headline. The model's
label ordering was verified directly against its Hugging Face config
(`{0: positive, 1: negative, 2: neutral}`) to rule out a code bug. It's not; this
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
