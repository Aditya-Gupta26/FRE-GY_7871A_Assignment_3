"""Shared paths, constants, and verified facts for the Iran war-risk heteroskedasticity project.

Every hard-coded date/fact below was verified against a live source during project
planning on 2026-09-18 (see comments), rather than recalled from model knowledge. Don't
extend these lists from memory alone, verify new entries the same way.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "report"
for _d in (DATA_RAW, DATA_PROCESSED, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (NYU FRE-GY 7871 coursework research script; contact: ag11023@nyu.edu)"
}
REQUEST_DELAY_SECONDS = 10.0  # GDELT explicitly asks for >=1 request/5s (confirmed via a
# live 429 response during planning); widened to 10s since bursts of requests during
# interactive testing were observed to trigger longer soft-throttling windows in practice

# ---------------------------------------------------------------------------
# Analysis window
# ---------------------------------------------------------------------------
# Fixed end date (not a floating "today") for reproducibility, see README.
ANALYSIS_START = "2026-01-01"
ANALYSIS_END = "2026-09-17"

# ---------------------------------------------------------------------------
# GDELT DOC 2.0 API
# ---------------------------------------------------------------------------
# Endpoint confirmed live during planning (2026-09-18): a real timelinevol query over
# 2026-01-01..2026-09-13 returned a daily volume series that independently spikes to
# 12.42 on 2026-02-28 from a ~0.5 baseline, i.e. the war's onset is visible in raw
# GDELT news volume with zero manual curation.
GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# Boolean query (GDELT DOC 2.0 syntax: space = AND, parentheses + OR for alternation,
# quotes for exact phrases) built to avoid a bare "Iran" keyword match, which would pull
# in unrelated Iran news (culture, unrelated domestic politics, etc.). Restricted to
# English-language sources for consistency with the FinBERT tone-scoring stage
# (documented limitation: non-English coverage exists but is out of scope for Phase A).
GDELT_QUERY = (
    'Iran (war OR strike OR strikes OR missile OR missiles OR "Strait of Hormuz" '
    'OR Hormuz OR nuclear OR IAEA OR ceasefire OR blockade OR sanctions OR IRGC '
    'OR Khamenei OR tanker OR tankers OR Natanz) sourcelang:english'
)

# ---------------------------------------------------------------------------
# Financial variables (Phase A, 13 variables; Phase B adds the GSW liquidity premium)
# ---------------------------------------------------------------------------
# yfinance tickers, verified live during planning (2026-09-18) against Yahoo Finance /
# issuer fact sheets. MES (VanEck Gulf States ETF) was checked and found defunct, so
# EIS (iShares MSCI Israel ETF, active, fact sheet dated 2026-06-30) is used as the
# regional/front-line equity proxy instead.
YFINANCE_TICKERS = {
    "brent_oil": "BZ=F",
    "wti_oil": "CL=F",
    "gold": "GC=F",
    "dollar_index": "DX-Y.NYB",
    "vix": "^VIX",
    "sp500": "^GSPC",
    "em_equity": "EEM",
    "israel_equity": "EIS",
}

# FRED series IDs, verified live during planning (2026-09-18) against FRED's own series
# pages (T10YIE, BAMLC0A4CBBB, BAMLH0A0HYM2, DTWEXBGS). DGS2/DGS10 are canonical, stable
# FRED constant-maturity Treasury series codes.
FRED_SERIES = {
    "treasury_2y": "DGS2",
    "treasury_10y": "DGS10",
    "breakeven_10y": "T10YIE",
    "bbb_spread": "BAMLC0A4CBBB",
    "hy_spread": "BAMLH0A0HYM2",
    "broad_dollar": "DTWEXBGS",  # cross-check vs. yfinance DX-Y.NYB
}

# The primary normalizing variable x1 (all headline sensitivities in Table 2 are
# measured relative to a fixed-size move in this variable), chosen to match Rigobon &
# Sack (2003) exactly for direct comparability to their published results.
NORMALIZING_VARIABLE = "treasury_2y"  # FRED_SERIES["treasury_2y"] = DGS2

# A parallel, secondary normalizing-variable experiment (doesn't replace the primary
# choice above). Brent crude is the most direct, least ambiguous transmission channel
# for Iran/Strait-of-Hormuz risk specifically, so re-running the same estimator with
# oil as x1 is a natural robustness/comparison exercise given the war's oil-supply-shock
# character, without giving up comparability to the original paper's Treasury-yield choice.
NORMALIZING_VARIABLES_TO_RUN = [NORMALIZING_VARIABLE, "brent_oil"]

# ---------------------------------------------------------------------------
# Confound calendar
# ---------------------------------------------------------------------------
# 2026 FOMC meeting dates, verified live during planning (2026-09-18) directly against
# federalreserve.gov/monetarypolicy/fomccalendars.htm and individual per-meeting pages
# (fomcpresconf20260318.htm, fomcpresconf20260429.htm, fomcpresconf20260617.htm,
# fomcpresconf20260729.htm all confirmed live on federalreserve.gov). Used to flag days
# where a non-Iran macro shock could contaminate the H/L classification.
FOMC_2026_MEETING_DATES = [
    "20260128",  # Jan 27-28 (decision announced on the 28th)
    "20260318",  # Mar 17-18
    "20260429",  # Apr 28-29
    "20260617",  # Jun 16-17
    "20260729",  # Jul 28-29
    "20260916",  # Sep 15-16
]

# Additional non-Iran macro/geopolitical shock dates identified during corpus review
# should be appended here as they are found, each with a one-line source comment.
# This list is deliberately seeded thin rather than guessed, and is expected to grow
# once classify_regimes.py surfaces borderline-intensity days that need investigation.
OTHER_CONFOUND_DATES = []

CONFOUND_CALENDAR = sorted(set(FOMC_2026_MEETING_DATES) | set(OTHER_CONFOUND_DATES))

# ---------------------------------------------------------------------------
# Iran war-risk 2026 timeline (validation reference only, NOT the classification
# mechanism itself). Used solely to sanity-check that NLP-derived H-days land on days
# we independently know were high-intensity, the same face-validity role Table 1 played
# in the original paper). Verified live during planning (2026-09-18) against Wikipedia
# ("2026 Iran war", "Timeline of the 2026 Iran war", "2026 Strait of Hormuz crisis"),
# the UK House of Commons Library briefing on the 2026 US-Iran ceasefire/nuclear talks,
# and Britannica's "2026 Iran war" entry.
# ---------------------------------------------------------------------------
IRAN_WAR_TIMELINE = [
    ("2026-01-10", "Iranian security forces violently suppress protests; nuclear talks deteriorating"),
    ("2026-02-10", "Large US military buildup in the region begins, described as largest since 2003"),
    ("2026-02-27", "US authorizes 'Operation Epic Fury'; embassy evacuations begin"),
    ("2026-02-28", "US/Israel launch 'Operation Epic Fury'/'Roaring Lion' strikes; Khamenei killed; Iran retaliates with ~170 missiles"),
    ("2026-03-08", "Brent crude passes $100/bbl for first time in 4 years"),
    ("2026-03-09", "Iran claims control of Strait of Hormuz; shipping insurance rates spike 4-6x"),
    ("2026-03-24", "Sustained strikes continue; Natanz nuclear site confirmed damaged by IAEA"),
    ("2026-04-08", "Two-week ceasefire begins; Israel resumes Lebanon strikes same day"),
    ("2026-04-13", "Islamabad Talks fail; US imposes its own naval blockade"),
    ("2026-04-21", "IMO reports ~20,000 mariners / ~2,000 ships stranded in the Persian Gulf"),
    ("2026-06-17", "US and Iran sign 14-point MOU ending the war (Hormuz navigation, nuclear/missile, sanctions)"),
    ("2026-07-08", "Ceasefire/MOU collapses after Iran attacks commercial vessels in the strait"),
    ("2026-08-25", "Continued sporadic tanker attacks in the Persian Gulf"),
    ("2026-09-13", "Renewed tanker incident; war described as still ongoing"),
]
