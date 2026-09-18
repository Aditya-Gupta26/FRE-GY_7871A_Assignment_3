"""Assemble the standalone PDF report: methodology, Table 1, Table 2/3 for each
normalizing variable (primary: 2yr Treasury yield, matching the original paper;
secondary: Brent crude, motivated by the war's oil-supply-shock character), the
Iraq-vs-Iran comparison (treasury_2y run only), NLP evaluation, robustness summaries,
discussion, and limitations. Builds an HTML file from live processed data then renders
it to PDF via weasyprint.
"""
import base64
import json

import pandas as pd

from config import ANALYSIS_END, ANALYSIS_START, DATA_PROCESSED, NORMALIZING_VARIABLES_TO_RUN, PROJECT_ROOT, REPORT_DIR
import make_table1
import make_table2
import make_table3

FIGURE1_PATH = PROJECT_ROOT / "notebooks" / "figure1_intensity_timeline.png"
FIGURE2_PATH = PROJECT_ROOT / "notebooks" / "figure2_oil_vs_treasury.png"


def image_data_uri(path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def robustness_block_html(normalizing_variable: str) -> str:
    parts = []
    stab_path = DATA_PROCESSED / f"robustness_stability_{normalizing_variable}.parquet"
    sens_path = DATA_PROCESSED / f"robustness_threshold_sensitivity_{normalizing_variable}.parquet"
    summ_path = DATA_PROCESSED / f"robustness_summary_{normalizing_variable}.json"
    if stab_path.exists():
        st = pd.read_parquet(stab_path)
        parts.append("<h4>First-half vs. second-half coefficient stability</h4>")
        parts.append(st.to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0, na_rep="-"))
    if sens_path.exists():
        sens = pd.read_parquet(sens_path)
        if not sens.empty:
            pivot = sens.pivot(index="variable", columns="H_quantile", values="coef")
            parts.append("<h4>H/L threshold sensitivity</h4>")
            parts.append(pivot.to_html(float_format=lambda x: f"{x:.3f}", border=0, na_rep="-"))
    if summ_path.exists():
        with open(summ_path) as f:
            summary = json.load(f)
        parts.append(f'<p class="note">Summary: {summary}</p>')
    return "\n".join(parts)


def normalizing_variable_section(normalizing_variable: str, is_primary: bool) -> str:
    table2_html, comparison_html = make_table2.build(normalizing_variable)
    table3_html = make_table3.build(normalizing_variable)
    label = f"{normalizing_variable}" + (" (primary, matches Rigobon &amp; Sack 2003)" if is_primary
                                          else " (secondary experiment: Brent crude as the direct Iran/Hormuz transmission channel)")
    section = f"""
    <h2>Results — normalizing variable: {label}</h2>
    <h3>Table 2: Estimated Impact of Increase in Iran War Risk</h3>
    {table2_html}
    <p class="note">Columns Eq(6)/Eq(7) are the direct ΔΩ-ratio estimators;
    IV ω1/ω2/ω3 are the equivalent instrumental-variables estimates
    (ω3 is a genuine two-instrument overidentified specification with a Sargan/J
    test). Bootstrap SE resamples H- and L-days separately, 500 replications.</p>
    """
    if comparison_html:
        section += f"""
        <h3>Comparison to Rigobon &amp; Sack (2003): Iraq 2003 vs. Iran 2026</h3>
        {comparison_html}
        """
    section += f"""
    <h3>Table 3: Variance Decomposition</h3>
    {table3_html}
    <h3>Robustness Checks</h3>
    {robustness_block_html(normalizing_variable)}
    """
    return section


def build_html() -> str:
    table1_html = make_table1.build()

    nlp_eval = {}
    eval_path = DATA_PROCESSED / "nlp_eval_report.json"
    if eval_path.exists():
        with open(eval_path) as f:
            nlp_eval = json.load(f)

    fig_uri = image_data_uri(FIGURE1_PATH) if FIGURE1_PATH.exists() else ""
    fig2_uri = image_data_uri(FIGURE2_PATH) if FIGURE2_PATH.exists() else ""

    nlp_eval_block = ""
    if nlp_eval.get("status") == "ok":
        nlp_eval_block = f"""
        <p>Lexicon relevance filter evaluated against a {nlp_eval['n_labeled']}-article
        reference set: <b>precision = {nlp_eval['precision']:.2f}</b>,
        <b>recall = {nlp_eval['recall']:.2f}</b> (tp={nlp_eval['tp']}, fp={nlp_eval['fp']},
        fn={nlp_eval['fn']}, tn={nlp_eval['tn']}). Reference labels were produced by
        Claude reading each sampled headline and judging Iran-war-risk relevance
        (see AI_USE.md) -- not independent human annotation.</p>
        """

    normalizing_sections = "\n".join(
        normalizing_variable_section(nv, is_primary=(i == 0))
        for i, nv in enumerate(NORMALIZING_VARIABLES_TO_RUN)
    )

    html = f"""
    <html><head><meta charset="utf-8"><style>
    body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 40px; color: #222; }}
    h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 32px; border-bottom: 1px solid #ccc; }}
    h3 {{ font-size: 14px; margin-top: 20px; }} h4 {{ font-size: 12px; margin-top: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 11px; margin: 10px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
    th {{ background: #f2f2f2; }} td:first-child, th:first-child {{ text-align: left; }}
    img {{ max-width: 100%; }}
    .note {{ font-size: 11px; color: #555; }}
    </style></head><body>

    <h1>The Effects of Iran War Risk on Global Financial Markets (2026)</h1>
    <p class="note">Replication of Rigobon &amp; Sack (2003), "The Effects of War Risk on
    U.S. Financial Markets," applying their heteroskedasticity-based identification
    method to the 2026 Iran war, with the H/L "war news" day classification derived from
    an NLP pipeline over GDELT-indexed news coverage rather than hand-picked from press
    commentary. Analysis window: {ANALYSIS_START} to {ANALYSIS_END}.</p>

    <h2>Figure 1: Iran War-Risk News Intensity</h2>
    <img src="{fig_uri}">

    <h2>Table 1: H/L Regime Dates</h2>
    {table1_html}

    <h2>NLP Component Evaluation</h2>
    {nlp_eval_block}
    <p class="note">Inter-method correlation between the three article-level daily
    scores (lexicon, TF-IDF tone, FinBERT) is reported in nlp_eval_report.json.</p>

    {normalizing_sections}

    <h2>Figure 2: Why Brent Oil Identifies the Model Better Than the 2yr Treasury Yield</h2>
    <img src="{fig2_uri}">
    <p class="note">Panel (a): the identification condition itself -- a normalizing
    variable's own variance should jump on war-news days; Brent oil's does (11.2x),
    the 2yr Treasury yield's barely moves (1.3x). Panel (b): |t-statistics| for every
    other variable's estimated sensitivity, by normalizing-variable choice -- the
    oil-normalized specification clears conventional significance (|t|>2) almost
    everywhere the Treasury-normalized one doesn't. Panel (c): a scale-free instability
    score (swing across three alternative H-day thresholds, relative to each variable's
    own typical magnitude) computed for every variable, not just one example -- the
    Treasury specification's typical swing sits at the boundary where a sign flip
    becomes plausible; the oil specification's sits mostly below it.</p>

    <h2>Discussion</h2>
    <p><b>Two normalizing-variable specifications tell different stories.</b> The primary
    specification (2-year Treasury yield, matching Rigobon &amp; Sack 2003 exactly) is
    weakly identified: the yield's own variance is not clearly elevated on H-days in this
    sample, unlike Iraq 2003 where it was roughly 6x higher on war-news days. The
    secondary specification (Brent crude) is stable and statistically strong across
    nearly every variable.</p>
    <p><b>Economic interpretation.</b> Against the oil anchor, equities fall, volatility
    and the dollar rise, and credit spreads widen -- a classic risk-off pattern. But
    Treasury yields and inflation breakevens <i>rise</i>, not fall, with war-risk-driven
    oil moves -- the opposite mechanism from Iraq 2003's flight-to-safety shock. The 2026
    Iran war appears to transmit primarily as an oil supply shock (inflationary, hawkish
    for rates), consistent with its actual character as a Strait-of-Hormuz disruption to
    global energy supply.</p>
    <p><b>A counterintuitive, robust result: gold falls</b> (coefficient consistently
    negative, t≈-3.4). One plausible reading is that broad-dollar strength (itself
    significant, t=6.1) mechanically pressures dollar-denominated gold even as "fear"
    demand might otherwise push it up -- flagged as an open question, not resolved.</p>

    <h2>Limitations</h2>
    <ul>
      <li>The core identifying assumption -- that only the Iran war-risk factor's
      variance shifts between H and L days -- was operationalized via a confound
      calendar (FOMC meetings, etc.) but cannot be exhaustively verified; flagged days
      are reported in Table 1's "Confound flag" column.</li>
      <li>The on-the-run 10-year Treasury liquidity premium (present in the original
      paper's 9-variable set) is omitted from this Phase-A analysis; see Phase B for the
      Gürkaynak-Sack-Wright-based extension.</li>
      <li>News-to-market trading-calendar alignment uses a documented next-trading-day
      roll-forward convention; perfect alignment across US/Gulf/Israeli market hours is
      not achievable with daily data.</li>
      <li>GDELT's artlist mode caps at 250 records/request, which can bind on the
      highest-intensity days; the primary intensity signal instead uses GDELT's
      uncapped timeline series specifically to avoid this bias.</li>
      <li>The 2yr-Treasury-normalized specification shows weak identification (the
      normalizing variable's own variance is not clearly elevated on H-days) -- reported
      alongside the Brent-oil-normalized specification, which is stable and economically
      coherent, rather than suppressed, since the contrast is itself informative about
      the war's transmission channel (oil-supply shock vs. classic flight-to-safety).</li>
      <li>Reported t-stats/p-values are not corrected for multiple comparisons across
      13 variables x 3 estimators x 2 normalizing-variable specifications.</li>
    </ul>

    </body></html>
    """
    return html


def main():
    html = build_html()
    html_path = REPORT_DIR / "report.html"
    html_path.write_text(html)
    print(f"Saved HTML -> {html_path}")

    from weasyprint import HTML
    pdf_path = REPORT_DIR / "report.pdf"
    HTML(string=html).write_pdf(str(pdf_path))
    print(f"Saved PDF -> {pdf_path}")


if __name__ == "__main__":
    main()
