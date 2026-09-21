"""Assemble the standalone PDF report: methodology, Table 1, Table 2/3 for each
normalizing variable (primary: 2yr Treasury yield, matching the original paper;
secondary: Brent crude, motivated by the war's oil-supply-shock character), the
Iraq-vs-Iran comparison (treasury_2y run only), NLP evaluation, robustness summaries,
discussion, and limitations. Builds an HTML file from live processed data then renders
it to PDF via weasyprint.

Math is rendered to small PNGs using matplotlib's mathtext (no external LaTeX install
needed, works fine since weasyprint just needs a static image, not a live browser).
"""
import base64
import io
import json

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    ANALYSIS_END, ANALYSIS_START, DATA_PROCESSED, DATA_RAW, NORMALIZING_VARIABLES_TO_RUN,
    PROJECT_ROOT, REPORT_DIR, US_OIL_PRODUCTION_CONTEXT,
)
import make_table1
import make_table2
import make_table3
import make_table_three_regime

FIGURE1_PATH = PROJECT_ROOT / "notebooks" / "figure1_intensity_timeline.png"
FIGURE2_PATH = PROJECT_ROOT / "notebooks" / "figure2_oil_vs_treasury.png"
FIGURE3_PATH = PROJECT_ROOT / "notebooks" / "figure3_var_covar.png"

_EQ_CACHE = {}


def eq(tex: str, fontsize: float = 14, block: bool = True) -> str:
    """Render a LaTeX-ish math string to an inline base64 PNG using matplotlib's
    mathtext engine. No matrix environments (mathtext does not support them), so
    matrix relations are written out as separate scalar equations instead."""
    key = (tex, fontsize)
    if key not in _EQ_CACHE:
        fig = plt.figure()
        t = fig.text(0, 0, f"${tex}$", fontsize=fontsize)
        fig.canvas.draw()
        bbox = t.get_window_extent()
        width, height = bbox.width / fig.dpi, bbox.height / fig.dpi
        plt.close(fig)
        fig = plt.figure(figsize=(width, height))
        fig.text(0, 0, f"${tex}$", fontsize=fontsize)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        _EQ_CACHE[key] = base64.b64encode(buf.getvalue()).decode()
    style = "display:block;margin:10px auto;" if block else "vertical-align:middle;"
    return f'<img src="data:image/png;base64,{_EQ_CACHE[key]}" style="{style}">'


def image_data_uri(path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def false_negative_section() -> str:
    """Novel-phrasing analysis: what vocabulary is the lexicon relevance filter
    actually missing, and is it genuinely new war-specific language or just a narrow
    hand-built list skipping something generic."""
    path = DATA_PROCESSED / "false_negative_terms.parquet"
    if not path.exists():
        return ""
    terms = pd.read_parquet(path)
    if terms.empty:
        return ""
    table_html = terms.to_html(index=False, border=0, na_rep="-")
    return f"""
    <h3>What vocabulary is the relevance filter actually missing</h3>
    <p>Looking only at false negatives that already contain the word "iran" but still
    got no relevance-term hit, so this is not just re-showing the already-known
    missing-the-word-iran problem from above, this is genuinely missed vocabulary:</p>
    {table_html}
    <ul class="points">
      <li>A term here being new-war-specific (a fresh operation codename, a new
      sanctions program, a new weapons system) means even an updated, general-purpose
      NLP tool would likely miss it too, it is not something our hand-built list did
      wrong specifically.</li>
      <li>A term here being generic (a place name, a diplomacy word) just means our
      narrow list happened to skip something a slightly broader list would have
      caught, an easy, low-risk fix.</li>
    </ul>
    """


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
    parts.append("""
    <ul class="points">
      <li>Stability check splits the war into two halves and re-estimates in each.
      If a variable's coefficient jumps around a lot or flips sign between the two
      halves, that variable's estimate is not to be trusted much, the war's phases
      were pretty different (initial strikes vs. later ceasefire-and-collapse) so
      this is a real check, not a formality.</li>
      <li>Threshold sensitivity redoes the whole H/L split at three different cutoffs
      (85th/90th/95th percentile of intensity) and shows how much the coefficient
      moves. Small movement means the result does not depend on us picking exactly
      the top decile as H.</li>
    </ul>
    """)
    return "\n".join(parts)


def normalizing_variable_section(normalizing_variable: str, is_primary: bool) -> str:
    table2_html, comparison_html = make_table2.build(normalizing_variable)
    table3_html = make_table3.build(normalizing_variable)
    label = f"{normalizing_variable}" + (" (primary, matches Rigobon &amp; Sack 2003)" if is_primary
                                          else " (secondary experiment: Brent crude as the direct Iran/Hormuz transmission channel)")
    section = f"""
    <h2>Results, normalizing variable: {label}</h2>
    <p>For every other variable x_j we get three estimates of the same war-risk
    sensitivity coefficient d, all from the same Omega_H and Omega_L matrices:</p>
    {eq(r"\hat{d}=\Delta\Omega_{22}/\Delta\Omega_{21}\ \ (Eq.6)\quad\quad \hat{d}=\Delta\Omega_{21}/\Delta\Omega_{11}\ \ (Eq.7)")}
    <p>plus the IV versions (omega1, omega2, and the combined omega3 which also gives
    a Sargan/J overidentification test). If the model is a good fit for a variable,
    all of these should land close to each other.</p>
    <h3>Table 2: Estimated Impact of Increase in Iran War Risk</h3>
    {table2_html}
    <ul class="points">
      <li>Eq(6) and Eq(7) are two different formulas for the same coefficient, they
      only agree if the model's assumptions actually hold for that variable, so a
      big gap between them is itself a warning sign (see the dedicated section below
      comparing this across the two normalizing variables).</li>
      <li>IV omega1/omega2/omega3 columns are the same coefficient again, done as a
      real IV regression so we get proper t-stats. Look at the t-stat column, not
      just the coefficient, to judge if a result is usable.</li>
      <li>Sargan p-value below 0.05 means the omega1 and omega3 instruments disagree
      more than chance would explain, a soft sign the identifying assumption is
      shaky for that variable.</li>
      <li>Bootstrap SE resamples H-days and L-days separately, 500 times, and is a
      second, non-analytic check on the standard errors.</li>
    </ul>
    """
    if comparison_html:
        section += f"""
        <h3>Comparison to Rigobon &amp; Sack (2003): Iraq 2003 vs. Iran 2026</h3>
        {comparison_html}
        <ul class="points">
          <li>Same direction column just checks if the sign matches the 2003 Iraq
          result. A "No" is not automatically wrong, the two wars are different
          (oil shock vs. flight to safety, see Discussion), it is just flagged so it
          is not missed.</li>
        </ul>
        """
    section += f"""
    <h3>Table 3: Variance Decomposition</h3>
    <p>Following the original paper, this is a lower-bound estimate of how much of
    each variable's own variance is explained by the war-risk factor:</p>
    {eq(r"\%\ explained = \hat{d}^2\cdot\Delta Var(x_1)\ /\ Var(x_j)")}
    {table3_html}
    <ul class="points">
      <li>Values above 100% do happen here and are noted as a sign of a noisy or
      weak estimate for that variable, not a literal claim that war risk explains
      more than all of the variance.</li>
      <li>"% explained (H days)" and "(all days)" differ because the denominator
      changes, H-day variance is naturally higher, so the same predicted variance
      is a smaller share of it than of the full-sample variance.</li>
    </ul>
    <h3>Robustness Checks</h3>
    {robustness_block_html(normalizing_variable)}
    """
    return section


def three_regime_hypothesis_check() -> str:
    """The brief's exact hypothesis, checked directly: bad war news should mean
    yields and oil up, equities down, good war news the reverse. This looks at raw
    average daily changes by regime (not the IV sensitivity coefficients above, which
    answer a different question, how xj moves per unit move in x1), so it is a direct,
    independent check of the brief's own stated expectation, not a restatement of
    Table 2."""
    panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")
    is_bad = panel["is_H"] & (panel["direction"] == "Increased")
    is_good = panel["is_H"] & (panel["direction"] == "Decreased")
    is_l = panel["is_L"]
    n_bad, n_good, n_l = int(is_bad.sum()), int(is_good.sum()), int(is_l.sum())

    check_vars = {
        "brent_oil": "oil", "treasury_2y": "2yr yield", "treasury_10y": "10yr yield",
        "bbb_spread": "BBB spread", "hy_spread": "HY spread", "sp500": "S&P 500",
        "em_equity": "EM equities", "israel_equity": "Israel equities", "vix": "VIX",
    }
    rows = []
    for var, label in check_vars.items():
        col = f"d_{var}"
        rows.append({
            "Variable": label,
            "Mean Δ, bad news": panel.loc[is_bad, col].mean(),
            "Mean Δ, good news": panel.loc[is_good, col].mean(),
            "Mean Δ, calm (L)": panel.loc[is_l, col].mean(),
        })
    table = pd.DataFrame(rows)
    table_html = table.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")

    # A rough scorecard: does each variable's bad-news sign match the brief's expected
    # direction (oil/yields/spreads up, equities/vix... brief only names oil, yields,
    # equities explicitly, so score those three groups), and does good news flip it.
    expect_up_on_bad = {"brent_oil", "treasury_2y", "treasury_10y", "bbb_spread", "hy_spread"}
    expect_down_on_bad = {"sp500", "em_equity", "israel_equity"}
    hits = 0
    checked = 0
    for var in expect_up_on_bad | expect_down_on_bad:
        bad_mean = panel.loc[is_bad, f"d_{var}"].mean()
        if pd.isna(bad_mean):
            continue
        checked += 1
        if var in expect_up_on_bad and bad_mean > 0:
            hits += 1
        elif var in expect_down_on_bad and bad_mean < 0:
            hits += 1

    return f"""
    <h2>Extension: splitting H-days into bad vs. good war news</h2>
    <h3>Checking the brief's exact hypothesis directly: bad news up for oil and
    yields, down for equities, good news the reverse</h3>
    <p>This is a different, more direct check than the IV coefficients below, it just
    looks at the raw average daily change in each variable, split by bad-news H-days
    ({n_bad} days), good-news H-days ({n_good} days), and calm L-days ({n_l} days),
    with no normalizing variable involved at all:</p>
    {table_html}
    <ul class="points">
      <li>On bad news: oil and both yields move up, both credit spreads widen, the
      S&amp;P 500 falls, matching the brief's stated expectation on {hits}/{checked}
      of the variables it names directly.</li>
      <li>On good news the pattern mostly reverses, oil and yields fall, spreads
      narrow, and the S&amp;P 500 rises sharply (+{table.loc[table['Variable']=='S&P 500', 'Mean Δ, good news'].values[0]:.1f}
      on an average good-news day), a clean, if small-sample, confirmation that
      direction of news matters in the way the brief expects.</li>
      <li>Not everything lines up (Israel equities fall on both bad and good news
      here, EM equities barely move on bad news), and n=5 bad-news days is genuinely
      thin, so read this as a real, directionally-consistent signal, not a settled
      result.</li>
    </ul>
    """


def three_regime_section(normalizing_variable: str) -> str:
    """Extension: split H-days into bad war news (war risk escalating) vs. good war
    news (de-escalating), re-run the same estimator on bad-vs-L, good-vs-L, and
    bad-vs-good. This sits on top of the main H-vs-L Table 2 above, it does not
    replace it."""
    panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")
    n_bad = int((panel["is_H"] & (panel["direction"] == "Increased")).sum())
    n_good = int((panel["is_H"] & (panel["direction"] == "Decreased")).sum())
    n_l = int(panel["is_L"].sum())

    pair_titles = {
        "bad_vs_L": "Bad war news vs. calm days",
        "good_vs_L": "Good war news vs. calm days",
        "bad_vs_good": "Bad war news vs. good war news",
    }
    tables_html = ""
    for pair, title in pair_titles.items():
        path = DATA_PROCESSED / f"table_three_regime_{pair}_{normalizing_variable}.parquet"
        if not path.exists():
            continue
        table_html = make_table_three_regime.build(pair, normalizing_variable)
        tables_html += f"<h4>{title}</h4>\n{table_html}\n"

    return f"""
    <h3>Full estimator on the same split, normalizing variable: {normalizing_variable}</h3>
    <p><b>Read the N columns before anything else in this section.</b> Bad-news H-days
    = {n_bad}, good-news H-days = {n_good}, L-days = {n_l}. These are small groups on
    top of an already-small H set, so every coefficient below is a rough signal, not
    something to lean on the way the main Table 2 numbers above can be leaned on. This
    is the same spirit as flagging the treasury_2y weak-identification result instead
    of hiding it, small-N problems get shown, not smoothed over.</p>
    <p>"Bad" (war risk escalating) and "good" (war risk de-escalating) come from the
    `direction` column already used in Table 1, built off GDELT's own day-over-day tone
    change, so it exists for every day, not just the ones with article-level text. The
    same Eq(6)/(7) and IV omega1/omega2/omega3 machinery from the main Table 2 is
    reused as-is here, just on these smaller subsets.</p>
    {tables_html}
    <ul class="points">
      <li>bad_vs_L and good_vs_L answer: does the direction of the news matter for the
      size or sign of the market reaction, or does any high-variance war-news day move
      markets about the same regardless of whether the news itself was good or bad.</li>
      <li>bad_vs_good is the most direct version of that question, comparing the two
      escalation directions to each other instead of each to calm days separately.</li>
      <li>If a variable's coefficient flips sign between bad_vs_L and good_vs_L, that
      is actually informative (it means direction matters), it is only a problem if
      the flip looks like pure noise given how few days are behind it.</li>
    </ul>
    """


def identification_alternatives_section() -> str:
    """Direct answer to: is heteroskedasticity-based identification the best approach
    here, and what else could we have used."""
    articles_path = DATA_RAW / "gdelt_articles.parquet"
    polymarket_note = ""
    polymarket_hits = 0
    if articles_path.exists():
        articles = pd.read_parquet(articles_path)
        poly = articles[articles["title"].fillna("").str.contains("polymarket", case=False)]
        polymarket_hits = len(poly)
        if polymarket_hits:
            example = poly["title"].iloc[0]
            polymarket_note = f"""
            <p>Checked, not just guessed: {polymarket_hits} article(s) in our own
            corpus already mention Polymarket, for example "{example}". A
            prediction-market instrument is not a hypothetical here, the raw
            material is already sitting in gdelt_articles.parquet, unused.</p>
            """
        else:
            polymarket_note = """
            <p>Checked, not just guessed: this run's corpus sample happens to have
            zero Polymarket-mentioning articles (a previous, differently-chunked
            sample of the same underlying news did have some, see AI_USE.md), so
            this specific corpus doesn't hand us the instrument for free this time.
            The idea itself doesn't depend on our corpus though, a real prediction
            market's odds series can just be pulled directly rather than mined out
            of scraped headlines.</p>
            """
    return f"""
    <h2>Is heteroskedasticity-based identification the best approach here?</h2>
    <p>Short answer: it is a reasonable primary choice for this project, but not
    obviously the only good one, and it is worth being honest about where it is
    strong and where it is not.</p>
    <p><b>Why it fits here.</b> We never have to sign or size any individual headline,
    which matters a lot given how mixed war coverage actually reads day to day (see
    the ceasefire-spike discussion under Figure 1). It also does not need an excluded
    instrument, which would be hard to defend for something as pervasive as war risk
    touching almost every market variable at once.</p>
    <p><b>Where it is weak.</b> The whole thing rests on one assumption we cannot
    check directly (see the assumption section above), and this report already has
    two independent pieces of evidence that it is cracking for the treasury_2y
    specification: the Eq(6)/Eq(7) disagreement and the Sargan rejections.</p>
    <p><b>Alternatives worth naming:</b></p>
    <ul class="points">
      <li><b>Narrative event-study with hand-picked dates.</b> The original papers'
      own baseline approach. More interpretable, but brings back the subjective
      day-picking this whole NLP pipeline was built to remove.</li>
      <li><b>A continuous text-based shock index as a direct regressor</b> (in the
      style of the Baker-Bloom-Davis Economic Policy Uncertainty index). Gives a
      signed, sized measure of war risk instead of a binary flag, but then the whole
      exercise depends on trusting that index to actually measure war risk correctly,
      which is exactly the quantification problem heteroskedasticity ID exists to
      sidestep.</li>
      <li><b>A market-priced instrument.</b> Something like Polymarket's war/ceasefire
      odds, continuous, real-time, and priced by people with money on the outcome,
      arguably a better proxy for "the market's own read of war risk" than anything
      built from text.{polymarket_note}</li>
      <li><b>GARCH or regime-switching directly on the financial series.</b> Models the
      variance shift structurally instead of assuming a clean two-bucket split, at the
      cost of real specification risk (model order, number of regimes) that
      heteroskedasticity ID avoids by construction.</li>
      <li><b>Local projections or a structural VAR with sign restrictions.</b> Flexible
      on dynamics (this whole project is static, same-day only, no lag structure), but
      sign restrictions need exactly the kind of "which direction does war risk move
      things" assumption this project was trying to avoid needing.</li>
    </ul>
    <p><b>Our actual call:</b> keep heteroskedasticity ID as the primary method, it
    matches both reference papers and the assignment's own approach. But a Polymarket-
    odds instrument looks like the strongest, cheapest robustness cross-check to add,
    {"since the data is already sitting unused in our own corpus" if polymarket_hits
    else "even though this particular corpus sample didn't happen to surface any usable mentions of it"}.</p>
    """


def eq10_comparison_section() -> str:
    """Direct answer to: does the IV form of Eq(10)/(7) actually match the closed-form
    ratio the way the paper's own footnote says it should, and does that hold the
    same way for both normalizing variables?"""
    panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")
    rows_summary = []
    tables = {}
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        h = panel.loc[panel["is_H"], f"d_{nv}"].dropna()
        l = panel.loc[panel["is_L"], f"d_{nv}"].dropna()
        t2 = pd.read_parquet(DATA_PROCESSED / f"table2_estimates_{nv}.parquet")
        t2 = t2[["variable", "eq7_direct", "iv_w1_coef"]].copy()
        t2["ratio_iv_over_eq7"] = t2["iv_w1_coef"] / t2["eq7_direct"]
        tables[nv] = t2
        rows_summary.append({
            "Normalizing variable": nv,
            "Mean daily change on H days": round(h.mean(), 4),
            "Mean daily change on L days": round(l.mean(), 4),
            "Std dev of daily change (all days)": round(panel[f"d_{nv}"].std(), 4),
        })
    summary_df = pd.DataFrame(rows_summary)
    summary_html = summary_df.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")

    treasury_tbl = tables["treasury_2y"].to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}", na_rep="-")
    oil_tbl = tables["brent_oil"].to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}", na_rep="-")

    return f"""
    <h2>Eq(10): does the paper's own equivalence actually hold here?</h2>
    <p>The paper (Rigobon &amp; Sack 2003, p.6) writes the omega1 IV estimator as a
    closed form and states it is identical to Eq(7):</p>
    {eq(r"\hat{d}=\frac{Cov_H(\Delta x_1,\Delta x_2)-Cov_L(\Delta x_1,\Delta x_2)}{Var_H(\Delta x_1)-Var_L(\Delta x_1)}\ \ (Eq.10)")}
    <p>That equivalence is only exact if Delta x has zero mean (the paper says this
    outright in its own footnote 2), which real market data will not do exactly. Our
    code computes Eq(7) as a plain covariance ratio (which de-means within each
    regime) and computes the IV version (omega1_coef in Table 2) as a real pooled
    2SLS regression with an intercept, so the two can drift apart whenever the H-day
    and L-day means of the normalizing variable are not close to each other relative
    to its own spread:</p>
    {summary_html}
    <p>2-year Treasury yield changes are tiny to begin with, so even a small
    difference between its H-day and L-day mean is large relative to its own
    variance, and Eq(7) and the IV omega1 estimate pull apart a lot for several
    variables. Brent crude's day-to-day moves are much bigger, so the same kind of
    mean difference barely matters and the two estimates stay close, which is what
    the paper's own algebra predicts when its assumptions are closer to holding.</p>
    <h4>treasury_2y: Eq(7) vs. IV omega1</h4>
    {treasury_tbl}
    <h4>brent_oil: Eq(7) vs. IV omega1</h4>
    {oil_tbl}
    <ul class="points">
      <li>ratio_iv_over_eq7 close to 1.0 is good, it means the IV route and the
      closed-form route agree like the paper says they should.</li>
      <li>For treasury_2y a few variables (dollar_index, broad_dollar, vix) have
      ratios wildly away from 1.0, this is on top of the weak-identification
      evidence already shown in Figure 2, not a separate new problem.</li>
      <li>For brent_oil almost every ratio sits reasonably close to 1.0, gold is the
      one exception, which lines up with gold being flagged as the odd result out
      in the Discussion.</li>
    </ul>
    """


def _own_variance_ratio(normalizing_variable: str) -> float:
    with open(DATA_PROCESSED / f"event_study_omegas_{normalizing_variable}.json") as f:
        omegas = json.load(f)["results"]
    xj0 = next(iter(omegas))
    var_h = omegas[xj0]["omega_H"][0][0]
    var_l = omegas[xj0]["omega_L"][0][0]
    return var_h / var_l


def variance_covar_figure_section() -> str:
    fig3_uri = image_data_uri(FIGURE3_PATH) if FIGURE3_PATH.exists() else ""
    oil_ratio = _own_variance_ratio("brent_oil")
    treasury_ratio = _own_variance_ratio("treasury_2y")
    return f"""
    <h2>Figure 3: raw variance and covariance on war-news vs. calm days</h2>
    <p>Every estimate in this report comes out of Omega_H and Omega_L, the covariance
    matrices of daily changes computed separately on H-days and L-days:</p>
    {eq(r"\Omega \equiv E\left([\Delta x_1\ \Delta x_2]'\,[\Delta x_1\ \Delta x_2]\right)")}
    <p>This figure just plots those raw numbers directly, instead of a ratio or a
    t-stat, so it is possible to see what is actually feeding the estimator.</p>
    <img src="{fig3_uri}">
    <ul class="points">
      <li>Top row: the normalizing variable's own variance on L-days vs. H-days.
      Brent oil's variance jumps about {oil_ratio:.1f}x on war-news days, the Treasury
      yield's only moves about {treasury_ratio:.1f}x, the same identification gap
      Figure 2 shows, here in raw units instead of a normalized ratio.</li>
      <li>Bottom row: covariance of the normalizing variable with every other
      variable, H-days vs. L-days. S&amp;P 500 and gold dominate the y-axis in both
      panels purely because they are large, unstandardized numbers (price-level
      changes vs. a small yield or a $ oil move), not because they are the most
      "explained" variables, Table 2/3's ratios already correct for this scale
      difference.</li>
    </ul>
    """


def assumption_section() -> str:
    panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")
    n_h = int(panel["is_H"].sum())
    n_l = int(panel["is_L"].sum())
    n_confound_h = int((panel["is_H"] & panel["confound_flag"]).sum())
    n_confound_l = int((panel["is_L"] & panel["confound_flag"]).sum())
    sargan_counts = {}
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        t2 = pd.read_parquet(DATA_PROCESSED / f"table2_estimates_{nv}.parquet")
        sargan_counts[nv] = int((t2["iv_w3_sargan_pvalue"] < 0.05).sum())
    return f"""
    <h2>The assumption on "non-war" effects, and what our results say about it</h2>
    <p>The whole method rests on one assumption (Rigobon &amp; Sack 2003, p.5): only
    the variance of the war-risk factor z1 changes between H-days and L-days.
    Everything else, monetary policy news, other macro data, unrelated shocks, is
    assumed to keep the same variance in both sets of days. That is what lets us
    blame the entire change in Omega on war risk alone:</p>
    {eq(r"\Delta\Omega_{11}=\Delta\sigma^2(z_1)\qquad \Delta\Omega_{21}=\hat{d}\cdot\Delta\sigma^2(z_1)\qquad \Delta\Omega_{22}=\hat{d}^2\cdot\Delta\sigma^2(z_1)")}
    <p>We cannot verify this assumption directly since z1 is unobserved by design,
    but we can look for cracks in it:</p>
    <ul class="points">
      <li><b>Confound calendar.</b> Of {n_h} H-days, {n_confound_h} land on a known
      FOMC date. Of {n_l} L-days, {n_confound_l} do. So the obvious, known
      confound (Fed meetings) is basically not contaminating the H set, which is
      good, but this only covers confounds we thought to list in advance, it says
      nothing about a confound we did not think of.</li>
      <li><b>Eq(6) vs. Eq(7) disagreement.</b> covered in detail above, this is
      really a live test of the same assumption: if some other factor's variance
      also moved on our H-days, the ΔΩ matrix stops having the clean shape the
      assumption predicts, and Eq(6)/Eq(7) stop agreeing. They disagree a lot for
      treasury_2y and mostly agree for brent_oil.</li>
      <li><b>Sargan/J rejections.</b> {sargan_counts.get('treasury_2y', 0)}/13
      variables reject at 5% under treasury_2y, {sargan_counts.get('brent_oil', 0)}/13
      under brent_oil. Same story again, more rejections means more variables where
      the two instruments (omega1, omega2) are not telling a consistent story, which
      is what you would expect if a non-war factor is also driving Omega_H versus
      Omega_L for that variable.</li>
    </ul>
    <p><b>What this predicts, put simply:</b> the treasury_2y specification is more
    likely picking up a mix of war risk and other macro noise (rate expectations,
    Fed-adjacent moves not on our confound calendar), since the 2-year yield reacts
    to a lot more than just war risk day to day. Brent oil is a more direct,
    single-channel bet (Hormuz disruption to physical oil supply), so there are
    simply fewer other big factors competing to move its variance on the same days,
    and the assumption ends up closer to true for it. This is the same conclusion
    Figure 2 already points to, from a completely different angle.</p>
    """


def improvements_section(nlp_eval: dict, nlp_zero_pct: float) -> str:
    recall = nlp_eval.get("recall", float("nan"))
    precision = nlp_eval.get("precision", float("nan"))
    n_labeled = nlp_eval.get("n_labeled", "?")
    return f"""
    <h2>Where this analysis could be improved</h2>
    <ul class="points">
      <li><b>Learn the relevance filter instead of hand-coding it.</b> The lexicon's
      recall is only {recall:.2f} because it needs the literal word "iran" plus a
      fixed term list (see the NLP evaluation above and the false-negative check for
      concrete examples it misses). We already have {n_labeled} Claude-judged labels
      sitting there, a simple classifier trained on those (even just logistic
      regression on TF-IDF features) would likely beat the hand-built keyword rule
      without much extra work.</li>
      <li><b>Get a second, independent labeler for the eval set.</b> Right now
      relevance judgments are single-pass, from one LLM reading each headline once.
      A second human or model pass with an inter-annotator agreement score (Cohen's
      kappa) would tell us how much to trust precision/recall =
      {precision:.2f}/{recall:.2f} in the first place.</li>
      <li><b>Full VAR pre-whitening instead of per-variable AR(1).</b> The original
      paper filters serial correlation with a VAR across all variables together,
      here we only remove each variable's own lag-1 autocorrelation. A full VAR
      would also catch cross-variable lead-lag effects (e.g. oil today predicting
      equities tomorrow), which the current AR(1) step cannot see.</li>
      <li><b>Robust scaling for the zero-inflated NLP components.</b> Lexicon, tone
      and FinBERT are zero on about {nlp_zero_pct:.0%} of days then spike hard on the
      rest (see the Limitations note on z-scoring). A rank-based or median/IQR
      standardization would probably be less sensitive to those spikes than a
      mean/std z-score.</li>
      <li><b>Test more than 2 normalizing variables, and pick with a rule, not by
      hand.</b> We ran treasury_2y (for comparability to 2003) and brent_oil (on a
      theory-driven hunch). A cleaner approach: compute the Var_H/Var_L ratio for
      every candidate variable up front (Figure 2 panel a's logic) and pick whichever
      clears some threshold, before looking at any downstream coefficients.</li>
      <li><b>A placebo test on the H/L split itself.</b> Randomly relabel H/L many
      times (keeping the same counts) and see how often Eq(6) and Eq(7) agree, or
      Sargan rejects, by pure chance. That would turn today's "these numbers seem
      more stable for oil than for treasury" into an actual p-value.</li>
      <li><b>Add the GSW liquidity premium (Phase B).</b> Already scoped in
      liquidity_premium_gsw.py but not run, this would bring the variable set fully
      in line with the original paper's 9 variables plus our 5 new ones.</li>
    </ul>
    """


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
        (see AI_USE.md), not independent human annotation.</p>
        <ul class="points">
          <li>High precision, low recall basically means the filter is picky, when
          it says an article is relevant it usually is, but it lets a lot of real,
          relevant articles slip past because they don't use its exact vocabulary.</li>
        </ul>
        """

    normalizing_sections = "\n".join(
        normalizing_variable_section(nv, is_primary=(i == 0))
        for i, nv in enumerate(NORMALIZING_VARIABLES_TO_RUN)
    )
    three_regime_sections = "\n".join(
        three_regime_section(nv) for nv in NORMALIZING_VARIABLES_TO_RUN
    )

    # Data-driven GDELT-cap severity numbers for the Limitations bullet below, computed
    # fresh each run instead of hardcoded, since these change with the analysis window.
    articles = pd.read_parquet(DATA_RAW / "gdelt_articles.parquet")
    articles["date"] = pd.to_datetime(articles["date"])
    n_days_with_articles = articles["date"].nunique()
    total_window_days = (pd.Timestamp(ANALYSIS_END) - pd.Timestamp(ANALYSIS_START)).days + 1
    n_days_without_articles = total_window_days - n_days_with_articles

    # Dynamic Discussion callouts, computed fresh each run instead of hardcoded.
    t2_oil = pd.read_parquet(DATA_PROCESSED / "table2_estimates_brent_oil.parquet").set_index("variable")
    gold_tstat = t2_oil.loc["gold", "iv_w1_tstat"]
    broad_dollar_tstat = t2_oil.loc["broad_dollar", "iv_w1_tstat"]

    # Dynamic NLP-quality callouts (recall/precision/max z-score/zero-inflation), used
    # in both the improvements and limitations sections below.
    di = pd.read_parquet(DATA_PROCESSED / "daily_intensity.parquet")
    nlp_zero_pct = di[["lexicon", "tone", "finbert"]].eq(0).mean().mean()
    nlp_max_z = di[["lexicon_z", "tone_z", "finbert_z"]].max().max()

    html = f"""
    <html><head><meta charset="utf-8"><style>
    @page {{ size: A4 landscape; margin: 15mm; }}
    body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; color: #222; }}
    h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 32px; border-bottom: 1px solid #ccc; }}
    h3 {{ font-size: 14px; margin-top: 20px; }} h4 {{ font-size: 12px; margin-top: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 9.5px; margin: 10px 0; table-layout: auto; }}
    th, td {{ border: 1px solid #ddd; padding: 3px 5px; text-align: right; word-break: break-word; }}
    th {{ background: #f2f2f2; }} td:first-child, th:first-child {{ text-align: left; }}
    img {{ max-width: 100%; }}
    .note {{ font-size: 11px; color: #555; }}
    ul.points {{ font-size: 12px; }}
    ul.points li {{ margin-bottom: 4px; }}
    </style></head><body>

    <h1>The Effects of Iran War Risk on Global Financial Markets (2026)</h1>
    <p class="note">This is a replication of Rigobon &amp; Sack (2003), "The Effects
    of War Risk on U.S. Financial Markets," using their heteroskedasticity-based
    identification method on the 2026 Iran war. The main change from the original
    paper is that the H/L "war news" day classification comes out of an NLP pipeline
    run over GDELT-indexed news, instead of being picked by hand from press
    commentary. Analysis window: {ANALYSIS_START} to {ANALYSIS_END}.</p>

    <h2>Figure 1: Iran War-Risk News Intensity</h2>
    <p>The composite intensity score plotted below combines GDELT's own news volume
    with three NLP scores (lexicon, TF-IDF tone, FinBERT), each z-scored and summed
    with fixed weights:</p>
    {eq(r"I_t = 0.30\,z(vol_{pct}) + 0.20\,z(vol_{raw}) + 0.15\,z(lex) + 0.15\,z(tone) + 0.20\,z(finbert)")}
    <img src="{fig_uri}">

    <h3>A possible explanation for some visible counterintuitive results</h3>
    <p>2026-04-08, the day the ceasefire begins, still comes out as an H-day (top
    decile intensity) rather than calm, and that is not a bug. The composite index
    measures news-driven variance and attention, not direction, so a sudden ceasefire
    announcement is itself a large, uncertain, breaking-news event and can spike the
    index just as a sudden attack would. It also is not a clean "good news" day even
    on the ground, the independently-verified war timeline (see
    IRAN_WAR_TIMELINE in config.py, checked against Wikipedia/Britannica/House of
    Commons Library sources during planning, not derived from the sparse article
    sample) records that Israel resumed strikes on Lebanon that same day, so the
    theatre widened right as the US-Iran ceasefire took hold. This fits the
    identification strategy rather than breaking it, Rigobon &amp; Sack only need
    elevated variance on H-days, not a particular sign of news, so a high-uncertainty
    day with a ceasefire in one theatre and fresh strikes in another is a fair H-day
    even though it looks odd at first glance.</p>

    <h2>Table 1: H/L Regime Dates</h2>
    {table1_html}
    <ul class="points">
      <li>Regime column is H (top decile intensity, war-news day) or L (matched
      calm day picked close to an H-day, following the same logic the original
      paper used).</li>
      <li>Confound flag marks a day that also happens to be an FOMC date, it is kept
      in the sample and just flagged, not removed, so this can be checked rather
      than hidden.</li>
      <li>This table only lists H/L days that actually have a headline available.
      Most H/L days do not (see the Limitations note on the GDELT artlist cap
      below for why) and are left out of this display rather than shown blank,
      the full H/L date list, headline or not, is still in
      table1_regime_dates.csv and used as-is everywhere else in this report.</li>
    </ul>

    <h2>NLP Component Evaluation</h2>
    {nlp_eval_block}
    <p class="note">Inter-method correlation between the three article-level daily
    scores (lexicon, TF-IDF tone, FinBERT) is reported in nlp_eval_report.json.</p>

    {false_negative_section()}

    {normalizing_sections}

    {three_regime_hypothesis_check()}

    {three_regime_sections}

    {eq10_comparison_section()}

    {variance_covar_figure_section()}

    <h2>Figure 2: Why Brent Oil Identifies the Model Better Than the 2yr Treasury Yield</h2>
    <img src="{fig2_uri}">
    <p class="note">Panel (a): the identification condition itself, a normalizing
    variable's own variance should jump on war-news days; Brent oil's does
    ({_own_variance_ratio("brent_oil"):.1f}x), the 2yr Treasury yield's only moves
    {_own_variance_ratio("treasury_2y"):.1f}x. Panel (b): |t-statistics| for every
    other variable's estimated sensitivity, by normalizing-variable choice, the
    oil-normalized specification clears conventional significance (|t|&gt;2) almost
    everywhere the Treasury-normalized one doesn't. Panel (c): a scale-free instability
    score (swing across three alternative H-day thresholds, relative to each variable's
    own typical magnitude) computed for every variable, not just one example, the
    Treasury specification's typical swing sits at the boundary where a sign flip
    becomes plausible; the oil specification's sits mostly below it.</p>
    <ul class="points">
      <li>Read panel (a) first, it is the actual precondition for everything else,
      if the normalizing variable's own variance does not jump on H-days, none of
      the downstream coefficients for that choice can be trusted much.</li>
    </ul>

    {assumption_section()}

    <h2>Discussion</h2>
    <p><b>Two normalizing-variable specifications tell different stories.</b> The
    primary specification (2-year Treasury yield, matching Rigobon &amp; Sack 2003
    exactly) turns out weakly identified, the yield's own variance is not clearly
    elevated on H-days in this sample, unlike Iraq 2003 where it was roughly 6x
    higher. The secondary specification (Brent crude) is stable and statistically
    strong across nearly every variable.</p>
    <p><b>Economic interpretation.</b> Against the oil anchor, equities fall,
    volatility and the dollar rise, and credit spreads widen, a fairly classic
    risk-off pattern. But Treasury yields and inflation breakevens rise, not fall,
    with war-risk-driven oil moves, the opposite mechanism from Iraq 2003's flight
    to safety shock. The 2026 Iran war looks like it transmits mainly as an oil
    supply shock (inflationary, hawkish for rates), which fits its actual character
    as a Strait-of-Hormuz disruption to global energy supply.</p>
    <p><b>A counterintuitive but robust result: gold falls</b> (coefficient
    negative, t is about {gold_tstat:.1f}). One plausible reading is that broad
    dollar strength (itself significant, t={broad_dollar_tstat:.1f}) mechanically
    pressures dollar-denominated gold even while "fear" demand might otherwise push
    it up, flagged here as an open question, not something we're claiming to have
    solved.</p>
    <p><b>Why yields rise instead of fall, one more piece of context.</b> The US now
    produces roughly {US_OIL_PRODUCTION_CONTEXT['us_crude_output_2026_forecast_mmbd']}
    million barrels of crude a day, against roughly
    {US_OIL_PRODUCTION_CONTEXT['us_crude_output_2003_avg_mmbd']} million in 2003 (EIA,
    checked live, sources in config.py). That is a genuinely different starting point
    for the US as an oil producer. A bigger domestic cushion plausibly weakens the
    flight-to-quality channel that dominated in 2003, a Hormuz shock now shows up more
    through inflation and rate expectations than through a pure safety bid for
    Treasuries. This is offered as a plausible reason behind the pattern we already
    see (yields and breakevens rising, not falling), not a new claim the coefficients
    themselves prove on their own.</p>

    {identification_alternatives_section()}

    {improvements_section(nlp_eval, nlp_zero_pct)}

    <h2>Limitations</h2>
    <ul>
      <li>The core identifying assumption, that only the Iran war-risk factor's
      variance shifts between H and L days, was checked with a confound calendar
      (FOMC meetings etc.) but cannot be fully verified, flagged days are reported
      in Table 1's "Confound flag" column. See the dedicated assumption section
      above for what our results actually suggest about this.</li>
      <li><b>Each of the three NLP scoring methods has its own blind spots.</b> The
      lexicon filter needs the literal word "iran" plus a fixed term list, so a
      genuinely escalatory article about the same conflict gets silently dropped if
      it does not use that word, e.g. "All six crew members killed after US
      refuelling aircraft crashes in Western Iraq, confirms CENTCOM" scores a real
      escalation intensity but is marked not relevant and excluded, purely because it
      never says "Iran," it says Iraq and CENTCOM instead. That is the direct cause
      of the filter's low recall ({nlp_eval.get('recall', float('nan')):.2f}, above),
      and it means the lexicon component undercounts true intensity whenever
      coverage is framed around a linked theatre or actor rather than Iran by name
      (see the false-negative check above for more examples). The TF-IDF tone
      method's relevance threshold (cosine similarity above 0.08) was picked by eye,
      not calibrated against the eval set the way the lexicon filter was, so its
      precision/recall are simply unknown. FinBERT applies no relevance filter at
      all, it scores every article GDELT's query already returned, so its daily
      score reflects a broader, differently-filtered set of articles than the other
      two, not quite a "same articles, three opinions" setup.</li>
      <li><b>Z-scoring fixes mean and variance across the five composite components,
      not their shape.</b> The two GDELT volume series are continuous and nonzero
      every day, the three NLP-scored series are zero on about {nlp_zero_pct:.0%} of
      days and spike hard on the rest, lexicon's z-score hits {nlp_max_z:.1f} on its
      most extreme day, well above any other component's peak. Since z-scoring only
      fixes mean and variance, a zero-inflated, heavy-tailed component can
      occasionally move the weighted sum by more than its assigned weight would
      suggest, on the specific days it fires.</li>
      <li>Even with the per-method caveats above, no single one of the five signals
      is used alone for H/L classification or Figure 1, the weighted, z-scored
      composite of all five is the only intensity measure used throughout, on the
      idea that combining several imperfectly-filtered, differently-biased signals
      is more robust than trusting any one method's quirks. This is a design choice,
      not a claim that the composite itself has no issues.</li>
      <li>The on-the-run 10-year Treasury liquidity premium (present in the original
      paper's 9-variable set) is left out of this Phase-A analysis, see
      liquidity_premium_gsw.py for the scoped-but-not-run Phase B extension.</li>
      <li>News-to-market trading-calendar alignment uses a documented next-trading-day
      roll-forward rule, perfect alignment across US/Gulf/Israeli market hours is not
      really achievable with daily data.</li>
      <li><b>The GDELT artlist cap turned out to bind much harder than expected.</b>
      Each 14-day chunk request is capped at 250 records and sorted earliest-first, so
      in practice the entire 250-article quota for a chunk gets used up by that
      chunk's first day alone, every other day in that 14-day window ends up with
      zero articles in gdelt_articles.parquet. Checked directly against the raw data,
      only {n_days_with_articles} of the roughly {total_window_days} days in the
      analysis window have any article-level text at all. This is why most rows in
      Table 1 show "no article-level data for this date" instead of a headline, and it
      means the three NLP-scored composite components (lexicon, tone, FinBERT) are
      genuinely zero, not just sparse, on the other {n_days_without_articles} days. The
      primary intensity signal and H/L classification are not affected by this,
      they run on GDELT's separate, uncapped daily timeline series specifically
      because of this kind of cap, but the NLP components' real contribution to the
      composite index is smaller and patchier than their assigned weights (0.15,
      0.15, 0.20) alone would suggest. This was checked and consciously left alone
      when this window was rebased to Feb 28 onward, since the three-regime extension
      above only needs the dense tone-based direction signal, not article text.</li>
      <li>The 2yr-Treasury-normalized specification is weakly identified (its own
      variance is not clearly elevated on H-days), reported alongside the
      Brent-oil-normalized specification rather than suppressed, since the contrast
      is itself informative about how the war actually transmits (oil supply shock
      vs. classic flight to safety).</li>
      <li>Reported t-stats and p-values are not corrected for multiple comparisons
      across 13 variables x 3 estimators x 2 normalizing-variable specifications.</li>
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
