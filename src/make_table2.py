"""Render Table 2 (estimated impact of Iran war risk on each financial variable) as an
HTML fragment for each normalizing variable, plus the explicit side-by-side comparison
to Rigobon & Sack (2003)'s published Iraq-2003 results (comparable only for the
treasury_2y-normalized run, since that matches the original paper's own normalization;
the Brent-oil-normalized run has no 2003 analog and is reported on its own)."""
import pandas as pd

from config import DATA_PROCESSED, NORMALIZING_VARIABLE, NORMALIZING_VARIABLES_TO_RUN

# Rigobon & Sack (2003) Table 2 results, normalized to a 25bp drop in the 2yr yield
# (their own display scale) -- transcribed directly from the paper for the comparison
# table. Only meaningful against our treasury_2y-normalized run.
IRAQ_2003_RESULTS = {
    "treasury_10y": {"coef": -0.26, "tstat": 11.85, "units": "pp chg"},
    "breakeven_10y": {"coef": -0.11, "tstat": 3.45, "units": "pp chg"},
    "sp500": {"coef": -3.76, "tstat": 2.90, "units": "pct chg"},
    "bbb_spread": {"coef": 0.05, "tstat": 3.79, "units": "pp chg"},
    "hy_spread": {"coef": 0.34, "tstat": 5.40, "units": "pp chg"},
    "wti_oil": {"coef": 0.77, "tstat": 2.44, "units": "$ chg (12mo futures)"},  # brent_oil is the closer real analog but paper used oil futures generally
    "gold": {"coef": 1.30, "tstat": 0.26, "units": "$ chg"},
    "dollar_index": {"coef": -0.44, "tstat": 2.22, "units": "pct chg"},
}


def df_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x), border=0, na_rep="-")


def build(normalizing_variable: str = NORMALIZING_VARIABLE) -> tuple[str, str | None]:
    table2 = pd.read_parquet(DATA_PROCESSED / f"table2_estimates_{normalizing_variable}.parquet")
    display = table2[[
        "variable", "eq6_direct", "eq7_direct", "iv_w1_coef", "iv_w1_tstat",
        "iv_w2_coef", "iv_w2_tstat", "iv_w3_coef", "iv_w3_tstat", "iv_w3_sargan_pvalue",
        "bootstrap_se", "n_obs",
    ]].rename(columns={
        "variable": "Variable", "eq6_direct": "Eq(6)", "eq7_direct": "Eq(7)",
        "iv_w1_coef": "IV ω1 coef", "iv_w1_tstat": "IV ω1 t-stat",
        "iv_w2_coef": "IV ω2 coef", "iv_w2_tstat": "IV ω2 t-stat",
        "iv_w3_coef": "IV ω3 coef", "iv_w3_tstat": "IV ω3 t-stat",
        "iv_w3_sargan_pvalue": "Sargan p-value", "bootstrap_se": "Bootstrap SE", "n_obs": "N",
    })
    html_table2 = df_to_html_table(display)

    html_comparison = None
    if normalizing_variable == "treasury_2y":
        comp_rows = []
        for _, row in table2.iterrows():
            var = row["variable"]
            our_coef, our_t = row["iv_w1_coef"], row["iv_w1_tstat"]
            if var in IRAQ_2003_RESULTS:
                orig = IRAQ_2003_RESULTS[var]
                same_dir = (orig["coef"] > 0) == (our_coef > 0) if pd.notna(our_coef) else None
                comp_rows.append({
                    "Variable": var, "Iraq 2003 coef": orig["coef"], "Iraq 2003 t-stat": orig["tstat"],
                    "Iran 2026 coef": our_coef, "Iran 2026 t-stat": our_t,
                    "Same direction?": "Yes" if same_dir else ("No" if same_dir is not None else "-"),
                })
            else:
                comp_rows.append({
                    "Variable": var, "Iraq 2003 coef": None, "Iraq 2003 t-stat": None,
                    "Iran 2026 coef": our_coef, "Iran 2026 t-stat": our_t,
                    "Same direction?": "New variable (no 2003 analog)",
                })
        comparison = pd.DataFrame(comp_rows)
        html_comparison = df_to_html_table(comparison)
        with open(DATA_PROCESSED / "table2_comparison.html", "w") as f:
            f.write(html_comparison)

    with open(DATA_PROCESSED / f"table2_{normalizing_variable}.html", "w") as f:
        f.write(html_table2)
    print(f"Saved Table 2 ({normalizing_variable}) HTML" + (" + Iraq-vs-Iran comparison" if html_comparison else ""))
    return html_table2, html_comparison


if __name__ == "__main__":
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        build(nv)
