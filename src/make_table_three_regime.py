"""Render the three-regime extension tables (bad_vs_L, good_vs_L, bad_vs_good) as HTML
fragments, one per (pair, normalizing_variable). Same rendering pattern as make_table2.py,
just with n_pos/n_neg shown first since that's the number a reader needs to see before
trusting anything else in the row."""
import pandas as pd

from config import DATA_PROCESSED, NORMALIZING_VARIABLE, NORMALIZING_VARIABLES_TO_RUN

PAIRS = ["bad_vs_L", "good_vs_L", "bad_vs_good"]


def df_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x), border=0, na_rep="-")


def build(pair: str, normalizing_variable: str = NORMALIZING_VARIABLE) -> str:
    path = DATA_PROCESSED / f"table_three_regime_{pair}_{normalizing_variable}.parquet"
    table = pd.read_parquet(path)
    display = table[[
        "variable", "n_pos", "n_neg", "eq6_direct", "eq7_direct",
        "iv_w1_coef", "iv_w1_tstat", "iv_w2_coef", "iv_w2_tstat",
        "iv_w3_coef", "iv_w3_tstat", "iv_w3_sargan_pvalue", "bootstrap_se", "n_obs",
    ]].rename(columns={
        "variable": "Variable", "n_pos": "N (first group)", "n_neg": "N (second group)",
        "eq6_direct": "Eq(6)", "eq7_direct": "Eq(7)",
        "iv_w1_coef": "IV ω1 coef", "iv_w1_tstat": "IV ω1 t-stat",
        "iv_w2_coef": "IV ω2 coef", "iv_w2_tstat": "IV ω2 t-stat",
        "iv_w3_coef": "IV ω3 coef", "iv_w3_tstat": "IV ω3 t-stat",
        "iv_w3_sargan_pvalue": "Sargan p-value", "bootstrap_se": "Bootstrap SE", "n_obs": "N",
    })
    html = df_to_html_table(display)
    out_path = DATA_PROCESSED / f"table_three_regime_{pair}_{normalizing_variable}.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved three-regime table ({pair}, {normalizing_variable}) HTML -> {out_path}")
    return html


if __name__ == "__main__":
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        for pair in PAIRS:
            build(pair, nv)
