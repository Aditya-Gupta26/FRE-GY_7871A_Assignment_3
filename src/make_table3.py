"""Render Table 3 (variance decomposition) as an HTML fragment, per normalizing variable."""
import pandas as pd

from config import DATA_PROCESSED, NORMALIZING_VARIABLE, NORMALIZING_VARIABLES_TO_RUN


def df_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    return df.to_html(index=False, float_format=lambda x: float_fmt.format(x), border=0, na_rep="-")


def build(normalizing_variable: str = NORMALIZING_VARIABLE) -> str:
    table3 = pd.read_parquet(DATA_PROCESSED / f"table3_variance_decomp_{normalizing_variable}.parquet")
    display = table3.rename(columns={
        "variable": "Variable", "var_L": "Var. on L days", "var_H": "Var. on H days",
        "predicted_delta_var": "Predicted ΔVar", "pct_explained_H_days": "% explained (H days)",
        "pct_explained_all_days": "% explained (all days)",
    }).copy()
    for pct_col in ["% explained (H days)", "% explained (all days)"]:
        display[pct_col] = (display[pct_col] * 100).round(1)
    html = df_to_html_table(display)
    out_path = DATA_PROCESSED / f"table3_{normalizing_variable}.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved Table 3 ({normalizing_variable}) HTML -> {out_path}")
    return html


if __name__ == "__main__":
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        build(nv)
