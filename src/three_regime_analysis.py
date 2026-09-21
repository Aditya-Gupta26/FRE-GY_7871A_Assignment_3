"""Extension: split H-days into "bad war news" (war risk escalating) and "good war
news" (war risk de-escalating), then re-run the same heteroskedasticity estimator on
bad-vs-L, good-vs-L, and bad-vs-good, on top of (not instead of) the main H-vs-L Table 2.
This is additive: the main Table 2/3 pipeline is untouched.

Reuses run_regressions.py's build_instruments/iv_estimate/bootstrap_ci as-is (they
hardcode the is_H/is_L column names) by building a copy of the panel with those two
columns temporarily overwritten to whichever pair of masks is being tested. This avoids
duplicating any estimator logic. event_study.compute_omega already takes an arbitrary
boolean mask, so it needs no wrapper either.

Regime definition: the `direction` column already in master_panel_filtered.parquet
("Increased" = war risk increasing = bad/escalatory news, "Decreased" = war risk
decreasing = good/de-escalatory news), computed from GDELT's own uncapped daily tone
series in classify_regimes.py, so it is defined every day, not limited to the sparse
article-level corpus.

These splits are small by construction, a subset of an already-small H set, so every
output row carries its own n_pos/n_neg counts up front. Read those before trusting any
coefficient here the way the main Table 2 numbers can be trusted.
"""
import pandas as pd

from config import DATA_PROCESSED, FRED_SERIES, NORMALIZING_VARIABLES_TO_RUN, YFINANCE_TICKERS
from event_study import compute_omega
from run_regressions import bootstrap_ci, build_instruments, direct_estimators, iv_estimate

ALL_VARIABLES = list(YFINANCE_TICKERS.keys()) + list(FRED_SERIES.keys())


def _masked_panel(panel: pd.DataFrame, pos_mask: pd.Series, neg_mask: pd.Series) -> pd.DataFrame:
    """A copy of panel with is_H/is_L overwritten to the given pair of masks, so every
    existing is_H/is_L-hardcoded function in run_regressions.py works unmodified."""
    sub_panel = panel.copy()
    sub_panel["is_H"] = pos_mask
    sub_panel["is_L"] = neg_mask
    return sub_panel


def run_pair(panel: pd.DataFrame, pos_mask: pd.Series, neg_mask: pd.Series,
             normalizing_variable: str, other_variables: list[str]) -> pd.DataFrame:
    sub_panel = _masked_panel(panel, pos_mask, neg_mask)
    n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())

    rows = []
    for xj in other_variables:
        omega_pos = compute_omega(panel, pos_mask, normalizing_variable, xj)
        omega_neg = compute_omega(panel, neg_mask, normalizing_variable, xj)
        direct = direct_estimators(omega_pos, omega_neg)

        sub = build_instruments(sub_panel, normalizing_variable, xj)
        try:
            iv_w1 = iv_estimate(sub, normalizing_variable, xj, ["omega1"])
        except Exception as e:
            iv_w1 = {"error": str(e)}
        try:
            iv_w2 = iv_estimate(sub, normalizing_variable, xj, ["omega2"])
        except Exception as e:
            iv_w2 = {"error": str(e)}
        try:
            iv_w3 = iv_estimate(sub, normalizing_variable, xj, ["omega1", "omega2"])
        except Exception as e:
            iv_w3 = {"error": str(e)}
        boot = bootstrap_ci(sub_panel, normalizing_variable, xj)

        rows.append({
            "variable": xj, "n_pos": n_pos, "n_neg": n_neg,
            "eq6_direct": direct["eq6"], "eq7_direct": direct["eq7"],
            "iv_w1_coef": iv_w1.get("coef"), "iv_w1_tstat": iv_w1.get("tstat"),
            "iv_w2_coef": iv_w2.get("coef"), "iv_w2_tstat": iv_w2.get("tstat"),
            "iv_w3_coef": iv_w3.get("coef"), "iv_w3_tstat": iv_w3.get("tstat"),
            "iv_w3_sargan_pvalue": iv_w3.get("sargan_pvalue"),
            "bootstrap_se": boot.get("se"),
            "n_obs": iv_w1.get("n_obs"),
        })
    return pd.DataFrame(rows)


def main(normalizing_variable: str) -> dict[str, pd.DataFrame]:
    panel = pd.read_parquet(DATA_PROCESSED / "master_panel_filtered.parquet")
    other_variables = [v for v in ALL_VARIABLES if v != normalizing_variable]

    is_H_bad = panel["is_H"] & (panel["direction"] == "Increased")
    is_H_good = panel["is_H"] & (panel["direction"] == "Decreased")
    is_L = panel["is_L"]

    print(f"[{normalizing_variable}] three-regime split: bad-news H-days={int(is_H_bad.sum())}, "
          f"good-news H-days={int(is_H_good.sum())}, L-days={int(is_L.sum())}")

    pairs = {
        "bad_vs_L": (is_H_bad, is_L),
        "good_vs_L": (is_H_good, is_L),
        "bad_vs_good": (is_H_bad, is_H_good),
    }
    results = {}
    for name, (pos_mask, neg_mask) in pairs.items():
        table = run_pair(panel, pos_mask, neg_mask, normalizing_variable, other_variables)
        out_path = DATA_PROCESSED / f"table_three_regime_{name}_{normalizing_variable}.parquet"
        table.to_parquet(out_path, index=False)
        print(f"  {name}: saved -> {out_path}")
        results[name] = table
    return results


if __name__ == "__main__":
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        main(nv)
