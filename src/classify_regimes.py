"""Classify each business day as H (elevated war-risk-news variance), L (calm), or
neither, from the composite intensity index. This is the data-driven replacement for
Rigobon & Sack's hand-picked Table 1.

Rule (locked in with the user): top-decile composite intensity = H; a matched-size set
of the lowest-intensity days *near* each H-day = L, mirroring the original paper's own
stated logic ("choose L days as close as possible to, but not included in," the H days,
to control for other factors).

Confound handling: days coinciding with CONFOUND_CALENDAR (FOMC meetings etc.) are
flagged, not silently dropped, so the identifying assumption (that H/L variance shifts
are attributable to Iran war risk specifically, not some other common shock) can be
inspected and, if needed, those dates excluded in a robustness pass.
"""
import numpy as np
import pandas as pd

from config import CONFOUND_CALENDAR, DATA_PROCESSED, IRAN_WAR_TIMELINE

H_QUANTILE = 0.90  # top decile = H


def build_labels() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / "daily_intensity.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    threshold = df["composite_intensity"].quantile(H_QUANTILE)
    df["is_H"] = df["composite_intensity"] >= threshold
    n_H = int(df["is_H"].sum())
    print(f"H threshold (top {1 - H_QUANTILE:.0%}): composite_intensity >= {threshold:.3f} "
          f"-> {n_H}/{len(df)} days")

    # L: for each H-day, take the nearest non-H days (by calendar distance) with the
    # lowest intensity, until we have a matched-size L set, excluding any day already
    # used. This operationalizes "close to H days" while still selecting genuinely calm
    # days (not just adjacent ones, which could still be elevated during sustained
    # multi-week clusters).
    non_H = df[~df["is_H"]].copy()
    low_cutoff = non_H["composite_intensity"].quantile(0.5)  # below-median calm pool
    calm_pool = non_H[non_H["composite_intensity"] <= low_cutoff].copy()

    df["is_L"] = False
    used_idx = set()
    h_dates = df.loc[df["is_H"], "date"].tolist()
    for h_date in h_dates:
        calm_pool["dist"] = (calm_pool["date"] - h_date).abs()
        candidates = calm_pool[~calm_pool.index.isin(used_idx)].nsmallest(1, "dist")
        if not candidates.empty:
            used_idx.add(candidates.index[0])
    df.loc[df.index.isin(used_idx), "is_L"] = True
    n_L = int(df["is_L"].sum())
    print(f"L set (nearest calm-pool day per H-day): {n_L} days")

    # Confound flagging (not exclusion by default, documented and inspectable).
    confound_dates = pd.to_datetime(CONFOUND_CALENDAR, format="%Y%m%d")
    df["confound_flag"] = df["date"].isin(confound_dates)
    n_confound_H = int((df["is_H"] & df["confound_flag"]).sum())
    n_confound_L = int((df["is_L"] & df["confound_flag"]).sum())
    print(f"Confound-flagged days: {df['confound_flag'].sum()} total "
          f"({n_confound_H} in H, {n_confound_L} in L)")

    # Direction (descriptive only, mirrors Table 1's "War Risk: Increased/Decreased"
    # column; the estimator itself uses only variance, never direction). Rule: a
    # day-over-day drop in GDELT's average tone score signals escalating coverage.
    df["tone_change"] = df["avg_tone"].diff()
    df["direction"] = np.where(df["tone_change"] < -0.1, "Increased",
                        np.where(df["tone_change"] > 0.1, "Decreased", "Unclear"))

    # Face-validity cross-check against the known milestone timeline.
    milestones = pd.DataFrame(IRAN_WAR_TIMELINE, columns=["date", "event"])
    milestones["date"] = pd.to_datetime(milestones["date"])
    check = milestones.merge(df[["date", "is_H", "composite_intensity"]], on="date", how="left")
    print("\nMilestone cross-check (should mostly land in H):")
    print(check.to_string(index=False))

    out_path = DATA_PROCESSED / "regime_labels.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} days -> {out_path}")

    # Table-1-equivalent export.
    table1 = df[df["is_H"] | df["is_L"]][
        ["date", "composite_intensity", "direction", "is_H", "is_L", "confound_flag"]
    ].sort_values("date")
    table1_path = DATA_PROCESSED / "table1_regime_dates.csv"
    table1.to_csv(table1_path, index=False)
    print(f"Saved Table-1-equivalent ({len(table1)} rows) -> {table1_path}")

    return df


if __name__ == "__main__":
    build_labels()
