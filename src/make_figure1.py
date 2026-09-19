"""Figure 1: daily war-risk news intensity timeline with H-days marked and known
milestones annotated. This is the visual analog of Rigobon & Sack's Table 1."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from config import DATA_PROCESSED, IRAN_WAR_TIMELINE, PROJECT_ROOT

KEY_MILESTONES = [  # subset annotated on the chart to avoid clutter
    ("2026-02-28", "Strikes begin"),
    ("2026-04-08", "Ceasefire begins"),
    ("2026-06-17", "US-Iran MOU signed"),
    ("2026-07-08", "MOU collapses"),
]


def main():
    df = pd.read_parquet(DATA_PROCESSED / "regime_labels.parquet")
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(df["date"], df["composite_intensity"], color="#444444", linewidth=1.0, label="Composite intensity")
    h_days = df[df["is_H"]]
    l_days = df[df["is_L"]]
    ax.scatter(h_days["date"], h_days["composite_intensity"], color="#c0392b", s=22, zorder=5, label="H (war-risk news spike)")
    ax.scatter(l_days["date"], l_days["composite_intensity"], color="#2980b9", s=18, zorder=4, label="L (calm)")

    for date_str, label in KEY_MILESTONES:
        d = pd.Timestamp(date_str)
        ax.axvline(d, color="#999999", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.annotate(label, xy=(d, ax.get_ylim()[1]), xytext=(3, -12), textcoords="offset points",
                    fontsize=8, rotation=90, va="top", color="#555555")

    ax.set_title("Iran War-Risk News Intensity, Jan–Sep 2026 (GDELT + NLP composite)")
    ax.set_ylabel("Composite intensity index (z-score weighted)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = PROJECT_ROOT / "notebooks" / "figure1_intensity_timeline.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
