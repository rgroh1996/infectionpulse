"""Render the README figure from the committed benchmark exports (no API calls)."""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {
    "south": "South",
    "east": "East",
    "north_west": "North-West",
    "central_west": "Central-West",
}
LABELS = {
    "LightGBM": "LightGBM, same 1,000 rows",
    "XGBoost": "XGBoost, same 1,000 rows",
    "LightGBM full-history": "LightGBM, full history",
    "XGBoost full-history": "XGBoost, full history",
    "Persistence": "Latest observation",
}


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import DateFormatter, MonthLocator

    forecasts = pd.read_csv(
        ROOT / "reports/are_forecasts.csv", parse_dates=["target_start"]
    )
    summary = pd.read_csv(ROOT / "reports/benchmark_summary.csv").sort_values("mae")
    tab = forecasts[forecasts.model == "TabPFN-3.5"]

    sns.set_theme(style="whitegrid", font_scale=0.9)
    blue = sns.color_palette()[0]
    fig = plt.figure(figsize=(14, 6), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[2.2, 1], wspace=0.04)
    grid = left.add_gridspec(2, 2)

    for i, (region, name) in enumerate(REGIONS.items()):
        ax = left.add_subplot(grid[i // 2, i % 2])
        rows = tab[tab.region_id == region].sort_values("target_start")
        ax.fill_between(
            rows.target_start,
            rows.q10,
            rows.q90,
            color=blue,
            alpha=0.25,
            linewidth=0,
            label="TabPFN 80% range",
        )
        ax.plot(rows.target_start, rows.q50, color=blue, label="TabPFN forecast")
        ax.plot(
            rows.target_start,
            rows.target_incidence,
            color="black",
            lw=1,
            label="Reported",
        )
        ax.set_title(name)
        ax.set_ylim(0, 12500)
        ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(DateFormatter("%b %y"))
        if i % 2 == 0:
            ax.set_ylabel("ARE per 100,000")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    ax = right.subplots()
    colors = [blue if m == "TabPFN-3.5" else "0.7" for m in summary.model]
    ax.barh(summary.model.replace(LABELS), summary.mae, color=colors)
    ax.invert_yaxis()
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=3, fontsize=9)
    ax.set_xlabel("Mean absolute error")
    ax.set_xlim(0, summary.mae.max() * 1.18)
    ax.grid(axis="y", visible=False)

    fig.savefig(ROOT / "docs/assets/results.png", dpi=150, bbox_inches="tight")
    print("Wrote docs/assets/results.png")


if __name__ == "__main__":
    main()
