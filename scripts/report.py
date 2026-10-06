"""Export reproducible benchmark tables and a compact submission figure."""

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    status = json.loads((ROOT / "models/respiratory/latest_run.json").read_text())
    folder = Path(status["path"])
    summary = pd.DataFrame(status["summary"]).sort_values("mae")
    destination = ROOT / "reports"
    destination.mkdir(exist_ok=True)
    summary.to_csv(destination / "benchmark_summary.csv", index=False)
    predictions = pd.read_parquet(folder / "predictions.parquet")
    predictions[
        [
            "model",
            "region_id",
            "issue_at",
            "target_start",
            "horizon_weeks",
            "q10",
            "q50",
            "q90",
            "target_incidence",
        ]
    ].sort_values(["model", "region_id", "target_start"]).to_csv(
        destination / "are_forecasts.csv", index=False, float_format="%.1f"
    )
    ladder_path = folder / "development_ladder/results.csv"
    ladder = pd.read_csv(ladder_path) if ladder_path.exists() else pd.DataFrame()
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1.4, 1, 1]}
    )
    colors = [
        "#087e8b" if name == "TabPFN-3.5" else "#9eb4be" for name in summary.model
    ]
    axes[0].barh(summary.model, summary.mae, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE · ARE incidence per 100,000 (lower is better)")
    axes[0].set_title(
        f"Locked historical test · {status['eligible_rows']}/{status['planned_rows']} issued"
    )
    if not ladder.empty:
        palette = {"TabPFN-3.5": "#087e8b", "LightGBM": "#6e7b8f", "XGBoost": "#c98239"}
        for axis, (origin, origin_rows) in zip(axes[1:], ladder.groupby("origin")):
            for name, group in origin_rows.groupby("model"):
                group = group.sort_values("context_size")
                axis.plot(
                    group.context_size,
                    group.mae,
                    marker="o",
                    label=name,
                    color=palette[name],
                )
            axis.set_xscale("log")
            ticks = sorted(origin_rows.context_size.unique())
            axis.set_xticks(ticks, [str(int(tick)) for tick in ticks])
            axis.xaxis.set_minor_locator(NullLocator())
            axis.set_xlabel("Context examples · calibration additional")
            axis.set_ylabel("Development MAE")
            axis.legend()
            axis.set_title(f"Exploratory · {origin}\nOne seed; fixed test cohort")
        ladder.to_csv(destination / "sample_efficiency.csv", index=False)
    else:
        for axis in axes[1:]:
            axis.axis("off")
        axes[1].text(
            0.05,
            0.6,
            "Sample-efficiency experiment not completed.\nNo sample-efficiency claim is made.",
            transform=axes[1].transAxes,
        )
    fig.suptitle(
        f"InfectionPulse · TabPFN-3.5 · evaluation {status['status']}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(destination / "benchmark_results.png", dpi=160)
    plt.close(fig)
    comparisons = status.get("paired_comparisons", [])
    baseline = summary[summary.model != "TabPFN-3.5"].iloc[0]
    tab = summary[summary.model == "TabPFN-3.5"]
    claim = "TabPFN evaluation incomplete; no comparative claim is established."
    if not tab.empty:
        improvement = 1 - float(tab.iloc[0].mae) / float(baseline.mae)
        paired = next((c for c in comparisons if c["competitor"] == baseline.model), {})
        ci = paired.get("ci95")
        strong = (
            status["status"] == "complete"
            and improvement >= 0.05
            and ci is not None
            and ci[0] > 0
        )
        claim = (
            f"TabPFN MAE improvement versus the strongest baseline ({baseline.model}): {improvement:.1%}. "
            f"The predeclared >=5% and positive paired 95% interval criterion {'passes' if strong else 'does not pass'}."
        )
    lines = [
        "# Measured ARE benchmark",
        "",
        claim,
        "",
        f"Run `{status['run_id']}`; status **{status['status']}**. "
        f"{status['eligible_rows']} of {status['planned_rows']} planned region-weeks issued; "
        f"abstentions: `{json.dumps(status.get('abstention_counts', {}))}`.",
        "",
        "| Model | MAE | RMSE | Brier | Raw 10–90% coverage | Adjusted 80% coverage | Adjusted width |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.model} | {row.mae:.2f} | {row.rmse:.2f} | {row.brier:.4f} | {getattr(row, 'raw_coverage80', float('nan')):.1%} | {row.coverage80:.1%} | {row.interval_width:.2f} |"
        )
    lines += [
        "",
        "![Measured results](benchmark_results.png)",
        "",
        "## Interpretation",
        "",
        "Raw coverage describes the q10–q90 band shown by the app and used by its precaution rule. "
        "Adjusted coverage and width describe the separately evaluated conformal interval; they must not be attributed to the raw band.",
        "",
        "Forecast features in the locked test use actual historical source vintages. "
        "Publication times use Git commit timestamps as a proxy. Pre-September-2023 development/training "
        "features are explicitly reconstructed; they are not prospective evidence. Labels use the first "
        "published vintage at least 35 days after target Sunday. Four-week block bootstrap intervals "
        "preserve time dependence and keep all regions together.",
        "",
        "The exploratory ladder has one seed and two predetermined development origins, with N=2000 "
        "only where enough history exists. It cannot establish robust small-sample superiority on its own.",
        "",
        "These metrics validate community surveillance forecasts, not personal infection probabilities "
        "or the clinical effectiveness of the traffic-light policy.",
        "",
        "## Probability readiness",
        "",
        "Numerical community-event probabilities are withheld unless all predeclared readiness checks pass. "
        "The reliability-error ceiling is 0.05; passing the discrimination metrics alone is insufficient.",
        "",
        "```json",
        json.dumps(status.get("probability_readiness", {}), indent=2),
        "```",
        "",
        "## Paired comparisons",
        "",
        "```json",
        json.dumps(comparisons, indent=2),
        "```",
        "",
    ]
    (destination / "benchmark.md").write_text("\n".join(lines))
    portable = {k: v for k, v in status.items() if k != "path"}
    (destination / "benchmark.json").write_text(json.dumps(portable, indent=2) + "\n")
    print(claim)
    print(
        "Wrote reports/benchmark.md, benchmark.json, benchmark_summary.csv, benchmark_results.png and are_forecasts.csv"
    )


if __name__ == "__main__":
    main()
