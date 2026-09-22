#!/usr/bin/env python3
"""Plot schema sizes at single-column print size. Requires matplotlib and numpy."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "plots" / "schema_size_by_dataset"

GROUPS = [
    ("bird_dev", "BIRD dev"),
    ("spider_dev", "Spider dev"),
    ("spider_test", "Spider test"),
    ("bird_interact_lite", "BIRD-Interact Lite"),
    ("bird_interact_full", "BIRD-Interact Full"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the mean number of original and RC-filtered schema columns "
            "per question for each evaluation set."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="DIN/DAIL linking-details JSONL containing successful filter column counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output path without an extension; both PDF and PNG are written "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )
    return parser.parse_args()


def load_sizes(path: Path) -> list[dict[str, float | int | str]]:
    values: dict[str, list[tuple[int, int]]] = defaultdict(list)

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            schema_filter = record.get("filter", {})
            if schema_filter.get("status") != "succeeded":
                continue

            group = record.get("group")
            if group not in {key for key, _ in GROUPS}:
                continue

            full = schema_filter.get("full_columns")
            filtered = schema_filter.get("filtered_columns")
            if not isinstance(full, int) or not isinstance(filtered, int):
                raise ValueError(f"Missing column counts at {path}:{line_number}")
            if full <= 0 or not 0 <= filtered <= full:
                raise ValueError(
                    f"Invalid column counts at {path}:{line_number}: "
                    f"full={full}, filtered={filtered}"
                )
            values[group].append((full, filtered))

    rows: list[dict[str, float | int | str]] = []
    for group, label in GROUPS:
        pairs = values.get(group, [])
        if not pairs:
            raise ValueError(f"No successful schema-filter records for {group}")
        original = mean(full for full, _ in pairs)
        filtered = mean(kept for _, kept in pairs)
        rows.append(
            {
                "group": group,
                "label": label,
                "questions": len(pairs),
                "original": original,
                "filtered": filtered,
                "ratio_of_means_pct": 100.0 * filtered / original,
                "mean_question_retained_pct": mean(
                    100.0 * kept / full for full, kept in pairs
                ),
            }
        )
    return rows


def plot(rows: list[dict[str, float | int | str]], output: Path) -> None:
    # Keep Matplotlib's cache inside a writable temporary directory on managed hosts.
    cache_dir = Path(tempfile.gettempdir()) / "rc-schema-size-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [str(row["label"]) for row in rows]
    original = np.array([float(row["original"]) for row in rows])
    filtered = np.array([float(row["filtered"]) for row in rows])
    y = np.arange(len(rows))
    bar_height = 0.30
    # Design at the final single-column width, rather than shrinking a 7.2-in
    # figure and all its text by half. The fixed canvas also keeps PDF sizing
    # predictable when included at width=\columnwidth in LaTeX.
    figure, axis = plt.subplots(figsize=(3.35, 2.65))
    figure.subplots_adjust(left=0.325, right=0.985, bottom=0.17, top=0.855)

    axis.barh(
        y - bar_height / 2,
        original,
        height=bar_height,
        color="#B8BEC7",
        edgecolor="#8F98A4",
        linewidth=0.35,
        label="Original schema",
    )
    axis.barh(
        y + bar_height / 2,
        filtered,
        height=bar_height,
        color="#2F6B9A",
        label="RC-filtered schema",
    )

    label_offset = original.max() * 0.018
    for index, value in enumerate(original):
        axis.text(
            value + label_offset,
            y[index] - bar_height / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            color="#374151",
            fontsize=8,
        )
    for index, value in enumerate(filtered):
        axis.text(
            value + label_offset,
            y[index] + bar_height / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            color="#1F4E73",
            fontsize=8,
        )

    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, original.max() * 1.30)
    axis.xaxis.set_major_locator(
        plt.MaxNLocator(nbins=4, steps=[1, 2, 5, 10], integer=True)
    )
    axis.set_xlabel("Mean number of columns", labelpad=5)
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#9CA3AF")
    axis.tick_params(axis="y", length=0, pad=4)
    axis.tick_params(axis="x", length=2.5, width=0.6, color="#9CA3AF")
    # A shared row above the plot leaves the short filtered bars unobstructed.
    handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.5,
        columnspacing=1.1,
    )

    output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Do not use bbox_inches="tight": it changes the physical canvas width.
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load_sizes(args.input)
    plot(rows, args.output)

    print(
        "Dataset\tQuestions\tOriginal\tFiltered\t"
        "Filtered/Original means\tMean per-question retained"
    )
    for row in rows:
        print(
            f"{row['label']}\t{row['questions']}\t{row['original']:.2f}\t"
            f"{row['filtered']:.2f}\t{row['ratio_of_means_pct']:.2f}%\t"
            f"{row['mean_question_retained_pct']:.2f}%"
        )
    print(
        "Unweighted mean of per-question retention across datasets: "
        f"{mean(float(row['mean_question_retained_pct']) for row in rows):.2f}%"
    )


if __name__ == "__main__":
    main()
