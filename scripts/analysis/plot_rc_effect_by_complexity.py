#!/usr/bin/env python3
"""Plot RC effects in compact, print-sized layouts (matplotlib and numpy)."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "plots" / "rc_effect_by_complexity"
# spconf.sty: (178 mm text width - 6 mm column separation) / 2.
COLUMN_WIDTH_IN = 86 / 25.4

DATASETS = []
STAGES = {}


def load_input(path: Path):
    """Read five ordered labels and Generation/Revision percentage series.

    JSON: {"datasets": [five labels], "stages": {"Generation": {
    "original": [five numbers], "rc": [five numbers], "gain": [five numbers]},
    "Revision": {the same fields}}}. Gains are supplied explicitly so plots
    preserve the upstream rounding/aggregation convention.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    labels, stages = data["datasets"], data["stages"]
    if len(labels) != 5 or not all(isinstance(label, str) for label in labels):
        raise ValueError("datasets must contain exactly five string labels in display order")
    if set(stages) != {"Generation", "Revision"}:
        raise ValueError("stages must contain Generation and Revision")
    for stage in stages.values():
        for field in ("original", "rc", "gain"):
            values = stage[field]
            if len(values) != 5 or not all(isinstance(v, (int, float)) and
                    not isinstance(v, bool) and math.isfinite(v) for v in values):
                raise ValueError(f"{field} must contain five finite numbers")
    return labels, {name: stages[name] for name in ("Generation", "Revision")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Original and RC accuracy over five ordered evaluation sets.",
        epilog=load_input.__doc__,
    )
    parser.add_argument("--input", type=Path, required=True, help="JSON labels and precomputed stage percentages.")
    parser.add_argument(
        "--layout",
        choices=("side-by-side", "merged", "bars", "lines", "lines-pair"),
        default="side-by-side",
        help="Two panels (default), merged curves, bars, compact lines, or a compact line pair.",
    )
    parser.add_argument(
        "--gain-angle",
        type=int,
        choices=(15, 30, 45),
        default=45,
        help="Gain-label angle in degrees for the bars layout (default: 45).",
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


def make_side_by_side_figure(plt, np):
    x = np.arange(len(DATASETS))
    # Shared labels and legend keep the pair within one 86-mm paper column.
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(COLUMN_WIDTH_IN, 1.85),
        sharey=True,
    )
    figure.subplots_adjust(left=0.12, right=0.955, bottom=0.285, top=0.72, wspace=0.16)

    for axis, (stage, values) in zip(axes, STAGES.items()):
        original = np.array(values["original"])
        rc = np.array(values["rc"])
        gains = np.array(values["gain"])

        axis.plot(
            x,
            original,
            color="#858B93",
            marker="o",
            linestyle="--",
            linewidth=0.9,
            markersize=2.6,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label="Original",
        )
        axis.plot(
            x,
            rc,
            color="#2F6B9A",
            marker="s",
            linestyle="-",
            linewidth=1.1,
            markersize=2.6,
            label="With RC",
        )
        axis.fill_between(x, original, rc, color="#2F6B9A", alpha=0.10)

        for index, (upper, gain) in enumerate(zip(rc, gains)):
            axis.annotate(
                f"+{gain:.2f}",
                (x[index], upper),
                # The two nearly equal Spider values need staggered labels.
                xytext=(0, -7 if index == 4 else 10 if index == 1 else 3),
                textcoords="offset points",
                ha="left" if index == 0 else "right" if index == 4 else "center",
                va="top" if index == 4 else "bottom",
                color="#1F4E73",
                fontsize=6.5,
            )

        axis.set_title(stage, pad=6)
        axis.set_xticks(x, DATASETS)
        axis.set_xlim(-0.05, 4.10)
        axis.set_ylim(0, 120)
        axis.set_yticks([0, 50, 100])
        axis.tick_params(axis="both", length=2.5, width=0.6, pad=2)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)

    figure.text(0.025, 0.50, "SQL accuracy (%)", rotation=90, ha="center", va="center")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.0),
        ncol=2,
        handlelength=1.8,
        columnspacing=1.8,
    )
    figure.supxlabel("Increasing empirical difficulty →", y=0.035, fontsize=7)
    return figure


def make_merged_figure(plt, np):
    """Use color for stage and line/marker fill for Original versus RC."""
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.98))
    figure.subplots_adjust(left=0.135, right=0.977, bottom=0.365, top=0.80)
    x = np.arange(len(DATASETS))
    stage_styles = {
        "Generation": {"color": "#2870A0", "marker": "o"},
        "Revision": {"color": "#C16A2D", "marker": "s"},
    }
    handles = {}
    # Keep exact x/y coordinates even where the Spider series nearly coincide.
    # Draw squares underneath circles so both marker shapes remain visible.
    for stage in ("Revision", "Generation"):
        values = STAGES[stage]
        style = stage_styles[stage]
        handles[(stage, "original")] = axis.plot(
            x, np.asarray(values["original"]),
            color=style["color"], marker=style["marker"],
            linestyle=(0, (2.7, 1.7)) if stage == "Generation" else (0, (1.2, 1.5)),
            linewidth=1.0, markersize=4.0, markerfacecolor="white",
            markeredgewidth=0.85, zorder=3,
        )[0]
        handles[(stage, "rc")] = axis.plot(
            x, np.asarray(values["rc"]),
            color=style["color"], marker=style["marker"], linestyle="-",
            linewidth=1.25, markersize=3.4, markeredgewidth=0.6, zorder=4,
        )[0]

    axis.set_xticks(x, DATASETS)
    axis.set_xlim(-0.35, 4.35)
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel("SQL accuracy (%)", labelpad=3)
    axis.tick_params(axis="both", length=2.1, width=0.55, pad=2)
    axis.tick_params(axis="x", labelsize=7)
    axis.tick_params(axis="y", labelsize=6.8)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.4, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.55)
    axis.spines["bottom"].set_linewidth(0.55)

    legend_order = [
        ("Generation", "original"), ("Generation", "rc"),
        ("Revision", "original"), ("Revision", "rc"),
    ]
    figure.legend(
        [handles[key] for key in legend_order],
        [f"{stage}: {'Original' if variant == 'original' else 'RC'}"
         for stage, variant in legend_order],
        frameon=False, loc="upper center", bbox_to_anchor=(0.53, 1.005),
        ncol=2, fontsize=6.8, handlelength=2.0, handletextpad=0.55,
        columnspacing=1.35, labelspacing=0.45,
    )

    # Aligned gain rows preserve all ten numbers without crowding the curves.
    # Delta denotes the originally reported gain in percentage points.
    for (stage, short_label), row_y in zip(
        [("Generation", "Gen. Δ"), ("Revision", "Rev. Δ")], [0.168, 0.108],
    ):
        color = stage_styles[stage]["color"]
        figure.text(0.025, row_y, short_label, ha="left", va="center",
                    fontsize=6.5, color=color)
        for point_x, gain in zip(x, STAGES[stage]["gain"]):
            figure_x = figure.transFigure.inverted().transform(
                axis.transData.transform((point_x, 0))
            )[0]
            figure.text(figure_x, row_y, f"+{gain:.2f}", ha="center",
                        va="center", fontsize=7, color=color)

    figure.text(0.54, 0.035, "Increasing empirical difficulty →",
                ha="center", va="center", fontsize=7)
    return figure


def make_bars_figure(plt, np, gain_angle: int = 45):
    """One bar per stage/dataset: Original body plus the RC gain, not two totals."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Keep print-sized text while reclaiming vertical space from the header
    # and readout. Do not scale the whole figure down in LaTeX.
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.25))
    figure.subplots_adjust(left=0.135, right=0.977, bottom=0.265, top=0.695)
    x = np.arange(len(DATASETS))
    width = 0.28
    stage_styles = {
        "Generation": {"color": "#2870A0", "pale": "#C8DAE7", "offset": -0.17},
        "Revision": {"color": "#C16A2D", "pale": "#EDD6C4", "offset": 0.17},
    }
    for stage, style in stage_styles.items():
        values = STAGES[stage]
        original = np.asarray(values["original"])
        rc = np.asarray(values["rc"])
        if np.any(rc < original):
            raise ValueError("The gain-cap bar layout requires RC >= Original.")
        centers = x + style["offset"]
        axis.bar(centers, original, width=width, color=style["pale"],
                 edgecolor=style["color"], linewidth=0.5, zorder=3)
        # Source gains were rounded separately. Subtract the displayed totals
        # here so each bar top is exactly RC; retain reported gains in labels.
        axis.bar(centers, rc - original, bottom=original, width=width,
                 color=style["color"], linewidth=0, zorder=3)
        axis.hlines(original, centers - width / 2 - 0.025,
                    centers + width / 2 + 0.025, color="#313840",
                    linewidth=0.75, linestyles=(0, (2, 1.3)), zorder=4)
        axis.hlines(rc, centers - width / 2, centers + width / 2,
                    color="#313840", linewidth=0.45, zorder=4)
        # At 15 degrees the labels are wider. Separate the pair horizontally,
        # retaining the same vertical offset and avoiding staggered rows.
        label_x_offset = (-7.5 if stage == "Generation" else -2.5) if gain_angle == 15 else -3
        for center, top, gain in zip(centers, rc, values["gain"]):
            # Give every label the same angle and vertical bar-top offset.
            # Rotation separates paired gains without alternating label rows.
            axis.annotate(
                f"+{gain:.2f}", (center, top),
                xytext=(label_x_offset, 1.5), textcoords="offset points",
                ha="left", va="bottom", rotation=gain_angle, rotation_mode="anchor",
                fontsize=6.0, color=style["color"], annotation_clip=False,
                zorder=5,
            )

    axis.set_xticks(x, [name.replace("\n", " ") for name in DATASETS])
    axis.set_xlim(-0.40, 4.40)
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel("SQL accuracy (%)", labelpad=3)
    axis.tick_params(axis="both", length=2.1, width=0.55, pad=2)
    axis.tick_params(axis="x", labelsize=7)
    axis.tick_params(axis="y", labelsize=6.8)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.4, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.55)
    axis.spines["bottom"].set_linewidth(0.55)

    figure.legend(
        [Patch(facecolor=style["color"]) for style in stage_styles.values()]
        + [Line2D([], [], color="#313840", linewidth=0.75, linestyle=(0, (2, 1.3))),
           Line2D([], [], color="#313840", linewidth=0.45)],
        list(stage_styles) + ["Original", "With RS"],
        frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.035),
        ncol=4, fontsize=7, handlelength=1.3, handletextpad=0.45,
        columnspacing=0.95,
    )

    figure.text(0.54, 0.060, "Evaluation set (increasing empirical difficulty →)",
                ha="center", va="center", fontsize=7)
    return figure


def make_lines_figure(plt, np):
    """Four identifiable curves in the same compact footprint as the bars."""
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.25))
    figure.subplots_adjust(left=0.135, right=0.977, bottom=0.265, top=0.685)
    x = np.arange(len(DATASETS))
    styles = {
        "Generation": {"color": "#2870A0", "marker": "o", "dash": (0, (3.3, 2.3))},
        "Revision": {"color": "#C16A2D", "marker": "s", "dash": (0, (1.2, 1.8))},
    }
    handles = {}
    # Retain the exact dataset coordinates. Differently sized circles/squares
    # and distinct dash patterns keep near-coincident curves distinguishable.
    for variant in ("original", "rc"):
        for stage in ("Revision", "Generation"):
            style = styles[stage]
            original = variant == "original"
            handles[(stage, variant)] = axis.plot(
                x, STAGES[stage][variant], color=style["color"],
                linestyle=style["dash"] if original else "-",
                linewidth=0.95 if original else 1.15,
                marker=style["marker"],
                markersize=(4.3 if stage == "Revision" else 3.1) if original
                else (3.2 if stage == "Revision" else 2.8),
                markerfacecolor="white" if original else style["color"],
                markeredgecolor=style["color"] if original else "white",
                markeredgewidth=0.8 if original else 0.4,
                zorder=3 if original else 4,
            )[0]

    axis.set_xticks(x, [name.replace("\n", " ") for name in DATASETS])
    axis.set_xlim(-0.40, 4.40)
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel("SQL accuracy (%)", labelpad=3)
    axis.tick_params(axis="both", length=2.1, width=0.55, pad=2)
    axis.tick_params(axis="x", labelsize=7)
    axis.tick_params(axis="y", labelsize=6.8)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.4, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.55)
    axis.spines["bottom"].set_linewidth(0.55)

    # Put the gains in a single band above their dataset columns. These are
    # figure labels, not extra data points, and never obscure the four curves.
    for stage, style in styles.items():
        generation = stage == "Generation"
        for point_x, gain in zip(x, STAGES[stage]["gain"]):
            figure_x = figure.transFigure.inverted().transform(
                axis.transData.transform((point_x, 0))
            )[0]
            figure.text(
                figure_x + (-0.007 if generation else 0.007), 0.695,
                f"+{gain:.2f}", ha="right" if generation else "left", va="bottom",
                rotation=30, rotation_mode="default", fontsize=6,
                color=style["color"],
            )

    legend_order = [("Generation", "original"), ("Generation", "rc"),
                    ("Revision", "original"), ("Revision", "rc")]
    figure.legend(
        [handles[key] for key in legend_order],
        ["Gen. Original", "Gen. RC", "Rev. Original", "Rev. RC"],
        frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.035),
        ncol=4, fontsize=6.5, handlelength=1.8, handletextpad=0.45,
        columnspacing=0.9,
    )
    figure.text(0.54, 0.060, "Increasing empirical difficulty →",
                ha="center", va="center", fontsize=7)
    return figure


def make_lines_pair_figure(plt, np):
    """Separate stages horizontally, with just Original and RC in each panel."""
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, 1.25), sharey=True)
    figure.subplots_adjust(left=0.115, right=0.974, bottom=0.32, top=0.885, wspace=0.19)
    x = np.arange(len(DATASETS))
    # Point offsets for these fixed publication data: checked against the
    # rotated text, both curves, markers, panel titles, and axes boundaries.
    gain_offsets = {
        "Generation": [(4, -3), (0, -2), (0, -2), (-5, 1), (-6, 3)],
        "Revision": [(4, -3), (0, -2), (0, -2), (-3, 4), (-4, 3)],
    }
    for axis, (stage, values) in zip(axes, STAGES.items()):
        axis.plot(
            x, values["original"], color="#7A818B", linestyle=(0, (3, 2)),
            linewidth=0.95, marker="o", markersize=2.7,
            markerfacecolor="white", markeredgewidth=0.7, label="Original",
            zorder=3,
        )
        axis.plot(
            x, values["rc"], color="#2870A0", linestyle="-",
            linewidth=1.15, marker="s", markersize=2.8,
            markeredgecolor="white", markeredgewidth=0.35, label="With RS",
            zorder=4,
        )
        axis.set_xticks(x, DATASETS)
        axis.set_xlim(-0.30, 4.30)
        axis.set_ylim(0, 100)
        axis.set_yticks([0, 50, 100])
        axis.tick_params(axis="both", length=2.1, width=0.55, pad=2)
        axis.tick_params(axis="x", labelsize=6)
        axis.tick_params(axis="y", labelsize=6.5)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.4, alpha=0.85)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.55)
        axis.spines["bottom"].set_linewidth(0.55)
        # The lower-left area is empty in both stages. Place panel titles
        # there to avoid adding a heading row to the paper's figure height.
        axis.text(0.055, 0.07, stage, transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=7,
                  color="#313840")
        for index, gain in enumerate(values["gain"]):
            # High-accuracy datasets have no headroom for rotated text above
            # RC. Use the clear area below Original there, and above RC for
            # the two lower-accuracy datasets. All labels stay inside axes.
            below = index < 3
            anchor_y = values["original" if below else "rc"][index]
            axis.annotate(
                f"+{gain:.2f}", (x[index], anchor_y),
                xytext=gain_offsets[stage][index],
                textcoords="offset points", ha="center",
                va="top" if below else "bottom", rotation=-45,
                rotation_mode="default", fontsize=6, color="#2870A0",
                annotation_clip=True, zorder=5,
            )

    figure.text(0.025, 0.505, "SQL accuracy (%)", rotation=90,
                ha="center", va="center", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.54, 1.045), ncol=2, fontsize=7,
                  handlelength=2.0, handletextpad=0.6, columnspacing=1.8)
    figure.text(0.54, 0.060, "Evaluation set (increasing empirical difficulty →)",
                ha="center", va="center", fontsize=7)
    return figure


def plot(output: Path, layout: str = "side-by-side", gain_angle: int = 45) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "rc-complexity-matplotlib"
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
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "legend.fontsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if layout == "side-by-side":
        figure = make_side_by_side_figure(plt, np)
    elif layout == "merged":
        figure = make_merged_figure(plt, np)
    elif layout == "bars":
        figure = make_bars_figure(plt, np, gain_angle)
    elif layout == "lines":
        figure = make_lines_figure(plt, np)
    elif layout == "lines-pair":
        figure = make_lines_pair_figure(plt, np)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the exact physical column width rather than cropping the canvas.
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def main() -> None:
    global DATASETS, STAGES
    args = parse_args()
    DATASETS, STAGES = load_input(args.input)
    plot(args.output, args.layout, args.gain_angle)


if __name__ == "__main__":
    main()
