"""Selected naturalness A/B figures from item-level and full-pool estimates.

Intervals describe instruction sampling conditional on the observed human panel
and LLM scores; they do not include uncertainty from a new panel or judge rerun.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from .publication_style import (
    STYLE,
    COLUMN_WIDTH,
    clean_axes,
    lift_digit_labels,
    tighten_bottom_margin,
)

COLORS = ("#44AA99", "#CC79A7")  # Cloud J teal and naturalness pink
LABELS = ("Human panel", "LLM judge")
QUANTITIES = ("human_mean", "llm_score")
MARKERS = ("o", "s")


def _estimates(report, names):
    rows = report.loc[names, ["estimate", "CI_95_low", "CI_95_high"]]
    values = rows.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Estimates and interval endpoints must be finite.")
    means, lower, upper = values.T
    errors = np.vstack((means - lower, upper - means))
    if (errors < 0).any():
        raise ValueError("Each interval must contain its point estimate.")
    return means, errors


def save_figure(fig, stem):
    """Export PDF/SVG and 600-dpi PNG, preserving exact canvas dimensions."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        with plt.rc_context(STYLE):
            fig.savefig(stem.with_suffix(f".{extension}"), dpi=600, bbox_inches=None)


def plot_naturalness_evidence(report, variants):
    """Return separate A/B figures; preserve each evaluation's estimand."""
    with plt.rc_context(STYLE):
        figures = []
        axes = []
        for _ in range(2):
            fig, ax = plt.subplots(figsize=((COLUMN_WIDTH - 0.10) / 2, 2.3))
            fig.subplots_adjust(left=0.23, right=0.98, bottom=0.30, top=0.80)
            figures.append(fig)
            axes.append(ax)

        def value_label(ax, x, mean, color):
            ax.text(
                x,
                mean / 2,
                f"{mean:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="#111111",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 0.4},
                zorder=5,
            )

        for j, (quantity, label, color) in enumerate(zip(QUANTITIES, LABELS, COLORS)):
            means, errors = _estimates(
                report, [f"baseline_{quantity}", f"ours_{quantity}"]
            )
            x = np.arange(2) + (j - 0.5) * 0.32
            axes[0].bar(
                x,
                means,
                width=0.29,
                yerr=errors,
                color=color,
                edgecolor="#333333",
                linewidth=0.4,
                label=label,
                error_kw={
                    "ecolor": "#222222",
                    "capsize": 2,
                    "elinewidth": 0.7,
                    "capthick": 0.7,
                },
            )
            for position, mean in zip(x, means):
                value_label(axes[0], position, mean, color)
        axes[0].set_xticks([0, 1], ["Baseline", "DIGIT\n(incremental)"])
        axes[0].legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.025),
            ncol=1,
            fontsize=6,
            borderaxespad=0,
            handlelength=1.5,
            columnspacing=1.2,
        )

        for x, method in enumerate(["baseline", "one-shot", "incremental"]):
            data = variants[variants.method.eq(method)]
            scenario_means = data.groupby("scenario").naturalness.mean()
            mean = scenario_means.mean()
            axes[1].bar(
                x,
                mean,
                width=0.55,
                color=COLORS[1],
                edgecolor="#333333",
                linewidth=0.4,
                label="LLM judge" if x == 0 else None,
            )
            axes[1].scatter(
                x + np.random.default_rng(42).uniform(-0.16, 0.16, len(scenario_means)),
                scenario_means,
                color="#222222",
                alpha=0.55,
                s=4,
                edgecolors="none",
                zorder=3,
                label="Scenario means" if x == 0 else None,
            )
            value_label(axes[1], x, mean, COLORS[1])
        axes[1].set_xticks(
            [0, 1, 2], ["Baseline", "DIGIT\n(one-shot)", "DIGIT\n(incremental)"]
        )
        # Center the long two-line labels and tilt them to avoid collisions.
        axes[1].set_xlim(-0.5, 2.5)
        for label in axes[1].get_xticklabels():
            label.set_rotation(35)
            label.set_ha("center")
            label.set_multialignment("center")
            label.set_rotation_mode("default")
        handles, labels = axes[1].get_legend_handles_labels()
        order = [labels.index("LLM judge"), labels.index("Scenario means")]
        axes[1].legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.025),
            ncol=1,
            fontsize=6,
            borderaxespad=0,
            handlelength=1.5,
        )
        for ax in axes:
            ax.set_ylim(0, 5)
            ax.set_yticks(range(6))
            ax.set_ylabel("Mean naturalness (1–5)", fontsize=7, labelpad=2)
            ax.tick_params(axis="y", pad=2, labelsize=6)
            ax.tick_params(axis="x", labelsize=5.5)
            clean_axes(ax, "y")
            lift_digit_labels(ax)
        for fig in figures:
            tighten_bottom_margin(fig)
        # Paired subfigures need aligned axes despite different tick-label heights.
        bottom = max(fig.subplotpars.bottom for fig in figures)
        for fig in figures:
            fig.subplots_adjust(bottom=bottom)
        return tuple(figures), tuple(axes)
