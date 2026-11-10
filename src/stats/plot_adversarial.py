"""Publication failure progression and common-deadline discovery overlaps."""

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.optimize import brentq

from .adversarial_analysis import BASELINE
from .publication_style import (
    BLUE,
    ORANGE,
    GRAY,
    STYLE,
    COLUMN_WIDTH,
    HALF_COLUMN_WIDTH,
    clean_axes,
)

COLORS = (GRAY, BLUE, ORANGE)


def _intersection_area(distance, r1, r2):
    if distance >= r1 + r2:
        return 0.0
    if distance <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a = math.acos(np.clip((distance**2 + r1**2 - r2**2) / (2 * distance * r1), -1, 1))
    b = math.acos(np.clip((distance**2 + r2**2 - r1**2) / (2 * distance * r2), -1, 1))
    radicand = (
        (-distance + r1 + r2)
        * (distance + r1 - r2)
        * (distance - r1 + r2)
        * (distance + r1 + r2)
    )
    return r1**2 * a + r2**2 * b - 0.5 * math.sqrt(max(0.0, radicand))


def plot_failure_progression(
    progression,
    metric="cumulative_failures",
    model="claude-opus-4.6",
    scenario_count=25,
):
    """One cumulative-count figure; baseline expected, DIGIT ordered/observed.

    All methods share an absolute attempt axis and stop at DIGIT's completion
    for both metrics. The dotted reference marks that budget, not elapsed time.
    Cumulative failures are counts, not scenario-balanced effectiveness rates.
    Both metrics use one full text column: COLUMN_WIDTH × 2.3 inches. Export
    with save_figure to preserve these dimensions without automatic cropping.
    Use the shared palette for lines, markers, and numeric labels:
    one-shot #0072B2, incremental #E69F00, baseline grey.
    """
    if metric not in ("cumulative_failures", "discovered_scenarios"):
        raise ValueError("Unknown progression metric")
    settings = [BASELINE, f"{model} / one-shot", f"{model} / incremental"]
    groups = [
        progression.loc[progression.setting.eq(s)].sort_values("attempts")
        for s in settings
    ]
    if any(g.empty for g in groups):
        raise ValueError("Require baseline and both DIGIT progression curves")
    completion = int(groups[1].attempts.max())
    if groups[2].attempts.max() != completion:
        raise ValueError("DIGIT variants must have the same attempt budget")
    groups = [g.loc[g.attempts.le(completion)] for g in groups]
    maximum = int(max(g.attempts.max() for g in groups))
    labels = ["Baseline", "DIGIT (one-shot)", "DIGIT (incremental)"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
        fig.subplots_adjust(left=0.15, right=0.965, bottom=0.19, top=0.96)
        reference_max = max(
            g.loc[g.attempts.eq(completion), metric].max() for g in groups
        )
        # Stop the budget guide at the data, keeping it out of the legend.
        ax.vlines(
            completion,
            0,
            reference_max,
            color="0.65",
            linestyle=":",
            linewidth=0.7,
            zorder=1,
        )
        for i, (group, label, color) in enumerate(zip(groups, labels, COLORS)):
            ax.step(
                group.attempts,
                group[metric],
                where="post",
                label=label,
                color=color,
                linewidth=1.0,
                linestyle="--" if i == 0 else "-",
                zorder=2 + i,
            )
            marks = group.loc[
                group.attempts.isin(
                    [
                        3 * scenario_count,
                        6 * scenario_count,
                        completion,
                        int(group.attempts.max()),
                    ]
                )
            ]
            ax.plot(
                marks.attempts,
                marks[metric],
                linestyle="none",
                color=color,
                marker=("o", "s", "^")[i],
                markersize=3,
                markerfacecolor="white",
                markeredgewidth=0.7,
                zorder=6 + i,
            )
        ax.set_xlim(0, maximum * 1.025)
        ticks = sorted(
            {
                0,
                3 * scenario_count,
                6 * scenario_count,
                completion,
                2 * completion,
                maximum,
            }
        )
        ax.set_xticks([t for t in ticks if t <= maximum])
        ax.set_xlabel("Evaluated attempts", fontsize=7, labelpad=3)
        if metric == "cumulative_failures":
            upper = max(g[metric].max() for g in groups)
            ax.set_ylim(0, max(5, math.ceil(upper / 10) * 10))
            ax.set_ylabel("Cumulative failure outcomes", fontsize=7, labelpad=3)
        else:
            ax.set_ylim(0, 20)
            ax.set_yticks(np.arange(0, 21, 5))
            ax.set_ylabel("Scenarios with ≥1 failure", fontsize=7, labelpad=3)
        ax.tick_params(labelsize=6.5, pad=2)
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=6.5,
            handlelength=2.0,
            borderaxespad=0.5,
            labelspacing=0.35,
        )
        clean_axes(ax)
        # Retain the selected completion-label offsets and also label the
        # intermediate three- and six-attempt-per-scenario budgets (75/150).
        offsets = (
            [(-6, -10), (-6, 8), (-6, -10)]
            if metric == "cumulative_failures"
            else [(-6, -10), (-6, 8), (-6, 5)]
        )
        budgets = sorted({3 * scenario_count, 6 * scenario_count, completion})
        for budget in budgets:
            for i, (group, color) in enumerate(zip(groups, COLORS)):
                row = group.loc[group.attempts.eq(budget), metric]
                if row.empty:
                    continue
                value = float(row.iloc[0])
                text = f"{value:.2f}" if i == 0 else f"{value:.0f}"
                offset = offsets[i]
                if budget != completion and i == 2:
                    if metric == "cumulative_failures":
                        offset = (6, 14)
                    else:
                        offset = (6, 4 if budget == 3 * scenario_count else -7)
                ax.annotate(
                    text,
                    (budget, value),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7,
                    color="#555555" if i == 0 else color,
                    ha="right" if offset[0] < 0 else "left",
                    va="center",
                    zorder=10,
                )
        return fig, ax


def plot_failure_overlap(
    frame,
    budget_comparisons,
    model="claude-opus-4.6",
    schedules=("one_shot_batch", "incremental_pipeline_1"),
    panel=None,
):
    """Area-proportional expected overlap at one common modeled deadline.

    Use the time-budget results, not full observed discovery sets.
    Fractional regions are conditional expectations over baseline subsets;
    DIGIT outcomes are replayed only through the deadline.
    """
    if panel not in (None, "one-shot", "incremental"):
        raise ValueError("panel must be None, 'one-shot', or 'incremental'")
    if len(schedules) != 2:
        raise ValueError("Select one schedule for each DIGIT method")
    common = budget_comparisons[
        budget_comparisons.comparison == "same_time_common_deadline"
    ]
    selected = []
    scenarios = set(frame.loc[frame.setting == BASELINE, "scenario"])
    for method, schedule in zip(["one-shot", "incremental"], schedules):
        setting = f"{model} / {method}"
        rows = common[(common.schedule == schedule) & (common.setting == setting)]
        if len(rows) != 1:
            raise ValueError(
                f"Require one common-deadline result for {setting}: {schedule}"
            )
        if set(frame.loc[frame.setting == setting, "scenario"]) != scenarios:
            raise ValueError("Set diagrams require the same scenario universe")
        selected.append(rows.iloc[0])
    if (
        not np.isfinite([r.deadline_hours for r in selected]).all()
        or selected[0].deadline_hours < 0
    ):
        raise ValueError("Deadline must be finite and nonnegative")
    if not np.isclose(selected[0].deadline_hours, selected[1].deadline_hours):
        raise ValueError("Both panels must use the same time budget")
    if not np.isclose(
        selected[0].expected_baseline_failure_scenarios,
        selected[1].expected_baseline_failure_scenarios,
    ):
        raise ValueError("Both panels must use the same baseline budget")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.65))
        for ax, method, color, row in zip(
            axes, ["one-shot", "incremental"], COLORS[1:], selected
        ):
            base = float(row.expected_baseline_failure_scenarios)
            other = float(row.target_failure_scenarios)
            left_only = float(row.expected_baseline_only_scenarios)
            right_only = float(row.expected_target_only_scenarios)
            shared = other - right_only
            neither = len(scenarios) - left_only - right_only - shared
            regions = np.array([left_only, right_only, shared, neither])
            if (
                not np.isfinite(regions).all()
                or (regions < -1e-8).any()
                or not np.isclose(base, left_only + shared)
            ):
                raise ValueError("Invalid expected overlap counts")
            ax.set_aspect("equal")
            ax.axis("off")
            if not base or not other or np.isclose(shared, min(base, other)):
                ax.text(
                    0.5,
                    0.5,
                    f"Baseline only: {left_only:.2f}\nBoth: {shared:.2f}\nDIGIT ({method}) only: {right_only:.2f}\nNeither: {neither:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue
            r1, r2 = math.sqrt(base / math.pi), math.sqrt(other / math.pi)
            distance = (
                (r1 + r2 + 0.25)
                if shared == 0
                else brentq(
                    lambda d: _intersection_area(d, r1, r2) - shared,
                    abs(r1 - r2) + 1e-10,
                    r1 + r2,
                )
            )
            ax.add_patch(
                Circle(
                    (0, 0),
                    r1,
                    facecolor=plt.matplotlib.colors.to_rgba(COLORS[0], 0.3),
                    edgecolor="#555555",
                    linewidth=0.6,
                )
            )
            ax.add_patch(
                Circle(
                    (distance, 0),
                    r2,
                    facecolor=plt.matplotlib.colors.to_rgba(color, 0.22),
                    edgecolor="#333333",
                    linewidth=0.6,
                )
            )
            ax.text(
                (-r1 + min(r1, distance - r2)) / 2,
                0,
                f"{left_only:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
            ax.text(
                (max(r1, distance - r2) + distance + r2) / 2,
                0,
                f"{right_only:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
            if shared:
                ax.text(
                    (max(-r1, distance - r2) + min(r1, distance + r2)) / 2,
                    0,
                    f"{shared:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )
            ax.text(
                -r1 * 0.65,
                -max(r1, r2) - 0.35,
                "Baseline",
                ha="center",
                va="top",
                fontsize=6.5,
            )
            ax.text(
                distance + r2 * 0.65,
                -max(r1, r2) - 0.35,
                f"DIGIT\n({method})",
                ha="center",
                va="top",
                fontsize=6.5,
            )
            ax.text(
                (distance + r2 - r1) / 2,
                max(r1, r2) + 0.25,
                f"Neither: {neither:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
            ax.set_xlim(-r1 - 0.4, distance + r2 + 0.4)
            ax.set_ylim(-max(r1, r2) - 1.15, max(r1, r2) + 0.65)
        # Identical physical area scale across the two comparisons.
        xspan = max(ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes)
        yspan = max(ax.get_ylim()[1] - ax.get_ylim()[0] for ax in axes)
        for ax in axes:
            cx = sum(ax.get_xlim()) / 2
            cy = sum(ax.get_ylim()) / 2
            ax.set_xlim(cx - xspan / 2, cx + xspan / 2)
            ax.set_ylim(cy - yspan / 2, cy + yspan / 2)
        fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.06, wspace=0.08)
        if panel is not None:
            index = ["one-shot", "incremental"].index(panel)
            axes[1 - index].remove()
            axes = np.array([axes[index]])
            fig.set_size_inches(HALF_COLUMN_WIDTH, 1.65)
            axes[0].set_position([0.03, 0.06, 0.94, 0.92])
        return fig, axes
