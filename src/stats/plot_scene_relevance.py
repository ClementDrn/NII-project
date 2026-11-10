"""Publication scene-reference bars and object-type clouds.

Matches the notebook's selected figures; exported through save_figure.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch
from .plot_naturalness import COLORS as EVAL_COLORS, save_figure


def plot_scene_relevance(scene_summary, scene_ci):
    """Balanced means and supplied CIs; shading and hatches mark subsets."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    scene_order = list(scene_summary.index)
    scene_labels = ["Baseline", "DIGIT\n(one-shot)", "DIGIT\n(incremental)"]
    scene_figure, scene_ax = plt.subplots(figsize=(3.5, 2.9))
    x = np.arange(len(scene_order))
    width = 0.32
    for j, (noun, color) in enumerate(zip(["object", "modifier"], EVAL_COLORS)):
        positions = x + (j - 0.5) * (width + 0.03)
        total = scene_summary[f"{noun}_count"].to_numpy()
        flagged = scene_summary[f"{noun}_hallucination_count"].to_numpy()
        scene_ax.bar(
            positions, total, width, color=color, edgecolor="#333333", linewidth=0.35
        )
        # Separate translucent shading from the fully opaque hatch strokes.
        scene_ax.bar(
            positions,
            flagged,
            width,
            color=(0, 0, 0, 0.22),
            edgecolor="none",
            linewidth=0,
        )
        hallucination_bars = scene_ax.bar(
            positions,
            flagged,
            width,
            facecolor="none",
            edgecolor="#111111",
            hatch="////",
            linewidth=0.6,
            hatch_linewidth=0.8,
        )
        # Matplotlib centers strokes on patch edges. Clip to each bar's own
        # rectangle so the outer half of the outline cannot exceed its bounds.
        for bar in hallucination_bars:
            bar.set_clip_path(bar.get_path(), bar.get_transform())
        for metric, values, offset, ecolor in [
            (f"{noun}_count", total, 0, "#222222"),
            (f"{noun}_hallucination_count", flagged, 0.055, "#222222"),
        ]:
            ci = scene_ci.xs(metric, level="metric").reindex(scene_order)
            scene_ax.errorbar(
                positions + offset,
                values,
                yerr=np.vstack(
                    [
                        values - ci.CI_95_low.to_numpy(),
                        ci.CI_95_high.to_numpy() - values,
                    ]
                ),
                fmt="none",
                ecolor=ecolor,
                capsize=2,
                linewidth=0.7,
            )
        for pos, mean, pct, high in zip(
            positions,
            total,
            scene_summary[f"{noun}_hallucination_pct"],
            scene_ci.xs(f"{noun}_count", level="metric")
            .reindex(scene_order)
            .CI_95_high,
        ):
            scene_ax.text(pos, high + 0.025, f"{mean:.2f}", ha="center", fontsize=6.5)
            # Use axes-relative height to keep percentage labels outside the data area.
            scene_ax.text(
                pos,
                -0.045,
                f"{pct:.1f}%",
                transform=scene_ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=5.5,
                clip_on=False,
            )
    scene_ax.set_xticks(x, scene_labels)
    scene_ax.set_ylabel("Mean count per instruction", fontsize=8)
    scene_ax.tick_params(axis="both", labelsize=7, length=2)
    scene_ax.tick_params(axis="x", pad=15)
    scene_ax.set_ylim(
        0,
        scene_ci.loc[
            (slice(None), ["object_count", "modifier_count"]), "CI_95_high"
        ].max()
        + 0.13,
    )
    scene_figure.legend(
        handles=[
            Patch(facecolor=EVAL_COLORS[0], edgecolor="#333333", label="Objects"),
            Patch(facecolor=EVAL_COLORS[1], edgecolor="#333333", label="Modifiers"),
            Patch(
                facecolor="none",
                edgecolor="#111111",
                hatch="////",
                hatch_linewidth=0.8,
                label="Hallucinated subset",
            ),
        ],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        frameon=False,
        fontsize=6.3,
        handlelength=1.3,
        columnspacing=0.8,
    )
    scene_ax.spines[["top", "right"]].set_visible(False)
    scene_ax.set_axisbelow(True)
    scene_ax.grid(axis="y", alpha=0.2)
    scene_figure.subplots_adjust(left=0.16, right=0.99, bottom=0.23, top=0.86)
    return scene_figure, scene_ax


def plot_object_type_clouds(object_type_frequency):
    """Common-scale mention areas, with matching letter IDs on all circles."""
    object_type_order = list(object_type_frequency.index)
    object_type_ranking = (
        object_type_frequency.mean().sort_values(ascending=False).index
    )
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch

    active_types = [
        t for t in object_type_ranking if object_type_frequency[t].max() > 0
    ]
    # Distinct categorical colors; names/letters identify types without relying on color.
    cloud_palette = [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
        "#999999",
        "#332288",
        "#88CCEE",
        "#44AA99",
        "#AA4499",
        "#DDCC77",
        "#882255",
        "#117733",
    ]
    type_colors = {
        t: cloud_palette[i % len(cloud_palette)] for i, t in enumerate(active_types)
    }
    type_ids = {t: chr(ord("A") + i) for i, t in enumerate(active_types)}

    # Deterministic spiral packing; radii are sqrt(frequency), so area is proportional.
    def pack_type_cloud(frequencies):
        placed = []
        for name, value in (
            frequencies[frequencies > 0].sort_values(ascending=False).items()
        ):
            radius = np.sqrt(value)
            for step in range(50000):
                angle = step * 2.399963229728653
                distance = 0.10 * np.sqrt(step)
                cx, cy = distance * np.cos(angle), distance * np.sin(angle)
                if all(
                    np.hypot(cx - px, cy - py) >= radius + pr + 0.22
                    for _, _, px, py, pr in placed
                ):
                    placed.append((name, value, cx, cy, radius))
                    break
            else:
                raise RuntimeError("Could not pack object-type cloud")
        # Rotate the existing non-overlapping packing so car and driving-lane
        # circles share a horizontal centerline. Rotation preserves all areas,
        # pairwise distances, and the arrangement of the smaller circles.
        anchors = {name: (cx, cy) for name, _, cx, cy, _ in placed}
        if "car" in anchors and "driving_lane" in anchors:
            origin = np.array(anchors["car"])
            delta = np.array(anchors["driving_lane"]) - origin
            angle = np.pi - np.arctan2(delta[1], delta[0])
            cosine, sine = np.cos(angle), np.sin(angle)
            rotated = []
            for name, value, cx, cy, radius in placed:
                dx, dy = np.array([cx, cy]) - origin
                rotated.append(
                    (
                        name,
                        value,
                        cosine * dx - sine * dy,
                        sine * dx + cosine * dy,
                        radius,
                    )
                )
            placed = rotated
        return placed

    type_clouds = [
        pack_type_cloud(object_type_frequency.loc[setting])
        for setting in object_type_order
    ]
    # Center each cloud tightly, while keeping a common data-to-physical-size scale.
    cloud_bounds = [
        (
            min(cx - r for _, _, cx, cy, r in cloud),
            max(cx + r for _, _, cx, cy, r in cloud),
            min(cy - r for _, _, cx, cy, r in cloud),
            max(cy + r for _, _, cx, cy, r in cloud),
        )
        for cloud in type_clouds
    ]
    # A common rectangular viewport avoids the empty corners of square axes.
    cloud_width = max(right - left for left, right, bottom, top in cloud_bounds) * 1.035
    cloud_height = (
        max(top - bottom for left, right, bottom, top in cloud_bounds) * 1.035
    )
    figure_width = 7.2
    axes_width = figure_width * 0.99 / (3 + 2 * 0.035)
    bottom_inches, top_inches = 0.79, 0.03
    figure_height = axes_width * cloud_height / cloud_width + bottom_inches + top_inches
    type_figure, type_axes = plt.subplots(1, 3, figsize=(figure_width, figure_height))
    for ax, cloud, bounds, label in zip(
        type_axes,
        type_clouds,
        cloud_bounds,
        ["Baseline", "DIGIT\n(one-shot)", "DIGIT\n(incremental)"],
    ):
        for name, value, cx, cy, radius in cloud:
            circle = Circle(
                (cx, cy),
                radius,
                facecolor=type_colors[name],
                edgecolor="#333333",
                linewidth=0.35,
            )
            ax.add_patch(circle)
            # Large bubbles show object names and frequency; small bubbles retain letter IDs.
            name_label = name.replace("_", "\n")
            # Long single words keep a separate ID line to fit their circles.
            prefix = (
                f"{type_ids[name]}.\n"
                if name in {"intersection", "shoulder"}
                else f"{type_ids[name]}. "
            )
            text = (
                prefix + name_label + f"\n{value:.1f}"
                if value >= 7
                else str(type_ids[name])
            )
            # Choose black or white text by relative luminance for readable labels.
            rgb = np.array(plt.matplotlib.colors.to_rgb(type_colors[name]))
            linear_rgb = np.where(
                rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
            )
            luminance = linear_rgb @ np.array([0.2126, 0.7152, 0.0722])
            text_color = "white" if luminance < 0.179 else "#111111"
            annotation = ax.text(
                cx,
                cy,
                text,
                ha="center",
                va="center",
                fontsize=(
                    (5.2 if name in {"intersection", "shoulder"} else 6.5)
                    if value >= 7
                    else 6
                ),
                color=text_color,
                bbox=dict(
                    facecolor=type_colors[name], edgecolor="none", alpha=1, pad=0.6
                ),
            )
            # Keep label backgrounds inside even the smallest circles.
            annotation.get_bbox_patch().set_clip_path(circle)
            annotation.get_bbox_patch().set_clip_on(True)
        left, right, bottom, top = bounds
        center_x, center_y = (left + right) / 2, (bottom + top) / 2
        ax.set(
            xlim=(center_x - cloud_width / 2, center_x + cloud_width / 2),
            ylim=(center_y - cloud_height / 2, center_y + cloud_height / 2),
            aspect="equal",
        )
        ax.axis("off")
        ax.text(
            0.5,
            -0.035,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
    type_figure.legend(
        handles=[
            Patch(
                facecolor=type_colors[t],
                edgecolor="#333333",
                label=f"{type_ids[t]}. {t.replace('_', ' ')}",
            )
            for t in active_types
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=7,
        frameon=False,
        fontsize=6.8,
        columnspacing=1.1,
        handlelength=1.4,
    )
    type_figure.subplots_adjust(
        top=1 - top_inches / figure_height,
        bottom=bottom_inches / figure_height,
        left=0.005,
        right=0.995,
        wspace=0.035,
    )
    return type_figure, type_axes
