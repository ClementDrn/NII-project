"""Shared styling at final paper size (inches); no automatic export cropping."""

COLUMN_WIDTH = 3.5
HALF_COLUMN_WIDTH = COLUMN_WIDTH / 2
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#999999"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "hatch.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": None,
}


def clean_axes(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="0.90", linewidth=0.5)


def lift_digit_labels(ax):
    """Align method labels leftward and lift unequal-line DIGIT blocks.

    Offsets are physical points, so they survive vector export and DPI changes.
    The longer incremental block needs more correction than the one-shot block.
    Call once after configuring tick labels and their rotation.
    """
    from matplotlib.transforms import ScaledTranslation
    import math

    for label in ax.get_xticklabels():
        text = label.get_text()
        if text != "Baseline" and not text.startswith("DIGIT\n"):
            continue
        rotation = abs(label.get_rotation())
        if not rotation:
            continue
        points = 0.0
        if text.startswith("DIGIT\n") and rotation:
            points = 6.0 if "incremental" in text else 3.5
            points *= abs(math.sin(math.radians(rotation))) / math.sin(math.radians(40))
        left_points = 8.0
        label.set_transform(
            label.get_transform()
            + ScaledTranslation(
                -left_points / 72, points / 72, ax.figure.dpi_scale_trans
            )
        )


def tighten_bottom_margin(fig, padding_points=4):
    """Expand the plot downward, leaving a small gap below its tick labels.

    Measure after rotation and offsets so each panel uses only the space its
    labels need. Keep the canvas width/height and the top of the axes fixed.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [
        label
        for ax in fig.axes
        for label in ax.get_xticklabels()
        if label.get_visible() and label.get_text()
    ]
    if not labels:
        return
    lowest = min(label.get_window_extent(renderer).y0 for label in labels)
    padding = padding_points * fig.dpi / 72
    bottom = fig.subplotpars.bottom - (lowest - padding) / fig.bbox.height
    fig.subplots_adjust(bottom=max(0, bottom))
