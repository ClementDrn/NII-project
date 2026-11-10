"""Numerical exports and run provenance, separate from calculations."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .config import input_path


def markdown_table(frame):
    """Small dependency-free Markdown formatter; numerical CSVs stay separate."""
    shown = frame.reset_index() if not isinstance(frame.index, pd.RangeIndex) else frame

    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}" if np.isfinite(value) else ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(str, shown.columns)) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_tables(tables, output):
    directory = Path(output) / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    # CSVs retain numerical precision. Human-readable publication formatting
    # is applied to separate copies and cannot feed back into calculations.
    for name, table in tables.items():
        table.to_csv(
            directory / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex)
        )
        if name in {"table_I", "table_II", "table_III"}:
            display = table.copy()
            if name == "table_I":
                from .adversarial_analysis import format_model_summary

                display = format_model_summary(display)
            if name == "table_II":
                display = display[
                    [
                        "schedule",
                        "deadline_hours",
                        "baseline_completed_attempts",
                        "novelty_percent",
                    ]
                ]
            if name == "table_III":
                display["human_mean"] = display.human_mean.map(lambda v: f"{v:.2f}")
            (directory / f"{name}.md").write_text(
                markdown_table(display) + "\n", encoding="utf-8"
            )


def write_outputs(config, tables, output, command, inputs, figure_names=()):
    """Export computed tables, an output inventory, and input provenance.

    The summary links to numerical tables and figures for researchers to explore.
    """
    output = Path(output)
    write_tables(tables, output)

    # Capture the environment and exact inputs so researchers can rerun the
    # calculations or investigate differences between their own runs.
    versions = {}
    for name in ("digit", "numpy", "pandas", "scipy", "matplotlib", "PyYAML"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "source checkout" if name == "digit" else "unavailable"

    paths = [Path(config["config_path"]), *(input_path(config, p) for p in inputs)]

    # Inventory only files produced by this invocation. Other workflows may
    # already have written additional files into the same output directory.
    outputs = ["manifest.json", "report.md"]
    for name in tables:
        outputs.append(f"tables/{name}.csv")
        if name in {"table_I", "table_II", "table_III"}:
            outputs.append(f"tables/{name}.md")

    for name in figure_names:
        outputs.extend(
            f"figures/{name}.{extension}" for extension in ("pdf", "svg", "png")
        )
    if figure_names:
        outputs.append("figures/captions.md")

    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        command=command,
        configuration=config,
        python=platform.python_version(),
        dependencies=versions,
        inputs=[
            dict(path=str(path.resolve()), sha256=sha256(path.read_bytes()).hexdigest())
            for path in paths
        ],
        analysis_sources={
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        outputs=sorted(outputs),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Link directly to computed outputs. Statistical definitions and assumptions
    # live in the guide and figure captions rather than being repeated per run.
    lines = [
        "# Analysis outputs",
        "",
        f"Workflow: `{command}`. Generated {len(tables)} numerical tables and "
        f"{len(figure_names)} figures.",
        "",
        "## Tables",
        "",
    ]
    for name in tables:
        links = f"[CSV](tables/{name}.csv)"
        if name in {"table_I", "table_II", "table_III"}:
            links += f" · [formatted table](tables/{name}.md)"
        lines.append(f"- `{name}`: {links}")

    if figure_names:
        lines += ["", "## Figures", ""]
        for name in figure_names:
            links = " · ".join(
                f"[{extension.upper()}](figures/{name}.{extension})"
                for extension in ("png", "pdf", "svg")
            )
            lines.append(f"- `{name}`: {links}")
        lines += [
            "",
            "See [figure definitions](figures/captions.md) for the plotted quantities and intervals.",
        ]

    lines += [
        "",
        "## Reproduce this run",
        "",
        "The [manifest](manifest.json) records the resolved study configuration, "
        "random seeds, input and analysis-source hashes, and dependency versions.",
        "",
        "The analysis guide describes statistical definitions, assumptions, and "
        "the distinction between observed outcomes, conditional expectations, "
        "and modeled timing estimates.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
