"""Offline command-line workflows for the paper and new exported experiments."""

import argparse
from pathlib import Path

from . import calculations
from .config import load_config
from .naturalness import analyze_naturalness
from .reporting import write_outputs
from .scene import analyze_scene
from .validation import load_results

COMMANDS = (
    "reproduce",
    "validate",
    "effectiveness",
    "complementarity",
    "naturalness",
    "scene",
    "timing",
    "diagnostics",
)


def run_analysis(command, config, output, figures=True):
    """Run a named workflow; return numerical tables and generated figure names.

    The configuration comes from :func:`load_config`. Calculation functions are
    also callable directly when filesystem outputs are not wanted.
    """
    if command not in COMMANDS:
        raise ValueError(f"Unknown analysis command: {command}")
    tables, inputs, frame = {}, [], None

    # Timing needs only modeled durations. Every other workflow first shares
    # the same validated experiment table and input-provenance record.
    all_results = command in {
        "reproduce",
        "validate",
        "effectiveness",
        "complementarity",
        "naturalness",
        "scene",
        "diagnostics",
    }
    if all_results:
        frame = load_results(config)
        inputs.extend(config["results"])
        tables["input_validation"] = frame.groupby(
            ["model", "method", "repetition"]
        ).agg(attempts=("run", "size"), scenarios=("scenario", "nunique"))

    # Dispatch calculation groups independently so individual commands and the
    # full reproduction use exactly the same statistical implementations.
    if command in {"reproduce", "effectiveness"}:
        tables.update(calculations.effectiveness(frame, config))
    if command in {"reproduce", "complementarity", "diagnostics"}:
        tables.update(
            calculations.complementarity(
                frame, config, sensitivities=command == "diagnostics"
            )
        )

    # Validation runs the annotation checks too, but publishes only the audit
    # table instead of presenting partial analytical results.
    if command in {"reproduce", "naturalness", "validate"}:
        natural, _ = analyze_naturalness(frame, config)
        inputs.extend(
            config["naturalness"][k] for k in ("instructions", "human", "llm")
        )
        if command != "validate":
            tables.update(natural)
    if command in {"reproduce", "scene", "validate"}:
        scene = analyze_scene(frame, config)
        if command != "validate":
            tables.update(scene)
    if command == "timing":
        summary, schedules = calculations.timing(config)
        tables["timing_summary"] = summary
        import pandas as pd

        tables["modeled_completion_schedule"] = pd.concat(
            [
                s.assign(schedule=name, setting=setting)
                for name, (setting, s) in schedules.items()
            ],
            ignore_index=True,
        )
    if command == "diagnostics":
        tables.update(calculations.diagnostics(frame, config))

    # Import plotting only when requested. Numerical-only runs do not need to
    # initialize a graphics backend or create figure output files.
    figure_names = []
    if figures and command not in {
        "validate",
        "timing",
        "diagnostics",
        "effectiveness",
    }:
        from .plotting import create_figures

        figure_names = create_figures(tables, frame, config, output)

    write_outputs(config, tables, output, command, inputs, figure_names)
    return tables, figure_names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "scripts/stats/paper.yaml",
        help="YAML study profile (default: checkout's scripts/stats/paper.yaml)",
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Override the profile's input data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Export calculations and reports without rendering",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.data_dir)
        tables, figures = run_analysis(
            args.command, config, args.output_dir, not args.no_figures
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(
        f"Wrote {len(tables)} numerical tables and {len(figures)} figures to {args.output_dir.resolve()}"
    )
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")
    return 0
