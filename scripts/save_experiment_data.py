import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from untitled.data.job_results import ExperimentResult, JobResults


FIELDNAMES = [
    "model",
    "method",
    "repetition",
    "scenario",
    "run",
    "redacted?",
    "errored?",
    "adversarial?",
    "instruction",
    "category",
    "start",
    "end",
    "reasoning",
]


def _load_prompt(run_dir: Path) -> Dict[str, Any]:
    prompts_file = run_dir / "custom_prompts.json"
    try:
        with prompts_file.open("r", encoding="utf-8") as file:
            prompts = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if isinstance(prompts, list) and prompts and isinstance(prompts[0], dict):
        return prompts[0]
    return {}


def _experiment_rows(job_dir: Path, ground_truth_cache_dir: Path) -> Iterable[Dict[str, Any]]:
    results = JobResults(job_dir, ground_truth_cache_dir)

    for scenario in results.get_all_scenarios():
        experiments = {experiment.run_id: experiment for experiment in scenario.experiments}
        scenario_dir = job_dir / f"sce_{scenario.scenario_id:03d}"

        for run_dir in sorted(scenario_dir.glob("run_*")):
            if not run_dir.is_dir():
                continue
            try:
                run_id = int(run_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            experiment = experiments.get(run_id)
            if experiment is None:
                continue

            prompt = _load_prompt(run_dir)
            start = prompt.get("injection_time", experiment.t_start)
            duration = prompt.get("injection_duration")
            end = experiment.t_end
            if duration is not None and start is not None:
                end = float(start) + float(duration)

            yield {
                "model": "N/A",
                "method": "N/A",
                "repetition": "N/A",
                "scenario": scenario.scenario_id,
                "run": run_id,
                "redacted?": 0,
                "errored?": int(experiment.is_error),
                "adversarial?": int(
                    scenario.classify_run(experiment) == ExperimentResult.ADVERSARIAL
                ),
                "instruction": prompt.get("text", ""),
                "category": "UNKNOWN",
                "start": start if start is not None else "",
                "end": end if end is not None else "",
                "reasoning": prompt.get("reasoning", ""),
            }


def save_experiment_data(
    job_dir: Path, output_path: Path, ground_truth_cache_dir: Path
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in _experiment_rows(job_dir, ground_truth_cache_dir):
            writer.writerow(row)
            row_count += 1
    return row_count


def _output_path(
    output: Path | None, experiment_dir: Path, job_dir: Path, project_root: Path
) -> Path:
    job_suffix = job_dir.name.removeprefix("job_")
    experiment_name = experiment_dir.name

    if output is None:
        return project_root / "data" / f"{experiment_name}_job_{job_suffix}.csv"
    if output.suffix.lower() == ".csv":
        return output.with_name(f"{output.stem}_job_{job_suffix}{output.suffix}")
    return output / f"{experiment_name}_job_{job_suffix}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Save experiment runs as CSV data")
    parser.add_argument("experiment_dir", type=Path, help="Path containing job directories")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output CSV base path or directory (default: data/{experiment}_job_{id}.csv)",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("cache"),
        help="Path to the ground truth cache directory (default: cache)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    experiment_dir = (
        args.experiment_dir
        if args.experiment_dir.is_absolute()
        else project_root / args.experiment_dir
    )
    output = args.output
    if output is not None and not output.is_absolute():
        output = project_root / output
    ground_truth_cache_dir = (
        args.gt_dir if args.gt_dir.is_absolute() else project_root / args.gt_dir
    )

    job_dirs = sorted(
        job_dir
        for job_dir in experiment_dir.glob("job_*")
        if job_dir.is_dir()
    )
    if not job_dirs:
        parser.error(f"No job directories found in {experiment_dir}")

    for job_dir in job_dirs:
        output_path = _output_path(output, experiment_dir, job_dir, project_root)
        row_count = save_experiment_data(job_dir, output_path, ground_truth_cache_dir)
        print(f"Wrote {row_count} experiment runs to {output_path}")


if __name__ == "__main__":
    main()