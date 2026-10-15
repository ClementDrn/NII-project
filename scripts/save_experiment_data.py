import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from untitled.core.base import LOG, SIMULATION_FPS
from untitled.data.hallucinations import check_hallucinations_in_instruction
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


HALLUCINATION_FIELDNAMES = [
    "object_count",
    "modifier_count",
    "object_hallucination_count",
    "modifier_hallucination_count",
]


def _load_injection_metadata(
    run_dir: Path,
    job_dir: Path,
    ground_truth_cache_dir: Path,
    scenario_id: int,
    ground_truth_index: int | None,
    injection_time: Any,
) -> Dict[str, Any] | None:
    """
    Loads the last saved frame at or before the instruction injection.
    Tries the experiment run, then local and cached ground truth metadata.
    """
    try:
        injection_step = float(injection_time) * SIMULATION_FPS
    except (TypeError, ValueError):
        LOG.warning(f"Cannot check hallucinations for {run_dir}: invalid injection time '{injection_time}'.")
        return None

    # JobResults may omit the index if the ground truth result file is missing.
    # The experiment info still identifies the ground truth used for generation.
    if ground_truth_index is None:
        try:
            with (run_dir / "experiment_info.json").open("r", encoding="utf-8") as file:
                experiment_info = json.load(file)
            ground_truth_index = int(experiment_info.get("ground_truth_index", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            ground_truth_index = 0

    ground_truth_relative_dir = Path(f"sce_{scenario_id:03d}") / f"run_{ground_truth_index:03d}"
    metadata_dirs = [
        run_dir / "metadata",
        job_dir / "ground_truths" / ground_truth_relative_dir / "metadata",
        ground_truth_cache_dir / ground_truth_relative_dir / "metadata",
    ]
    for metadata_dir in metadata_dirs:
        # Filenames are simulator step numbers. Never use a future frame.
        candidate_files = []
        for metadata_path in metadata_dir.glob("*.json"):
            try:
                step = int(metadata_path.stem)
            except ValueError:
                continue
            if step <= injection_step:
                candidate_files.append((step, metadata_path))

        if not candidate_files:
            continue
        _, metadata_path = max(candidate_files)
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, ValueError):
            LOG.warning(f"Cannot read injection metadata from {metadata_path}.")
            continue
        if isinstance(metadata, dict):
            scene = metadata.get("scene_objects") if "scene_objects" in metadata else metadata.get("scene")
            if isinstance(scene, list):
                return metadata

    LOG.warning(f"No injection metadata available for {run_dir}; hallucination metrics will be N/A.")
    return None


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


def _experiment_rows(
    job_dir: Path,
    ground_truth_cache_dir: Path,
    *,
    hallucinations: bool = False,
    model: str = "N/A",
    method: str = "N/A",
    repetition: str = "N/A",
) -> Iterable[Dict[str, Any]]:
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

            row = {
                "model": model,
                "method": method,
                "repetition": repetition,
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

            # Evaluate the instruction against the scene when it was injected.
            if hallucinations:
                metadata = _load_injection_metadata(
                    run_dir, job_dir, ground_truth_cache_dir,
                    scenario.scenario_id, experiment.ground_truth_index, start,
                )
                if metadata is None:
                    row.update({field_name: "N/A" for field_name in HALLUCINATION_FIELDNAMES})
                else:
                    row.update(check_hallucinations_in_instruction(row["instruction"], metadata))

            yield row


def save_experiment_data(
    job_dir: Path,
    output_path: Path,
    ground_truth_cache_dir: Path,
    *,
    hallucinations: bool = False,
    model: str = "N/A",
    method: str = "N/A",
    repetition: str = "N/A",
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(FIELDNAMES)
        if hallucinations:
            fieldnames.extend(HALLUCINATION_FIELDNAMES)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in _experiment_rows(
            job_dir, ground_truth_cache_dir,
            hallucinations=hallucinations, model=model, method=method, repetition=repetition,
        ):
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
    parser.add_argument(
        "--hallucinations", "-H", action="store_true",
        help="Check instruction hallucinations using injection metadata and append four metric columns",
    )
    parser.add_argument("--model", "-M", default="N/A", help="Model label written to each row (default: N/A)")
    parser.add_argument("--method", "-m", default="N/A", help="Method label written to each row (default: N/A)")
    parser.add_argument("--repetition", "-r", default="N/A", help="Repetition label written to each row (default: N/A)")
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
        row_count = save_experiment_data(
            job_dir, output_path, ground_truth_cache_dir,
            hallucinations=args.hallucinations,
            model=args.model, method=args.method, repetition=args.repetition,
        )
        print(f"Wrote {row_count} experiment runs to {output_path}")


if __name__ == "__main__":
    main()
