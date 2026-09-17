"""Regenerate and rerun selected iterations of an incremental experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


EXPERIMENT_DIR = Path("/data1/cdarne/digit/output/simlingo/bench2drive/1/08-16-opus-4.6-incremental-n1")
FILTER_ID = "undersampled_filter"
GENERATOR_ID = "claude-opus-4.6_single_step_incremental"
MODEL_INSTANCE_ID = "claude-opus-4.6_0"

# Entries are ordered so that each later generation sees the repaired prior run.
RUNS_TO_REPAIR = {
    18: (2,),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=EXPERIMENT_DIR,
        help="Experiment directory containing job_000/sce_XXX directories",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        action="append",
        dest="scenarios",
        help="Only repair this scenario; may be specified more than once",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        action="append",
        dest="run_indices",
        help="Only repair this run index; may be specified more than once",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without calling the model or simulator",
    )
    return parser.parse_args()


def _run(command: list[str], *, dry_run: bool) -> str:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return ""
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout


def main() -> int:
    args = _parse_args()
    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise SystemExit(f"Experiment directory does not exist: {experiment_dir}")
    results_dir = experiment_dir / "job_000"
    if not results_dir.is_dir():
        raise SystemExit(f"Experiment job directory does not exist: {results_dir}")

    scenarios = args.scenarios or list(RUNS_TO_REPAIR)
    unknown = sorted(set(scenarios) - set(RUNS_TO_REPAIR))
    if unknown:
        raise SystemExit(f"Scenario(s) not in repair list: {', '.join(map(str, unknown))}")

    repo_root = Path(__file__).resolve().parents[1]
    generator_script = repo_root / "scripts" / "generate_next_instruction.py"
    simulator_script = repo_root / "scripts" / "run_simulation.py"
    for scenario_id in scenarios:
        scenario_dir = results_dir / f"sce_{scenario_id:03d}"
        ground_truth_dir = repo_root / "cache" / f"sce_{scenario_id:03d}" / "run_000"
        if not scenario_dir.is_dir():
            raise SystemExit(f"Scenario directory does not exist: {scenario_dir}")
        if not ground_truth_dir.is_dir():
            raise SystemExit(f"Ground-truth directory does not exist: {ground_truth_dir}")

        run_indices = RUNS_TO_REPAIR[scenario_id]
        if args.run_indices is not None:
            run_indices = tuple(run_index for run_index in run_indices if run_index in args.run_indices)
        for run_index in run_indices:
            instruction_file = scenario_dir / f"rerun_instruction_{run_index:03d}.json"
            generate_command = [
                sys.executable,
                str(generator_script),
                "--ground_truth_cache_dir", str(ground_truth_dir),
                "--scenario_results_dir", str(scenario_dir),
                "--run_index", str(run_index),
                "--filter_id", FILTER_ID,
                "--generator_id", GENERATOR_ID,
                "--model_instance_id", MODEL_INSTANCE_ID,
                "--experiment_nature", "adversarial",
                "--num_instructions_to_generate", "1",
                "--incremental",
            ]
            generated = _run(generate_command, dry_run=args.dry_run)
            if args.dry_run:
                print(
                    "$ "
                    + " ".join(
                        [
                            sys.executable,
                            str(simulator_script),
                            "--instruction_filepath",
                            str(instruction_file),
                            "--output_dir",
                            str(experiment_dir),
                            "--run_index",
                            str(run_index),
                        ]
                    ),
                    flush=True,
                )
                continue

            try:
                prompts = json.loads(generated)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Generator returned invalid JSON for sce_{scenario_id} run {run_index}: {exc}") from exc
            if not isinstance(prompts, list) or len(prompts) != 1:
                raise SystemExit(f"Expected one generated prompt for sce_{scenario_id} run {run_index}")
            instruction_file.write_text(
                json.dumps({"scenario_id": scenario_id, **prompts[0]}),
                encoding="utf-8",
            )
            target_run_dir = scenario_dir / f"run_{run_index:03d}"
            if target_run_dir.exists():
                shutil.rmtree(target_run_dir)

            simulate_command = [
                sys.executable,
                str(simulator_script),
                "--instruction_filepath", str(instruction_file),
                "--output_dir", str(experiment_dir),
                "--run_index", str(run_index),
            ]
            _run(simulate_command, dry_run=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())