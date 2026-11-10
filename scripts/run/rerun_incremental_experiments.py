"""Regenerate selected incremental runs from an explicit repair plan.

Existing run directories are preserved as timestamped backups before rerunning.
Use --dry-run to inspect the commands without calling models or changing data.
"""

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_plan(path):
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            "Repair plan must be a nonempty object mapping scenario IDs to run indices"
        )
    plan = {}
    for scenario, runs in payload.items():
        scenario_id = int(scenario)
        if not 0 <= scenario_id <= 219 or not isinstance(runs, list) or not runs:
            raise ValueError(f"Invalid scenario or run list: {scenario}")
        if any(type(run) is not int or run < 0 for run in runs) or len(
            set(runs)
        ) != len(runs):
            raise ValueError(
                f"Run indices must be unique nonnegative integers: {scenario}"
            )
        plan[scenario_id] = sorted(runs)
    return plan


def run(command, dry_run):
    print("$ " + shlex.join(command), flush=True)
    if dry_run:
        return ""
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument(
        "--plan",
        required=True,
        type=Path,
        help="JSON object mapping scenario IDs to run indices",
    )
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    parser.add_argument("--filter-id", required=True)
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--model-instance-id", required=True)
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.job_index < 0:
        parser.error("--job-index must be nonnegative")
    plan = load_plan(args.plan)
    results_dir = args.experiment_dir.resolve() / f"job_{args.job_index:03d}"
    # Validate the entire plan before generating requests or changing results.
    for scenario in plan:
        for directory in (
            results_dir / f"sce_{scenario:03d}",
            args.cache_dir / f"sce_{scenario:03d}/run_000",
        ):
            if not directory.is_dir():
                parser.error(f"Directory does not exist: {directory}")
    for scenario, runs in plan.items():
        scenario_dir = results_dir / f"sce_{scenario:03d}"
        for index in runs:
            instruction_file = scenario_dir / f"rerun_instruction_{index:03d}.json"
            generate = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/vlm/generate_next_instruction.py"),
                "--ground_truth_cache_dir",
                str(args.cache_dir / f"sce_{scenario:03d}/run_000"),
                "--scenario_results_dir",
                str(scenario_dir),
                "--run_index",
                str(index),
                "--filter_id",
                args.filter_id,
                "--generator_id",
                args.generator_id,
                "--model_instance_id",
                args.model_instance_id,
                "--experiment_nature",
                "adversarial",
                "--num_instructions_to_generate",
                "1",
                "--incremental",
            ]
            generated = run(generate, args.dry_run)
            if not args.dry_run:
                prompts = json.loads(generated)
                if (
                    not isinstance(prompts, list)
                    or len(prompts) != 1
                    or not isinstance(prompts[0], dict)
                ):
                    raise ValueError(
                        f"Expected one generated prompt for scenario {scenario}, run {index}"
                    )
                instruction_file.write_text(
                    json.dumps({"scenario_id": scenario, **prompts[0]})
                )
                target = scenario_dir / f"run_{index:03d}"
                if target.exists():
                    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    target.rename(scenario_dir / f"{target.name}.backup-{suffix}")
            simulate = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/run/run_simulation.py"),
                "--instruction_filepath",
                str(instruction_file),
                "--output_dir",
                str(args.experiment_dir),
                "--run_index",
                str(index),
                "--job-index",
                str(args.job_index),
            ]
            run(simulate, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
