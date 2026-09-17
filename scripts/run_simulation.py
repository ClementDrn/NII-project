from dataclasses import dataclass
from typing import Any, List
import argparse
import json
from pathlib import Path
from typing import Optional

from untitled.core.base import LOG
from untitled.generator.instruction import InjectionWindow, Instruction
from untitled.simulator.simulator import Simulator
from untitled.simulator.ads import ADS


@dataclass
class SimulationJob:
    scenario_id: int
    instruction: Optional[Instruction]
    repeat_count: int = 1


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simulation with hardcoded jobs or a JSON file of extracted instructions"
    )
    parser.add_argument(
        "--instruction_filepath",
        type=Path,
        default=None,
        help="Path to instruction JSON (flattened extracted records or custom_prompts.json format)",
    )
    parser.add_argument(
        "--repeat_count",
        type=_positive_int,
        default=1,
        help="Number of repeated runs for a single legacy instruction input",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Existing simulation root to write into instead of a new timestamped directory",
    )
    parser.add_argument(
        "--run_index",
        type=_non_negative_int,
        default=None,
        help="Explicit run index for a single batch instruction job",
    )
    args = parser.parse_args()

    # Validate instruction file path if provided.
    if args.instruction_filepath is not None:
        if not args.instruction_filepath.exists():
            parser.error(f"--instruction_filepath does not exist: {args.instruction_filepath}")
        if not args.instruction_filepath.is_file():
            parser.error(f"--instruction_filepath must be a file: {args.instruction_filepath}")

    if args.run_index is not None and args.instruction_filepath is None:
        parser.error("--run_index requires --instruction_filepath")
    if args.run_index is not None and args.repeat_count != 1:
        parser.error("--run_index cannot be combined with --repeat_count")
    if args.output_dir is not None and not args.output_dir.is_dir():
        parser.error(f"--output_dir must be an existing directory: {args.output_dir}")

    return args


def _build_instruction_from_item(item: dict[str, Any]) -> Instruction:
    text = str(item.get("instruction", item.get("text", ""))).strip()
    if not text:
        raise ValueError("Instruction JSON must contain non-empty 'instruction' or 'text'")

    injection_window_data = item.get("injection_window")
    if isinstance(injection_window_data, dict):
        start_timestamp = float(injection_window_data.get("start_timestamp", 0.0))
        end_timestamp = float(injection_window_data.get("end_timestamp", start_timestamp))
    else:
        start_timestamp = float(item.get("injection_time", 0.0))
        injection_duration = max(0.0, float(item.get("injection_duration", 0.0)))
        end_timestamp = start_timestamp + injection_duration

    return Instruction(
        rank=int(item.get("rank", 1)),
        injection_window=InjectionWindow(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
        text=text,
        target_module_logic=str(item.get("target_module_logic", "")),
        reasoning=str(item.get("reasoning", "")),
    )


def _load_instruction_from_file(instruction_filepath: Path) -> Instruction:
    with open(instruction_filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"Instruction file is empty: {instruction_filepath}")
        item = payload[0]
        if len(payload) > 1:
            LOG.warning(
                f"Instruction file contains {len(payload)} entries; only the first one will be used."
            )
    elif isinstance(payload, dict):
        item = payload
    else:
        raise ValueError("Instruction file must contain a JSON object or a non-empty JSON array")

    if not isinstance(item, dict):
        raise ValueError("Instruction item must be a JSON object")

    return _build_instruction_from_item(item)


def _load_simulation_jobs_from_file(instruction_filepath: Path, repeat_count: int) -> List[SimulationJob]:
    with open(instruction_filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"Instruction file is empty: {instruction_filepath}")

        jobs: List[SimulationJob] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError("Each instruction item must be a JSON object")

            if "scenario_id" not in item:
                raise ValueError("Batch instruction JSON must contain 'scenario_id' for each item")

            instruction = _build_instruction_from_item(item)
            jobs.append(
                SimulationJob(
                    scenario_id=int(item["scenario_id"]),
                    instruction=Instruction(
                        rank=instruction.rank if instruction.rank > 0 else index + 1,
                        injection_window=instruction.injection_window,
                        text=instruction.text,
                        target_module_logic=instruction.target_module_logic,
                        reasoning=instruction.reasoning,
                    ),
                    repeat_count=repeat_count,
                )
            )

        return jobs

    if isinstance(payload, dict):
        if "scenario_id" in payload:
            scenario_id = int(payload["scenario_id"])
        else:
            raise ValueError("Single instruction JSON must contain 'scenario_id' or be used with hardcoded jobs")

        instruction = _build_instruction_from_item(payload)
        return [
            SimulationJob(
                scenario_id=scenario_id,
                instruction=instruction,
                repeat_count=repeat_count,
            )
        ]

    raise ValueError("Instruction file must contain a JSON object or a non-empty JSON array")

# --- Define simulation jobs with instructions --------------------------------

default_jobs: List[SimulationJob] = [
    SimulationJob(
        scenario_id=18,
        instruction=None,
        repeat_count=1
    )
]

# --- Main --------------------------------------------------------------------
def main():
    args = _parse_args()

    # Build runtime jobs from CLI input when provided.
    runtime_jobs: List[SimulationJob] = []
    if args.instruction_filepath is not None:
        runtime_jobs = _load_simulation_jobs_from_file(args.instruction_filepath, args.repeat_count)
    else:
        # No CLI override, use hardcoded jobs.
        runtime_jobs = default_jobs

    LOG.info("Initializing simulator...")
    simulator = Simulator(ADS("simlingo"), True, base_dir=args.output_dir)
    
    for job_index, job in enumerate(runtime_jobs):
        LOG.info(f"Running scenario {job.scenario_id} with instruction: '{(job.instruction.text if job.instruction else 'None')}'...")
        # Set job and scenario context
        simulator.set_job_context(job_index, job.scenario_id)       # TODO: Remove scenario_id from this method
        simulator.set_scenario(job.scenario_id)     # TODO: Group this method with run_scenario and remove scenario_id from set_job_context

        # Run simulation
        for i in range(job.repeat_count):
            LOG.info(f"Starting run {i+1}/{job.repeat_count} for scenario {job.scenario_id}...")
            run_index = args.run_index if args.run_index is not None else None
            simulator.run_scenario(job.instruction, run_id=run_index)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
