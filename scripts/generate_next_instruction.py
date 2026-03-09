"""
Generate the next instruction to inject depending on the given context.

The context includes:
- Directory of ground truth scenario cache
- Directory with this scenario results (optional, relevant for incremental approach)
- Run index
- Filter ID
- Generator ID
- Model instance ID
- Experiment nature (adversarial vs corrective)
- Number of instructions to generate
- If we are using the incremental approach

Usage:
python scripts/generate_next_instruction.py \
    --ground_truth_cache_dir <path_to_ground_truth_cache_run_dir> \
    --scenario_results_dir <path_to_results_scenario_dir> \
    --run_index <run_index> \
    --filter_id <filter_id> \
    --generator_id <generator_id> \
    --model_instance_id <model_instance_id> \
    --experiment_nature <adversarial_or_corrective> \
    --num_instructions_to_generate <num_instructions_to_generate> \
    --incremental

Outputs the generated instruction to stdout as a JSON string which follows the custom_prompts.json format.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

from untitled.core.base import CONFIG, LOG, SIMULATION_FPS
from untitled.core.settings import GeneratorConfig, get_filter_config_by_id, get_generator_config_by_id, get_model_instance_config_by_id
from untitled.filter.embedder import DINOEmbedder
from untitled.filter.filter import cluster_frames
from untitled.generator.experiment_nature import ExperimentNature
from untitled.generator.instruction import InjectionWindow, Instruction, InstructionResult
from untitled.generator.multi_step_vlm_instruction_generator import MultiStepVLMInstructionGenerator
from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
from untitled.simulator.simulation_data import SimulationData, SimulationFrameData, load_metadata
from untitled.simulator.simulation_result import SimulationResult


def _positive_int(value: str) -> int:
    # Argparse validator for strictly positive integer values.
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    # Argparse validator for non-negative integer values.
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _parse_args() -> argparse.Namespace:
    # CLI arguments for selecting cache inputs and generation mode.
    parser = argparse.ArgumentParser(description="Generate the next instruction to inject")
    parser.add_argument(
        "--ground_truth_cache_dir",
        required=True,
        type=Path,
        help="Ground-truth run directory (e.g. cache/sce_077/run_000)",
    )
    parser.add_argument(
        "--scenario_results_dir",
        type=Path,
        default=None,
        help="Scenario results directory containing run_XXX dirs (required for incremental mode)",
    )
    parser.add_argument(
        "--run_index",
        required=True,
        type=_non_negative_int,
        help="Run index to use from ground truth cache",
    )
    parser.add_argument("--filter_id", required=True, type=str, help="Filter ID from config.yaml")
    parser.add_argument("--generator_id", required=True, type=str, help="Generator ID from config.yaml")
    parser.add_argument("--model_instance_id", required=True, type=str, help="Model instance ID from config.yaml")
    parser.add_argument(
        "--num_instructions_to_generate",
        required=True,
        type=_positive_int,
        help="Number of instructions to include in output payload",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Enable incremental generation with previous instruction outcomes",
    )
    parser.add_argument(
        "--experiment_nature",
        default="adversarial",
        choices=["adversarial", "corrective"],
        help="Experiment nature to use for task selection",
    )
    args = parser.parse_args()

    # Dependent-argument validation: incremental mode requires scenario results input.
    if args.incremental and args.scenario_results_dir is None:
        parser.error("--scenario_results_dir is required when --incremental is enabled")

    # Path validation: ground-truth cache directory must exist and be a directory.
    if not args.ground_truth_cache_dir.exists():
        parser.error(f"--ground_truth_cache_dir does not exist: {args.ground_truth_cache_dir}")
    if not args.ground_truth_cache_dir.is_dir():
        parser.error(f"--ground_truth_cache_dir must be a directory: {args.ground_truth_cache_dir}")

    # Path validation for incremental mode.
    if args.incremental and args.scenario_results_dir is not None:
        if not args.scenario_results_dir.exists():
            parser.error(f"--scenario_results_dir does not exist: {args.scenario_results_dir}")
        if not args.scenario_results_dir.is_dir():
            parser.error(f"--scenario_results_dir must be a directory: {args.scenario_results_dir}")

    # ID validation.
    try:
        get_filter_config_by_id(CONFIG, args.filter_id)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        get_generator_config_by_id(CONFIG, args.generator_id)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        get_model_instance_config_by_id(CONFIG, args.model_instance_id)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def _parse_run_id(run_dir: Path) -> Optional[int]:
    # Expected naming convention: run_XXX
    try:
        return int(run_dir.name.split("_")[1])
    except (IndexError, ValueError):
        return None


def _fetch_scenario_id_from_path(run_dir: Path) -> int:
    # Expected path structure: .../sce_XXX/run_XXX
    try:
        return int(run_dir.parent.name.split("_")[1])
    except (IndexError, ValueError):
        LOG.warning(f"Failed to parse scenario ID from path: {run_dir}")
        return -1


def _sort_key_for_run_dir(run_dir: Path) -> int:
    run_id = _parse_run_id(run_dir)
    return run_id if run_id is not None else -1


def _load_simulation_data(run_dir: Path) -> SimulationData:
    # Read synchronized image + metadata logs from a single run.
    image_dir = run_dir / "images"
    metadata_dir = run_dir / "metadata"

    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    image_by_stem = {
        image_path.stem: image_path
        for image_path in image_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    metadata_by_stem = {
        metadata_path.stem: metadata_path
        for metadata_path in metadata_dir.glob("*.json")
        if metadata_path.is_file()
    }

    # Match frames by filename stem (e.g. 000120.png <-> 000120.json).
    shared_stems = sorted(
        set(image_by_stem.keys()) & set(metadata_by_stem.keys()),
        key=lambda stem: int(stem),
    )
    if not shared_stems:
        raise RuntimeError(f"No matching image/metadata pairs found in run: {run_dir}")

    frames: list[SimulationFrameData] = []
    for stem in shared_stems:
        image_path = image_by_stem[stem]
        # Reuse project metadata loader to keep field-cleaning behavior identical.
        metadata = load_metadata(metadata_by_stem[stem])
        timestamp = metadata.step * (1.0 / SIMULATION_FPS)
        frames.append(
            SimulationFrameData(
                image_name=image_path.name,
                metadata=metadata,
                timestamp=timestamp,
            )
        )

    return SimulationData(image_dir=image_dir, frame_count=len(frames), frames=frames)


def _apply_filter_to_simulation_data(
    simulation_data: SimulationData,
    max_frames_per_minute: int,
    clustering_threshold: float,
    max_cluster_count: int,
) -> SimulationData:
    # 1) Optional temporal undersampling.
    undersampled = simulation_data
    if max_frames_per_minute > 0 and simulation_data.frame_count > 1:
        first_step = simulation_data.frames[0].metadata.step
        second_step = simulation_data.frames[1].metadata.step
        step_delta = max(1, second_step - first_step)
        actual_frame_rate = SIMULATION_FPS / step_delta
        log_frame_total_per_minute = actual_frame_rate * 60

        if max_frames_per_minute < log_frame_total_per_minute:
            interval = max(1, int(log_frame_total_per_minute / max_frames_per_minute))
            sampled_frames = [
                frame for index, frame in enumerate(simulation_data.frames)
                if index % interval == 0
            ]
            undersampled = SimulationData(
                image_dir=simulation_data.image_dir,
                frame_count=len(sampled_frames),
                frames=sampled_frames,
            )

    # 2) Similarity clustering (same as pipeline behavior).
    embedder = DINOEmbedder()
    clusters = cluster_frames(undersampled, embedder, clustering_threshold, max_cluster_count)
    filtered_frames = [cluster.get_data().frames[0] for cluster in clusters] if clusters else []

    return SimulationData(
        image_dir=undersampled.image_dir,
        frame_count=len(filtered_frames),
        frames=filtered_frames,
    )


def _select_generator_for_model(
    generator_id: str,
    model_instance_id: str,
    is_incremental: bool,
    instruction_count: int,
) -> GeneratorConfig:
    # Resolve selected generator + model instance from config.
    configured_generator = get_generator_config_by_id(CONFIG, generator_id)
    instance = get_model_instance_config_by_id(CONFIG, model_instance_id)

    # Check incremental mode consistency.
    if bool(configured_generator.is_incremental) != is_incremental:
        raise ValueError(
            f"Generator '{generator_id}' has is_incremental={configured_generator.is_incremental}, "
            f"but CLI incremental={is_incremental}."
        )

    # Validate model instance compatibility with selected generator.
    if not configured_generator.steps:
        raise ValueError(f"Generator '{generator_id}' has no configured steps")

    step_ids = [step.id for step in configured_generator.steps]
    if any(step_id not in instance.step_ids for step_id in step_ids):
        raise ValueError(
            f"Model instance '{model_instance_id}' is not linked to all required steps of generator '{generator_id}'"
        )

    if any(step.model_name != instance.model_name for step in configured_generator.steps):
        raise ValueError(
            f"Model instance '{model_instance_id}' model_name='{instance.model_name}' is incompatible "
            f"with generator '{generator_id}' step model(s)"
        )

    if configured_generator.design == "single_step_vlm" and instance.type != "vlm":
        raise ValueError(f"Generator '{generator_id}' requires a VLM instance for single_step_vlm design")

    # Use selected generator and override instruction count for this call.
    selected = deepcopy(configured_generator)
    selected.instruction_count_per_generation = max(1, instruction_count)
    # Configure global active context expected by generator internals.
    CONFIG.generator = selected
    CONFIG.active_model_instances = [instance]
    return selected


def _evaluate_instruction_success(
    experiment_result: Optional[SimulationResult],
    ground_truth_result: Optional[SimulationResult],
    nature: ExperimentNature,
) -> bool:
    # Mirrors pipeline success logic used for incremental learning context.
    if experiment_result is None or ground_truth_result is None:
        return False

    if nature == ExperimentNature.ADVERSARIAL:
        return ground_truth_result.scores.score_penalty > experiment_result.scores.score_penalty
    return ground_truth_result.scores.score_penalty < experiment_result.scores.score_penalty


def _build_previous_instruction_results(
    scenario_results_dir: Path,
    current_run_index: int,
    nature: ExperimentNature,
    scenario_id: int,
    ground_truth_result: Optional[SimulationResult],
) -> list[InstructionResult]:
    # Recover previous instructions from scenario runs for incremental prompts.
    previous_results: list[InstructionResult] = []

    run_dirs = sorted(
        [path for path in scenario_results_dir.glob("run_*") if path.is_dir()],
        key=_sort_key_for_run_dir,
    )

    for run_dir in run_dirs:
        run_id = _parse_run_id(run_dir)
        # Only include strictly previous runs.
        if run_id is None or run_id >= current_run_index:
            continue

        prompts_file = run_dir / "custom_prompts.json"
        if not prompts_file.exists():
            continue

        try:
            prompt_items = json.loads(prompts_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOG.warning(f"Skipping invalid JSON file: {prompts_file}")
            continue

        if not prompt_items or not isinstance(prompt_items, list):
            continue

        # `custom_prompts.json` is an array; current simulator uses first element.
        prompt_item = prompt_items[0] if isinstance(prompt_items[0], dict) else {}
        text = str(prompt_item.get("text", "")).strip()
        if not text:
            continue

        injection_time = float(prompt_item.get("injection_time", 0.0))
        injection_duration = max(0.0, float(prompt_item.get("injection_duration", 0.0)))

        instruction = Instruction(
            rank=1,
            injection_window=InjectionWindow(
                start_timestamp=injection_time,
                end_timestamp=injection_time + injection_duration,
            ),
            text=text,
            target_module_logic=str(prompt_item.get("target_module_logic", "")),
            reasoning=str(prompt_item.get("reasoning", "")),
        )

        # Compare experiment against ground truth to annotate success/failure.
        experiment_result = SimulationResult.load_from_json_file(run_dir / "res.json", scenario_id)
        was_successful = _evaluate_instruction_success(experiment_result, ground_truth_result, nature)
        previous_results.append(InstructionResult(instruction=instruction, was_successful=was_successful))

    return previous_results


def _instruction_to_custom_prompt(instruction: Instruction) -> dict:
    # Keep exact payload keys expected by simulator `custom_prompts.json` loader.
    return {
        "text": instruction.text,
        "flag": 2,
        "injection_time": instruction.injection_window.start_timestamp,
        "injection_duration": instruction.injection_window.end_timestamp - instruction.injection_window.start_timestamp,
        "target_module_logic": instruction.target_module_logic,
        "reasoning": instruction.reasoning,
    }


async def _generate(args: argparse.Namespace) -> list[dict]:
    # Get experiment nature from args
    experiment_nature = ExperimentNature.from_string(args.experiment_nature)

    # Fetch scenario ID from directory path, for metadata
    gt_run_dir = args.ground_truth_cache_dir
    scenario_id = _fetch_scenario_id_from_path(gt_run_dir)

    # Load the source run from cache and its baseline result.
    gt_simulation_data = _load_simulation_data(gt_run_dir)
    gt_result = SimulationResult.load_from_json_file(gt_run_dir / "res.json", scenario_id)

    # Apply configured filter (undersampling + clustering) to GT samples.
    filter_config = get_filter_config_by_id(CONFIG, args.filter_id)
    CONFIG.filter = filter_config
    gt_simulation_data = _apply_filter_to_simulation_data(
        simulation_data=gt_simulation_data,
        max_frames_per_minute=filter_config.max_frames_per_minute,
        clustering_threshold=filter_config.clustering_threshold,
        max_cluster_count=filter_config.max_cluster_count,
    )

    # Select generator design based on args
    generator_config: GeneratorConfig = _select_generator_for_model(
        generator_id=args.generator_id,
        model_instance_id=args.model_instance_id,
        is_incremental=args.incremental,
        instruction_count=args.num_instructions_to_generate,
    )

    # If incremental, build prior instruction outcomes
    previously_tried_instructions = None
    if args.incremental:
        previously_tried_instructions = _build_previous_instruction_results(
            scenario_results_dir=args.scenario_results_dir,
            current_run_index=args.run_index,
            nature=experiment_nature,
            scenario_id=scenario_id,
            ground_truth_result=gt_result,
        )

    # Instantiate generator based on selected design
    if generator_config.design == "single_step_vlm":
        generator = SingleStepVLMInstructionGenerator(experiment_nature, args.incremental)
    elif generator_config.design == "multi_step_vlm":
        generator = MultiStepVLMInstructionGenerator(experiment_nature, args.incremental)
    else:
        raise ValueError(f"Unsupported generator design: {generator_config.design}")

    # Ensure model resources are opened/closed cleanly around the request.
    await generator.init()
    try:
        instruction_set = await generator.generate_instructions_from_images(
            data=gt_simulation_data,
            failure_text=gt_result.failure_text if gt_result else None,
            was_route_completed=gt_result.was_route_completed if gt_result else None,
            previously_tried_instructions=previously_tried_instructions,
        )
    finally:
        await generator.free()

    if instruction_set is None or not instruction_set.instructions:
        raise RuntimeError("Instruction generation returned no valid instruction")

    # Return up to requested count (or fewer if model returned less).
    count = min(args.num_instructions_to_generate, len(instruction_set.instructions))
    return [_instruction_to_custom_prompt(instr) for instr in instruction_set.instructions[:count]]


def main() -> int:
    # Entrypoint: print JSON payload to stdout for caller consumption.
    args = _parse_args()
    try:
        prompts = asyncio.run(_generate(args))
    except Exception as exc:
        LOG.error(f"Failed to generate instruction: {exc}")
        return 1

    print(json.dumps(prompts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
