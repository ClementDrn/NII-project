"""
Run scenarios in the jobs file with API batching.

Warning: This script is intended for one-shot instruction generation experiments only.
Also, it was only made to work with Anthropic Claude.
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx

from untitled.core.base import CONFIG, LOG, PROJECT_ROOT, SIMULATION_FPS
from untitled.core.job_handler import Job, JobHandler
from untitled.core.settings import (
    GeneratorStepConfig,
    configure_settings_for_job,
    get_generator_config_by_id,
    get_model_instance_config_by_id,
)
from untitled.generator.batch import (
    Batch,
    BatchRequest,
    BatchRequestResult,
    BatchRequestStatus,
    BatchStatus,
)
from untitled.generator.instruction import InstructionSet, parse_instructions_from_json
from untitled.generator.instruction_generator import get_api_parameters_from_config
from untitled.generator.model import ModelClient, create_model_client, parse_model_response
from untitled.generator.task import Task
from untitled.generator.task_manager import TaskManager
from untitled.filter.embedder import DINOEmbedder, Embedder
from untitled.filter.filter import cluster_frames
from untitled.simulator.simulation_data import SimulationData, SimulationFrameData
from untitled.simulator.simulator import Simulator


# --- Experiment configuration -----------------------------------------------

REPETITION_COUNT = 3
PROPOSALS_PER_REQUEST = 3
REQUESTS_PER_BATCH = 6      # max batch size in 256MB and a single request can hold up to 40MB of data.
GROUND_TRUTH_RUN_ID = 0
DOWNLOAD_MAX_ATTEMPTS = 5
DOWNLOAD_INITIAL_DELAY_SECONDS = 5.0
DOWNLOAD_MAX_DELAY_SECONDS = 60.0
DEFAULT_JOBS_FILE = PROJECT_ROOT / "jobs.yaml"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"


# --- Batch context -----------------------------------------------------------

@dataclass(frozen=True)
class RequestContext:
    job_id: int
    scenario_id: int


@dataclass
class RunningBatch:
    batch: Batch
    contexts: dict[str, RequestContext]


def _parse_args() -> argparse.Namespace:
    # Allow alternate job and cache locations for local experiments.
    parser = argparse.ArgumentParser(
        description="Run all one-shot Claude instruction experiments"
    )
    parser.add_argument(
        "--jobs-file",
        type=Path,
        default=DEFAULT_JOBS_FILE,
        help="Path to the jobs YAML file",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing cached ground-truth runs",
    )
    return parser.parse_args()


def _load_simulation_data(
    simulator: Simulator,
    scenario_id: int,
    cache_dir: Path,
    frames_per_minute: int = 0,
) -> SimulationData:
    """Load synchronized frames and metadata from a cached ground-truth run."""
    # Ground-truth cache directories use the same scenario/run layout as the simulator.
    run_dir = cache_dir / f"sce_{scenario_id:03d}" / f"run_{GROUND_TRUTH_RUN_ID:03d}"
    image_paths = simulator.get_image_logs(
        0,
        scenario_id,
        GROUND_TRUTH_RUN_ID,
        run_base_dir=run_dir,
        frames_per_minute=frames_per_minute,
    )
    metadata = simulator.get_metadata_logs(
        0,
        scenario_id,
        GROUND_TRUTH_RUN_ID,
        run_base_dir=run_dir,
        frames_per_minute=frames_per_minute,
    )

    # Validate that the cached data is complete and consistent.
    if not image_paths:
        raise RuntimeError(f"No cached images found in {run_dir}")
    if not metadata:
        raise RuntimeError(f"No cached metadata found in {run_dir}")
    if len(image_paths) != len(metadata):
        raise RuntimeError(
            f"Cached image/metadata count mismatch in {run_dir}: "
            f"{len(image_paths)} images, {len(metadata)} metadata files"
        )

    # Build a SimulationData object with frame-level timestamps and metadata.
    frames = [
        SimulationFrameData(
            image_name=image_path.name,
            metadata=metadata[index],
            timestamp=metadata[index].step / SIMULATION_FPS,
        )
        for index, image_path in enumerate(image_paths)
    ]
    data = SimulationData(
        frame_count=len(frames),
        image_dir=image_paths[0].parent,
        frames=frames,
    )

    return data


def _filter_simulation_data(
    simulation_data: SimulationData,
    embedder: Embedder | None,
) -> SimulationData:
    """Apply the configured pipeline frame filter to cached simulation data."""
    # Check filter config
    if CONFIG.filter is None:
        return simulation_data
    if embedder is None:
        raise RuntimeError("Filter configuration is active but no embedder was created")

    # Cluster similar frames
    clusters = cluster_frames(
        simulation_data,
        embedder,
        CONFIG.filter.clustering_threshold,
        CONFIG.filter.max_cluster_count,
    )
    filtered_data = SimulationData(
        image_dir=simulation_data.image_dir,
        frame_count=len(clusters),
        frames=[cluster.get_data().frames[0] for cluster in clusters],
    )
    LOG.info(
        "Filtered scenario frames from %d to %d using threshold %.3f",
        simulation_data.frame_count,
        filtered_data.frame_count,
        CONFIG.filter.clustering_threshold,
    )

    return filtered_data


def _build_batch_requests(
    simulator: Simulator,
    task: Task,
    scenarios: list[int],
    cache_dir: Path,
) -> tuple[list[BatchRequest], dict[str, RequestContext]]:
    """Build one cached instruction-generation request per scenario/repetition."""
    requests: list[BatchRequest] = []
    contexts: dict[str, RequestContext] = {}
    simulation_data_cache: dict[int, SimulationData] = {}
    embedder = DINOEmbedder() if CONFIG.filter is not None else None

    # Requests are build for each scenario (and then for each repetition)
    for scenario_id in scenarios:
        try:
            # Load simulation data with undersampling
            simulation_data = _load_simulation_data(
                simulator,
                scenario_id,
                cache_dir,
                frames_per_minute=(CONFIG.filter.max_frames_per_minute if CONFIG.filter else 0),
            )
            simulation_data_cache[scenario_id] = _filter_simulation_data(
                simulation_data,
                embedder,
            )
        except (OSError, RuntimeError, ValueError) as error:
            LOG.error("Could not prepare scenario %03d: %s", scenario_id, error)
            continue

        # Repeat the same request multiple times
        for repetition in range(REPETITION_COUNT):
            request_id = f"job_{repetition:03d}-sce_{scenario_id:03d}"
            data = simulation_data_cache[scenario_id]

            prompt_request = task.prepare_prompt(
                future=asyncio.get_running_loop().create_future(),
                images=[data.image_dir / frame.image_name for frame in data.frames],
                frames=data.frames,
            )
            # Task.prepare_prompt marks the static scene inputs for Anthropic prompt caching.
            requests.append(
                BatchRequest(
                    request_id=request_id,
                    system_text=prompt_request.request.system_text,
                    user_content=prompt_request.request.user_content,
                )
            )
            contexts[request_id] = RequestContext(repetition, scenario_id)

    return requests, contexts


def _chunk_requests(
    requests: list[BatchRequest],
    contexts: dict[str, RequestContext],
) -> list[tuple[list[BatchRequest], dict[str, RequestContext]]]:
    """Split requests into native batch-sized chunks while preserving context."""
    chunks = []
    # Anthropic batches are submitted in groups of scenarios x repetitions.
    for start in range(0, len(requests), REQUESTS_PER_BATCH):
        request_chunk = requests[start : start + REQUESTS_PER_BATCH]
        context_chunk = {request.request_id: contexts[request.request_id] for request in request_chunk}
        chunks.append((request_chunk, context_chunk))
    return chunks


def _get_planning_step(job: Job) -> GeneratorStepConfig:
    # The one-shot script uses the planner step from the configured generator.
    generator_config = get_generator_config_by_id(CONFIG, job.generator)
    for step in generator_config.steps:
        if step.id == "planning":
            return step
    raise ValueError(f"Generator '{job.generator}' has no planning step")


def _parse_instruction_set(result: BatchRequestResult) -> InstructionSet | None:
    # A failed request has no model response to parse or simulate.
    if result.status != BatchRequestStatus.SUCCEEDED:
        LOG.error("Request %s failed with status %s", result.request_id, result.status)
        return None

    raw_response = parse_model_response(result.text)
    instruction_set = parse_instructions_from_json(raw_response)
    # Do not create simulation runs from incomplete planner output.
    if instruction_set is None or len(instruction_set.instructions) < PROPOSALS_PER_REQUEST:
        count = len(instruction_set.instructions) if instruction_set else 0
        LOG.error(
            "Request %s returned %d valid instructions; expected %d",
            result.request_id,
            count,
            PROPOSALS_PER_REQUEST,
        )
        return None
    return instruction_set


async def _submit_batches(
    client: ModelClient,
    requests: list[BatchRequest],
    contexts: dict[str, RequestContext],
) -> list[RunningBatch]:
    running_batches: list[RunningBatch] = []
    # Submit every batch before polling so Anthropic can process them concurrently.
    for request_chunk, context_chunk in _chunk_requests(requests, contexts):
        batch = await client.batch(request_chunk)
        await batch.send()
        running_batches.append(RunningBatch(batch, context_chunk))
        LOG.info(
            "Submitted batch %s with %d requests",
            batch.get_id(),
            len(request_chunk),
        )
    return running_batches


async def _process_completed_batch(
    running_batch: RunningBatch,
    simulator: Simulator,
) -> int:
    simulation_count = 0
    processed_request_ids: set[str] = set()

    # Results are streamed, so a dropped connection can happen after some requests
    # have already been processed. Retry the stream without rerunning those requests.
    for attempt_index in range(DOWNLOAD_MAX_ATTEMPTS):
        try:
            async for result in running_batch.batch.download_results():
                if result.request_id in processed_request_ids:
                    continue

                context = running_batch.contexts.get(result.request_id)
                if context is None:
                    LOG.error("Received unknown request ID %s", result.request_id)
                    processed_request_ids.add(result.request_id)
                    continue

                instruction_set = _parse_instruction_set(result)
                if instruction_set is None:
                    processed_request_ids.add(result.request_id)
                    continue

                LOG.info(
                    "Processing request %s: job=%03d, scenario=%03d",
                    result.request_id,
                    context.job_id,
                    context.scenario_id,
                )

                simulator.set_job_context(context.job_id, context.scenario_id)
                simulator.set_scenario(context.scenario_id)
                # Proposal rank maps directly to the explicit simulator run ID.
                for proposal_index, instruction in enumerate(
                    instruction_set.instructions[:PROPOSALS_PER_REQUEST]
                ):
                    try:
                        simulator.run_scenario(instruction, run_id=proposal_index)
                        simulation_count += 1
                    except Exception as error:
                        LOG.error(
                            "Simulation failed for %s proposal %d: %s",
                            result.request_id,
                            proposal_index + 1,
                            error,
                        )
                processed_request_ids.add(result.request_id)
            return simulation_count
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError) as error:
            if attempt_index + 1 >= DOWNLOAD_MAX_ATTEMPTS:
                LOG.error(
                    "Could not download results for batch %s after %d attempts",
                    running_batch.batch.get_id(),
                    DOWNLOAD_MAX_ATTEMPTS,
                )
                raise

            delay = min(
                DOWNLOAD_MAX_DELAY_SECONDS,
                DOWNLOAD_INITIAL_DELAY_SECONDS * (2 ** attempt_index),
            )
            LOG.warning(
                "Results download for batch %s failed on attempt %d/%d (%s). "
                "Retrying in %.1fs.",
                running_batch.batch.get_id(),
                attempt_index + 1,
                DOWNLOAD_MAX_ATTEMPTS,
                error,
                delay,
            )
            await asyncio.sleep(delay)
    return simulation_count


async def _run(args: argparse.Namespace) -> int:
    # Use the first jobs.yaml entry as the source of scenarios and model settings.
    job_handler = JobHandler(str(args.jobs_file))
    jobs = job_handler.get_jobs()
    if not jobs:
        raise ValueError("jobs.yaml does not contain any jobs")
    job = jobs[0]
    if len(job.scenarios) == 0:
        raise ValueError(f"Expected some scenarios, found 0 in job '{job.name}'")

    configure_settings_for_job(CONFIG, job)
    planning_step = _get_planning_step(job)
    generator_config = get_generator_config_by_id(CONFIG, job.generator)
    if generator_config.instruction_count_per_generation != PROPOSALS_PER_REQUEST:
        raise ValueError(
            "The configured generator must produce exactly "
            f"{PROPOSALS_PER_REQUEST} instructions per request"
        )
    task = TaskManager.create_planner_task(job.experiment_nature, is_incremental=False)
    model_instance = get_model_instance_config_by_id(CONFIG, job.model_instances[0])
    client = create_model_client(model_instance, get_api_parameters_from_config(planning_step))
    if client is None:
        raise RuntimeError("Could not create the configured Claude model client")

    simulator = Simulator()
    try:
        # Prepare all cached prompts before opening the model connection.
        requests, contexts = _build_batch_requests(
            simulator, task, job.scenarios, args.cache_dir
        )
        expected_request_count = len(job.scenarios) * REPETITION_COUNT
        if len(requests) != expected_request_count:
            raise RuntimeError(
                f"Prepared {len(requests)} requests; expected {expected_request_count}. "
                "Check cached ground-truth data."
            )

        await client.open_connection()
        # Native Anthropic batches are independent of the model dispatcher's batch_size.
        running_batches = await _submit_batches(client, requests, contexts)
        simulation_count = 0
        # Poll every submitted batch until all terminal results have been processed.
        while running_batches:
            for running_batch in list(running_batches):
                status = await running_batch.batch.check_status()
                if status == BatchStatus.COMPLETED:
                    simulation_count += await _process_completed_batch(
                        running_batch, simulator
                    )
                    running_batches.remove(running_batch)
                elif status in (
                    BatchStatus.FAILED,
                    BatchStatus.CANCELLED,
                    BatchStatus.EXPIRED,
                ):
                    LOG.error(
                        "Batch %s ended with status %s",
                        running_batch.batch.get_id(),
                        status,
                    )
                    running_batches.remove(running_batch)
            if running_batches:
                await asyncio.sleep(15)

        LOG.info("Completed %d/%d simulations", simulation_count, expected_request_count * PROPOSALS_PER_REQUEST)
        return 0
    finally:
        # Always release the API connection and terminate any simulator processes.
        if client.get_state() == ModelClient.State.CONNECTED:
            await client.close_connection()
        simulator.free()


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except Exception as error:
        LOG.exception("One-shot Claude experiment run failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())