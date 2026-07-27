import asyncio
import json
from pathlib import Path
from untitled import simulator
from untitled.core.base import LOG, PROJECT_ROOT, CONFIG, SIMULATION_FPS
from untitled.core.job_handler import Job, JobHandler, copy_jobs_to_directory
from untitled.core.settings import configure_settings_for_job, copy_config_to_directory
from untitled.generator.instruction_generator import InstructionGenerator
from untitled.generator.experiment_nature import ExperimentNature
from untitled.simulator.simulator import Simulator
from untitled.simulator.simulation_data import SimulationData
from untitled.simulator.simulation_result import SimulationResult
from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
from untitled.generator.multi_step_vlm_instruction_generator import MultiStepVLMInstructionGenerator
from untitled.generator.rule_based_instruction_generator import RuleBasedInstructionGenerator
from untitled.simulator.simulation_data import SimulationFrameData
from untitled.generator.instruction import Instruction, InstructionCategory, InstructionResult, InstructionSet
from untitled.filter.filter import cluster_frames
from untitled.filter.embedder import DINOEmbedder, Embedder

from typing import Dict, List, Optional, Tuple, cast
import time


class Pipeline:
    """
    Pipeline to run simulation with VLM-generated instructions.
    
    This pipeline orchestrates job execution, running multiple scenarios
    with both ground truth (baseline) and experiment (VLM-instructed) runs.
    """

    def __init__(self, jobs_filepath: Optional[Path] = None):
        """Initialize pipeline components."""
        # TODO: Use underscore prefix for private attributes
        # Components
        self.simulator: Optional[Simulator] = None
        self.filter_embedder: Optional[Embedder] = None
        self.generator: Optional[InstructionGenerator] = None
        self.job_handler: Optional[JobHandler] = None

        # Constants
        self._ground_truth_dir_name: str = "ground_truths"
        default_jobs_filepath = PROJECT_ROOT / "jobs.yaml"
        self._jobs_filepath: Path = (
            Path(jobs_filepath) if jobs_filepath is not None else default_jobs_filepath
        )
        if not self._jobs_filepath.is_absolute():
            self._jobs_filepath = PROJECT_ROOT / self._jobs_filepath
        self._experiment_json_filename: str = "experiment_info.json"
        self._config_copied: bool = False
        self._supported_instruction_categories: List[InstructionCategory] = []

    async def init(self):
        """
        Initialize pipeline-level components (job handler only).
        
        Note: Simulator, generator, and filter_embedder are initialized per-job
        in _init_job_components() after the job context is configured.
        """
        # Initialize JobHandler
        
        self.job_handler = JobHandler(str(self._jobs_filepath))
        LOG.info(f"Loaded {len(self.job_handler)} jobs")

    async def _init_job_components(self, job: Job):
        """
        Initialize components for the current job context.
        
        Must be called after configure_settings_for_job() sets up
        CONFIG.simulator, CONFIG.generator, and CONFIG.filter.

        Args:
            job: The job for which to initialize components (used for context)
        """
        # Initialize simulator
        self.simulator = Simulator()

        # Initialize instruction generator based on config
        if CONFIG.generator is None:
            raise ValueError("No generator configured for job")

        if CONFIG.generator.design == "single_step_vlm":
            self.generator = SingleStepVLMInstructionGenerator(job.experiment_nature, CONFIG.generator.is_incremental)
            await self.generator.init()
        elif CONFIG.generator.design == "multi_step_vlm":
            self.generator = MultiStepVLMInstructionGenerator(job.experiment_nature, CONFIG.generator.is_incremental)
            await self.generator.init()
        elif CONFIG.generator.design == "rule_based":
            self.generator = RuleBasedInstructionGenerator(CONFIG.generator.dataset)
            await self.generator.init()
        else:
            raise ValueError(f"Unknown generator design: {CONFIG.generator.design}")

        # Initialize filter embedder only when filtering is enabled.
        if CONFIG.filter is not None:
            # WARN: Hardcoded is fine since we support only one filter type.
            self.filter_embedder = DINOEmbedder()

        # Cache supported categories to support category-cycling baseline generation.
        self._supported_instruction_categories = self.generator.get_supported_instruction_categories()

    def _is_rule_based_job(self) -> bool:
        return CONFIG.generator is not None and CONFIG.generator.design == "rule_based"

    def _ground_truth_result_key(self, scenario_id: int, run_id: int) -> tuple:
        """Build result key for ground-truth runs."""
        return (scenario_id, "gt", run_id)

    def _experiment_result_key(self, scenario_id: int, run_id: int) -> tuple:
        """Build result key for experiment runs."""
        return (scenario_id, "exp", run_id)

    def _get_experiment_run_id(self, job: Job, repeat_index: int, generation_index: int, instruction_index: int) -> int:
        """Compute stable experiment run id across repeats and generations (independent from ground truth IDs)."""
        per_repeat_run_count = job.instruction_generation_count * job.instruction_count_per_generation
        return (
            repeat_index * per_repeat_run_count
            + generation_index * job.instruction_count_per_generation
            + instruction_index
        )

    def _get_effective_repeat_count(self, job: Job) -> int:
        """Compute repeat count with baseline-specific uniform-category expansion."""
        repeat_count = max(1, job.repeat_count)
        if (
            self._is_rule_based_job()
            and job.repeat_for_uniform_instruction_category_distribution
            and self._supported_instruction_categories
        ):
            repeat_count = max(repeat_count, len(self._supported_instruction_categories))
        return repeat_count

    def _create_simulation_data_from_logs(
        self,
        job: Job,
        scenario_id: int,
        ground_truth_index: int,
        frames_per_minute: int,
    ) -> SimulationData:
        """
        Load simulation data from a ground truth run without frame clustering.
        """
        if self.job_handler is None:
            LOG.error("JobHandler not initialized. Cannot create simulation data from logs.")
            raise RuntimeError("JobHandler not initialized")
        if self.simulator is None:
            LOG.error("Simulator not initialized. Cannot create simulation data from logs.")
            raise RuntimeError("Simulator not initialized")
    
        cache_dir = self.job_handler.get_ground_truth_cache_dir(job.id, scenario_id, ground_truth_index)
        if cache_dir is not None:
            image_path_list = self.simulator.get_image_logs(job.id, scenario_id, ground_truth_index, cache_dir, frames_per_minute)
            metadata_list = self.simulator.get_metadata_logs(job.id, scenario_id, ground_truth_index, cache_dir, frames_per_minute)
        else:
            image_path_list = self.simulator.get_image_logs(job.id, scenario_id, ground_truth_index, frames_per_minute=frames_per_minute)
            metadata_list = self.simulator.get_metadata_logs(job.id, scenario_id, ground_truth_index, frames_per_minute=frames_per_minute)

        if not image_path_list:
            raise RuntimeError("Images not found in simulation logs")
        if not metadata_list:
            raise RuntimeError("Metadata not found in simulation logs")
        if len(image_path_list) != len(metadata_list):
            raise RuntimeError("Mismatch between images and metadata counts")

        return SimulationData(
            frame_count=len(image_path_list),
            image_dir=image_path_list[0].parent,
            frames=[
                SimulationFrameData(
                    image_name=img_path.name,
                    metadata=metadata_list[i],
                    timestamp=metadata_list[i].step * (1 / SIMULATION_FPS),
                )
                for i, img_path in enumerate(image_path_list)
            ],
        )

    async def _cleanup_job_components(self):
        """
        Cleanup components after a job completes.
        """
        # Free simulator
        if self.simulator:
            self.simulator.free()
            self.simulator = cast(Simulator, None)

        # Free generator
        if self.generator:
            await self.generator.free()
            self.generator = cast(InstructionGenerator, None)

        # Filter embedder doesn't need explicit cleanup
        self.filter_embedder = cast(Embedder, None)
        self._supported_instruction_categories = []

    async def stop(self):
        """
        Stop and free all pipeline components.
        
        Delegates to _cleanup_job_components() for component cleanup.
        """
        await self._cleanup_job_components()

    def _run_ground_truth(self, job: Job, scenario_id: int, run_id: int) -> Tuple[int, int]:
        """
        Run a single ground truth (baseline) scenario without instructions.
        
        Args:
            job: The job being executed
            scenario_id: The ID of the scenario being executed
            run_id: The run index (0-based)
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        # None check
        if self.simulator is None:
            LOG.error("Simulator not initialized")
            return (0, 1)
        if self.job_handler is None:
            LOG.error("JobHandler not initialized")
            return (0, 1)

        # Get cache directory if ground truth is cached
        cache_dir = self.job_handler.get_ground_truth_cache_dir(job.id, scenario_id, run_id)
        
        # Skip ground truth run if its logs are already cached
        if cache_dir is not None:
            LOG.info(f"  Ground truth run {run_id} is already cached. Skipping.")
            # Still collect the result from cache
            result = self.simulator.get_result(job.id, scenario_id, run_id, cache_dir)
            if result:
                job.results[self._ground_truth_result_key(scenario_id, run_id)] = result
            return (1, 0)
        try:
            LOG.info(f"  Ground truth run {run_id} ({run_id + 1} / {job.ground_truth_count})")
            actual_run_id = self.simulator.run_scenario()
            LOG.info(f"    Ground truth run {run_id} completed")
            # Collect the result using the actual run_id from the simulator
            result = self.simulator.get_result(job.id, scenario_id, actual_run_id)
            if result:
                job.results[self._ground_truth_result_key(scenario_id, run_id)] = result
            return (1, 0)
        except Exception as e:
            LOG.error(f"    Ground truth run {run_id} failed: {e}")
            return (0, 1)
        
    async def _cluster_frames_for_run(self, job: Job, scenario_id: int, run_id: int) -> SimulationData:
        """
        Cluster frames from a ground truth run for a scenario to reduce redundancy.
        
        Args:
            job: The job being executed
            scenario_id: The ID of the scenario being executed
            run_id: The generation/run index used to select the ground truth (modulo ground_truth_count)
        
        Returns:
            Filtered SimulationData with clustered frames
        """
        # None check
        if self.simulator is None:
            LOG.error("Simulator not initialized")
            raise RuntimeError("Simulator not initialized")
        if CONFIG.filter is None:
            # No filtering requested: return unclustered simulation data.
            ground_truth_index = run_id % job.ground_truth_count
            simulation_data = self._create_simulation_data_from_logs(
                job,
                scenario_id,
                ground_truth_index,
                frames_per_minute=0,
            )
            return simulation_data
        if self.filter_embedder is None:
            LOG.error("Filter embedder not initialized")
            raise RuntimeError("Filter embedder not initialized")
        if self.job_handler is None:
            LOG.error("JobHandler not initialized")
            raise RuntimeError("JobHandler not initialized")

        # Select ground truth based on run_id, cycling through available ground truths
        ground_truth_index = run_id % job.ground_truth_count
        fpm = CONFIG.filter.max_frames_per_minute
        simulation_data = self._create_simulation_data_from_logs(job, scenario_id, ground_truth_index, fpm)

        # Cluster similar frames to reduce redundancy
        threshold = CONFIG.filter.clustering_threshold
        max_cluster_count = CONFIG.filter.max_cluster_count
        clusters = cluster_frames(simulation_data, self.filter_embedder, threshold, max_cluster_count)

        # Create a new SimulationData with only the first frame from each cluster
        filtered_simulation_data = SimulationData(
            image_dir=simulation_data.image_dir,
            frame_count=len(clusters),
            frames=[cluster.get_data().frames[0] for cluster in clusters]
        )
        LOG.debug(f"    Reduced {simulation_data.frame_count} frames to {filtered_simulation_data.frame_count} frames using similarity threshold {threshold}")

        return filtered_simulation_data

    async def _generate_instruction_for_experiment(
        self, 
        job: Job, 
        scenario_id: int, 
        generation_index: int, 
        simulation_data: SimulationData,
        ground_truth_result: Optional[SimulationResult] = None,
        previously_tried_instructions: Optional[List[InstructionResult]] = None,
        instruction_category: InstructionCategory = InstructionCategory.NONE,
        forced_timestamp: Optional[float] = None,
    ) -> Tuple[bool, Optional[InstructionSet]]:
        """
        Generate VLM instruction for a single experiment (async, queues the prompt).
        
        Args:
            job: The job being executed
            scenario_id: The ID of the scenario being executed
            generation_index: The generation index (0-based, relative to instruction generations only)
            simulation_data: The simulation data containing frames and metadata
            ground_truth_result: Optional ground truth simulation result (for corrective tasks)
            previously_tried_instructions: List of previous instruction results (for incremental generation)
            
        Returns:
            Tuple of (success, InstructionSet or None)
        """
        if self.generator is None:
            LOG.error("Instruction generator not initialized. Cannot generate instruction for experiment.")
            return (False, None)

        try:
            LOG.info(f"  Generating instruction set {generation_index} ({generation_index + 1} / {job.instruction_generation_count})")
            
            # Extract failure info from ground truth result (for corrective tasks)
            failure_text = None
            was_route_completed = None
            if ground_truth_result is not None:
                failure_text = ground_truth_result.failure_text
                was_route_completed = ground_truth_result.was_route_completed
                LOG.debug(f"    Ground truth: route_completed={was_route_completed}, failure_text={failure_text[:100] if failure_text else 'None'}...")
            
            LOG.debug(f"    Generating instructions from images...")
            instruction_set: InstructionSet = await self.generator.generate_instructions(
                simulation_data,
                failure_text=failure_text,
                was_route_completed=was_route_completed,
                previously_tried_instructions=previously_tried_instructions,
                instruction_category=instruction_category,
                timestamp=forced_timestamp,
            )
            if not instruction_set or not instruction_set.instructions:
                LOG.error(f"    Failed to generate instruction from VLM")
                raise RuntimeError("VLM instruction generation failed")

            if self._is_rule_based_job() and job.instruction_count_per_generation > 1:
                desired_count = job.instruction_count_per_generation
                collected_instructions: List[Instruction] = []
                for index in range(desired_count):
                    if index == 0:
                        candidate_set = instruction_set
                    else:
                        candidate_set = await self.generator.generate_instructions(
                            simulation_data,
                            failure_text=failure_text,
                            was_route_completed=was_route_completed,
                            previously_tried_instructions=previously_tried_instructions,
                            instruction_category=instruction_category,
                            timestamp=forced_timestamp,
                        )

                    if candidate_set is None or not candidate_set.instructions:
                        break

                    candidate = candidate_set.instructions[0]
                    collected_instructions.append(Instruction(
                        rank=len(collected_instructions) + 1,
                        text=candidate.text,
                        injection_window=candidate.injection_window,
                        target_module_logic=candidate.target_module_logic,
                        reasoning=candidate.reasoning,
                        category=candidate.category,
                    ))

                if collected_instructions:
                    if len(collected_instructions) < desired_count:
                        LOG.warning(
                            "    Generated only %d/%d instruction(s) for category %s",
                            len(collected_instructions),
                            desired_count,
                            instruction_category.name,
                        )
                    instruction_set = InstructionSet(instructions=collected_instructions)
            
            LOG.debug(f"    Generated instruction set with {len(instruction_set.instructions)} test case(s)")
            for i in range(1, len(instruction_set.instructions) + 1):
                instruction = instruction_set.get_instruction_by_rank(i)
                if instruction is not None:
                    LOG.debug(f"    - Instruction {i}: Window=[{instruction.injection_window.start_timestamp:.2f}s, {instruction.injection_window.end_timestamp:.2f}s], Text='{instruction.text}'")
                else:
                    LOG.debug(f"    - Instruction {i}: None")

            return (True, instruction_set)
            
        except Exception as e:
            LOG.error(f"    Generation of instruction set {generation_index} failed: {e}")
            return (False, None)

    def _run_experiment(self, job: Job, scenario_id: int, run_id: int, instruction: Instruction) -> Tuple[int, int]:
        """
        Run a single experiment with a pre-generated VLM instruction.
        
        Args:
            job: The job being executed
            scenario_id: The ID of the scenario being executed
            run_id: The experiment run index (0-based, absolute)
            instruction: The instruction to use
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        # Simulator is required
        if self.simulator is None:
            LOG.error("Simulator not initialized. Cannot run experiment.")
            return (0, 1)

        try:
            experiment_index = run_id
            LOG.info(f"  Experiment run {run_id} ({experiment_index + 1} / {job.instruction_count_per_generation * job.instruction_generation_count})")
            
            if instruction is None:
                LOG.error(f"    No instruction available for this experiment")
                return (0, 1)
            
            # Run scenario with instruction
            LOG.debug(f"    Running scenario with instruction...")
            actual_run_id = self.simulator.run_scenario(instruction, run_id=run_id)
            
            LOG.info(f"    Experiment run {run_id} completed")
            # Collect the result using the actual run_id from the simulator
            result = self.simulator.get_result(job.id, scenario_id, actual_run_id)
            if result:
                job.results[self._experiment_result_key(scenario_id, run_id)] = result
            return (1, 0)
            
        except Exception as e:
            LOG.error(f"    Experiment run {run_id} failed: {e}")
            return (0, 1)

    def _evaluate_instruction_success(
        self,
        job: Job,
        experiment_result: SimulationResult,
        ground_truth_result: Optional[SimulationResult]
    ) -> bool:
        """
        Evaluate whether an instruction was successful based on experiment results.
        
        Args:
            job: The job being executed (contains experiment_nature)
            experiment_result: The result of running the experiment with the instruction
            ground_truth_result: The ground truth result to compare against
            
        Returns:
            True if the instruction achieved its goal, False otherwise
        """
        if ground_truth_result is None:
            # Cannot determine success without ground truth
            return False
        
        gt_penalty = ground_truth_result.scores.score_penalty
        exp_penalty = experiment_result.scores.score_penalty
        
        if job.experiment_nature == ExperimentNature.ADVERSARIAL:
            # Adversarial success: experiment penalty is lower (more infractions)
            return exp_penalty < gt_penalty
        else:
            # Corrective success: experiment penalty is higher (fewer infractions)
            return exp_penalty > gt_penalty

    def _run_scenario_ground_truths(self, job: Job, scenario_id: int) -> Tuple[int, int]:
        """
        Run all ground truth runs for a single scenario (Stage 1: CARLA only).
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID to run
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        # Simulator is required
        if self.simulator is None:
            LOG.error("Simulator not initialized. Cannot run scenario ground truths.")
            return (0, 1)

        LOG.info("-" * 80)
        LOG.info(f"Job {job.id} | Scenario {scenario_id} | Ground Truth Stage")
        LOG.info("-" * 80)
        
        successful = 0
        failed = 0
        
        try:
            # Set job and scenario context for directory structure
            self.simulator.set_job_context(job.id, scenario_id)
            self.simulator.set_scenario(scenario_id)
            
            # Run ground truth (baseline) runs
            LOG.info(f"Running {job.ground_truth_count} ground truth run(s)...")
            for run_index in range(job.ground_truth_count):
                gt_success, gt_failed = self._run_ground_truth(job, scenario_id, run_index)
                successful += gt_success
                failed += gt_failed
            
            LOG.info(f"Scenario {scenario_id} ground truth stage completed")
            
        except Exception as e:
            LOG.error(f"Scenario {scenario_id} ground truth stage failed: {e}")
        
        return (successful, failed)

    async def _generate_scenario_instructions(
        self, 
        job: Job, 
        scenario_id: int,
        previously_tried_instructions: Optional[List[InstructionResult]] = None,
        repeat_index: int = 0,
    ) -> Dict[int, Optional[InstructionSet]]:
        """
        Generate VLM instructions for all experiments in a scenario (Stage 2: VLM only).
        
        Each generation uses a different ground truth run (cycling through available ones)
        to ensure diversity when multiple ground truths exist.
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID
            previously_tried_instructions: List of instruction results from previous repeats (modified in-place)
            
        Returns:
            Dictionary mapping experiment_index to InstructionSet or None
        """
        # Simulator is required
        # FIXME: Why is simulator required? Is this an error?
        if self.simulator is None:
            LOG.error("Simulator not initialized. Cannot generate scenario instructions.")
            return {i: None for i in range(job.instruction_generation_count)}

        LOG.info("-" * 80)
        LOG.info(f"Job {job.id} | Scenario {scenario_id} | VLM Instruction Generation Stage")
        LOG.info("-" * 80)
        
        instructions: Dict[int, Optional[InstructionSet]] = {}
        
        try:
            # Set job and scenario context
            # FIXME: Why do we set simulator's context when we use the generator? Is this needed?
            self.simulator.set_job_context(job.id, scenario_id)
            self.simulator.set_scenario(scenario_id)
            
            # Cache simulation data per ground truth to avoid redundant clustering
            simulation_data_cache: Dict[int, SimulationData] = {}
            
            # Generate instructions for all experiments
            LOG.info(f"Generating instructions: {job.instruction_generation_count} generation call(s)...")
            
            for generation_index in range(job.instruction_generation_count):
                # Cluster frames from the appropriate ground truth for this generation
                # Cycle through ground truths for diverse ground_truth_index selection
                ground_truth_index = generation_index % job.ground_truth_count
                LOG.debug(f"  Generation {generation_index} using ground truth run {ground_truth_index}")
                
                # Use cached simulation data if available, otherwise cluster and cache
                if ground_truth_index in simulation_data_cache:
                    simulation_data = simulation_data_cache[ground_truth_index]
                else:
                    simulation_data = await self._cluster_frames_for_run(job, scenario_id, generation_index)
                    simulation_data_cache[ground_truth_index] = simulation_data
                
                # Get ground truth result for this generation (for corrective tasks)
                ground_truth_result: Optional[SimulationResult] = job.results.get(
                    self._ground_truth_result_key(scenario_id, ground_truth_index)
                )
                if ground_truth_result is None and job.experiment_nature == ExperimentNature.CORRECTIVE:
                    LOG.warning(f"No ground truth result found for scenario {scenario_id} run {ground_truth_index}, corrective task may lack failure context")
                
                # Pass previously tried instructions for incremental generation (limiting is done in generator)
                requested_category = InstructionCategory.NONE
                if (
                    self._is_rule_based_job()
                    and job.repeat_for_uniform_instruction_category_distribution
                    and self._supported_instruction_categories
                ):
                    category_index = repeat_index % len(self._supported_instruction_categories)
                    requested_category = self._supported_instruction_categories[category_index]

                success, instruction_set = await self._generate_instruction_for_experiment(
                    job, scenario_id, generation_index, simulation_data, ground_truth_result,
                    previously_tried_instructions=previously_tried_instructions,
                    instruction_category=requested_category,
                )

                # Fallback for forced categories that have no valid window in this scenario.
                if (not success or instruction_set is None or len(instruction_set.instructions) == 0) and requested_category != InstructionCategory.NONE:
                    if self._is_rule_based_job() and job.repeat_for_uniform_instruction_category_distribution:
                        LOG.warning(
                            f"Scenario {scenario_id} generation {generation_index}: no valid instruction for category {requested_category.name}, skipping"
                        )
                        success = False
                        instruction_set = None
                    else:
                        LOG.warning(
                            f"Scenario {scenario_id} generation {generation_index}: no valid instruction for category {requested_category.name}, retrying without forced category"
                        )
                        success, instruction_set = await self._generate_instruction_for_experiment(
                            job,
                            scenario_id,
                            generation_index,
                            simulation_data,
                            ground_truth_result,
                            previously_tried_instructions=previously_tried_instructions,
                            instruction_category=InstructionCategory.NONE,
                        )

                if success:
                    instructions[generation_index] = instruction_set
                else:
                    instructions[generation_index] = None

                # Serialize chosen ground truth index for debugging
                # Make sure directory exists and write JSON data
                # Use has_job_subdir=False to write to experiment directory, not ground_truths
                debug_run_id = self._get_experiment_run_id(job, repeat_index, generation_index, 0)
                experiment_json_dir: Path = self.simulator.get_run_dir(scenario_id, debug_run_id, has_job_subdir=False)
                experiment_json_path: Path = experiment_json_dir / self._experiment_json_filename
                experiment_json_dir.mkdir(parents=True, exist_ok=True)
                actual_category = None
                actual_instruction_text = None
                if instruction_set is not None and instruction_set.instructions:
                    actual_category = instruction_set.instructions[0].category.name
                    actual_instruction_text = instruction_set.instructions[0].text
                json_data = {
                    "ground_truth_index": ground_truth_index,
                    "generation_index": generation_index,
                    "repeat_index": repeat_index,
                    "requested_category": requested_category.name,
                    "actual_category": actual_category,
                    "actual_instruction": actual_instruction_text,
                    "generation_success": bool(success and instruction_set and instruction_set.instructions),
                }
                with open(experiment_json_path, "w") as f:
                    json.dump(json_data, f)
            
            LOG.info(f"Scenario {scenario_id} instruction generation completed")
            
        except Exception as e:
            LOG.error(f"Scenario {scenario_id} instruction generation failed: {e}")
        
        return instructions

    def _run_scenario_experiments(self, job: Job, scenario_id: int, instructions: Dict[int, Optional[InstructionSet]], repeat_index: int) -> Tuple[int, int]:
        """
        Run all experiment runs for a single scenario with pre-generated instructions (Stage 3: CARLA again).
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID to run
            instructions: Dictionary mapping experiment_index to InstructionSet or None
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        # Simulator is required
        if self.simulator is None:
            LOG.error("Simulator not initialized. Cannot run scenario experiments.")
            return (0, 1)

        LOG.info("-" * 80)
        LOG.info(f"Job {job.id} | Scenario {scenario_id} | Experiment Execution Stage")
        LOG.info("-" * 80)
        
        successful = 0
        failed = 0
        
        try:
            # Set job and scenario context
            self.simulator.set_job_context(job.id, scenario_id)
            self.simulator.set_scenario(scenario_id)

            # Check how many instructions to run per experiment
            instruction_count_per_generation = job.instruction_count_per_generation
            if instruction_count_per_generation < 1:
                LOG.warning(f"Invalid instruction_count_per_generation={instruction_count_per_generation}, defaulting to 1")
                instruction_count_per_generation = 1

            # Run experiment runs with pre-generated instructions
            LOG.info(f"Running {job.instruction_generation_count} instruction generation call(s)...")
            for generation_index in range(job.instruction_generation_count):
                # Get instruction set for this experiment
                instruction_set = instructions.get(generation_index)
                if instruction_set is None:
                    LOG.warning(f"  No instruction set available for instruction generation {generation_index}, skipping run")
                    failed += 1
                    continue
                
                # Determine number of runs and their IDs
                total_run_count = instruction_count_per_generation
                if total_run_count > len(instruction_set.instructions):
                    LOG.warning(f"  Requested {total_run_count} instructions for instruction generation {generation_index}, but only {len(instruction_set.instructions)} available. Reducing to available count.")
                    total_run_count = len(instruction_set.instructions)

                # Run the specified number of instructions for this experiment
                for instruction_index in range(total_run_count):
                    run_id = self._get_experiment_run_id(job, repeat_index, generation_index, instruction_index)
                    instruction = instruction_set.instructions[instruction_index]
                    exp_success, exp_failed = self._run_experiment(job, scenario_id, run_id, instruction)
                    successful += exp_success
                    failed += exp_failed
            
            LOG.info(f"Scenario {scenario_id} experiment stage completed")
            
        except Exception as e:
            LOG.error(f"Scenario {scenario_id} experiment stage failed: {e}")
        
        return (successful, failed)

    def _collect_instruction_results(
        self,
        job: Job,
        scenario_id: int,
        instructions: Dict[int, Optional[InstructionSet]],
        tried_instructions: List[InstructionResult],
        repeat_index: int,
    ) -> None:
        """
        Collect experiment results into InstructionResult objects for incremental generation.
        
        After experiments are run, this method evaluates each instruction's success
        and appends the results to tried_instructions for use in subsequent repeats.
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID
            instructions: Dictionary mapping generation_index to InstructionSet
            tried_instructions: List to append InstructionResult objects to (modified in-place)
        """
        for generation_index, instruction_set in instructions.items():
            if instruction_set is None:
                continue
            
            # Get ground truth result for comparison
            ground_truth_index = generation_index % job.ground_truth_count
            ground_truth_result = job.results.get(
                self._ground_truth_result_key(scenario_id, ground_truth_index)
            )
            
            # Collect results for each instruction in the set
            instruction_count = min(job.instruction_count_per_generation, len(instruction_set.instructions))
            for i in range(instruction_count):
                instruction = instruction_set.instructions[i]
                run_id = self._get_experiment_run_id(job, repeat_index, generation_index, i)
                exp_result = job.results.get(self._experiment_result_key(scenario_id, run_id))
                
                if exp_result:
                    was_successful = self._evaluate_instruction_success(job, exp_result, ground_truth_result)
                    tried_instructions.append(InstructionResult(
                        instruction=instruction,
                        was_successful=was_successful
                    ))
                    LOG.debug(f"  Collected result for run {run_id}: {'success' if was_successful else 'failure'}")

    async def _run_job(self, job: Job) -> Tuple[int, int]:
        """
        Run a single job with all its scenarios using 3-stage approach:
        1. CARLA Stage: Run all ground truths for all scenarios
        2. VLM Stage: Generate instructions for all experiments (activates VLM if needed)
        3. CARLA Stage: Run all experiments with generated instructions
        
        Args:
            job: The job to execute
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        # Configure settings for this job (sets CONFIG.generator, CONFIG.filter, CONFIG.simulator)
        configure_settings_for_job(CONFIG, job)
        
        # Initialize components for this job context
        await self._init_job_components(job)

        # Simulator is required for all execution stages
        if self.simulator is None:
            LOG.error("Simulator not initialized. Cannot run job.")
            return (0, 1)
        
        # Copy settings to simulator's output directory for record-keeping (once)
        # NOTE: Instead of having a "once" flag, we could copy each job by one in their respective job directory, this is less straigthforward though.
        if not self._config_copied:
            simulator_base_dir = self.simulator.get_base_dir()
            copy_config_to_directory(PROJECT_ROOT, simulator_base_dir, LOG)
            copy_jobs_to_directory(self._jobs_filepath, simulator_base_dir)
            self._config_copied = True
        
        LOG.info("=" * 80)
        LOG.info(f"Starting Job {job.id}: '{job.name}'")
        LOG.info(f"Scenarios: {len(job.scenarios)}, Ground truth runs: {job.ground_truth_count}, Instruction generations: {job.instruction_generation_count}, "
                 f"Experiment runs: {job.instruction_count_per_generation * job.instruction_generation_count}, Repeats: {job.repeat_count}")
        LOG.info(f"Experiment nature: {job.experiment_nature.value}")
        if CONFIG.generator:
            LOG.info(f"Generator: {CONFIG.generator.id} ({CONFIG.generator.design})")
        if CONFIG.filter:
            LOG.info(f"Filter: {CONFIG.filter.id}")
        if CONFIG.simulator:
            LOG.info(f"Simulator: {CONFIG.simulator.id} ({CONFIG.simulator.type})")
        LOG.info("=" * 80)
        
        job_start_time = time.time()
        job_successful = 0
        job_failed = 0
        
        # Set experiment nature on generator
        if self.generator is not None:
            self.generator.set_experiment_nature(job.experiment_nature)
        
        # Stage 1: CARLA - Run all ground truths for all scenarios
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info("STAGE 1: CARLA Ground Truth Runs")
        LOG.info("=" * 80)

        # Set base subdir for ground truth runs to keep them organized and separate from experiments
        self.simulator.set_base_job_subdir(Path(self._ground_truth_dir_name))
        
        # TODO: Add parallelized VLM instruction generation during ground truth runs
        for scenario_id in job.scenarios:
            gt_success, gt_failed = self._run_scenario_ground_truths(job, scenario_id)
            job_successful += gt_success
            job_failed += gt_failed

        # TODO: Add async VLM instruction generation during ground truth runs
        # Initialize tracking for repeat loop
        repeat_count = self._get_effective_repeat_count(job)
        stop_on_success = job.stop_repeat_on_success
        
        # Track previously tried instructions per scenario (for incremental generation across repeats)
        scenario_tried_instructions: Dict[int, List[InstructionResult]] = {sid: [] for sid in job.scenarios}
        # Track which scenarios have achieved success (for stop_repeat_on_success)
        scenario_success: Dict[int, bool] = {sid: False for sid in job.scenarios}
        
        # Repeat loop for instruction generation + experiment execution
        for repeat_index in range(repeat_count):
            # Check if we should stop early (all scenarios have success)
            if stop_on_success and all(scenario_success.values()):
                LOG.info(f"All scenarios achieved success, stopping at repeat {repeat_index}")
                break
                
            LOG.info("")
            LOG.info("=" * 80)
            LOG.info(f"REPEAT {repeat_index + 1} / {repeat_count}")
            LOG.info("=" * 80)
            
            # Stage 2: VLM - Generate instructions for all experiments
            LOG.info("")
            LOG.info("=" * 80)
            LOG.info(f"STAGE 2: VLM Instruction Generation (Repeat {repeat_index + 1})")
            LOG.info("=" * 80)
            
            # Enable generator's full speed mode as the simulator is idle now (async)
            if self.generator is not None:
                enable_full_speed_mode_task = asyncio.create_task(self.generator.enable_full_speed_mode())
                await enable_full_speed_mode_task

            # Generate instructions for all scenarios
            # Each generation will cluster frames from a different ground truth run (modulo ground_truth_count)
            all_instructions = {}  # Map: scenario_id -> {experiment_index -> Instruction}
            for scenario_id in job.scenarios:
                # Skip scenarios that already achieved success (if stop_on_success is enabled)
                if stop_on_success and scenario_success[scenario_id]:
                    LOG.debug(f"Skipping scenario {scenario_id} (already achieved success)")
                    continue
                    
                LOG.debug(f"Generating instructions for scenario {scenario_id}...")
                scenario_instructions = await self._generate_scenario_instructions(
                    job, scenario_id, 
                    previously_tried_instructions=scenario_tried_instructions[scenario_id],
                    repeat_index=repeat_index,
                )
                all_instructions[scenario_id] = scenario_instructions
            
            # Disable full speed mode once all instructions are generated
            if self.generator is not None:
                await self.generator.disable_full_speed_mode()

            # Unset base subdir as we are done with running accessing ground truths
            self.simulator.set_base_job_subdir(Path(""))
            
            # Stage 3: CARLA - Run all experiments with generated instructions
            LOG.info("")
            LOG.info("=" * 80)
            LOG.info(f"STAGE 3: CARLA Experiment Runs (Repeat {repeat_index + 1})")
            LOG.info("=" * 80)
            for scenario_id in job.scenarios:
                # Skip scenarios that already achieved success (if stop_on_success is enabled)
                if stop_on_success and scenario_success[scenario_id]:
                    continue
                    
                if scenario_id not in all_instructions:
                    continue
                    
                exp_success, exp_failed = self._run_scenario_experiments(job, scenario_id, all_instructions[scenario_id], repeat_index)
                job_successful += exp_success
                job_failed += exp_failed
                
                # Collect instruction results from this repeat for incremental generation in next repeat
                self._collect_instruction_results(
                    job, scenario_id, all_instructions[scenario_id],
                    scenario_tried_instructions[scenario_id],
                    repeat_index,
                )
                
                # Check if this scenario achieved success in this repeat
                # Look at the tried_instructions to see if any was successful
                for instruction_result in scenario_tried_instructions[scenario_id]:
                    if instruction_result.was_successful:
                        scenario_success[scenario_id] = True
                        LOG.info(f"Scenario {scenario_id} achieved success!")
                        break
        
        # Job summary
        job_elapsed = time.time() - job_start_time
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info(f"Job {job.id} '{job.name}' completed in {job_elapsed:.1f}s ({job_elapsed/3600:.2f}h)")
        LOG.info(f"  Successful: {job_successful}, Failed: {job_failed}")
        LOG.info("=" * 80)
        
        # Print detailed results summary
        self._print_job_results_summary(job)
        
        # Cleanup job components
        await self._cleanup_job_components()
        
        return (job_successful, job_failed)

    def _print_job_results_summary(self, job: Job) -> None:
        """
        Print a detailed summary of job results, comparing ground truth vs experiments.
        
        An instruction is considered "adversarial" if it caused the experiment's 
        penalty score to be lower than the ground truth's penalty score
        (ground_truth_score > experiment_score means more penalty = worse = adversarial success).
        
        Args:
            job: The job with collected results
        """
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info(f"RESULTS SUMMARY - Job {job.id}: '{job.name}'")
        LOG.info("=" * 80)
        
        if not job.results:
            LOG.info("No results collected.")
            return
        
        total_experiments = 0
        adversarial_count = 0
        corrective_count = 0

        for scenario_id in job.scenarios:
            LOG.info(f"")
            LOG.info(f"Scenario {scenario_id}:")
            LOG.info("-" * 40)
            
            # Get ground truth results (average if multiple)
            gt_penalty_scores = []
            gt_infractions_total = 0
            for run_id in range(job.ground_truth_count):
                gt_result: Optional[SimulationResult] = job.results.get(
                    self._ground_truth_result_key(scenario_id, run_id)
                )
                if gt_result:
                    gt_penalty_scores.append(gt_result.scores.score_penalty)
                    gt_infractions_total += gt_result.infractions.total_count
            
            if gt_penalty_scores:
                avg_gt_penalty = sum(gt_penalty_scores) / len(gt_penalty_scores)
                LOG.info(f"  Ground Truth: penalty={avg_gt_penalty:.3f} (avg of {len(gt_penalty_scores)} run(s)), infractions={gt_infractions_total}")
            else:
                avg_gt_penalty = None
                LOG.info(f"  Ground Truth: No results available")
            
            # Get experiment results and compare (all repeats)
            scenario_experiment_run_ids = sorted(
                run_id
                for key in job.results.keys()
                if len(key) == 3 and key[0] == scenario_id and key[1] == "exp"
                for run_id in [key[2]]
            )
            for run_id in scenario_experiment_run_ids:
                exp_result: Optional[SimulationResult] = job.results.get(
                    self._experiment_result_key(scenario_id, run_id)
                )
                
                if exp_result:
                    total_experiments += 1
                    exp_penalty = exp_result.scores.score_penalty
                    exp_infractions = exp_result.infractions.total_count
                    
                    # Determine if adversarial (lower penalty = more infractions = adversarial success) or corrective
                    # score_penalty is a multiplier: 1.0 = no infractions, <1.0 = infractions occurred
                    change_marker = ""
                    if avg_gt_penalty is not None:
                        if avg_gt_penalty > exp_penalty:
                            adversarial_count += 1
                            change_marker = " [ADVERSARIAL]"
                        elif avg_gt_penalty < exp_penalty:
                            corrective_count += 1
                            change_marker = " [IMPROVED]"
                    
                    # Count specific collision types for clarity
                    collisions = (
                        len(exp_result.infractions.collisions_vehicle) +
                        len(exp_result.infractions.collisions_pedestrian) +
                        len(exp_result.infractions.collisions_layout)
                    )
                    
                    LOG.info(
                        f"  Run {run_id}: penalty={exp_penalty:.3f}, "
                        f"infractions={exp_infractions}, collisions={collisions}{change_marker}"
                    )
                else:
                    LOG.info(f"  Run {run_id}: No result available")
        
        # Overall summary
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info("INSTRUCTION EFFECTIVENESS SUMMARY")
        LOG.info("=" * 80)
        if total_experiments > 0:
            adversarial_rate = (adversarial_count / total_experiments) * 100
            corrective_rate = (corrective_count / total_experiments) * 100
            LOG.info(f"Total experiments: {total_experiments}")
            LOG.info(f"Instruction success rate: {(adversarial_rate if job.experiment_nature == ExperimentNature.ADVERSARIAL else corrective_rate):.1f}%")
            LOG.info(f"Instruction failure rate: {(100 - adversarial_rate if job.experiment_nature == ExperimentNature.ADVERSARIAL else 100 - corrective_rate):.1f}%")
            LOG.info(f"Adversarial results: {adversarial_count}")
            LOG.info(f"Corrective results: {corrective_count}")
            LOG.info(f"No change results: {total_experiments - adversarial_count - corrective_count}")

        else:
            LOG.info("No experiment results to analyze.")
        LOG.info("=" * 80)

    async def start(self):
        """
        Start the pipeline execution.
        
        Executes all jobs defined in jobs.yaml using a 3-stage approach:
        1. CARLA Stage: Run all ground truths
        2. VLM Stage: Generate instructions (with GPU if local)
        3. CARLA Stage: Run all experiments
        """
        if not self.job_handler or not self.job_handler.get_jobs():
            LOG.error("No jobs loaded. Cannot start pipeline.")
            return
        
        # Statistics tracking
        total_runs = 0
        successful_runs = 0
        failed_runs = 0

        
        try:
            # Iterate through all jobs
            for job in self.job_handler.get_jobs():
                job_successful, job_failed = await self._run_job(job)
                total_runs += job_successful + job_failed
                successful_runs += job_successful
                failed_runs += job_failed
        
        except KeyboardInterrupt:
            LOG.warning("Pipeline interrupted by user (Ctrl+C)")
            LOG.info("Cleaning up...")
        
        except Exception as e:
            LOG.error(f"Pipeline failed with unexpected error: {e}")
            import traceback
            LOG.error(traceback.format_exc())
        
        finally:
            # Final summary
            LOG.info("")
            LOG.info("=" * 80)
            LOG.info("PIPELINE EXECUTION SUMMARY")
            LOG.info("=" * 80)
            LOG.info(f"Total runs: {total_runs}")
            LOG.info(f"Successful: {successful_runs}")
            LOG.info(f"Failed: {failed_runs}")
            if total_runs > 0:
                success_rate = (successful_runs / total_runs) * 100
                LOG.info(f"Success rate: {success_rate:.1f}%")
            LOG.info("=" * 80)
