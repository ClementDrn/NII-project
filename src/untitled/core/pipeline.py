import asyncio
from untitled.core.base import LOG, PROJECT_ROOT, CONFIG, SIMULATION_FPS
from untitled.core.job_handler import Job, JobHandler
from untitled.generator.instruction_generator import InstructionGenerator
from untitled.generator.experiment_nature import ExperimentNature
from untitled.simulator.simulator import Simulator
from untitled.simulator.simulation_data import SimulationData
from untitled.simulator.simulation_result import SimulationResult
from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
from untitled.generator.multi_step_vlm_instruction_generator import MultiStepVLMInstructionGenerator
from untitled.simulator.simulation_data import SimulationFrameData
from untitled.generator.instruction import Instruction, InstructionSet
from untitled.filter.filter import cluster_frames
from untitled.filter.embedder import DINOEmbedder, Embedder

from typing import Dict, List, Optional, Tuple
import time


class Pipeline:
    """
    Pipeline to run simulation with VLM-generated instructions.
    
    This pipeline orchestrates job execution, running multiple scenarios
    with both ground truth (baseline) and experiment (VLM-instructed) runs.
    """

    def __init__(self):
        """Initialize pipeline components."""
        self.simulator: Simulator = None
        self.filter_embedder: Embedder = None
        self.generator: InstructionGenerator = None
        self.job_handler: JobHandler = None

    async def init(self):
        """
        Initialize all pipeline components.
        """
        # Initialize JobHandler
        jobs_file = PROJECT_ROOT / "jobs.yaml"
        self.job_handler = JobHandler(str(jobs_file))
        LOG.info(f"Loaded {len(self.job_handler)} jobs")

        # Initialize simulator
        self.simulator = Simulator()

        # Initialize instruction generator based on config
        if CONFIG.generator.design == "single_step_vlm":
            self.generator = SingleStepVLMInstructionGenerator()
            await self.generator.init()
        elif CONFIG.generator.design == "multi_step_vlm":
            self.generator = MultiStepVLMInstructionGenerator()
            await self.generator.init()
        else:
            raise ValueError(f"Unknown generator design: {CONFIG.generator.design}")

        # Initialize filter embedder
        self.filter_embedder = DINOEmbedder()

    async def stop(self):
        """
        Stop and free all pipeline components.
        """
        # Free simulator
        if self.simulator:
            self.simulator.free()
            self.simulator = None

        # Free VLM
        if self.generator:
            await self.generator.free()
            self.generator = None

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
        # Get cache directory if ground truth is cached
        cache_dir = self.job_handler.get_ground_truth_cache_dir(job.id, scenario_id, run_id)
        
        # Skip ground truth run if its logs are already cached
        if cache_dir is not None:
            LOG.info(f"  Ground truth run {run_id} is already cached. Skipping.")
            # Still collect the result from cache
            result = self.simulator.get_result(job.id, scenario_id, run_id, cache_dir)
            if result:
                job.results[(scenario_id, run_id)] = result
            return (1, 0)
        try:
            LOG.info(f"  Ground truth run {run_id} ({run_id + 1} / {job.ground_truth_count})")
            actual_run_id = self.simulator.run_scenario()
            LOG.info(f"    Ground truth run {run_id} completed")
            # Collect the result using the actual run_id from the simulator
            result = self.simulator.get_result(job.id, scenario_id, actual_run_id)
            if result:
                job.results[(scenario_id, run_id)] = result
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
        # Select ground truth based on run_id, cycling through available ground truths
        ground_truth_index = run_id % job.ground_truth_count
        cache_dir = self.job_handler.get_ground_truth_cache_dir(job.id, scenario_id, ground_truth_index)
        
        # Get image and metadata from previously collected ground truth
        fpm = CONFIG.filter.max_frames_per_minute     # undersampling
        LOG.debug(f"    Fetching images and metadata from ground truth run {ground_truth_index}...")
        image_path_list = self.simulator.get_image_logs(job.id, scenario_id, ground_truth_index, cache_dir, fpm)
        metadata_list = self.simulator.get_metadata_logs(job.id, scenario_id, ground_truth_index, cache_dir, fpm)
        
        if not image_path_list:
            LOG.error(f"    Failed to get images from simulation logs")
            raise RuntimeError("Images not found in simulation logs")
        
        if not metadata_list:
            LOG.error(f"    Failed to get metadata from simulation logs")
            raise RuntimeError("Metadata not found in simulation logs")

        # Number of images and metadata entries must match
        if len(image_path_list) != len(metadata_list):
            LOG.error(f"    Mismatch between number of images ({len(image_path_list)}) and metadata entries ({len(metadata_list)})")
            raise RuntimeError("Mismatch between images and metadata counts")
        
        # Create initial SimulationData
        simulation_data = SimulationData(
            frame_count=len(image_path_list),
            image_dir=image_path_list[0].parent,
            frames=[SimulationFrameData(image_name=img_path.name, metadata=metadata_list[i], timestamp=metadata_list[i].step * (1 / SIMULATION_FPS))
                    for i, img_path in enumerate(image_path_list)]
        )

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
        ground_truth_result: Optional[SimulationResult] = None
    ) -> Tuple[bool, Optional[InstructionSet]]:
        """
        Generate VLM instruction for a single experiment (async, queues the prompt).
        
        Args:
            job: The job being executed
            scenario_id: The ID of the scenario being executed
            generation_index: The generation index (0-based, relative to instruction generations only)
            simulation_data: The simulation data containing frames and metadata
            ground_truth_result: Optional ground truth simulation result (for corrective tasks)
            
        Returns:
            Tuple of (success, InstructionSet or None)
        """
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
            instruction_set: InstructionSet = await self.generator.generate_instructions_from_images(
                simulation_data,
                failure_text=failure_text,
                was_route_completed=was_route_completed
            )
            if not instruction_set:
                LOG.error(f"    Failed to generate instruction from VLM")
                raise RuntimeError("VLM instruction generation failed")
            
            LOG.debug(f"    Generated instruction set with {len(instruction_set.instructions)} test case(s)")
            for i in range(1, len(instruction_set.instructions) + 1):
                instruction = instruction_set.get_instruction_by_rank(i)
                LOG.debug(f"    - Instruction {i}: Window=[{instruction.start_frame_index}, {instruction.end_frame_index}], Text='{instruction.text}'")

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
        try:
            experiment_index = run_id - job.ground_truth_count
            LOG.info(f"  Experiment run {run_id} ({experiment_index + 1} / {job.instruction_count_per_generation * job.instruction_generation_count})")
            
            if instruction is None:
                LOG.error(f"    No instruction available for this experiment")
                return (0, 1)
            
            # Run scenario with instruction
            LOG.debug(f"    Running scenario with instruction...")
            actual_run_id = self.simulator.run_scenario(instruction)
            
            LOG.info(f"    Experiment run {run_id} completed")
            # Collect the result using the actual run_id from the simulator
            result = self.simulator.get_result(job.id, scenario_id, actual_run_id)
            if result:
                job.results[(scenario_id, run_id)] = result
            return (1, 0)
            
        except Exception as e:
            LOG.error(f"    Experiment run {run_id} failed: {e}")
            return (0, 1)

    def _run_scenario_ground_truths(self, job: Job, scenario_id: int) -> Tuple[int, int]:
        """
        Run all ground truth runs for a single scenario (Stage 1: CARLA only).
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID to run
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
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

    async def _generate_scenario_instructions(self, job: Job, scenario_id: int) -> Dict[int, Optional[InstructionSet]]:
        """
        Generate VLM instructions for all experiments in a scenario (Stage 2: VLM only).
        
        Each generation uses a different ground truth run (cycling through available ones)
        to ensure diversity when multiple ground truths exist.
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID
            
        Returns:
            Dictionary mapping experiment_index to InstructionSet or None
        """
        LOG.info("-" * 80)
        LOG.info(f"Job {job.id} | Scenario {scenario_id} | VLM Instruction Generation Stage")
        LOG.info("-" * 80)
        
        instructions: Dict[int, Optional[InstructionSet]] = {}
        
        try:
            # Set job and scenario context
            self.simulator.set_job_context(job.id, scenario_id)
            self.simulator.set_scenario(scenario_id)
            
            # Cache simulation data per ground truth to avoid redundant clustering
            simulation_data_cache: Dict[int, SimulationData] = {}
            
            # Generate instructions for all experiments
            LOG.info(f"Generating instructions: {job.instruction_generation_count} generation call(s)...")
            for generation_index in range(job.instruction_generation_count):
                # Cluster frames from the appropriate ground truth for this generation
                ground_truth_index = generation_index % job.ground_truth_count
                LOG.debug(f"  Generation {generation_index} using ground truth run {ground_truth_index}")
                
                # Use cached simulation data if available, otherwise cluster and cache
                if ground_truth_index in simulation_data_cache:
                    simulation_data = simulation_data_cache[ground_truth_index]
                else:
                    simulation_data = await self._cluster_frames_for_run(job, scenario_id, generation_index)
                    simulation_data_cache[ground_truth_index] = simulation_data
                
                # Get ground truth result for this generation (for corrective tasks)
                ground_truth_result: Optional[SimulationResult] = job.results.get((scenario_id, ground_truth_index))
                if ground_truth_result is None and job.experiment_nature == ExperimentNature.CORRECTIVE:
                    LOG.warning(f"No ground truth result found for scenario {scenario_id} run {ground_truth_index}, corrective task may lack failure context")
                
                success, instruction_set = await self._generate_instruction_for_experiment(
                    job, scenario_id, generation_index, simulation_data, ground_truth_result
                )
                if success:
                    instructions[generation_index] = instruction_set
                else:
                    instructions[generation_index] = None
            
            LOG.info(f"Scenario {scenario_id} instruction generation completed")
            
        except Exception as e:
            LOG.error(f"Scenario {scenario_id} instruction generation failed: {e}")
        
        return instructions

    def _run_scenario_experiments(self, job: Job, scenario_id: int, instructions: Dict[int, Optional[InstructionSet]]) -> Tuple[int, int]:
        """
        Run all experiment runs for a single scenario with pre-generated instructions (Stage 3: CARLA again).
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID to run
            instructions: Dictionary mapping experiment_index to InstructionSet or None
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
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
                    LOG.error(f"  No instruction set available for instruction generation {generation_index}, skipping run")
                    failed += 1
                    continue
                
                # Determine number of runs and their IDs
                base_run_id = job.ground_truth_count + generation_index * instruction_count_per_generation
                total_run_count = instruction_count_per_generation
                if total_run_count > len(instruction_set.instructions):
                    LOG.warning(f"  Requested {total_run_count} instructions for instruction generation {generation_index}, but only {len(instruction_set.instructions)} available. Reducing to available count.")
                    total_run_count = len(instruction_set.instructions)

                # Run the specified number of instructions for this experiment
                for run_id in range(base_run_id, base_run_id + total_run_count):
                    instruction = instruction_set.instructions[run_id - base_run_id]
                    exp_success, exp_failed = self._run_experiment(job, scenario_id, run_id, instruction)
                    successful += exp_success
                    failed += exp_failed
            
            LOG.info(f"Scenario {scenario_id} experiment stage completed")
            
        except Exception as e:
            LOG.error(f"Scenario {scenario_id} experiment stage failed: {e}")
        
        return (successful, failed)

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
        LOG.info("=" * 80)
        LOG.info(f"Starting Job {job.id}: '{job.name}'")
        LOG.info(f"Scenarios: {len(job.scenarios)}, Ground truth runs: {job.ground_truth_count}, Instruction generations: {job.instruction_generation_count}, "
                 f"Experiment runs: {job.instruction_count_per_generation * job.instruction_generation_count}")
        LOG.info(f"Experiment nature: {job.experiment_nature.value}")
        LOG.info("=" * 80)
        
        job_start_time = time.time()
        job_successful = 0
        job_failed = 0
        
        # Set experiment nature on generator
        self.generator.set_experiment_nature(job.experiment_nature)
        
        # Stage 1: CARLA - Run all ground truths for all scenarios
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info("STAGE 1: CARLA Ground Truth Runs")
        LOG.info("=" * 80)
        # TODO: Add parallelized VLM instruction generation during ground truth runs
        for scenario_id in job.scenarios:
            gt_success, gt_failed = self._run_scenario_ground_truths(job, scenario_id)
            job_successful += gt_success
            job_failed += gt_failed
        
        # TODO: Add async VLM instruction generation during ground truth runs
        # Stage 2: VLM - Generate instructions for all experiments
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info("STAGE 2: VLM Instruction Generation")
        LOG.info("=" * 80)
        
        # Enable generator's full speed mode as the simulator is idle now (async)
        enable_full_speed_mode_task = asyncio.create_task(self.generator.enable_full_speed_mode())
        await enable_full_speed_mode_task
        
        # Generate instructions for all scenarios
        # Each generation will cluster frames from a different ground truth run (modulo ground_truth_count)
        all_instructions = {}  # Map: scenario_id -> {experiment_index -> Instruction}
        for scenario_id in job.scenarios:
            LOG.debug(f"Generating instructions for scenario {scenario_id}...")
            scenario_instructions = await self._generate_scenario_instructions(job, scenario_id)
            all_instructions[scenario_id] = scenario_instructions
        
        # Disable full speed mode once all instructions are generated
        # await enable_full_speed_mode_task
        await self.generator.disable_full_speed_mode()
        
        # Stage 3: CARLA - Run all experiments with generated instructions
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info("STAGE 3: CARLA Experiment Runs")
        LOG.info("=" * 80)
        for scenario_id in job.scenarios:
            exp_success, exp_failed = self._run_scenario_experiments(job, scenario_id, all_instructions[scenario_id])
            job_successful += exp_success
            job_failed += exp_failed
        
        # Job summary
        job_elapsed = time.time() - job_start_time
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info(f"Job {job.id} '{job.name}' completed in {job_elapsed:.1f}s ({job_elapsed/3600:.2f}h)")
        LOG.info(f"  Successful: {job_successful}, Failed: {job_failed}")
        LOG.info("=" * 80)
        
        # Print detailed results summary
        self._print_job_results_summary(job)
        
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
        adversarial_successes = 0
        
        for scenario_id in job.scenarios:
            LOG.info(f"")
            LOG.info(f"Scenario {scenario_id}:")
            LOG.info("-" * 40)
            
            # Get ground truth results (average if multiple)
            gt_penalty_scores = []
            gt_infractions_total = 0
            for run_id in range(job.ground_truth_count):
                gt_result: Optional[SimulationResult] = job.results.get((scenario_id, run_id))
                if gt_result:
                    gt_penalty_scores.append(gt_result.scores.score_penalty)
                    gt_infractions_total += gt_result.infractions.total_count
            
            if gt_penalty_scores:
                avg_gt_penalty = sum(gt_penalty_scores) / len(gt_penalty_scores)
                LOG.info(f"  Ground Truth: penalty={avg_gt_penalty:.3f} (avg of {len(gt_penalty_scores)} run(s)), infractions={gt_infractions_total}")
            else:
                avg_gt_penalty = None
                LOG.info(f"  Ground Truth: No results available")
            
            # Get experiment results and compare
            for generation_index in range(job.instruction_generation_count):
                run_id = job.ground_truth_count + generation_index
                exp_result: Optional[SimulationResult] = job.results.get((scenario_id, run_id))
                
                if exp_result:
                    total_experiments += 1
                    exp_penalty = exp_result.scores.score_penalty
                    exp_infractions = exp_result.infractions.total_count
                    
                    # Determine if adversarial (lower penalty = more infractions = adversarial success)
                    # score_penalty is a multiplier: 1.0 = no infractions, <1.0 = infractions occurred
                    is_adversarial = False
                    adversarial_marker = ""
                    if avg_gt_penalty is not None:
                        if avg_gt_penalty > exp_penalty:
                            is_adversarial = True
                            adversarial_successes += 1
                            adversarial_marker = " [ADVERSARIAL]"
                        elif avg_gt_penalty < exp_penalty:
                            adversarial_marker = " [IMPROVED]"
                    
                    # Count specific collision types for clarity
                    collisions = (
                        len(exp_result.infractions.collisions_vehicle) +
                        len(exp_result.infractions.collisions_pedestrian) +
                        len(exp_result.infractions.collisions_layout)
                    )
                    
                    LOG.info(
                        f"  Run {run_id}: penalty={exp_penalty:.3f}, "
                        f"infractions={exp_infractions}, collisions={collisions}{adversarial_marker}"
                    )
                else:
                    LOG.info(f"  Run {run_id}: No result available")
        
        # Overall summary
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info("INSTRUCTION EFFECTIVENESS SUMMARY")
        LOG.info("=" * 80)
        if total_experiments > 0:
            adversarial_rate = (adversarial_successes / total_experiments) * 100
            LOG.info(f"Total experiments: {total_experiments}")
            LOG.info(f"Adversarial successes: {adversarial_successes}")     # TODO: Also consider corrective instructions
            LOG.info(f"Adversarial rate: {adversarial_rate:.1f}%")
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
