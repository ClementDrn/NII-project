import asyncio
from untitled.core.base import LOG, PROJECT_ROOT, CONFIG, SIMULATION_FPS
from untitled.core.job_handler import Job, JobHandler
from untitled.generator.instruction_generator import InstructionGenerator
from untitled.simulator.simulator import Simulator
from untitled.simulator.simulation_data import SimulationData
from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
from untitled.generator.multi_step_vlm_instruction_generator import MultiStepVLMInstructionGenerator
from untitled.simulator.simulation_data import SimulationFrameData
from untitled.generator.instruction import Instruction
from untitled.filter.filter import cluster_frames
from untitled.filter.embedder import DINOEmbedder, Embedder

from typing import Dict, List, Tuple
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
        # Skip ground truth run if its logs are already cached
        if self.job_handler.is_ground_truth_cached(job.id, scenario_id, run_id):
            LOG.info(f"  Ground truth run {run_id} is already cached. Skipping.")
            return (1, 0)
        try:
            LOG.info(f"  Ground truth run {run_id} ({run_id + 1} / {job.ground_truth_count})")
            self.simulator.run_scenario()
            LOG.info(f"    Ground truth run {run_id} completed")
            return (1, 0)
        except Exception as e:
            LOG.error(f"    Ground truth run {run_id} failed: {e}")
            return (0, 1)
        
    async def _cluster_frames_for_scenario(self, job: Job, scenario_id: int) -> SimulationData:
        """
        Cluster frames from ground truth runs for a scenario to reduce redundancy.
        
        Args:
            job: The job being executed
            scenario_id: The ID of the scenario being executed
        
        Returns:
            Filtered SimulationData with clustered frames
        """
        # Get ground truth cache directory if it exists 
        # WARN: Ground truth run index is hardcoded to 0 for now
        ground_truth_index = 0
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
            frames=[cluster.frames[0] for cluster in clusters]
        )
        LOG.debug(f"    Reduced {simulation_data.frame_count} frames to {filtered_simulation_data.frame_count} frames using similarity threshold {threshold}")

        return filtered_simulation_data

    async def _generate_instruction_for_experiment(self, job: Job, scenario_id: int, experiment_index: int, simulation_data: SimulationData) -> Tuple[bool, Instruction]:
        """
        Generate VLM instruction for a single experiment (async, queues the prompt).
        
        Args:
            job: The job being executed
            scenario_id: The ID of the scenario being executed
            experiment_index: The experiment index (0-based, relative to experiments only)
            simulation_data: The simulation data containing frames and metadata
            
        Returns:
            Tuple of (success, instruction or None)
        """
        try:
            LOG.info(f"  Generating instruction for experiment {experiment_index} ({experiment_index + 1} / {job.experiment_count})")
            
            LOG.debug(f"    Generating instructions from images...")
            instruction_set = await self.generator.generate_instructions_from_images(simulation_data)
            if not instruction_set:
                LOG.error(f"    Failed to generate instruction from VLM")
                raise RuntimeError("VLM instruction generation failed")
            
            # Get the top-ranked instruction
            top_instruction = instruction_set.get_top_instruction()
            if not top_instruction:
                LOG.error(f"    No valid instructions in generated set")
                raise RuntimeError("No valid instructions generated")
            
            LOG.debug(f"    Generated instruction set with {len(instruction_set.instructions)} test case(s)")
            LOG.debug(f"    Using top-ranked instruction (rank {top_instruction.rank}): '{top_instruction.adversarial_instruction}'")
            LOG.debug(f"    Target: {top_instruction.target_module_logic}")
            LOG.debug(f"    Injection window: {top_instruction.injection_window.start_timestamp}s - {top_instruction.injection_window.end_timestamp}s")
            
            return (True, top_instruction)
            
        except Exception as e:
            LOG.error(f"    Instruction generation for experiment {experiment_index} failed: {e}")
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
            LOG.info(f"  Experiment run {run_id} ({experiment_index + 1} / {job.experiment_count})")
            
            if instruction is None:
                LOG.error(f"    No instruction available for this experiment")
                return (0, 1)
            
            # Run scenario with instruction
            LOG.debug(f"    Running scenario with instruction...")
            self.simulator.run_scenario(instruction)
            
            LOG.info(f"    Experiment run {run_id} completed")
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

    async def _generate_scenario_instructions(self, job: Job, scenario_id: int, simulation_data: SimulationData) -> Dict[int, Instruction]:
        """
        Generate VLM instructions for all experiments in a scenario (Stage 2: VLM only).
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID
            simulation_data: The simulation data containing frames and metadata
            
        Returns:
            Dictionary mapping experiment_index to Instruction
        """
        LOG.info("-" * 80)
        LOG.info(f"Job {job.id} | Scenario {scenario_id} | VLM Instruction Generation Stage")
        LOG.info("-" * 80)
        
        instructions = {}
        
        try:
            # Set job and scenario context
            self.simulator.set_job_context(job.id, scenario_id)
            self.simulator.set_scenario(scenario_id)
            
            # Generate instructions for all experiments
            LOG.info(f"Generating instructions for {job.experiment_count} experiment(s)...")
            for experiment_index in range(job.experiment_count):
                success, instruction = await self._generate_instruction_for_experiment(job, scenario_id, experiment_index, simulation_data)
                if success:
                    instructions[experiment_index] = instruction
                else:
                    instructions[experiment_index] = None
            
            LOG.info(f"Scenario {scenario_id} instruction generation completed")
            
        except Exception as e:
            LOG.error(f"Scenario {scenario_id} instruction generation failed: {e}")
        
        return instructions

    def _run_scenario_experiments(self, job: Job, scenario_id: int, instructions: Dict[int, Instruction]) -> Tuple[int, int]:
        """
        Run all experiment runs for a single scenario with pre-generated instructions (Stage 3: CARLA again).
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID to run
            instructions: Dictionary mapping experiment_index to Instruction
            
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
            
            # Run experiment runs with pre-generated instructions
            LOG.info(f"Running {job.experiment_count} experiment run(s)...")
            for experiment_index in range(job.experiment_count):
                run_id = job.ground_truth_count + experiment_index
                instruction = instructions.get(experiment_index)
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
        LOG.info(f"Scenarios: {len(job.scenarios)}, Ground truth runs: {job.ground_truth_count}, Experiment runs: {job.experiment_count}")
        LOG.info("=" * 80)
        
        job_start_time = time.time()
        job_successful = 0
        job_failed = 0
        
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

        # Filter/Cluster frames before generating instructions
        # TODO: Parallelize this with simulation runs. We can cluster frames together as they arrive since the clustering is greedy.
        simulation_data_list: List[SimulationData] = []
        for scenario_id in job.scenarios:
            LOG.debug(f"Clustering frames for scenario {scenario_id} before instruction generation...")
            data = self._cluster_frames_for_scenario(job, scenario_id)
            simulation_data_list.append(data)
        
        # Enable generator's full speed mode as the simulator is idle now (async)
        enable_full_speed_mode_task = asyncio.create_task(self.generator.enable_full_speed_mode())
        await enable_full_speed_mode_task
        
        # Generate instructions for all scenarios
        all_instructions = {}  # Map: scenario_id -> {experiment_index -> Instruction}
        for scenario_id in job.scenarios:
            LOG.debug(f"Generating instructions for scenario {scenario_id}...")
            simulation_data = simulation_data_list[job.scenarios.index(scenario_id)]
            scenario_instructions = await self._generate_scenario_instructions(job, scenario_id, simulation_data)
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
        
        return (job_successful, job_failed)

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
