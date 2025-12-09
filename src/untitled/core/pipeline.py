import time
from untitled.core.base import LOG, PROJECT_ROOT, CONFIG
from untitled.core.job_handler import Job, JobHandler
from untitled.simulator.simulator import Simulator
from untitled.simulator.simulation_data import SimulationData
from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
from untitled.generator.multi_step_vlm_instruction_generator import MultiStepVLMInstructionGenerator
from untitled.simulator.simulation_data import SimulationFrameData


class Pipeline:
    """
    Pipeline to run simulation with VLM-generated instructions.
    
    This pipeline orchestrates job execution, running multiple scenarios
    with both ground truth (baseline) and experiment (VLM-instructed) runs.
    """

    def __init__(self):
        """Initialize pipeline components."""
        self.simulator = None
        self.generator = None
        self.job_handler = None

    def init(self):
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
        elif CONFIG.generator.design == "multi_step_vlm":
            self.generator = MultiStepVLMInstructionGenerator()
        else:
            raise ValueError(f"Unknown generator design: {CONFIG.generator.design}")

    def stop(self):
        """
        Stop and free all pipeline components.
        """
        # Free simulator
        if self.simulator:
            self.simulator.free()
            self.simulator = None

        # Free VLM
        if self.generator:
            self.generator.free()
            self.generator = None

    def _run_ground_truth(self, job: Job, scenario_id: int, run_id: int):
        """
        Run a single ground truth (baseline) scenario without instructions.
        
        Args:
            job: The job being executed
            scenario_id: The ID of the scenario being executed
            run_id: The run index (0-based)
            ground_truth_count: Total number of ground truth runs
            
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

    def _run_experiment(self, job: Job, scenario_id: int, run_id: int):
        """
        Run a single experiment with VLM-generated instruction.
        
        This involves:
        1. Fetching image and metadata from ground truth runs
        2. Generating instruction with VLM
        3. Running scenario again with instruction
        
        Args:
            job: The job being executed
            scenario_id: The ID of the scenario being executed
            run_id: The experiment run index (0-based)
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        try:
            LOG.info(f"  Experiment run {run_id} ({run_id - job.ground_truth_count + 1} / {job.experiment_count})")
            
            # Step 1: Get image and metadata from previously collected ground truth
            # WARN: Ground truth run index is hardcoded to 0 for now. Later, we may want to select the ground truth based on how the current experiment run is performing.
            ground_truth_index = 0
            LOG.debug(f"    Step 1: Fetching images and metadata from ground truth run {ground_truth_index}...")
            image_path_list = self.simulator.get_image_logs(job.id, scenario_id, ground_truth_index)
            metadata_list = self.simulator.get_metadata_logs(job.id, scenario_id, ground_truth_index)
            
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
            
            # Step 2: Generate instruction with VLM
            LOG.debug(f"    Step 2: Generating instruction with VLM...")
            simulation_data = SimulationData(
                frame_count=len(image_path_list),
                image_dir=image_path_list[0].parent,
                frames=[SimulationFrameData(image_name=img_path.name, metadata=metadata_list[i])
                        for i, img_path in enumerate(image_path_list)]
            )
            
            instruction_set = self.generator.generate_instructions_from_images(simulation_data)
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

            # Step 3: Run scenario again with instruction
            LOG.debug(f"    Step 3: Running scenario with instruction...")
            self.simulator.run_scenario(top_instruction)
            
            LOG.info(f"    Experiment run {run_id} completed")
            return (1, 0)
            
        except Exception as e:
            LOG.error(f"    Experiment run {run_id} failed: {e}")
            return (0, 1)

    def _run_scenario(self, job: Job, scenario_id):
        """
        Run a single scenario with all its ground truth and experiment runs.
        
        Args:
            job: The job being executed
            scenario_id: The scenario ID to run
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        LOG.info("-" * 80)
        LOG.info(f"Job {job.id} | Scenario {scenario_id}")
        LOG.info("-" * 80)
        
        successful = 0
        failed = 0
        
        try:
            # Set job and scenario context for directory structure
            self.simulator.set_job_context(job.id, scenario_id)
            self.simulator.set_scenario(scenario_id)

            run_index = 0
            run_total = job.ground_truth_count + job.experiment_count
            
            # Run ground truth (baseline) runs
            LOG.info(f"Running {job.ground_truth_count} ground truth run(s)...")
            while run_index < job.ground_truth_count:
                gt_success, gt_failed = self._run_ground_truth(job, scenario_id, run_index)
                successful += gt_success
                failed += gt_failed
                run_index += 1
            
            # Run experiment runs with VLM-generated instructions
            LOG.info(f"Running {job.experiment_count} experiment run(s)...")
            while run_index < run_total:
                exp_success, exp_failed = self._run_experiment(job, scenario_id, run_index)
                successful += exp_success
                failed += exp_failed
                run_index += 1
            
            LOG.info(f"Scenario {scenario_id} completed")
            
        except Exception as e:
            LOG.error(f"Scenario {scenario_id} failed: {e}")
            LOG.error(f"Skipping remaining runs for this scenario")
        
        return (successful, failed)

    def _run_job(self, job: Job):
        """
        Run a single job with all its scenarios.
        
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
        
        # Iterate through all scenarios in this job
        for scenario_id in job.scenarios:
            scenario_success, scenario_failed = self._run_scenario(job, scenario_id)
            job_successful += scenario_success
            job_failed += scenario_failed
        
        # Job summary
        job_elapsed = time.time() - job_start_time
        LOG.info("=" * 80)
        LOG.info(f"Job {job.id} '{job.name}' completed in {job_elapsed:.1f}s")
        LOG.info(f"  Successful: {job_successful}, Failed: {job_failed}")
        LOG.info("=" * 80)
        
        return (job_successful, job_failed)

    def start(self):
        """
        Start the pipeline execution.
        
        Executes all jobs defined in jobs.yaml, running each scenario with:
        1. Ground truth runs (baseline without instructions)
        2. Experiment runs (with VLM-generated instructions)
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
                job_successful, job_failed = self._run_job(job)
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

