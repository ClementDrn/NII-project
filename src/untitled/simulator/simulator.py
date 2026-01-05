import os
import subprocess
import signal
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from itertools import chain

from untitled.core.base import LOG, PROJECT_ROOT, CONFIG, SIMULATION_FPS
from untitled.simulator.ads import ADS
from untitled.simulator.simulation_data import SimulationMetadata, load_metadata
from untitled.generator.instruction import Instruction


# Benchmark configuration (Bench2Drive)
BENCHMARK_ROOT = PROJECT_ROOT / "vendor" / "ads" / "simlingo" / "Bench2Drive"
ROUTES_ROOT = PROJECT_ROOT / "vendor" / "ads" / "simlingo" / "leaderboard" / "data" / "bench2drive_split"

# Simulator paths
IMAGES_DIR_NAME = "images"
METADATA_DIR_NAME = "metadata"
VIZ_DIR_NAME = "viz"
LOG_OUT_FILE_NAME = "out.log"
LOG_ERR_FILE_NAME = "err.log"
LOG_RES_FILE_NAME = "res.json"
SCRIPT_RUN_FILE_NAME = "run.sh"
INSTRUCTION_FILE_NAME = "custom_prompts.json"


class Simulator:
    """
    CARLA Simulator wrapper for running autonomous driving scenarios.
    
    This class manages CARLA simulation execution, including:
    - Starting and stopping CARLA processes
    - Running scenarios with optional text instructions
    - Retrieving logged images and metadata from simulation runs
    """
    
    def __init__(self, ads: Optional[ADS] = None, debug: bool = None):
        """
        Initialize the CARLA simulator.
        
        Args:
            ads: ADS agent configuration. If None, creates default SimLingo agent.
            debug: Enable debug output (images with visualization). If None, uses config setting.
        """
        # Check required environment variables
        if not CONFIG.carla.path or not CONFIG.carla.conda_path:
            raise ValueError("CONDA_PATH and CARLA_PATH must be set in .env file")
        
        # Set debug mode (parameter overrides config)
        self.debug = debug if debug is not None else CONFIG.carla.debug
        
        # Initialize ADS agent (default to SimLingo)
        self.ads = ads if ads is not None else ADS("simlingo")
        
        # Validate ADS installation
        if not self.ads.validate_installation():
            LOG.warning("ADS installation validation failed. Continuing anyway...")
        
        # CARLA configuration
        self.carla_path: Path = Path(CONFIG.carla.path)
        self.conda_path: Path = Path(CONFIG.carla.conda_path)
        self.frame_rate: int = SIMULATION_FPS    # fps
        
        # Benchmark configuration (Bench2Drive)
        self.benchmark = "bench2drive"
        self.leaderboard_path = BENCHMARK_ROOT / "leaderboard"
        self.scenario_runner_path = BENCHMARK_ROOT / "scenario_runner"
        self.leaderboard_evaluator = (
            BENCHMARK_ROOT / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
        )
        
        # Scenario configuration
        self.route_path = ROUTES_ROOT
        self.seed = 1
        
        # Validate benchmark installation
        self._validate_benchmark_installation()
        
        # Output directories
        self.out_root = CONFIG.carla.output_path
        self.base_dir = self.out_root / self.ads.agent_name / self.benchmark / str(self.seed)
        
        # Get first route (hardcoded scenario)
        routes = sorted(
            [x for x in os.listdir(self.route_path) if x.endswith(".xml")],
            key=lambda x: int(x.split('_')[-1].split('.xml')[0])
        )
        if not routes:
            raise ValueError(f"No route files found in {self.route_path}")
        
        self.current_route = self.route_path / routes[0]
        self.route_id_str = routes[0].split("_")[-1][:-4].zfill(3)
        
        # Ports for CARLA
        self.port = 2000
        self.tm_port = 2500
        
        # Current run tracking
        self.run_counter = 0
        self.current_viz_path = None
        self.current_result_file = None
        self.current_log_file = None
        self.current_err_file = None
        self.current_image_dir = None
        self.current_metadata_dir = None
        
        # Job and scenario context tracking
        self.current_job_id = None
        self.current_scenario_id = None
        
        LOG.info(f"Simulator initialized with route: {self.current_route}")
        LOG.info(f"Route ID: {self.route_id_str}")
    
    def _validate_benchmark_installation(self):
        """Validate that the benchmark is properly installed."""
        required_paths = [
            self.route_path,
            self.leaderboard_evaluator,
            self.scenario_runner_path,
        ]
        
        missing = []
        for path in required_paths:
            if not path.exists():
                missing.append(str(path))
        
        if missing:
            LOG.error(f"Missing required benchmark paths:")
            for path in missing:
                LOG.error(f"  - {path}")
            raise ValueError("Benchmark installation validation failed")
        
        LOG.info("Benchmark installation validated successfully")
    
    def _calculate_scenario_base_dir(self, base_dir: Path, job_id: int, scenario_id: int) -> Path:
        """
        Calculate the base directory for a given scenario.
        
        Args:
            scenario_id: The scenario ID number
            job_id: The job ID number
        Returns:
            Path to the base directory
        """        
        job_dir = f"job_{job_id:03d}"
        sce_dir = f"sce_{scenario_id:03d}"
        return base_dir / job_dir / sce_dir
    
    def _calculate_run_base_dir(self, base_dir: Path, job_id: int, scenario_id: int, run_id: int) -> Path:
        """
        Calculate the base directory for a given job, scenario, and run.
        
        Args:
            base_dir: The root base directory
            job_id: The job ID number
            scenario_id: The scenario ID number
            run_id: The run ID number
        Returns:
            Path to the base directory
        """
        run_dir = f"run_{run_id:03d}"
        return self._calculate_scenario_base_dir(base_dir, job_id, scenario_id) / run_dir

    def set_job_context(self, job_id: int, scenario_id: int):
        """
        Set the current job and scenario context for directory structure.
        
        This method should be called before running scenarios to establish
        the proper directory hierarchy: job_XXX/sce_XXX/run_XXX
        
        Args:
            job_id: The job ID number
            scenario_id: The scenario ID number
        """
        self.current_job_id = job_id
        self.current_scenario_id = scenario_id
        LOG.debug(f"Set job context: job_{job_id:03d}/sce_{scenario_id:03d}")
    
    def set_scenario(self, scenario_id: int):
        """
        Set the current scenario to run.
        
        This method dynamically loads the route file for the specified scenario
        and resets the run counter.
        
        Args:
            scenario_id: The scenario ID (0-219 for Bench2Drive)
            
        Raises:
            ValueError: If scenario_id is out of valid range or file not found
        """
        # Validate scenario ID range
        if not (0 <= scenario_id <= 219):
            raise ValueError(f"Scenario ID {scenario_id} out of valid range (0-219)")
        
        # Build route file path
        # NOTE: In SimLingo's dataset, scenario IDs are zero-padded to 2 digits
        route_file = self.route_path / f"bench2drive_{scenario_id:02d}.xml"
        
        # Validate route file exists
        if not route_file.exists():
            raise ValueError(f"Route file not found: {route_file}")
        
        # Update current route
        self.current_route = route_file
        self.route_id_str = str(scenario_id).zfill(3)
        self.current_scenario_id = scenario_id
        
        # Reset run counter for new scenario
        self.run_counter = 0
        
        LOG.info(f"Set scenario to {scenario_id}: {route_file.name}")
    
    def _get_benchmark_python_paths(self, carla_root: Path) -> list[Path]:
        """Get PYTHONPATH entries for benchmark and CARLA.
        
        Args:
            carla_root: Path to CARLA root directory
            
        Returns:
            List of paths for benchmark infrastructure and CARLA
        """
        return [
            carla_root / "PythonAPI" / "carla",
            carla_root / "PythonAPI" / "carla" / "dist" / "carla-0.9.15-py3.7-linux-x86_64.egg",
            self.leaderboard_path,
            self.scenario_runner_path,
        ]
    
    def _build_environment_vars(self, viz_path: Path, img_path: Path, metadata_path: Path) -> Dict[str, str]:
        """Build environment variables for scenario execution.
        
        Args:
            viz_path: Path where visualization outputs should be saved
            img_path: Path where images should be saved
            metadata_path: Path where metadata should be saved
            
        Returns:
            Dictionary of environment variable names and values
        """
        # Get agent Python path
        agent_path = str(self.ads.get_agent_python_path())
        
        # Get benchmark and CARLA Python paths and cast to string
        benchmark_paths = self._get_benchmark_python_paths(self.carla_path)
        benchmark_paths = [str(p) for p in benchmark_paths]
        
        # Combine all Python paths (agent first for proper imports)
        python_paths = [agent_path] + benchmark_paths
        
        return {
            "CARLA_ROOT": str(self.carla_path),
            "PYTHONPATH": ":".join(python_paths),
            "SCENARIO_RUNNER_ROOT": str(self.scenario_runner_path),
            "SAVE_PATH": viz_path,
            "SAVE_PATH_IMG": img_path,
            "SAVE_PATH_METADATA": metadata_path,
            "DEBUG": "true" if self.debug else "false",
        }
    
    def _build_leaderboard_args(
        self,
        route_file: Path,
        result_file: Path,
        seed: int,
        port: int,
        tm_port: int,
        gpu_rank: int = 0,
        timeout: int = 600,
        repetitions: int = 1,
    ) -> list[str]:
        """Build command-line arguments for the CARLA leaderboard evaluator.
        
        Args:
            route_file: Path to the route XML file
            result_file: Path where results JSON will be saved
            seed: Random seed for traffic manager
            port: CARLA server port
            tm_port: Traffic manager port
            gpu_rank: GPU rank to use (default: 0)
            timeout: Timeout in seconds (default: 600)
            repetitions: Number of repetitions (default: 1)
            
        Returns:
            List of command-line arguments
        """
        # Get agent-specific arguments
        agent_args = self.ads.get_agent_args()
        
        # Build benchmark-specific arguments
        benchmark_args = [
            f"--routes={str(route_file)}",
            f"--repetitions={repetitions}",
            "--track=SENSORS",
            f"--checkpoint={str(result_file)}",
            f"--timeout={timeout}",
            f"--traffic-manager-seed={seed}",
            f"--port={port}",
            f"--gpu-rank={gpu_rank}",
            f"--traffic-manager-port={tm_port}",
        ]
        
        # Combine agent args with benchmark args
        return benchmark_args + agent_args
    
    def _kill_carla_processes(self):
        """Kill only CARLA server processes."""
        # Kill CarlaUE4.sh processes
        try:
            carla_pids = subprocess.check_output(
                "ps aux | grep CarlaUE4 | grep -v grep | awk '{print $2}'",
                shell=True
            ).decode().split()
            for pid in carla_pids:
                LOG.debug(f"Killing CARLA process {pid}")
                os.kill(int(pid), signal.SIGKILL)
        except subprocess.CalledProcessError:
            LOG.warning("Killing CARLA processes failed or no processes found.")

    def _kill_bench2drive_leaderboard_processes(self):
        """Kill only Bench2Drive leaderboard processes."""
        # Kill leaderboard_evaluator.py processes
        try:
            leaderboard_pids = subprocess.check_output(
                "ps aux | grep leaderboard_evaluator.py | grep -v grep | awk '{print $2}'",
                shell=True
            ).decode().split()
            for pid in leaderboard_pids:
                LOG.debug(f"Killing leaderboard process {pid}")
                os.kill(int(pid), signal.SIGKILL)
        except subprocess.CalledProcessError:
            LOG.warning("Killing leaderboard processes failed or no processes found.")
    
    def _create_run_script(
        self,
        instruction: Optional[Instruction] = None
    ) -> str:
        """
        Create a bash script to run the CARLA scenario.
        
        Args:
            instruction: Optional Instruction object to inject into the scenario
            
        Returns:
            Path to the created bash script
        """
        if self.current_job_id is None or self.current_scenario_id is None:
            LOG.error("Job and scenario context must be set before creating run script.")
            raise ValueError("Job and scenario context must be set before creating run script.")
        
        # Increment run counter
        run_id = self.run_counter
        self.run_counter += 1

        # Build directory structure based on job/scenario context
        run_base_dir = self._calculate_run_base_dir(self.base_dir, self.current_job_id, self.current_scenario_id, run_id)

        # Setup paths for this run
        # Directories
        run_base_dir.mkdir(parents=True, exist_ok=True)
        self.current_viz_path = run_base_dir / VIZ_DIR_NAME
        self.current_viz_path.mkdir(parents=True, exist_ok=True)
        self.current_image_dir = run_base_dir / IMAGES_DIR_NAME
        self.current_image_dir.mkdir(parents=True, exist_ok=True)
        self.current_metadata_dir = run_base_dir / METADATA_DIR_NAME
        self.current_metadata_dir.mkdir(parents=True, exist_ok=True)
        # Files
        self.current_result_file = run_base_dir / LOG_RES_FILE_NAME
        self.current_log_file = run_base_dir / LOG_OUT_FILE_NAME
        self.current_err_file = run_base_dir / LOG_ERR_FILE_NAME
        job_file = run_base_dir / SCRIPT_RUN_FILE_NAME

        # Env variables to export
        export_lines = ""
        
        # Handle instruction injection if provided
        if instruction is not None:
            # Store instruction in a temporary file
            instruction_file = run_base_dir / INSTRUCTION_FILE_NAME
            instruction_data = {
                "text": instruction.adversarial_instruction,
                "flag": 2,      # 0: safety, 1: instruction following as extra prompt, 2: instruction following as override, 3: other
                "injection_time": instruction.injection_window.start_timestamp,
                "injection_duration": instruction.injection_window.end_timestamp - instruction.injection_window.start_timestamp,
            }
            with open(instruction_file, 'w', encoding='utf-8') as f:
                json.dump([instruction_data], f)    # Store as array of instructions
            LOG.debug(f"Instruction saved to: {instruction_file}")

            # Prepare environment variable for instruction file
            export_lines += f"\nexport CUSTOM_PROMPTS_FILE={instruction_file}"
        
        # Build environment variables
        env_vars = self._build_environment_vars(self.current_viz_path, self.current_image_dir, self.current_metadata_dir)
        
        # Build export statements
        export_lines += "\n" + "\n".join([f"export {key}={value}" for key, value in env_vars.items()])
        
        # Build leaderboard arguments
        leaderboard_args = self._build_leaderboard_args(
            route_file=self.current_route,
            result_file=self.current_result_file,
            seed=self.seed,
            port=self.port,
            tm_port=self.tm_port,
            gpu_rank=0,
            timeout=600,
            repetitions=1,
        )
        
        # Format arguments for bash script
        args_str = " \\\n".join(leaderboard_args)
        
        # Create bash script
        script_content = f"""#!/bin/bash
# Deactivate any active virtual environment
deactivate 2>/dev/null || true

source ~/.bashrc
. {str(self.conda_path)}/etc/profile.d/conda.sh
conda activate {self.ads.get_conda_env()}

# Debug: check which Python is being used
# echo "Python location: $(which python)" >> {self.current_log_file}
# echo "Python version: $(python --version)" >> {self.current_log_file}

cd {self.ads.agent_root}

{export_lines}

python -u {str(self.leaderboard_evaluator)} {args_str} \\
> {self.current_log_file} 2> {self.current_err_file}
"""
        
        with open(job_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return str(job_file)
    
    def run_scenario(
        self,
        instruction: Optional[Instruction] = None
    ):
        """
        Run a CARLA scenario with the SimLingo agent.
        
        Args:
            instruction: Optional Instruction object to inject into the scenario
        """
        # Clean up any existing leaderboard processes
        # TODO: CARLA is also killed. We may want to keep it running between runs to save time.
        self._kill_bench2drive_leaderboard_processes()
        time.sleep(2)
        
        # NOTE: CARLA server is started automatically by the Bench2Drive leaderboard evaluator, so we don't need to start it ourselves
        # NOTE: The evaluator is started by SimLingo right now. In the future, we may want to start CARLA and the evaluator ourselves in order to choose what agent to use.
        
        # Create and run the scenario script
        job_file = self._create_run_script(instruction)
        
        if instruction:
            LOG.info(f"Running scenario with instruction at t={instruction.injection_window.start_timestamp}s: '{instruction.adversarial_instruction}'")
        else:
            LOG.info(f"Running scenario without instruction")
        
        LOG.debug(f"Executing job script: {job_file}")
        LOG.debug(f"Output log: {self.current_log_file}")
        LOG.debug(f"Error log: {self.current_err_file}")
        
        # Run the scenario with a clean environment (no virtual env interference)
        # Use bash -l to get a login shell with clean PATH from system defaults
        result = subprocess.run(
            ["bash", "-l", job_file],
            capture_output=False,
            env={
                'HOME': os.environ.get('HOME', ''),
                'USER': os.environ.get('USER', ''),
                'LOGNAME': os.environ.get('LOGNAME', ''),
                'SHELL': os.environ.get('SHELL', '/bin/bash'),
                'TERM': os.environ.get('TERM', 'xterm'),
                'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            }
        )
        
        if result.returncode != 0:
            LOG.error(f"Scenario execution failed with return code: {result.returncode}")
            LOG.error(f"Check error log: {self.current_err_file}")
        else:
            LOG.debug("Scenario execution completed")
        
        # Clean up leaderboard processes after run (keep CARLA running)
        time.sleep(2)
        self._kill_bench2drive_leaderboard_processes()
    
    def get_image_log(self, job_id: int, scenario_id: int, run_id: int, timestamp: float, run_base_dir: Path = None) -> Optional[Path]:
        """
        Retrieve the image path from the simulation logs at a specific timestamp.
        
        Args:
            job_id: The ID of the job that was executed
            scenario_id: The ID of the scenario that was executed
            run_id: The ID of the run that was executed
            timestamp: Time in seconds to retrieve the image from
            run_base_dir: Optional directory to override the run base directory
            
        Returns:
            Path to the image file, or None if not found
        """
        # Calculate run base directory if it was not provided
        if run_base_dir is None:
            run_base_dir = self._calculate_run_base_dir(self.base_dir, job_id, scenario_id, run_id)

        # Check if the image directory exists
        image_dir = run_base_dir / IMAGES_DIR_NAME
        if not image_dir.exists():
            LOG.error(f"Image directory does not exist: {image_dir}")
            return None
        
        # CARLA typically saves images with frame numbers
        frame_index = int(timestamp * self.frame_rate)
        
        # Common patterns for images
        # The exact naming convention depends on the agent's implementation
        possible_patterns = [
            f"{frame_index:06d}.png",
        ]
        
        # Look for image files in the image directory
        image_path = None
        for pattern in possible_patterns:
            image_path_candidate = image_dir / pattern
            if image_path_candidate.exists():
                LOG.debug(f"Found image at timestamp {timestamp}s: {image_path_candidate}")
                image_path = image_path_candidate
                break
        # If exact frame not found, list available files
        else:
            LOG.warning(f"Image not found for timestamp {timestamp}s (frame {frame_index})")
            LOG.debug(f"Tried fetching files: {[str(image_dir / p) for p in possible_patterns]}")
            available_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
            if available_files:
                LOG.debug(f"Sample of available image files: {[f.name for f in available_files[:5]]}")
        
        return image_path

    def get_image_logs(self, job_id: int, scenario_id: int, run_id: int, run_base_dir: Path = None, frames_per_minute: int = 0) -> List[Path]:
        """
        Retrieve all image paths from the simulation logs for a specific run.
        
        Args:
            job_id: The ID of the job that was executed
            scenario_id: The ID of the scenario that was executed
            run_id: The ID of the run that was executed
            run_base_dir: Optional directory to override the run base directory
            frames_per_minute: Optional number of logs to fetch for each minute of simulation (defaults to 0, for all frames)

        Returns:
            List of Paths to the image files
        """
        # Calculate run base directory if it was not provided
        if run_base_dir is None:
            run_base_dir = self._calculate_run_base_dir(self.base_dir, job_id, scenario_id, run_id)
        
        # Check if the image directory exists
        image_dir = run_base_dir / IMAGES_DIR_NAME
        if not image_dir.exists():
            LOG.error(f"Image directory does not exist: {image_dir}")
            return []
        
        # Retrieve all image files (assuming .png and .jpg formats)
        image_files = chain(image_dir.glob("*.png"), image_dir.glob("*.jpg"))
        
        # Sort by filename without extension. BUT: "5.png" should come before "10.png", so we sort by integer value
        sorted_image_files = sorted(image_files, key=lambda path: int(path.stem))

        if not sorted_image_files:
            LOG.warning(f"No image files found in directory: {image_dir}")
            return []
        else:
            LOG.debug(f"Found {len(sorted_image_files)} image files in directory: {image_dir}")
        
        # Undersampling: Filter out frames if frames_per_minute is specified
        if frames_per_minute > 0 and sorted_image_files and len(sorted_image_files) > 1:            
            # Calculate total log frames per minute
            first_frame_step = int(sorted_image_files[0].stem)
            second_frame_step = int(sorted_image_files[1].stem)
            actual_frame_rate = self.frame_rate / (second_frame_step - first_frame_step)
            log_frame_total_per_minute = actual_frame_rate * 60

            # Apply undersampling if needed
            if frames_per_minute > log_frame_total_per_minute:
                LOG.warning(f"Requested frames_per_minute {frames_per_minute} exceeds total frames per minute {log_frame_total_per_minute}. Returning all frames.")
            else:
                interval = log_frame_total_per_minute / frames_per_minute
                sorted_image_files = [
                    img for idx, img in enumerate(sorted_image_files) if idx % int(interval) == 0
                ]
                LOG.debug(f"Filtered image files to {len(sorted_image_files)} using frames_per_minute={frames_per_minute}")

        return sorted_image_files
    
    def get_metadata_log(self, job_id: int, scenario_id: int, run_id: int, timestamp: float, run_base_dir: Path = None) -> Optional[SimulationMetadata]:
        """
        Retrieve metadata from the simulation logs at a specific timestamp.
        
        Args:
            job_id: The ID of the job that was executed
            scenario_id: The ID of the scenario that was executed
            run_id: The ID of the run that was executed
            timestamp: Time in seconds to retrieve metadata from
            run_base_dir: Optional directory to override the run base directory
            
        Returns:
            Dictionary containing metadata, or None if not found
        """
        # Calculate run base directory if it was not provided
        if run_base_dir is None:
            run_base_dir = self._calculate_run_base_dir(self.base_dir, job_id, scenario_id, run_id)
        
        # Check if the metadata directory exists
        metadata_dir = run_base_dir / METADATA_DIR_NAME
        if not metadata_dir.exists():
            LOG.error(f"Metadata directory does not exist: {metadata_dir}")
            return None

        # Possible metadata file patterns
        frame_index = int(timestamp * self.frame_rate)
        possible_patterns = [
            f"{frame_index:06d}.json",
        ]
        
        # Look for metadata files in the metadata directory
        metadata_path = None
        for pattern in possible_patterns:
            metadata_path_candidate = metadata_dir / pattern
            if metadata_path_candidate.exists():
                LOG.debug(f"Found metadata at timestamp {timestamp}s: {metadata_path_candidate}")
                metadata_path = metadata_path_candidate
                break
        # If exact frame not found, list available files
        else:
            LOG.warning(f"Metadata not found for timestamp {timestamp}s (frame {frame_index})")
            LOG.debug(f"Tried fetching files: {[str(metadata_dir / p) for p in possible_patterns]}")
            available_files = list(metadata_dir.glob("*.json"))
            if available_files:
                LOG.debug(f"Sample of available metadata files: {[f.name for f in available_files[:5]]}")

        return metadata_path

    def get_metadata_logs(self, job_id: int, scenario_id: int, run_id: int, run_base_dir: Path = None, frames_per_minute: int = 0) -> List[SimulationMetadata]:
        """
        Retrieve all metadata from the simulation logs for a specific run.
        
        Args:
            job_id: The ID of the job that was executed
            scenario_id: The ID of the scenario that was executed
            run_id: The ID of the run that was executed
            run_base_dir: Optional directory to override the run base directory
            frames_per_minute: Optional number of logs to fetch for each minute of simulation (defaults to 0, for all frames)

        Returns:
            List of SimulationMetadata objects
        """
        # Calculate run base directory if it was not provided
        if run_base_dir is None:
            run_base_dir = self._calculate_run_base_dir(self.base_dir, job_id, scenario_id, run_id)
        
        # Check if the metadata directory exists
        metadata_dir = run_base_dir / METADATA_DIR_NAME
        if not metadata_dir.exists():
            LOG.error(f"Metadata directory does not exist: {metadata_dir}")
            return []
        
        # Retrieve all metadata files (assuming .json format)
        metadata_files = metadata_dir.glob("*.json")

        # Sort by filename without extension. BUT: "5.json" should come before "10.json", so we sort by integer value
        sorted_metadata_files = sorted(metadata_files, key=lambda path: int(path.stem))
        
        if not sorted_metadata_files:
            LOG.warning(f"No metadata files found in directory: {metadata_dir}")
            return []
        else:
            LOG.debug(f"Found {len(sorted_metadata_files)} metadata files in directory: {metadata_dir}")
        
        # Undersampling: Filter out frames if frames_per_minute is specified
        if frames_per_minute > 0 and sorted_metadata_files and len(sorted_metadata_files) > 1:            
            # Calculate total log frames per minute
            first_frame_step = int(sorted_metadata_files[0].stem)
            second_frame_step = int(sorted_metadata_files[1].stem)
            actual_frame_rate = self.frame_rate / (second_frame_step - first_frame_step)
            log_frame_total_per_minute = actual_frame_rate * 60

            # Apply undersampling if needed
            if frames_per_minute > log_frame_total_per_minute:
                LOG.warning(f"Requested frames_per_minute {frames_per_minute} exceeds total frames per minute {log_frame_total_per_minute}. Returning all metadata.")
            else:
                interval = log_frame_total_per_minute / frames_per_minute
                sorted_metadata_files = [
                    md for idx, md in enumerate(sorted_metadata_files) if idx % int(interval) == 0
                ]
                LOG.debug(f"Filtered metadata files to {len(sorted_metadata_files)} using frames_per_minute={frames_per_minute}")
        
        # Load metadata from files
        metadata_list = []
        for metadata_file in sorted_metadata_files:
            metadata = load_metadata(metadata_file)
            metadata_list.append(metadata)
        
        LOG.debug(f"Loaded {len(metadata_list)} metadata entries from directory: {metadata_dir}")
        
        return metadata_list

    def free(self):
        """Clean up any running processes."""
        LOG.info("Cleaning up simulator processes...")
        
        # Kill leaderboard processes first
        self._kill_bench2drive_leaderboard_processes()
        
        # Give leaderboard's atexit handler a moment to cleanup CARLA
        time.sleep(2)

        # Then kill any remaining CARLA processes
        self._kill_carla_processes()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.free()
        except:
            pass
