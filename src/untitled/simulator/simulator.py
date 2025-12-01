import os
import subprocess
import signal
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

from untitled.core.base import LOG, PROJECT_ROOT, CONFIG
from untitled.simulator.ads import ADS


class Simulator:
    """
    CARLA Simulator wrapper for running autonomous driving scenarios.
    
    This class manages CARLA simulation execution, including:
    - Starting and stopping CARLA processes
    - Running scenarios with optional text instructions
    - Retrieving logged images and metadata from simulation runs
    """
    
    def __init__(self, ads: Optional[ADS] = None):
        """
        Initialize the CARLA simulator.
        
        Args:
            ads: ADS agent configuration. If None, creates default SimLingo agent.
        """
        # Check required environment variables
        if not CONFIG.carla.path or not CONFIG.carla.username:
            raise ValueError("USERNAME and CARLA_PATH must be set in .env file")
        
        # Initialize ADS agent (default to SimLingo)
        self.ads = ads if ads is not None else ADS("simlingo")
        
        # Validate ADS installation
        if not self.ads.validate_installation():
            LOG.warning("ADS installation validation failed. Continuing anyway...")
        
        # CARLA configuration
        self.carla_path = CONFIG.carla.path
        self.username = CONFIG.carla.username
        
        # Scenario configuration
        self.route_path = self.ads.route_path
        self.seed = 1
        
        # Output directories
        self.out_root = CONFIG.carla.output_path
        self.base_dir = self.out_root / self.ads.agent_name / self.ads.benchmark / str(self.seed)
        
        # Create necessary directories
        self._setup_directories()
        
        # Get first route (hardcoded scenario)
        routes = sorted(
            [x for x in os.listdir(self.route_path) if x.endswith(".xml")],
            key=lambda x: int(x.split('_')[-1].split('.xml')[0])
        )
        if not routes:
            raise ValueError(f"No route files found in {self.route_path}")
        
        self.current_route = os.path.join(self.route_path, routes[0])
        self.route_id = routes[0].split("_")[-1][:-4].zfill(3)
        
        # Ports for CARLA
        self.port = 2000
        self.tm_port = 2500
        
        # Current run tracking
        self.run_counter = 0
        self.current_viz_path = None
        self.current_result_file = None
        self.current_log_file = None
        self.current_err_file = None
        
        LOG.info(f"Simulator initialized with route: {self.current_route}")
        LOG.info(f"Route ID: {self.route_id}")
    
    def _setup_directories(self):
        """Create necessary output directories."""
        dirs = ["run", "res", "out", "err", "viz"]
        for d in dirs:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)
    
    def _kill_carla_processes(self):
        """Kill only CARLA server processes."""
        # Kill CarlaUE4.sh processes
        try:
            carla_pids = subprocess.check_output(
                "ps aux | grep CarlaUE4 | grep -v grep | awk '{print $2}'",
                shell=True
            ).decode().split()
            for pid in carla_pids:
                LOG.info(f"Killing CARLA process {pid}")
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
                LOG.info(f"Killing leaderboard process {pid}")
                os.kill(int(pid), signal.SIGKILL)
        except subprocess.CalledProcessError:
            LOG.warning("Killing leaderboard processes failed or no processes found.")
    
    def _create_run_script(
        self,
        instruction: Optional[str] = None,
        injection_time: Optional[float] = None
    ) -> str:
        """
        Create a bash script to run the CARLA scenario.
        
        Args:
            instruction: Optional text instruction to inject into the scenario
            injection_time: Optional time (in seconds) to inject the instruction
            
        Returns:
            Path to the created bash script
        """
        self.run_counter += 1
        run_id = f"run_{self.run_counter:03d}"
        
        # Setup paths for this run
        self.current_viz_path = self.base_dir / "viz" / self.route_id / run_id
        self.current_viz_path.mkdir(parents=True, exist_ok=True)
        
        self.current_result_file = self.base_dir / "res" / f"{self.route_id}_{run_id}_res.json"
        self.current_log_file = self.base_dir / "out" / f"{self.route_id}_{run_id}_out.log"
        self.current_err_file = self.base_dir / "err" / f"{self.route_id}_{run_id}_err.log"
        job_file = self.base_dir / "run" / f"eval_{self.route_id}_{run_id}.sh"

        # Env variables to export
        export_lines = ""
        
        # Handle instruction injection if provided
        if instruction and injection_time is not None:
            # Store instruction in a temporary file
            instruction_file = self.current_viz_path / "custom_prompts.json"
            instruction_data = {
                "text": instruction,
                "flag": 2,      # 0: safety, 1: instruction following as extra prompt, 2: instruction following as override, 3: other
                "injection_time": injection_time,
                "injection_duration": 0.05  # 1 sec / 20 fps
            }
            with open(instruction_file, 'w', encoding='utf-8') as f:
                json.dump(instruction_data, f)
            LOG.info(f"Instruction saved to: {instruction_file}")

            # Prepare environment variable for instruction file
            export_lines += f"\nexport CUSTOM_PROMPTS_FILE={instruction_file}"
        
        # Get environment variables from ADS
        env_vars = self.ads.get_environment_vars(
            carla_root=self.carla_path,
            viz_path=str(self.current_viz_path)
        )
        
        # Build export statements
        export_lines += "\n".join([f"export {key}={value}" for key, value in env_vars.items()])
        
        # Get leaderboard arguments from ADS
        leaderboard_args = self.ads.get_leaderboard_args(
            route_file=self.current_route,
            result_file=str(self.current_result_file),
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
. /home/{self.username}/miniconda3/etc/profile.d/conda.sh   # WARN: will not work if anaconda is installed instead if miniconda
conda activate {self.ads.get_conda_env()}

# Debug: check which Python is being used
# echo "Python location: $(which python)" >> {self.current_log_file}
# echo "Python version: $(python --version)" >> {self.current_log_file}

cd {self.ads.agent_root}

{export_lines}

python -u {self.ads.leaderboard_evaluator} {args_str} \\
>> {self.current_log_file} 2>> {self.current_err_file}      # NOTE: Temporary change: append to logs instead of overwrite for easier debugging
"""
        
        with open(job_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return str(job_file)
    
    def run_scenario(
        self,
        instruction: Optional[str] = None,
        injection_time: Optional[float] = None
    ):
        """
        Run a CARLA scenario with the SimLingo agent.
        
        Args:
            instruction: Optional text instruction to inject into the scenario
            injection_time: Optional time (in seconds) when to inject the instruction
        """
        # Clean up any existing leaderboard processes
        # But not CARLA, as we want to keep it running between runs
        self._kill_bench2drive_leaderboard_processes()
        time.sleep(2)
        
        # NOTE: CARLA server is started automatically by the Bench2Drive leaderboard evaluator, so we don't need to start it ourselves
        
        # Create and run the scenario script
        job_file = self._create_run_script(instruction, injection_time)
        
        if instruction:
            LOG.info(f"Running scenario with instruction at t={injection_time}s: '{instruction}'")
        else:
            LOG.info(f"Running scenario without instruction")
        
        LOG.info(f"Executing job script: {job_file}")
        LOG.info(f"Output log: {self.current_log_file}")
        LOG.info(f"Error log: {self.current_err_file}")
        
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
    
    def get_image_log(self, timestep: float) -> Optional[Path]:
        """
        Retrieve the image path from the simulation logs at a specific timestep.
        
        Args:
            timestep: Time in seconds to retrieve the image from
            
        Returns:
            Path to the image file, or None if not found
        """
        if not self.current_viz_path:
            LOG.error("No scenario has been run yet. Call run_scenario() first.")
            return None
        
        # CARLA typically saves images with frame numbers
        # Assuming 20 fps: frame_index = timestep * 20
        frame_index = int(timestep * 20)
        
        # Look for image files in the viz directory
        # The exact naming convention depends on SimLingo's implementation
        # Common patterns: frame_{frame_index}.png, rgb_{frame_index}.jpg, etc.
        possible_patterns = [
            f"frame_{frame_index:06d}.png",
            f"rgb_{frame_index:06d}.png",
            f"rgb_{frame_index:06d}.jpg",
            f"{frame_index:06d}.png",
            f"frame_{frame_index}.png",
        ]
        
        for pattern in possible_patterns:
            image_path = self.current_viz_path / pattern
            if image_path.exists():
                LOG.info(f"Found image at timestep {timestep}s: {image_path}")
                return image_path
        
        # If exact frame not found, list available files
        LOG.warning(f"Image not found for timestep {timestep}s (frame {frame_index})")
        if self.current_viz_path.exists():
            available_files = list(self.current_viz_path.glob("*.png")) + \
                            list(self.current_viz_path.glob("*.jpg"))
            if available_files:
                LOG.info(f"Available image files: {[f.name for f in available_files[:5]]}")
        
        return None
    
    def get_metadata_log(self, timestep: float) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata from the simulation logs at a specific timestep.
        
        Args:
            timestep: Time in seconds to retrieve metadata from
            
        Returns:
            Dictionary containing metadata, or None if not found
        """
        if not self.current_result_file:
            LOG.error("No scenario has been run yet. Call run_scenario() first.")
            return None
        
        if not self.current_result_file.exists():
            LOG.error(f"Result file not found: {self.current_result_file}")
            return None
        
        try:
            with open(self.current_result_file, 'r', encoding='utf-8') as f:
                evaluation_data = json.load(f)
            
            # Extract relevant metadata
            # The structure depends on CARLA leaderboard output format
            frame_index = int(timestep * 20)  # Assuming 20 fps
            
            metadata = {
                "timestep": timestep,
                "frame_index": frame_index,
                "progress": evaluation_data.get("_checkpoint", {}).get("progress", []),
                "records": evaluation_data.get("_checkpoint", {}).get("records", []),
            }
            
            # Try to find frame-specific data if available
            # This depends on how SimLingo logs data
            if "frames" in evaluation_data:
                frames = evaluation_data["frames"]
                if frame_index < len(frames):
                    metadata["frame_data"] = frames[frame_index]
            
            LOG.info(f"Retrieved metadata for timestep {timestep}s")
            return metadata
            
        except json.JSONDecodeError as e:
            LOG.error(f"Failed to parse result file: {e}")
            return None
        except Exception as e:
            LOG.error(f"Error reading metadata: {e}")
            return None
    
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
