import os
from pathlib import Path
from typing import Dict, Any

from untitled.core.base import PROJECT_ROOT, LOG


# SimLingo-specific constants
AGENT_FILE = "team_code/agent_simlingo.py"
CHECKPOINT = "outputs/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"


class ADS:
    """
    Autonomous Driving System (ADS) agent configuration.
    
    This class encapsulates the agent-specific configuration for running
    in the CARLA simulator. Currently hardcoded to SimLingo.
    """
    
    def __init__(self, agent_name: str = "simlingo"):
        """
        Initialize the ADS agent configuration.
        
        Args:
            agent_name: Name of the agent to use (default: "simlingo")
        """
        if agent_name != "simlingo":
            raise ValueError(f"Only 'simlingo' agent is currently supported, got '{agent_name}'")
        
        self.agent_name = agent_name
        self.agent_root = PROJECT_ROOT / "vendor" / "ads" / "simlingo"
        
        # SimLingo-specific configuration
        self._setup_simlingo()
        
        LOG.info(f"ADS initialized: {self.agent_name}")
        LOG.info(f"Agent root: {self.agent_root}")
    
    def _setup_simlingo(self):
        """Setup SimLingo-specific configuration."""
        # Agent file and checkpoint
        self.agent_file = str(self.agent_root / AGENT_FILE)
        self.checkpoint = str(self.agent_root / CHECKPOINT)
        
        # Benchmark configuration
        self.benchmark = "bench2drive"
        self.route_path = str(self.agent_root / "leaderboard" / "data" / "bench2drive_split")
        
        # Leaderboard and scenario runner paths
        self.leaderboard_path = str(self.agent_root / "Bench2Drive" / "leaderboard")
        self.scenario_runner_path = str(self.agent_root / "Bench2Drive" / "scenario_runner")
        self.leaderboard_evaluator = str(
            self.agent_root / "Bench2Drive" / "leaderboard" / 
            "leaderboard" / "leaderboard_evaluator.py"
        )
    
    def get_python_paths(self, carla_root: str) -> list[str]:
        """
        Get the PYTHONPATH entries required for this agent.
        
        Args:
            carla_root: Path to CARLA root directory
            
        Returns:
            List of paths to add to PYTHONPATH
        """
        return [
            f"{carla_root}/PythonAPI/carla",
            f"{carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg",
            self.leaderboard_path,
            self.scenario_runner_path,
        ]
    
    def get_environment_vars(self, carla_root: str, viz_path: str) -> Dict[str, str]:
        """
        Get environment variables required for this agent.
        
        Args:
            carla_root: Path to CARLA root directory
            viz_path: Path where visualization outputs should be saved
            
        Returns:
            Dictionary of environment variable names and values
        """
        python_paths = self.get_python_paths(carla_root)
        
        return {
            "CARLA_ROOT": carla_root,
            "PYTHONPATH": ":".join(python_paths),
            "SCENARIO_RUNNER_ROOT": self.scenario_runner_path,
            "SAVE_PATH": viz_path,
        }
    
    def get_leaderboard_args(
        self,
        route_file: str,
        result_file: str,
        seed: int,
        port: int,
        tm_port: int,
        gpu_rank: int = 0,
        timeout: int = 600,
        repetitions: int = 1,
    ) -> list[str]:
        """
        Get command-line arguments for the CARLA leaderboard evaluator.
        
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
        return [
            f"--routes={route_file}",
            f"--repetitions={repetitions}",
            "--track=SENSORS",
            f"--checkpoint={result_file}",
            f"--timeout={timeout}",
            f"--agent={self.agent_file}",
            f"--agent-config={self.checkpoint}",
            f"--traffic-manager-seed={seed}",
            f"--port={port}",
            f"--gpu-rank={gpu_rank}",
            f"--traffic-manager-port={tm_port}",
        ]
    
    def get_conda_env(self) -> str:
        """
        Get the conda environment name for this agent.
        
        Returns:
            Conda environment name
        """
        return "simlingo"
    
    def validate_installation(self) -> bool:
        """
        Validate that the agent is properly installed.
        
        Returns:
            True if agent is properly installed, False otherwise
        """
        required_paths = [
            self.agent_root,
            Path(self.agent_file),
            Path(self.route_path),
            Path(self.leaderboard_evaluator),
        ]
        
        missing = []
        for path in required_paths:
            if not path.exists():
                missing.append(str(path))
        
        if missing:
            LOG.error(f"Missing required paths for {self.agent_name}:")
            for path in missing:
                LOG.error(f"  - {path}")
            return False
        
        # Check checkpoint (warning only, as it might not exist yet)
        if not Path(self.checkpoint).exists():
            LOG.warning(f"Checkpoint not found: {self.checkpoint}")
            LOG.warning("You may need to train or download the model first")
        
        return True
