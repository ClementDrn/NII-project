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
        self.agent_file = self.agent_root / AGENT_FILE
        self.checkpoint = self.agent_root / CHECKPOINT
    
    def get_agent_python_path(self) -> str:
        """
        Get the PYTHONPATH entry for this agent.
        
        Returns:
            Path to add to PYTHONPATH for agent imports
        """
        return str(self.agent_root)
    
    def get_agent_args(self) -> list[str]:
        """
        Get agent-specific command-line arguments.
        
        Returns:
            List of agent-specific command-line arguments
        """
        return [
            f"--agent={str(self.agent_file)}",
            f"--agent-config={str(self.checkpoint)}",
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
            self.agent_file,
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
        if not self.checkpoint.exists():
            LOG.warning(f"Checkpoint not found: {self.checkpoint}")
            LOG.warning("You may need to train or download the model first")
        
        return True
