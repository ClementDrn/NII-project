from untitled.core.logs import init_logging
from untitled.core.settings import load_config

from pathlib import Path


# Project paths
PROJECT_NAME = "untitled"
# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # this file is at src/untitled/core/base.py, so go up 3 levels

# Configuration settings
CONFIG = load_config(PROJECT_ROOT)

# Logger
LOG = init_logging(PROJECT_NAME, PROJECT_ROOT / "logs")


# EXPORTS
__all__ = [
    "PROJECT_NAME",
    "PROJECT_ROOT",
    "CONFIG",
    "LOG",
]
