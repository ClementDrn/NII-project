from untitled.core.logs import init_logging
from untitled.core.settings import load_config

import os
import logging
from pathlib import Path


# Project paths
PROJECT_NAME = "untitled"
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # this file is at src/untitled/core/base.py, so go up 3 levels
VENDOR_VLM_DIR = PROJECT_ROOT / "vendor" / "vlm"
VENDOR_CARLA_DIR = PROJECT_ROOT / "vendor" / "ads" / "carla"
VENDOR_LAMPILOT_DIR = PROJECT_ROOT / "vendor" / "data" / "LaMPilot" / "DbLv1"

# Logs directory
LOG_DIR = PROJECT_ROOT / "logs"

# Main application log file
if os.getenv("UNTITLED_ENABLE_DEBUG_LOGS", "false").lower() in ("true", "1", "yes"):
    console_level = logging.DEBUG
else:
    console_level = logging.INFO
LOG = init_logging(PROJECT_NAME, LOG_DIR, console_level=console_level)

# Log file for VLM requests and responses, since they can be large
VLM_LOG = init_logging("vlm", LOG_DIR, console_level=logging.INFO)

# Configuration settings
CONFIG = load_config(PROJECT_ROOT, LOG)

# Important general constants
# WARN: 20 FPS is the only supported framerate at the moment. SimLingo hardcodes Carla's FPS to 20 too. 
SIMULATION_FPS = 20  # Frames per second for simulation


# EXPORTS
__all__ = [
    "PROJECT_NAME",
    "PROJECT_ROOT",
    "CONFIG",
    "LOG",
    "VLM_LOG",
    "SIMULATION_FPS",
]
