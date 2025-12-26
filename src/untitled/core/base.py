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

# Logs
if os.getenv("UNTITLED_ENABLE_DEBUG_LOGS", "false").lower() in ("true", "1", "yes"):
    console_level = logging.DEBUG
else:
    console_level = logging.INFO
LOG_DIR = PROJECT_ROOT / "logs"
# Main application log file
LOG = init_logging(PROJECT_NAME, LOG_DIR, console_level=console_level)
# Intermediate prompts log file
INT_PROMPTS_LOG = init_logging("intermediate_prompts", LOG_DIR, console_level=logging.INFO)

# Configuration settings
CONFIG = load_config(PROJECT_ROOT, LOG)


# EXPORTS
__all__ = [
    "PROJECT_NAME",
    "PROJECT_ROOT",
    "CONFIG",
    "LOG",
]
