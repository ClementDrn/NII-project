"""Shared runtime state; importing DIGIT does not read config or create files."""

import logging
import os
from pathlib import Path

from digit.core.logs import init_logging, set_console_log_level
from digit.core.settings import Settings, load_config

PROJECT_NAME = "digit"
PROJECT_ROOT = Path(
    os.environ.get("DIGIT_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()
VENDOR_VLM_DIR = PROJECT_ROOT / "vendor" / "vlm"
VENDOR_CARLA_DIR = PROJECT_ROOT / "vendor" / "ads" / "carla"
VENDOR_LAMPILOT_DIR = PROJECT_ROOT / "vendor" / "data" / "LaMPilot" / "DbLv1"
LOG_DIR = PROJECT_ROOT / "logs"

LOG = logging.getLogger(PROJECT_NAME)
VLM_LOG = logging.getLogger("vlm")
CONFIG = Settings()
SIMULATION_FPS = 20
SKIP_SIMULATION = os.getenv("DIGIT_SKIP_SIMULATION", "false").lower() in (
    "true",
    "1",
    "yes",
)


def initialize_logging() -> None:
    """Enable file and console logging when an application is actually started."""
    init_logging(PROJECT_NAME, LOG_DIR, console_level=logging.INFO)
    init_logging("vlm", LOG_DIR, console_level=logging.INFO)
    if os.getenv("DIGIT_ENABLE_DEBUG_LOGS", "false").lower() in ("true", "1", "yes"):
        set_console_log_level(LOG, logging.DEBUG)


def initialize_runtime() -> Settings:
    """Load source-checkout configuration while preserving imported CONFIG references.

    Library consumers may instead construct Settings and pass configuration to
    their own components. Command-line entrypoints call this after parsing args,
    so --help never reads private configuration or initializes model services.
    """
    initialize_logging()
    settings = load_config(PROJECT_ROOT, LOG)
    CONFIG.__dict__.clear()
    CONFIG.__dict__.update(settings.__dict__)
    CONFIG.__pydantic_private__.clear()
    CONFIG.__pydantic_private__.update(settings.__pydantic_private__)
    return CONFIG


__all__ = [
    "PROJECT_NAME",
    "PROJECT_ROOT",
    "CONFIG",
    "LOG",
    "VLM_LOG",
    "SIMULATION_FPS",
    "SKIP_SIMULATION",
    "initialize_logging",
    "initialize_runtime",
]
