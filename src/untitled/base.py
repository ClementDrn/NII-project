from pathlib import Path
import logging
import colorlog

PROJECT_NAME = "untitled"

# Project root directory path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Create logs directory
logs_dir = PROJECT_ROOT / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)     # Ensure logs directory exists

# Configure logging to both file and console
LOG = logging.getLogger(PROJECT_NAME)
LOG.setLevel(logging.INFO)

# Log File handler
file_handler = logging.FileHandler(logs_dir / f"{PROJECT_NAME}.log", encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
file_handler.setFormatter(file_formatter)

# Log Console handler with colors
console_handler = colorlog.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s%(reset)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
console_handler.setFormatter(console_formatter)

# Add handlers to logger
LOG.addHandler(file_handler)
LOG.addHandler(console_handler)

__all__ = [
    "PROJECT_NAME",
    "PROJECT_ROOT",
    "LOG",
]
