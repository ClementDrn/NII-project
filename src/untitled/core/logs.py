from pathlib import Path
import logging
import colorlog


def init_logging(name: str, logs_dir: Path, console_level = logging.INFO) -> logging.Logger:
    """
    Initialize logging for the application.
    """
    # Create logs directory
    logs_dir.mkdir(parents=True, exist_ok=True)     # Ensure logs directory exists

    # Configure logging to both file and console
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG so file handler can capture all levels
    
    # Clear any existing handlers to prevent duplication
    logger.handlers.clear()

    # Log File handler
    file_handler = logging.FileHandler(logs_dir / f"{name}.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)

    # Log Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(console_level)
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
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Print initial log message
    logger.debug("--- New session started ----------------------------------------------------")

    return logger
