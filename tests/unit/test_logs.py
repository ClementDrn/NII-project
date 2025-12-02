import pytest
import logging
from pathlib import Path
import tempfile
import shutil
from untitled.core.logs import init_logging


@pytest.fixture
def temp_logs_dir():
    """Create a temporary directory for logs and clean up after test."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestInitLogging:
    """Test suite for init_logging function."""

    def test_creates_logs_directory(self, temp_logs_dir):
        """Test that the logs directory is created if it doesn't exist."""
        logs_dir = temp_logs_dir / "new_logs"
        assert not logs_dir.exists()
        
        init_logging("test_logger_dir", logs_dir)
        
        assert logs_dir.exists()
        assert logs_dir.is_dir()

    def test_creates_log_file(self, temp_logs_dir):
        """Test that a log file is created with the correct name."""
        logger_name = "test_app"
        init_logging(logger_name, temp_logs_dir)
        
        log_file = temp_logs_dir / f"{logger_name}.log"
        assert log_file.exists()
        assert log_file.is_file()

    def test_returns_logger_instance(self, temp_logs_dir):
        """Test that the function returns a Logger instance."""
        logger_name = "test_logger_instance"
        logger = init_logging(logger_name, temp_logs_dir)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == logger_name

    def test_logger_level_is_debug(self, temp_logs_dir):
        """Test that the logger's level is set to DEBUG."""
        logger = init_logging("test_logger_level", temp_logs_dir)
        
        assert logger.level == logging.DEBUG

    def test_has_two_handlers(self, temp_logs_dir):
        """Test that the logger has exactly two handlers (file and console)."""
        logger = init_logging("test_logger_handlers", temp_logs_dir)
        
        assert len(logger.handlers) == 2

    def test_file_handler_configuration(self, temp_logs_dir):
        """Test that the file handler is properly configured."""
        logger = init_logging("test_logger_file", temp_logs_dir)
        
        file_handler = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                file_handler = handler
                break
        
        assert file_handler is not None
        assert file_handler.level == logging.DEBUG
        assert file_handler.formatter is not None

    def test_console_handler_configuration(self, temp_logs_dir):
        """Test that the console handler is properly configured."""
        logger = init_logging("test_logger_console", temp_logs_dir)
        
        console_handler = None
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                console_handler = handler
                break
        
        assert console_handler is not None
        assert console_handler.level == logging.INFO

    def test_debug_message_written_to_file(self, temp_logs_dir):
        """Test that DEBUG messages are written to the log file."""
        logger_name = "test_debug"
        logger = init_logging(logger_name, temp_logs_dir)
        
        # Log a debug message
        test_message = "This is a debug message"
        logger.debug(test_message)
        
        # Force flush handlers
        for handler in logger.handlers:
            handler.flush()
        
        # Read log file
        log_file = temp_logs_dir / f"{logger_name}.log"
        log_content = log_file.read_text(encoding='utf-8')
        
        assert test_message in log_content
        assert "DEBUG" in log_content

    def test_debug_message_not_in_console(self, temp_logs_dir, capsys):
        """Test that DEBUG messages are not printed to the console."""
        logger_name = "test_console_debug"
        logger = init_logging(logger_name, temp_logs_dir)
        
        # Log a debug message
        test_message = "This is a debug message for console"
        logger.debug(test_message)
        
        # Capture console output
        captured = capsys.readouterr()
        assert test_message not in captured.out


    def test_info_message_written_to_file(self, temp_logs_dir):
        """Test that INFO messages are written to the log file."""
        logger_name = "test_info"
        logger = init_logging(logger_name, temp_logs_dir)
        
        test_message = "This is an info message"
        logger.info(test_message)
        
        # Force flush handlers
        for handler in logger.handlers:
            handler.flush()
        
        log_file = temp_logs_dir / f"{logger_name}.log"
        log_content = log_file.read_text(encoding='utf-8')
        
        assert test_message in log_content
        assert "INFO" in log_content

    def test_warning_message_written_to_file(self, temp_logs_dir):
        """Test that WARNING messages are written to the log file."""
        logger_name = "test_warning"
        logger = init_logging(logger_name, temp_logs_dir)
        
        test_message = "This is a warning message"
        logger.warning(test_message)
        
        # Force flush handlers
        for handler in logger.handlers:
            handler.flush()
        
        log_file = temp_logs_dir / f"{logger_name}.log"
        log_content = log_file.read_text(encoding='utf-8')
        
        assert test_message in log_content
        assert "WARNING" in log_content

    def test_error_message_written_to_file(self, temp_logs_dir):
        """Test that ERROR messages are written to the log file."""
        logger_name = "test_error"
        logger = init_logging(logger_name, temp_logs_dir)
        
        test_message = "This is an error message"
        logger.error(test_message)
        
        # Force flush handlers
        for handler in logger.handlers:
            handler.flush()
        
        log_file = temp_logs_dir / f"{logger_name}.log"
        log_content = log_file.read_text(encoding='utf-8')
        
        assert test_message in log_content
        assert "ERROR" in log_content

    def test_initial_debug_session_message(self, temp_logs_dir):
        """Test that the initial debug session message is written to the log file."""
        logger_name = "test_session"
        init_logging(logger_name, temp_logs_dir)
        
        log_file = temp_logs_dir / f"{logger_name}.log"
        log_content = log_file.read_text(encoding='utf-8')
        
        assert "New logging session" in log_content

    def test_multiple_log_levels_hierarchy(self, temp_logs_dir):
        """Test that different log levels are handled correctly based on hierarchy."""
        logger_name = "test_hierarchy"
        logger = init_logging(logger_name, temp_logs_dir)
        
        # Log messages at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Force flush handlers
        for handler in logger.handlers:
            handler.flush()
        
        log_file = temp_logs_dir / f"{logger_name}.log"
        log_content = log_file.read_text(encoding='utf-8')
        
        # All messages should be in the file (file handler level is DEBUG)
        assert "Debug message" in log_content
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content
