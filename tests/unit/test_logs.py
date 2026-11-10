import logging

import pytest

from digit.core.logs import init_logging, set_console_log_level


@pytest.fixture
def logger_factory(tmp_path, request):
    name = request.node.name
    logger = logging.getLogger(name)
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.propagate = False

    def create(console_level=logging.INFO):
        return init_logging(name, tmp_path / 'logs', console_level=console_level)

    yield create, tmp_path / 'logs' / f'{name}.log'
    for handler in logger.handlers:
        handler.close()
    logger.handlers, logger.level, logger.propagate = previous


def test_initializes_file_and_console_handlers(logger_factory):
    create, path = logger_factory
    logger = create()
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2
    handlers = {handler.name: handler for handler in logger.handlers}
    assert isinstance(handlers['file'], logging.FileHandler)
    assert handlers['file'].level == logging.DEBUG
    assert handlers['console'].level == logging.INFO
    assert 'New session started' in path.read_text()


@pytest.mark.parametrize(('level', 'visible'), [
    (logging.DEBUG, False), (logging.INFO, True),
    (logging.WARNING, True), (logging.ERROR, True),
])
def test_file_captures_all_levels_and_console_respects_threshold(logger_factory, capsys, level, visible):
    create, path = logger_factory
    logger = create()
    message = 'distinctive test message'
    logger.log(level, message)
    assert message in path.read_text()
    assert logging.getLevelName(level) in path.read_text()
    assert (message in capsys.readouterr().err) is visible


def test_changing_console_level_keeps_file_logging_debug(logger_factory, capsys):
    create, path = logger_factory
    logger = create()
    set_console_log_level(logger, logging.ERROR)
    logger.warning('file only warning')
    logger.error('visible error')
    output = capsys.readouterr().err
    assert 'file only warning' not in output
    assert 'visible error' in output
    assert 'file only warning' in path.read_text()
    assert next(handler for handler in logger.handlers if handler.name == 'file').level == logging.DEBUG
