import logging
import logging.handlers
import re
import sys
import types

import pytest

import mlptrain.log as log_module

_ENV_VAR = 'MLT_LOG_LEVEL'


@pytest.fixture
def throwaway_logger():
    """A logger that is not registered in the global logging manager."""
    logger = logging.Logger('mlptrain-test')
    logger.setLevel(logging.WARNING)
    return logger


@pytest.fixture
def without_coloredlogs(monkeypatch):
    """Make ``import coloredlogs`` raise ImportError.

    A None entry in sys.modules is treated by the import machinery as a
    failed import, so this exercises the plain StreamHandler fallback even
    when coloredlogs is installed (as it is in CI).
    """
    monkeypatch.setitem(sys.modules, 'coloredlogs', None)


@pytest.fixture
def stub_coloredlogs(monkeypatch):
    """Replace coloredlogs with a stub that installs a marker handler.

    A distinct handler is created per call because Logger.addHandler silently
    ignores a handler it already holds. Stubbing rather than skipping keeps
    this branch covered in environments without coloredlogs installed.
    """
    module = types.ModuleType('coloredlogs')

    def install(**kwargs):
        kwargs['logger'].addHandler(logging.NullHandler())

    module.install = install  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, 'coloredlogs', module)


def test_log_level_defaults_to_info(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)

    assert log_module._log_level() == logging.INFO


@pytest.mark.parametrize(
    'value, expected',
    [
        ('debug', logging.DEBUG),
        ('INfo', logging.INFO),
        ('WARNING', logging.WARNING),
        ('erroR', logging.ERROR),
        ('Critical', logging.CRITICAL),
    ],
)
def test_log_level_reads_env_var(monkeypatch, value, expected):
    monkeypatch.setenv(_ENV_VAR, value)

    assert log_module._log_level() == expected


# BASIC_FORMAT exists as an attribute of logging but is a string, not a level
@pytest.mark.parametrize('value', ['NONSENSE', 'BASIC_FORMAT', '', '25'])
def test_log_level_bad_value_falls_back_to_info(monkeypatch, capsys, value):
    monkeypatch.setenv(_ENV_VAR, value)

    assert log_module._log_level() == logging.INFO

    stdout = capsys.readouterr().out
    assert value in stdout
    assert 'INFO' in stdout


def test_add_handler_uses_coloredlogs_when_available(
    throwaway_logger, stub_coloredlogs
):
    log_module._add_handler(throwaway_logger)

    # coloredlogs owns the handler, and we must not add a second one
    assert len(throwaway_logger.handlers) == 1
    assert isinstance(throwaway_logger.handlers[0], logging.NullHandler)


def test_add_handler_falls_back_to_stream_handler(
    throwaway_logger, without_coloredlogs
):
    log_module._add_handler(throwaway_logger)

    assert len(throwaway_logger.handlers) == 1
    handler = throwaway_logger.handlers[0]

    assert isinstance(handler, logging.StreamHandler)
    assert handler.level == throwaway_logger.level


def test_add_handler_formats_records(
    throwaway_logger, without_coloredlogs, capsys
):
    log_module._add_handler(throwaway_logger)

    throwaway_logger.warning('a message')

    # StreamHandler defaults to stderr
    written = capsys.readouterr().err

    assert 'mlptrain-test' in written
    assert 'WARNING' in written
    assert 'a message' in written

    # The date is prefixed and the year is printed with only two digits
    assert re.match(r'^\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ', written)


def test_add_handler_is_idempotent(throwaway_logger, without_coloredlogs):
    """Re-configuring a logger must not duplicate its messages."""
    log_module._add_handler(throwaway_logger)
    log_module._add_handler(throwaway_logger)

    assert len(throwaway_logger.handlers) == 1


def test_logger_is_configured_once():
    assert len(log_module.logger.handlers) == 1


def test_logger_records_do_not_reach_root_handlers():
    """Records must be emitted exactly once, not also via the root logger."""
    root_probe = logging.handlers.BufferingHandler(capacity=100)
    mlp_probe = logging.handlers.BufferingHandler(capacity=100)

    root_logger = logging.getLogger()
    root_logger.addHandler(root_probe)
    log_module.logger.addHandler(mlp_probe)

    try:
        # CRITICAL so that the record is emitted whatever MLT_LOG_LEVEL is set to
        log_module.logger.critical('a message from the mlptrain logger')
    finally:
        root_logger.removeHandler(root_probe)
        log_module.logger.removeHandler(mlp_probe)

    assert len(mlp_probe.buffer) == 1
    assert (
        mlp_probe.buffer[0].getMessage()
        == 'a message from the mlptrain logger'
    )
    assert root_probe.buffer == []
