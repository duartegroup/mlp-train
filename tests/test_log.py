import importlib
import logging

import mlptrain.log as log_module


def test_logger_is_configured_without_root_propagation():
    logger = log_module.logger

    assert logger.name == 'mlptrain'
    assert logger.propagate is False


def test_logger_configuration_is_idempotent():
    logger = log_module.logger
    initial_handlers = list(logger.handlers)
    initial_owned = [
        handler
        for handler in logger.handlers
        if getattr(handler, '_mlptrain_handler', False)
    ]

    reloaded = importlib.reload(log_module)
    reloaded_logger = reloaded.logger
    reloaded_owned = [
        handler
        for handler in reloaded_logger.handlers
        if getattr(handler, '_mlptrain_handler', False)
    ]

    assert reloaded_logger is logger
    assert len(initial_owned) == 1
    assert reloaded_logger.handlers == initial_handlers
    assert len(reloaded_owned) == 1
    assert isinstance(reloaded_owned[0], logging.Handler)
