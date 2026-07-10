import mlptrain.log as log_module


def test_logger_is_configured_without_root_propagation():
    logger = log_module.logger

    assert logger.name == 'mlptrain'
    assert logger.propagate is False
