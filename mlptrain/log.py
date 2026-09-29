import logging
import os

_LOGGING_FORMAT = '%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'
_LOGGER_NAME = 'mlptrain'

# Print year with only two digits
_DATE_FORMAT = '%y-%m-%d %H:%M:%S'


# Users can specify the logging level by setting the MLT_LOG_LEVEL environment variable before
# running the python program. e.g. to only print warning message:
#
#     $ export MLT_LOG_LEVEL=WARNING
#
# Valid logging levels are defined as attributes of the logging module, see:
# https://docs.python.org/3/library/logging.html#logging-levels
def _log_level() -> int:
    level_name = os.environ.get('MLT_LOG_LEVEL', default='INFO').upper()
    allowed_levels = (
        'NOTSET',
        'DEBUG',
        'INFO',
        'WARNING',
        'ERROR',
        'CRITICAL',
    )
    if level_name not in allowed_levels:
        print(f'Invalid value of MLT_LOG_LEVEL: "{level_name}"')
        print('Falling back to INFO level')
        return logging.INFO

    return getattr(logging, level_name)


def _add_handler(logger: logging.Logger) -> None:
    # Configuring an already configured logger would duplicate every message
    if logger.handlers:
        return

    # Try and use colourful logs
    try:
        import coloredlogs

        coloredlogs.install(
            level=logger.level,
            logger=logger,
            fmt=_LOGGING_FORMAT,
            datefmt=_DATE_FORMAT,
            reconfigure=False,
        )
        return
    except ImportError:
        pass

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(_LOGGING_FORMAT, datefmt=_DATE_FORMAT)
    )
    handler.setLevel(logger.level)
    logger.addHandler(handler)


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_log_level())
    logger.propagate = False
    _add_handler(logger)

    return logger


logger = _configure_logger()
