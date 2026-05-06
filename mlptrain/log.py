import logging
import os
from typing import Optional

_LOGGING_FORMAT = '%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'
_LOGGER_NAME = 'mlptrain'
_HANDLER_MARKER = '_mlptrain_handler'

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
    try:
        return getattr(logging, level_name)
    except AttributeError:
        print(f'Invalid value of MLT_LOG_LEVEL: "{level_name}"')
        print('Falling back to INFO level')
        return logging.INFO


def _find_owned_handler(logger: logging.Logger) -> Optional[logging.Handler]:
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return handler

    return None


def _new_handler(logger: logging.Logger) -> logging.Handler:
    before_handlers = list(logger.handlers)

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
        for handler in logger.handlers:
            if handler not in before_handlers:
                return handler
    except ImportError:
        pass

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(_LOGGING_FORMAT, datefmt=_DATE_FORMAT)
    )
    logger.addHandler(handler)
    return handler


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_log_level())
    logger.propagate = False

    handler = _find_owned_handler(logger)
    if handler is None:
        handler = _new_handler(logger)
        setattr(handler, _HANDLER_MARKER, True)

    handler.setLevel(logger.level)

    return logger


logger = _configure_logger()
