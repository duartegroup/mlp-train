import logging
import os
from typing import Optional


_LOGGER_NAME = 'mlptrain'
_HANDLER_MARKER = '_mlptrain_handler'
_FORMAT = '%(name)-12s: %(levelname)-8s %(message)s'


def _log_level() -> int:
    level_name = os.environ.get('MLT_LOG_LEVEL', default='INFO').upper()
    return getattr(logging, level_name, logging.INFO)


def _find_owned_handler(logger: logging.Logger) -> Optional[logging.Handler]:
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return handler

    return None


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_log_level())
    logger.propagate = False

    handler = _find_owned_handler(logger)
    if handler is None:
        handler = None
        try:
            import coloredlogs

            coloredlogs.install(
                level=logger.level,
                logger=logger,
                fmt=_FORMAT,
                reconfigure=False,
            )
            handler = _find_owned_handler(logger)
            if handler is None and logger.handlers:
                handler = logger.handlers[-1]
        except ImportError:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(handler)

        if handler is not None:
            setattr(handler, _HANDLER_MARKER, True)
    else:
        handler.setLevel(logger.level)

    return logger


logger = _configure_logger()
