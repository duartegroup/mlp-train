import logging
import os

_LOGGING_FORMAT = '%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'

# Print year with only two digits
_DATE_FORMAT = '%y-%m-%d %H:%M:%S'

# Users can specify the logging level by setting the MLT_LOG_LEVEL environment variable before
# running the python program. e.g. to only print warning message:
#
#     $ export MLT_LOG_LEVEL=WARNING
#
# Valid logging levels are defined as attributes of the logging module, see:
# https://docs.python.org/3/library/logging.html#logging-levels
ll = os.environ.get('MLT_LOG_LEVEL', default='INFO')
try:
    _level = getattr(logging, ll)
except AttributeError:
    _level = logging.INFO
    print(f'Invalid value of MLT_LOG_LEVEL: "{ll}"')
    print('Falling back to INFO level')

logging.basicConfig(level=_level, format=_LOGGING_FORMAT, datefmt=_DATE_FORMAT)

logger = logging.getLogger('mlptrain')
# Try and use colourful logs
try:
    import coloredlogs

    coloredlogs.install(
        level=_level, logger=logger, fmt=_LOGGING_FORMAT, datefmt=_DATE_FORMAT
    )
except ImportError:
    pass
