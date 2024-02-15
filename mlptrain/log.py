import logging
import os

ll = os.environ.get('MLT_LOG_LEVEL', default='INFO')

logging.basicConfig(
    level=getattr(logging, ll),
    format='%(name)-12s: %(levelname)-8s %(message)s',
)
logger = logging.getLogger(__name__)

# Try and use colourful logs
try:
    import coloredlogs

    coloredlogs.install(level=getattr(logging, ll), logger=logger)
except ImportError:
    pass
