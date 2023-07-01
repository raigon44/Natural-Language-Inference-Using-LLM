import logging


def config_logging():
    return logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


config_logging()
logger = logging.getLogger(__name__)
