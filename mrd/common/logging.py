import logging
from pathlib import Path

LOG_FORMAT = "[%(name)s] %(message)s"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def set_logger(filename=None, level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.StreamHandler()]

    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="w"))

    logging.basicConfig(
        format=LOG_FORMAT,
        datefmt=TIME_FORMAT,
        level=level,
        handlers=handlers,
    )

    logger.info(f"Logger initialized. Output file: {filename}")
