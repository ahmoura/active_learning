import logging
import os
from pathlib import Path
from typing import Union


def get_seed() -> Union[int, None]:
    seed = os.environ.get("PYHARD_SEED")
    if seed is None:
        return seed
    else:
        try:
            return int(seed)
        except ValueError:
            return None


log_file = Path(__file__).parents[2] / "graphene.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

nh = logging.NullHandler()

formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")  # - %(name)s

nh.setFormatter(formatter)

logger.addHandler(nh)
