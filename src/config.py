import os
from pathlib import Path

# TODO: refactor to YAML

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
PATH_TO_DATA = os.path.join(ROOT_DIR, 'bucket/data')
PATH_TO_FEATURES = os.path.join(ROOT_DIR, 'bucket/features')
PATH_TO_MODELS = os.path.join(ROOT_DIR, 'bucket/models')

TEST_SIZE_PCT = 0.05
LAG = 60
N_FORECAST = 10
