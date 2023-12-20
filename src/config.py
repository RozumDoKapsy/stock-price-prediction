import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
PATH_TO_RAW_DATA = os.path.join(ROOT_DIR, 'data/raw')
