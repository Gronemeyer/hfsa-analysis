from pathlib import Path
from typing import Dict, Tuple
import re
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

'''
https://bids-website.readthedocs.io/en/latest/getting_started/folders_and_files/folders.html

Experiment/
├──
processed/
├──
data/ 
└── subject
    └── session
        └── datatype
            └── data

Experiment/
├──
data/
└── sub-01
    └── ses-01
        └── func
            └── YYYMMDD_HHMMSS_sub-01_ses-01_meso.tiff
'''

SUBJECT_REGEX = re.compile(r"sub-([A-Za-z0-9]+)")
SESSION_REGEX = re.compile(r"ses-([A-Za-z0-9]+)")

GLOB_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "meso.ome.tiff": ("raw", "meso_tiff"),
    "meso.ome.tiff_frame_metadata.json": ("raw", "meso_metadata"),
    "pupil.ome.tiff": ("raw", "pupil_tiff"),
    "pupil.ome.tiff_frame_metadata.json": ("raw", "pupil_metadata"),
    "treadmill_data.csv": ("raw", "encoder",),
    "*.psydat": ("raw", "psydat",),
    "configuration.csv": ("raw", "session_config",),
    "full.pickle": ("processed", "dlc_pupil"),
    "meso-mean-trace.csv": ("processed", "meso_trace"),
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
