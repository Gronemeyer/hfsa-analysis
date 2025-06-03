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

GLOB_PATTERNS: Dict[str, str] = {
    "meso.ome.tiff": "meso_tiff",
    "meso.ome.tiff_frame_metadata.json": "meso_metadata",
    "pupil.ome.tiff": "pupil_tiff",
    "pupil.ome.tiff_frame_metadata.json": "pupil_metadata",
    "treadmill_data.csv": "encoder",
    "*.psydat": "psydat",
    "configuration.csv": "session_config",
    "full.pickle": "dlc_pupil",
    "meso-mean-trace.csv": "meso_mean",
    "notes.txt": "notes",
}

METRIC_MAP = {
    'pupil_diameter': {
        'series':   lambda ds, sbj, sess: ds.pupil.pupil_diameter_mm.loc[(sbj, sess)][21:],
        'stats': {
            'Mean Pupil Diameter (mm)': lambda s: s.mean(),
            'Std Pupil Diameter (mm)':  lambda s: s.std()
        },
        'label':        'Pupil Diameter',
        'unit':         'mm',
        'color':        'blue',
        'mean_col':     'Mean Pupil Diameter (mm)',
        'std_col':      'Std Pupil Diameter (mm)'
    },
    'speed': {
        'series':   lambda ds, sbj, sess: ds.encoder.speed.loc[(sbj, sess)][21:]/10,
        'stats': {
            'Mean Speed (cm/s)': lambda s: s.mean(),
            'Std Speed (cm/s)':  lambda s: s.std()
        },
        'label':        'Speed',
        'unit':         'cm/s',
        'color':        'green',
        'mean_col':     'Mean Speed (cm/s)',
        'std_col':      'Std Speed (cm/s)'
    },
    'distance': {
        'series':   lambda ds, sbj, sess: ds.encoder.distance.loc[(sbj, sess)][21:],
        'stats': {
            'Total Distance (cm)': lambda s: (s.max() - s.min()) / 10
        },
        'label':        'Total Distance',
        'unit':         'cm',
        'color':        'red',
        'mean_col':     'Total Distance (cm)',
        'std_col':      None          # no std for distance
    },
    'meso_fluorescence': {
        'series':   lambda ds, sbj, sess: ds.meso.meso_tiff.loc[(sbj, sess)][21:],
        'stats': {
            'Mean Widefield Calcium Fluorescence': lambda s: s.mean(),
            'Std Widefield Calcium Fluorescence':  lambda s: s.std()
        },
        'label':        'Widefield Calcium Fluorescence',
        'unit':         'a.u.',
        'color':        'purple',
        'mean_col':     'Mean Widefield Calcium Fluorescence',
        'std_col':      'Std Widefield Calcium Fluorescence'
    }
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
