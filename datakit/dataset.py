from pathlib import Path
import datetime
from dataclasses import dataclass
from typing import Callable, Literal, Any

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
from datakit.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
import datakit.load as load
from datakit.utils import
app = typer.Typer()

@dataclass(frozen=True)
class Source:
    name: str
    kind: Literal["series", "dataframe"]
    data: Any            # e.g. attribute name or DataFrame
    loader: Callable = None
    structured: bool = False

@app.command()
def main(
    
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    experimenter: str = typer.Argument(..., help="Name of the experimenter"),

    input_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / f"{datetime.datetime.now().strftime('%y%m%d')}_dataset_JG.h5",
    # ----------------------------------------------
):
    logger.info("Mapping dataset...")
    loader = load.ExperimentData(input_path, verbose=False, validate=True)
    
    sources = [
        Source("meso",           "dataframe",   pd.read_pickle(r"C:/dev/hfsa-analysis/meso_means_npy.pkl")),
        Source("encoder",        "series",    "encoder",         loader=pd.read_csv),
        Source("notes",          "series",    "notes",           loader=load.read_session_notes,  structured=True),
        Source("session_config", "series",    "session_config",  loader=load.read_session_config, structured=True),
        Source("meso_meta",      "series",    "meso_metadata",   loader=load.camera_metadata),
        Source("pupil_meta",     "series",    "pupil_metadata",  loader=load.camera_metadata),
        Source("pupil",          "series",    "dlc_pupil",       loader=load.deeplabcut_pickle),
    ]

    logger.info("Registering dataset into memmory...")
    for src in tqdm(sources, desc="Registering sources"):
        if src.kind == "series":
            series = getattr(loader.data, src.data)
            loader.register_series(
                src.name, series,
                loader_func=src.loader,
                structured=src.structured
            )
        else:  # dataframe
            loader.register_dataframe(src.name, src.data)  
            

    # Load everything into one unified df
    
    full_df = loader.load_all()

    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
