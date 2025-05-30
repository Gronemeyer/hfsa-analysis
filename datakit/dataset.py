from pathlib import Path
import datetime
from dataclasses import dataclass
from typing import Callable, Literal, Any

from loguru import logger
from tqdm import tqdm
import typer

from IPython import embed

import pandas as pd
import io
from datakit.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
import datakit.load as load
from datakit.utils import DatasetProxy, log_analysis

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
    #experimenter: str = typer.Argument(..., help="Name of the experimenter"),

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
        Source("pupil",          "series",    "dlc_pupil",       loader=load.deeplabcut_pickle, structured=True),
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
            
    @log_analysis(
        name="dlc_pupil_means",
        provenance_logger=loader.provenance,
        source="pupil",
        description="Computes mean pupil size from DLC output",
        parameters={"confidence_threshold": 0.7,
                    "pixel_to_mm": 53.6}
    )
    def dlc_pupil_analysis(df_dlc_pupil):
        from datakit.proc import analyze_pupil_data
        df = analyze_pupil_data(df_dlc_pupil, confidence_threshold=0.7, pixel_to_mm=53.6)
        return df

    loader.register_analysis(name="dlc_pupil_means", source="pupil", function=dlc_pupil_analysis)
    # Load everything into one unified df
    
    full_df = loader.load_all()

    logger.success("Processing dataset complete.")
    full_df.to_hdf(r'processed\250507_HFSA_data.h5', 'HFSA')
    logger.info("Loading IPython terminal...")
    
    # Capture the DataFrame info into a string
    buf = io.StringIO()
    full_df.info(buf=buf)
    buf.seek(0)
    info_str = buf.read()

    dataset = DatasetProxy(full_df)

    # Launch IPython with a banner showing columns and dtypes
    embed(
        header=(
            "<<[[[[[=-=-=_{HFSA Dataset Analysis Terminal}_=-=-=]]]]]>>\n\n"
            "Dataset Information:\n"
            f"{info_str}\n"
            "Type `dataset.` + <TAB> to explore."
        ),
        local={
            'dataset': dataset,
            'pd': pd,
        },
    )
    # -----------------------------------------


if __name__ == "__main__":
    app()
