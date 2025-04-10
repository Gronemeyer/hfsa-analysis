from pathlib import Path
import datetime

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

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    experimenter: str = typer.Argument(..., help="Name of the experimenter"),

    input_path: Path = RAW_DATA_DIR / "20250408_HFSA_data.h5",
    output_path: Path = PROCESSED_DATA_DIR / f"{datetime.datetime.now().strftime('%y%m%d')}_dataset_JG.h5",
    # ----------------------------------------------
):

    loader = load.ExperimentData(input_path, verbose=True, validate=False)

    # Register sources
    loader.register_series("encoder", loader.encoder, loader_func=pd.read_csv)
    loader.register_series("meso", loader.meso_metadata, loader_func=load.camera_metadata)
    loader.register_series("pupil", loader.pupil_metadata, loader_func=load.camera_metadata)
    loader.register_dataframe("meso", pd.read_pickle("meso_means_npy.pkl"))
    loader.register_series("session_config", loader.session_config, loader_func=load.read_session_config)
    loader.register_series("notes", loader.notes, loader_func=load.read_session_notes, structured=True)
    
    @log_analysis(
        name="calcium_summary",
        provenance_logger=loader.provenance,
        source="meso",
        description="Computes calcium event summary statistics",
        parameters={"prominence": 1.5, "skip": 2}
    )
    def calcium_summary_analysis(df_meso):
        traces = analysis.get_normalized_meso_traces(df_meso, trace_key='meso_tiff', time_key='TimeReceivedByCore', skip=2)
        return analysis.compute_trace_summary_stats(traces)

    loader.register_analysis(name="calcium_summary", source="meso", function=calcium_summary_analysis)

    # Load everything into one unified df
    df_all = loader.load_all()

    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
