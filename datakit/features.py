from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from datakit.config import PROCESSED_DATA_DIR
import datakit.load as load

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    loader = load.ExperimentData(input_path, verbose=True, validate=False)

    # -----------------------------------------


if __name__ == "__main__":
    app()
