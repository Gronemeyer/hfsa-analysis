from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loguru import logger
from tqdm import tqdm
import typer

from datakit.config import (
    FIGURES_DIR, 
    PROCESSED_DATA_DIR
)

app = typer.Typer()
def plot_single_session_trace(
    df_all,
    subject,
    session,
    source='meso',
    trace_key='meso_means',
    time_key='TimeReceivedByCore',
    notes_source='notes',
    show_notes_text=False,
    show_summary_stats=True,
    skip=2,
    figsize=(10, 4),
    analysis_source='Analysis',
    ax=None
):
    """
    Plots a single session trace from df_all. Optionally includes:
    - Session notes
    - Summary statistics
    - Event markers (if precomputed)
    """
    idx = (subject, session)

    # Setup axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    try:
        # Get trace and time
        trace = np.array(df_all.loc[idx, (source, trace_key)])[skip:]
        time = pd.to_datetime(df_all.loc[idx, (source, time_key)])[skip:]
        time_from_start = (time - time[0]).total_seconds()

        ax.plot(time_from_start, trace, color='black', linewidth=1.2)

        # Plot events if stored
        if ('Analysis', 'EventFrames') in df_all.columns:
            event_frames = df_all.loc[idx, ('Analysis', 'EventFrames')]
            if isinstance(event_frames, (list, np.ndarray)):
                ax.scatter(time_from_start[event_frames], trace[event_frames],
                           color='red', s=20, label='Events')

        # Plot notes if available
        if notes_source in df_all.columns.get_level_values(0):
            notes_df = df_all.loc[idx, (notes_source, 'dataframe')]
            if isinstance(notes_df, pd.DataFrame):
                for _, note in notes_df.iterrows():
                    t_note = pd.to_datetime(note['timestamp'])
                    offset = (t_note - time[0]).total_seconds()
                    if time_from_start[0] <= offset <= time_from_start[-1]:
                        ax.axvline(offset, color='blue', linestyle='--', alpha=0.4)
                        if show_notes_text:
                            ax.text(offset, ax.get_ylim()[1] * 0.9, note['note'],
                                    rotation=90, fontsize=8, color='blue')

        # Show analysis summary (MeanZ, AUC, etc.)
        if show_summary_stats and analysis_source in df_all.columns.get_level_values(0):
            try:
                summary = df_all.loc[idx, analysis_source]
                text = "\n".join([
                    f"Mean: {summary['MeanZ']:.2f}",
                    f"SD: {summary['StdZ']:.2f}",
                    f"Events: {summary['NumEvents']}",
                    f"Peak: {summary['PeakAmplitude']:.2f}",
                    f"AUC: {summary['AUC']:.2f}",
                    f"Dur: {summary['MeanEventDuration']:.2f}s"
                ])
                ax.text(0.02, 0.98, text, transform=ax.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception:
                pass

        ax.set_title(f"{subject} - {session}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z-Scored Intensity")

        return fig, ax

    except Exception as e:
        ax.set_title(f"{subject} - {session} (Error)")
        ax.text(0.5, 0.5, str(e), ha='center', va='center', fontsize=8)
        return fig, ax

def generate_html_report(df_all, subject, output_dir, loader):
    
    import os
    import json
    from jinja2 import Environment, FileSystemLoader

    
    os.makedirs(output_dir, exist_ok=True)
    env = Environment(loader=FileSystemLoader(searchpath='.'))
    template = env.get_template(r'reports/templates/report_template.html')

    sessions = []
    subject_df = df_all.loc[df_all.index.get_level_values('Subject') == subject]

    for session in subject_df.index.get_level_values('Session').unique():
        idx = (subject, session)
        safe_session = str(session).replace(" ", "_")

        # === Plot trace
        fig_path = os.path.join(output_dir, f"{subject}_{safe_session}_trace.png")
        plot_single_session_trace(
            df_all,
            subject,
            session,
            source='meso',
            trace_key='meso_tiff',
            time_key='TimeReceivedByCore',
            notes_source='notes',
            show_notes_text=False,
            show_summary_stats=True,
            skip=2,
            figsize=(10, 4),
            analysis_source='Analysis',
            ax=None
        )

        plt.savefig(fig_path)
        plt.close()

        # === Session notes
        notes = []
        try:
            notes_df = df_all.loc[idx, ('notes', 'dataframe')]
            if isinstance(notes_df, pd.DataFrame):
                notes = notes_df.to_dict(orient='records')
        except KeyError:
            pass

        # === Summary stats
        stats_html = "<p>No analysis available.</p>"
        try:
            stats_df = df_all.loc[[idx], 'Analysis']
            stats_html = stats_df.to_html(classes='table table-striped', index=False)
        except Exception:
            pass

        sessions.append({
            "session": session,
            "plot_path": os.path.basename(fig_path),
            "notes": notes,
            "stats_table": stats_html
        })

    # === Provenance
    provenance_text = json.dumps(loader.provenance.log, indent=2)

    html = template.render(
        subject=subject,
        sessions=sessions,
        provenance=provenance_text
    )

    report_path = os.path.join(output_dir, f"{subject}_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ HTML report written to: {report_path}")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "250408_HFSA_data.h5",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):  
    
    from datakit.load import ExperimentData
    data = ExperimentData(input_path, verbose=True, validate=False)
    logger.info("Generating plot from data...")
    generate_html_report(data.processed, subject='STREHAB05', output_dir=output_path, loader=data)

    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")


if __name__ == "__main__":
    app()
