#%% IMPORTS

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bokeh
#%%
#  LOAD DATA FROM THE H5 FILE
from datakit.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
import os
dataset =pd.read_hdf(r'/Users/jakegronemeyer/hfsa-analysis/data/processed/250506_HFSA_data.h5', key='HFSA')
# path = os.environ.get('DATASET')
# dataset = pd.read_hdf(path, key='HFSA')
#%%

def print_dataset_info(dataset):
    """
    Print basic information about the dataset.
    """
    print(f"Dataset shape: {dataset.shape}")
    print(f"Columns: {dataset.columns.tolist()}")
    print(f"Index: {dataset.index.names}")
    print(f"Data types:\n{dataset.dtypes}")
    
def set_timestamps_to_datetime(dataset):
    """
    Convert timestamp columns to datetime format.
    
    ```python
    Subject    Session
    STREHAB02  '01'         DatetimeIndex(['2025-03-24 16:57:24.456272', '...
    STREHAB03  '01'         DatetimeIndex(['2025-03-25 17:34:53.303604', '...
    ```

    """
    dataset[('meso_meta', 'TimeReceivedByCore')] = pd.to_datetime(
        dataset[('meso_meta', 'TimeReceivedByCore')]
    )
    dataset[('pupil_meta', 'TimeReceivedByCore')] = pd.to_datetime(
        dataset[('pupil_meta', 'TimeReceivedByCore')]
    )
    
def make_time_attributes(dataset):
    """
    Create time index attributes for meso and pupil data.
    
    ```python
    Subject    Session
    STREHAB02  '01'         Index([ 0.0, 19.897, 39.986, ...
    STREHAB03  '01'         Index([ 0.0, 19.943, 39.922, ...
    ```
    
    """
    # Convert TimeReceivedByCore to datetime if not already done
    #set_timestamps_to_datetime(dataset)
    
    # Necessary for this dataset due to mistaken NaT entry in a single session's metadata.
    def compute_time_index(ts):
        # If ts is not indexable or empty, return empty array
        if not hasattr(ts, '__getitem__') or len(ts) == 0:
            return np.array([])
        # If the first timestamp is NaT, drop all NaT and re-test
        if pd.isnull(ts[0]):
            ts = ts[~pd.isnull(ts)]
            if len(ts) == 0:
                return np.array([])
        # Compute elapsed ms
        return np.round((ts - ts[0]).total_seconds() * 1000.0, 3)
    
    # Compute elapsed time in milliseconds from the first timestamp in each session
    times = dataset[('meso_meta', 'TimeReceivedByCore')]
    dataset.time_index_meso = times.apply(
        lambda ts: np.round((ts - ts[0]).total_seconds() * 1000.0, 3)
    )
    
    # Convert each pandas.Index in time_index_meso to a numpy array of floats
    dataset.time_index_meso = dataset.time_index_meso.apply(
        lambda idx: np.asarray(idx, dtype=float)
    )
    
    # Compute elapsed time in ms from the first pupil timestamp in each session, skipping the NaT entr
    times = dataset[('pupil_meta', 'TimeReceivedByCore')]

    dataset.time_index_pupil = times.apply(compute_time_index)

    # convert each pandas.Index in time_index_pupil to a numpy array of floats
    dataset.time_index_pupil = dataset.time_index_pupil.apply(
        lambda idx: np.asarray(idx, dtype=float)
    )

# dataset[('meso_meta', 'TimeReceivedByCore')] = dataset[('meso_meta', 'TimeReceivedByCore')].apply(
#     lambda arr: pd.to_datetime(arr)
# )

# dataset[('pupil_meta', 'TimeReceivedByCore')] = dataset[('pupil_meta', 'TimeReceivedByCore')].apply(
#     lambda arr: pd.to_datetime(arr)
# )


# times = dataset[('meso_meta', 'TimeReceivedByCore')]
# dataset.time_index_meso = times.apply(
#     lambda ts: np.round((ts - ts[0]).total_seconds() * 1000.0, 3)
# )
# # convert each pandas.Index in time_index_pupil to a numpy array of floats
# dataset.time_index_meso = dataset.time_index_meso.apply(
#     lambda idx: np.asarray(idx, dtype=float)
# )

make_time_attributes(dataset)
#%% 
'''
# Compute elapsed time in milliseconds from the first timestamp in each session


'''

dataset.duration = dataset.time_index_meso.iloc[-1] - dataset.time_index_meso.iloc[0]

#%% PLOT AN INTERACTIVE SINGLE SESSION FLUORESCENCE TRACE

from bokeh.plotting import figure, show
from bokeh.io import output_notebook, curdoc
from datetime import datetime

curdoc().theme = "dark_minimal"
meso_trace = dataset.meso.meso_tiff

output_notebook()


# Create a Bokeh figure
p = figure(title="Trace for ('STREHAB02', '10')", 
           x_axis_label='Time (seconds)', 
           y_axis_label='Trace Value', 
           width=950, height=400,
           background_fill_color=None,
           border_fill_color=None,
           )

# use the session’s time index in seconds for the x axis
session = ('STREHAB02', '10')
time_sec = dataset.time_index_meso[session] / 1000.0  # ms → sec

p.line(x=time_sec[1:], 
       y=dataset.meso.meso_tiff[session][1:], 
       line_width=0.8, color="limegreen", alpha=0.8)

# Show the plot
show(p)

#%% WRAP-AROUND DETECTION AND CLEANING FOR TEENSY TIMESTAMPS

# [FUNCTION] Unwrap Teensy micros() timestamps to fix wrap-around artifacts ---

def unwrap_teensy_timestamps(timestamps_us, rollover_val=2**32):
    """
    Fixes wrap-around artifacts in Teensy micros() data by detecting overflows
    and adding offset (2^32) after each wrap. 

    Parameters:
        timestamps_us (np.ndarray): Array of raw micros() timestamps
        rollover_val (int): Wrap-around threshold (default: 2^32 for micros())

    Returns:
        np.ndarray: Unwrapped timestamps in microseconds (monotonic)
    """
    timestamps_us = np.asarray(timestamps_us)
    diffs = np.diff(timestamps_us)
    rollover_indices = np.where(diffs < 0)[0]

    corrected = timestamps_us.copy()
    offset = 0

    for i in rollover_indices:
        offset += rollover_val
        corrected[i + 1:] += rollover_val

    return corrected


#%% PIPELINE V3

# [FUNCTIONS] PIPELINE V3: Standardized Encoder Cleaning & Interpolation ---

def clean_encoder_trace(row, micros_per_sec=1e6):
    """
    Clean encoder trace using meso.runner_time_ms duration as reference.
    Logs session-level metadata to ('Processed', ...) and ('Analysis', ...) columns.
    """
    ts = np.asarray(row[('encoder', 'timestamp')])
    speed = np.asarray(row[('encoder', 'speed')])
    distance = np.asarray(row[('encoder', 'distance')])

    # Reference duration from precomputed time index in milliseconds
    runner_time_ms = dataset.time_index_meso[row.name]

    expected_duration_us = (runner_time_ms[-1] - runner_time_ms[0]) * 1000

    wraparound_detected = np.any(np.diff(ts) < 0)
    if wraparound_detected:
        ts = unwrap_teensy_timestamps(ts)

    ts_end = ts[-1]
    ts_start = ts_end - expected_duration_us
    valid_mask = ts >= ts_start

    ts_clean = ts[valid_mask] - ts[valid_mask][0]
    speed_clean = speed[valid_mask]
    dist_clean = distance[valid_mask] - distance[valid_mask][0]

    frames_trimmed = len(ts) - len(ts_clean)
    cleaned_duration_sec = ts_clean[-1] / micros_per_sec

    return pd.Series({
        ('Processed', 'encoder_timestamp'): ts_clean,
        ('Processed', 'encoder_speed'): speed_clean,
        ('Processed', 'encoder_distance'): dist_clean,
        ('Analysis', 'EncoderFramesTrimmed'): frames_trimmed,
        ('Analysis', 'EncoderWraparoundDetected'): wraparound_detected,
        ('Analysis', 'CleanedEncoderDuration_s'): cleaned_duration_sec
    })


def interpolate_speed_to_runner_time(row):
    """
    Interpolates cleaned encoder speed to the meso runner_time_ms timebase.
    """
    ts = np.asarray(row[('Processed', 'encoder_timestamp')])
    speed = np.asarray(row[('Processed', 'encoder_speed')])
    # use precomputed time index in milliseconds
    runner_ms = dataset.time_index_meso[row.name]

    ts_ms = ts.astype(float) / 1000.0
    runner_ms = np.asarray(runner_ms, dtype=float)

    # use stepwise mapping: zeros between encoder samples, speed value at nearest past timestamp
    speed_series = pd.Series(speed, index=ts_ms)
    interp_series = speed_series.reindex(runner_ms, method='ffill', fill_value=0)
    interp_speed = interp_series.values

    return pd.Series({('Analysis', 'interp_encoder_speed'): interp_speed})


#%% VALIDATE CLEANED ENCODER SPEED AND DISTANCE WITH MATPLOTLIB VISUALIZATION

# [FUNCTIONS] Visualization of Cleaned Encoder Speed and Distance for Inspection -----

def plot_encoder_speed_and_distance(data, subject, session):
    """
    Plots raw vs cleaned encoder speed and distance for a given (Subject, Session)
    """
    idx = (subject, session)

    # Raw
    raw_ts = data.encoder.timestamp[idx] 
    raw_speed = data.encoder.speed[idx]
    raw_dist = data.encoder.distance[idx]

    # Cleaned
    clean_ts = data.loc[idx, ('Processed', 'encoder_timestamp')]
    clean_speed = data.loc[idx, ('Processed', 'encoder_speed')]
    clean_dist = data.loc[idx, ('Processed', 'encoder_distance')]
    # Convert microseconds to seconds
    raw_ts_s = raw_ts / 1e6
    clean_ts_s = clean_ts / 1e6
    
    gaps = np.diff(raw_ts_s)
    large_gaps = gaps > 0.5  # e.g., gaps > 500ms
    print(f"Number of gaps > 500ms: {np.sum(large_gaps)}")

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    #Speed Plot
    # interp_speed = data.loc[idx, ('Analysis', 'interp_encoder_speed')]
    # runner_ms = dataset.time_index_meso[idx] / 1000  # Convert to seconds
    # axs[0].step(runner_ms, interp_speed, where='post', label='Interpolated Speed (Step)', color='orange')


    axs[0].scatter(raw_ts_s, raw_speed, label='Raw Speed', color='grey', alpha=0.1, s=5)
    axs[0].scatter(clean_ts_s, clean_speed, label='Cleaned Speed', color='orange', alpha=0.3, s=5)
    axs[0].set_ylabel('Speed (mm/s)')
    axs[0].legend()
    axs[0].set_title(f'Encoder Speed & Distance: {subject}, Session {session}')
    axs[0].grid(True)

    # Distance Plot
    axs[1].scatter(raw_ts_s, raw_dist, label='Raw Distance', color='grey', alpha=0.1, s=5)
    axs[1].scatter(clean_ts_s, clean_dist, label='Cleaned Distance', color='orange', alpha=0.3, s=5)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Distance (mm)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# # Get all unique (Subject, Session) pairs
# session_tuples = dataset.index.unique()

# # Loop and plot for each
# for subject, session in session_tuples:
#     plot_encoder_speed_and_distance(dataset, subject, session)


# %% LOCOMOTION STATISTICS

# [FUNCTIONS] Detect Locomotion Bouts from Encoder Speed -----

def detect_locomotion_bouts(ts, speed, distance, 
                            speed_thresh=2, 
                            min_pause_ms=1000, 
                            min_bout_duration_ms=2000, 
                            micro_distance_thresh=60):
    """
    Detects locomotion bouts from encoder speed data.
    - ts: timestamps in microseconds
    - speed: speed array (mm/s)
    - distance: cumulative distance (mm)
    
    Returns: list of dicts per bout
    """
    ts = np.asarray(ts)
    speed = np.asarray(speed)
    distance = np.asarray(distance)

    # detect movement in either direction | abs value to account for bidirectioanlity of data
    is_moving = np.abs(speed) > speed_thresh

    # Group movement into bouts allowing short pauses
    bout_segments = []
    i = 0
    while i < len(is_moving):
        if is_moving[i]:
            start_idx = i
            while i < len(is_moving) and (
                is_moving[i] or (
                    (i < len(is_moving) - 1) and 
                    ((ts[i+1] - ts[i]) < min_pause_ms * 1000)
                )):
                i += 1
            end_idx = i
            bout_segments.append((start_idx, end_idx))
        else:
            i += 1

    bouts = []
    for start_idx, end_idx in bout_segments:
        start_time = ts[start_idx] / 1e6  # s
        end_time = ts[end_idx-1] / 1e6
        duration = end_time - start_time
        if duration * 1000 < min_bout_duration_ms:
            continue  # skip short blips

        dist = distance[end_idx-1] - distance[start_idx]
        is_micro = dist < micro_distance_thresh

        bouts.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'distance': dist,
            'is_micro': is_micro
        })

    return bouts


def compute_locomotion_per_session(row):
    ts = row[('Processed', 'encoder_timestamp')]
    speed = row[('Processed', 'encoder_speed')]
    distance = row[('Processed', 'encoder_distance')]

    bouts = detect_locomotion_bouts(ts, speed, distance)

    # Store the full list of bouts (can save it later if needed)
    total_distance = distance[-1]
    total_duration = ts[-1] / 1e6
    # average speed magnitude | abs value to account for bidirectioanlity of data
    avg_speed = np.mean(np.abs(speed))

    n_bouts = len([b for b in bouts if not b['is_micro']])
    n_micro = len([b for b in bouts if b['is_micro']])

    return pd.Series({
        ('Analysis', 'LocomotionBouts'): bouts,
        ('Analysis', 'TotalDistance_mm'): total_distance,
        ('Analysis', 'SessionDuration_s'): total_duration,
        ('Analysis', 'AverageSpeed_mmps'): avg_speed,
        ('Analysis', 'NumLocomotionBouts'): n_bouts,
        ('Analysis', 'NumMicroMovements'): n_micro
    })


# locomotion_stats = dataset.apply(compute_locomotion_per_session, axis=1)
# # drop overlapping locomotion stats columns so they get overwritten
# dataset = dataset.drop(columns=locomotion_stats.columns, errors='ignore')
# dataset = dataset.join(locomotion_stats)

#%% CALCULATE LOCOMOTION STATISTICS LONGFORM

# [FUNCTION] Convert session-level locomotion bout lists into a long-form DataFrame -----

def get_locomotion_bouts_longform(data):
    """
    Converts session-level locomotion bout lists into a long-form pandas DataFrame.
    Parameters

    Notes
    -----
    The input DataFrame must use a MultiIndex composed of (Subject, Session) for proper extraction
    of bout data.
    

    Subject     |   Session |   start_time  |  end_time  |  duration  |  distance  |  is_micro
    STREHAB02   |   01	    |   12.3	    |   17.8	 |   5.5	  |  42.1	   |  False
    STREHAB02	|   01	    |   20.1	    |   21.3	 |   1.2	  |  3.2	   |  True
    """
    
    records = []
    for (subject, session), row in data.iterrows():
        bouts = row.get(('Analysis', 'LocomotionBouts'), [])
        for bout in bouts:
            records.append({
                'Subject': subject,
                'Session': session,
                **bout  # includes start_time, end_time, duration, distance, is_micro
            })
    
    df_bouts = pd.DataFrame.from_records(records)
    return df_bouts

#dataset.bouts = get_locomotion_bouts_longform(dataset)


#%% CORRELATE LOCOMOTION WITH MESO SIGNAL

# [FUNCTION] Correlate Interpolated Locomotion with Meso Signal -----
def correlate_locomotion_with_meso(data, speed_thresh=5, bin_size_s=5):
    """
    Bins meso fluorescence and locomotion into consecutive `bin_size_s`-second windows,
    then computes the correlation between binned mean meso signal and binned
    locomotion occupancy (fraction of time moving) per session.
    Returns a DataFrame indexed by (Subject, Session).
    """

    records = []
    for (subject, session), row in data.iterrows():
        try:
            # timebase in seconds
            runner_time_ms = data.time_index_meso[(subject, session)]
            runner_time_s  = np.asarray(runner_time_ms, dtype=float) / 1000.0

            # continuous meso signal
            meso_signal = np.asarray(row[('meso', 'meso_tiff')])

            # encoder timestamps and speed in seconds
            ts_us = np.asarray(row[('Processed', 'encoder_timestamp')])
            speed = np.asarray(row[('Processed', 'encoder_speed')])
            ts_s = ts_us.astype(float) / 1e6

            # binary locomotion trace interpolated onto meso timebase
            locomotion_binary = (speed > speed_thresh).astype(int)
            locomotion_interp = np.interp(runner_time_s, ts_s, locomotion_binary, left=0, right=0)

            # define bin edges and assign each timepoint to a bin
            max_t = runner_time_s.max()
            edges = np.arange(0, max_t + bin_size_s, bin_size_s)
            bin_idx = np.digitize(runner_time_s, edges) - 1

            # compute per‐bin statistics
            n_bins = len(edges) - 1
            meso_means = np.zeros(n_bins, dtype=float)
            loco_frac  = np.zeros(n_bins, dtype=float)

            for b in range(n_bins):
                mask = bin_idx == b
                if not mask.any():
                    meso_means[b] = np.nan
                    loco_frac[b]  = np.nan
                else:
                    meso_means[b] = meso_signal[mask].mean()
                    loco_frac[b]  = locomotion_interp[mask].mean() * 100.0

            # drop bins with nan
            valid = ~np.isnan(meso_means) & ~np.isnan(loco_frac)
            corr = np.nan
            if valid.sum() > 1:
                corr = np.corrcoef(meso_means[valid], loco_frac[valid])[0, 1]

            records.append({
                'Subject': subject,
                'Session': session,
                'Correlation_Locomotion_Meso': corr,
                'MeanMeso_perBin': np.nanmean(meso_means),
                'StdMeso_perBin': np.nanstd(meso_means),
                'PercentMoving_perBin': np.nanmean(loco_frac)
            })
        except Exception:
            continue

    df = pd.DataFrame.from_records(records)
    df.set_index(['Subject', 'Session'], inplace=True)
    return df

# df_corr = correlate_locomotion_with_meso(dataset)
# # Join correlation results back into MultiIndexed structure
# dataset.update(df_corr)  # if the index matches exactly


# %% PLOT SESSION WITH LOCOMOTION BOUTS

# [FUNCTION] Plotting a single session with locomotion bouts highlighted -----
from bokeh.io import export_svgs
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, BoxAnnotation
from bokeh.layouts import column
import numpy as np

def plot_session_with_locomotion_bouts(
    data, subject, session, speed_thresh=5, output_svg: str = None
):
    """
    Plots mean fluorescence with locomotion bouts.
    If output_svg is a filepath, exports the figure as SVG instead of showing it.
    """
    row = data.loc[(subject, session)]
    meso = np.asarray(row[('meso', 'meso_tiff')])
    times = row[('meso_meta', 'TimeReceivedByCore')]
    runner_time = (times - times[0]).total_seconds()

    ts = row[('Processed', 'encoder_timestamp')] / 1000  # µs → s
    speed = np.asarray(row[('Processed', 'encoder_speed')])
    locomotion_binary = (speed > speed_thresh).astype(int)
    locomotion_interp = np.interp(runner_time, ts, locomotion_binary)

    source = ColumnDataSource({
        'time': runner_time[1:],
        'meso': meso[1:],
        'locomotion': locomotion_interp[1:]
    })

    p = figure(
        title=f"Fluorescence & Locomotion | {subject} Session {session}",
        x_axis_label="Time (s)",
        y_range=(6000, 10000),
        width=900, height=350,
        tools="pan,wheel_zoom,box_zoom,reset"
    )
    p.line('time', 'meso', source=source, color="green", legend_label="Ca²⁺ Signal")

    for bout in row.get(('Analysis', 'LocomotionBouts'), []):
        if not bout['is_micro']:
            span = BoxAnnotation(
                left=bout['start_time'],
                right=bout['end_time'],
                fill_alpha=0.1,
                fill_color='green'
            )
            p.add_layout(span)

    p.legend.location = "top_left"
    p.toolbar.autohide = True

    if output_svg:
        # enable SVG backend and export
        p.output_backend = "svg"
        export_svgs(p, filename=output_svg)
    else:
        show(p)


# Example usage:
# To display in notebook:
# plot_session_with_locomotion_bouts(dataset, "STREHAB02", "02")

# # To save as SVG:
# plot_session_with_locomotion_bouts(
#     dataset, "STREHAB02", "02", output_svg="session_STREHAB02_02_plot.svg"
# )

# %% PRODUCE SUMMARY STATISTICS

# [FUNCTION] Summarize session data into a DataFrame with key metrics -----


def summarize_session(data):
    records = []

    for (subject, session), row in data.iterrows():
        try:
            # Base metrics
            total_dist = row[('Analysis', 'TotalDistance_mm')]
            session_dur = row[('Analysis', 'SessionDuration_s')]
            avg_speed = row[('Analysis', 'AverageSpeed_mmps')]
            n_bouts = row[('Analysis', 'NumLocomotionBouts')]
            n_micro = row[('Analysis', 'NumMicroMovements')]
            corr = row.get(('Analysis', 'Correlation_Locomotion_Meso'), np.nan)

            # Meso trace and runner time
            meso = row[('meso', 'meso_tiff')]
            # Compute elapsed time in milliseconds from the first timestamp
            runner_time = dataset.time_index_meso[subject, session]

            # Locomotion binary trace
            ts = row[('Processed', 'encoder_timestamp')] / 1000
            speed = row[('Processed', 'encoder_speed')]
            moving_binary = (speed > 5).astype(int)
            moving_interp = np.interp(runner_time, ts, moving_binary)

            # Meso inside/outside movement
            meso = np.asarray(meso)
            moving = moving_interp >= 0.5
            still = ~moving

            meso_moving_mean = meso[moving].mean() if moving.any() else np.nan
            meso_moving_var = meso[moving].var() if moving.any() else np.nan
            meso_still_mean = meso[still].mean() if still.any() else np.nan
            meso_still_var = meso[still].var() if still.any() else np.nan

            # Bouts summary
            bouts = row[('Analysis', 'LocomotionBouts')]
            real_bouts = [b for b in bouts if not b['is_micro']]
            mean_bout_dur = np.mean([b['duration'] for b in real_bouts]) if real_bouts else np.nan
            #print(f'Subject: {subject} Session: {session} bouts: {bouts} | real bouts: {real_bouts} | mean_bout_dur: {mean_bout_dur}')
            mean_bout_dist = np.mean([b['distance'] for b in real_bouts]) if real_bouts else np.nan

            percent_moving = moving.mean() * 100

            records.append({
                'Subject': subject,
                'Session': session,
                'TotalDistance_mm': total_dist,
                'SessionDuration_s': session_dur,
                'AverageSpeed_mmps': avg_speed,
                'NumLocomotionBouts': n_bouts,
                'NumMicroMovements': n_micro,
                'MeanBoutDuration_s': mean_bout_dur,
                'MeanBoutDistance_mm': mean_bout_dist,
                'MesoMean_Moving': meso_moving_mean,
                'MesoVar_Moving': meso_moving_var,
                'MesoMean_Still': meso_still_mean,
                'MesoVar_Still': meso_still_var,
                'PercentTimeMoving': percent_moving,
                'Correlation_Locomotion_Meso': corr
            })

        except Exception as e:
            print(f"[WARN] Skipped ({subject}, {session}): {e}")
            continue

    df_summary = pd.DataFrame.from_records(records)
    df_summary.set_index(['Subject', 'Session'], inplace=True)
    # prefix summary columns under 'Analysis' group for consistency
    df_summary.columns = pd.MultiIndex.from_tuples(
        [('Analysis', col) for col in df_summary.columns]
    )
    return df_summary


#%%  PLOT ΔF/F₀ TRACE & KERNEL DENSITY ESTIMATE FOR STREHAB02 | 02

# [PLOTTER] ΔF/F₀ TRACE, KDE & EVENT QUANTIFICATION  -----

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def compute_dff_all_sessions(data, percentile=5):
    """
    Compute ΔF/F₀ for each (Subject, Session) from meso.meso_tiff
    and append the resulting arrays under the Analysis level.
    """
    def _dff(f_arr):
        arr = np.asarray(f_arr)[1:]                     # drop first frame
        f0 = np.percentile(arr, percentile)             # baseline
        return (arr - f0) / f0                          # ΔF/F₀

    data[('Analysis', 'meso_dff')] = data[('meso', 'meso_tiff')].apply(_dff)
    return data

#%% EVENT QUANTIFICATION FROM KERNEL DENSITY ESTIMATION

# [FUNCTION] Detect ΔF/F₀ events using KDE and thresholding -----

import seaborn as sns
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import binary_closing, label

def detect_events_vec(
    dff: np.ndarray,
    t_ms: np.ndarray,
    high_pct: float = 90,
    low_pct: float | None = None,
    window: int = 11,
    poly: int = 3,
    max_gap_ms: int = 1000,
    min_dur_ms: int = 500,
) -> list[tuple[int, int]]:
    if low_pct is None:
        low_pct = 0.75 * high_pct
    # 1) smooth
    smooth = savgol_filter(dff, window, poly)
    # 2) thresholds
    hi, lo = np.percentile(smooth, [high_pct, low_pct])
    hi_mask = smooth >= hi
    lo_mask = smooth >= lo
    # 3) grow seeds via binary closing
    seeds, _ = label(hi_mask)
    grown = binary_closing(seeds.astype(bool), structure=np.ones(int(max_gap_ms)))
    mask = grown & lo_mask
    # 4) extract events
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    events = []
    for s, e in zip(starts, ends):
        if (t_ms[e - 1] - t_ms[s]) >= min_dur_ms:
            events.append((s, e))
    return events

def detect_dff_events(
    data: pd.DataFrame,
    cutoff_pct: float = 90,
    hyst_lower_pct: float | None = None,
    sg_window: int = 11,
    sg_polyorder: int = 3,
    max_gap: int = 500,
    min_duration_ms: int = 500,
) -> pd.DataFrame:
    """
    Uses detect_events_vec on each ΔF/F₀ trace in data.
    Appends:
      ('Analysis','DffEventList'),
      ('Analysis','NumDffEvents'),
      ('Analysis','DffEventDurations_s')
    """
    if hyst_lower_pct is None:
        hyst_lower_pct = 0.75 * cutoff_pct

    lists, counts, durations = [], [], []

    for sess_key in data.index:
        dff_arr = np.asarray(data.loc[sess_key, ('Analysis', 'meso_dff')])
        t_ms    = np.asarray(data.time_index_meso[sess_key])
        if dff_arr.size == 0 or t_ms.size < dff_arr.size:
            lists.append([])
            counts.append(0)
            durations.append([])
            continue

        events = detect_events_vec(
            dff=dff_arr,
            t_ms=t_ms,
            high_pct=cutoff_pct,
            low_pct=hyst_lower_pct,
            window=sg_window,
            poly=sg_polyorder,
            max_gap_ms=max_gap,
            min_dur_ms=min_duration_ms
        )
        lists.append(events)
        counts.append(len(events))
        # convert to seconds
        durs = [(t_ms[e-1] - t_ms[s]) / 1000.0 for s, e in events]
        durations.append(durs)

    data[('Analysis', 'DffEventList')]        = pd.Series(lists,     index=data.index)
    data[('Analysis', 'NumDffEvents')]        = pd.Series(counts,    index=data.index)
    data[('Analysis', 'DffEventDurations_s')] = pd.Series(durations, index=data.index)
    return data


#%% PLOT ΔF/F₀ TRACE, KDE & EVENT QUANTIFICATION

# [PLOTTER] Loop plot all ΔF/F₀ sessions with detected events overlayed -----
def plot_all_dff_sessions(data, sg_window=2, sg_polyorder=1):
    """
    Loop through every (Subject, Session) in `data` and plot its ΔF/F₀ trace
    with detected events overlaid.
    """
    for subject, session in data.index:
        f_raw = data.loc[(subject, session), ('Analysis', 'meso_dff')]
        f = savgol_filter(f_raw, sg_window, sg_polyorder)
        evts = data.loc[(subject, session), ('Analysis', 'DffEventList')]
        
        plt.figure(figsize=(10, 3))
        plt.plot(f, color='navy', lw=1)
        for start, end in evts:
            plt.axvspan(start, end, color='orange', alpha=0.3)
        
        plt.xlabel('Frame Index')
        plt.ylabel('ΔF/F₀')
        plt.title(f'ΔF/F₀ & Events: {subject} | {session}')
        plt.tight_layout()
        plt.show()


#%% PLOT KDE

# [PLOTTER]] 
plt.figure(figsize=(10, 5))
plt.plot(x_grid, density, color='navy', lw=2)
plt.axvline(threshold, color='red', linestyle='--',
            label=f'{cutoff_pct}th pct ({threshold:.2f})')
plt.xlabel("ΔF/F₀")
plt.ylabel("Estimated Density")
plt.title(f"KDE & Event Threshold (Session {session_key[1]})")
plt.legend()
plt.grid(alpha=0.3)

# inset: trace with events marked
plt.figure(figsize=(10, 3))
times_s = np.arange(f.shape[0])  # or actual timebase if available
plt.plot(times_s, f, color='black', lw=1)
for start, end in events:
    plt.axvspan(start, end, color='orange', alpha=0.3)
plt.xlabel("Frame Index")
plt.ylabel("ΔF/F₀")
plt.title(f"ΔF/F₀ Trace & {n_events} Events > {cutoff_pct}th pct")
plt.tight_layout()
plt.show()


# %%

# ----- Compute 5th percentile of meso_tiff fluorescence trace and plot standard deviation across subjects -----


import numpy as np
import matplotlib.pyplot as plt

# pull out the meso_tiff column
meso = dataset[('meso','meso_tiff')]

# 1) compute 5th percentile in each (Subject,Session)
p5 = meso.apply(lambda arr: np.percentile(arr, 5))

# 2) compute std‐dev of those percentiles across sessions for each Subject
std_p5 = p5.groupby(level='Subject').mean()

# 3) plot
std_p5.plot(kind='bar', legend=False, color='C0')
plt.title('Std of 5th Percentile of meso_tiff by Subject')
plt.ylabel('Standard Deviation')
plt.xlabel('Subject')
plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import seaborn as sns

# Ensure fonts are embedded as outlines in the SVG
rcParams['svg.fonttype'] = 'none'

# Loop over all (Subject, Session) pairs
for subject, session in dataset.index.unique():
    # get time in seconds and fluorescence trace (skip the first frame)
    runner_ms = np.asarray(dataset.loc[(subject, session), ('meso', 'runner_time_ms')])[1:]
    times_s = runner_ms / 1000.0

    f = np.asarray(dataset.loc[(subject, session), ('meso', 'meso_tiff')])[1:]
    f_base = np.percentile(f, 5)
    dff = (f - f_base) / f_base

    # get locomotion bouts
    bouts = dataset.loc[(subject, session), ('Analysis', 'LocomotionBouts')]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times_s, dff, color='navy', lw=1)
    for bout in bouts:
        if not bout['is_micro']:
            ax.axvspan(bout['start_time'],
                       bout['end_time'],
                       color='green',
                       alpha=0.2)

    ax.set_title(f"{subject} | Session {session}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F₀")
    ax.set_ylim(-0.1, 0.6)       # set same y-axis limits for all plots
    ax.grid(True)

    # Ensure the output directory exists
    output_dir = Path("reports") / "250422_meso-dff_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the figure as an SVG into the reports folder
    filename = output_dir / f"dff_{subject}_{session}.svg"
    fig.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)
    
# %% Function Call Pipeline


#%% CLEAN ENCODER TRACES AND INTERPOLATE SPEED
# ------ Clean Encoder Traces and Interpolate Speed -----

# 1. Clean encoder traces
encoder_clean = dataset.apply(clean_encoder_trace, axis=1)

# 2. Merge cleaned output
data = dataset.copy(deep=True).join(encoder_clean)

# 3. Interpolate speed
interp_speed_df = data.apply(interpolate_speed_to_runner_time, axis=1)
# merge into data (which already contains the cleaned columns) and then overwrite dataset
data = data.join(interp_speed_df)
dataset = data
del(data)
# 4. Validation
if ('Analysis', 'interp_encoder_speed') in dataset.columns:
    sample = dataset[('Analysis', 'interp_encoder_speed')].iloc[0]
    runner_time = dataset.iloc[0][('meso_meta', 'TimeReceivedByCore')]
    expected_length = len(runner_time) if hasattr(runner_time, '__len__') else 1
    interp_length = len(sample) if hasattr(sample, '__len__') else 1
    print("Interpolation column exists.")
    print("Expected length:", expected_length)
    print("Interpolated speed length:", interp_length)
    print("Validation passed: Interpolated speed column is correct.")
else:
    print("Validation failed: Interpolated speed column is missing.")

# ------- Plot encoder speed and distance for each session -------
# Get all unique (Subject, Session) pairs
session_tuples = dataset.index.unique()

# Loop and plot for each
for subject, session in session_tuples:
    plot_encoder_speed_and_distance(dataset, subject, session)
# ----------------------------------------------------------------

#%% LOCOMOTION STATISTICS
# ------------ Detect Locomotion Bouts from Encoder Speed ---------------
locomotion_stats = dataset.apply(compute_locomotion_per_session, axis=1)
# drop overlapping locomotion stats columns so they get overwritten
dataset = dataset.drop(columns=locomotion_stats.columns, errors='ignore')
dataset = dataset.join(locomotion_stats)

dataset.bouts = get_locomotion_bouts_longform(dataset)

#%% CORRELATE LOCOMOTION WITH MESO SIGNAL
# ------------ Correlate Interpolated Locomotion with Meso Signal -------
df_corr = correlate_locomotion_with_meso(dataset)
# Join correlation results back into MultiIndexed structure
dataset.update(df_corr)  # if the index matches exactly


#%% PLOT SESSION WITH LOCOMOTION BOUTS
# ------ Plotting a single session with locomotion bouts highlighted -----

# To display in notebook:
plot_session_with_locomotion_bouts(dataset, "STREHAB02", "02")

# To save as SVG:
# plot_session_with_locomotion_bouts(
#     dataset, "STREHAB02", "02", output_svg="session_STREHAB02_02_plot.svg"
# )
# ------------------------------------------------------------------------

# %% PRODUCE SUMMARY STATISTICS
# ------ Summarize session data into a DataFrame with key metrics --------
df_summary = summarize_session(dataset)
# drop any existing Analysis summary columns to prevent overlap
dataset = dataset.drop(columns=df_summary.columns, errors='ignore')
dataset = dataset.join(df_summary)

#%% 
# Compute dF/F₀ for all sessions and update `data`
dataset = compute_dff_all_sessions(dataset)

# run detection
dataset = detect_dff_events(dataset, 
                            cutoff_pct=80, 
                            max_gap=100,
                            sg_window= 2,
                            sg_polyorder= 1,
                            hyst_lower_pct= 50,
                            min_duration_ms= 5000
                            )
# Call the function to visualize all sessions
plot_all_dff_sessions(dataset)


#%% BOKEH DASHBOARD FOR ALL SESSIONS
# build one row per session with 3 panels horizontally
session_rows = []
for subject, session in dataset.index:
    f1 = make_session_figure(dataset, subject, session)
    f2 = make_distance_figure(dataset, subject, session)
    f3 = make_stats_table(dataset, subject, session)
    session_rows.append(row(f1, f2, f3))

# stack each session-row vertically into the dashboard
dashboard = column(*session_rows)

# write out a single HTML file
output_file("all_sessions_dashboard.html", title="All Sessions Dashboard")
save(dashboard)

#%%

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import savgol_filter

def make_sample_data(duration_s=20, fs=100):
    """Generate synthetic ΔF/F trace: drift + random Gaussian events + noise."""
    t = np.linspace(0, duration_s * 1000, int(duration_s * fs))  # ms
    drift = 0.001 * t + 0.1 * np.sin(2 * np.pi * t / 10000)
    dff = drift.copy()
    np.random.seed(0)
    for center in np.random.uniform(0, t[-1], size=5):
        width = np.random.uniform(200, 1000)
        amp = np.random.uniform(0.2, 1.0)
        dff += amp * np.exp(-0.5 * ((t - center) / width)**2)
    dff += np.random.normal(scale=0.05, size=t.shape)
    return t, dff

def detect_events(dff, t_ms, cutoff_pct, hyst_lower_pct, sg_window, sg_polyorder, max_gap_ms, min_duration_ms):
    """Simplified ΔF/F event detection."""
    smooth = savgol_filter(dff, sg_window, sg_polyorder)
    high_th = np.percentile(smooth, cutoff_pct)
    low_th = np.percentile(smooth, hyst_lower_pct)
    mask = np.zeros_like(smooth, bool)
    in_evt = False
    for i, val in enumerate(smooth):
        if not in_evt and val >= high_th:
            in_evt, start = True, i
        elif in_evt and val <= low_th:
            mask[start:i] = True
            in_evt = False
    if in_evt:
        mask[start:] = True

    # fill short gaps
    dt = np.diff(t_ms, prepend=t_ms[0])
    gap_frames = int(max_gap_ms // np.median(dt))
    inv = ~mask
    d = np.diff(inv.astype(int), prepend=0, append=0)
    s_idx = np.where(d == 1)[0]
    e_idx = np.where(d == -1)[0]
    for s, e in zip(s_idx, e_idx):
        if (e - s) <= gap_frames:
            mask[s:e] = True

    # extract events
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    events = [(s, e) for s, e in zip(starts, ends) if (t_ms[e-1] - t_ms[s]) >= min_duration_ms]
    return smooth, high_th, low_th, mask, events

# Prepare data
t_ms, dff = make_sample_data()

# Initial parameters
init = {'cutoff_pct': 90, 'hyst_ratio': 0.75, 'sg_window': 11, 'sg_poly': 3,
        'max_gap': 500, 'min_dur': 500}

# Set up figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.4)

# Initial detection & plots
smooth, high_th, low_th, mask, _events = detect_events(
    dff, t_ms, init['cutoff_pct'], init['cutoff_pct']*init['hyst_ratio'],
    init['sg_window'], init['sg_poly'], init['max_gap'], init['min_dur']
)
_l_raw, = ax1.plot(t_ms, dff, label='Raw ΔF/F')
l_smooth, = ax1.plot(t_ms, smooth, label='Smoothed')
l_hi = ax1.hlines(high_th, t_ms[0], t_ms[-1], linestyles='--')
l_lo = ax1.hlines(low_th, t_ms[0], t_ms[-1], linestyles='--')
ax1.set_title('Raw vs Smoothed with Thresholds')
ax1.set_ylabel('ΔF/F')
ax1.legend()

l_mask, = ax2.plot(t_ms, mask.astype(int), drawstyle='steps-post')
ax2.set_title('Detected Event Mask')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Event (0/1)')

# Slider axes
axcolor = 'lightgoldenrodyellow'
ax_cut = plt.axes((0.1, 0.30, 0.8, 0.03), facecolor=axcolor)
ax_rat = plt.axes((0.1, 0.25, 0.8, 0.03), facecolor=axcolor)
ax_win = plt.axes((0.1, 0.20, 0.8, 0.03), facecolor=axcolor)
ax_poly = plt.axes((0.1, 0.15, 0.8, 0.03), facecolor=axcolor)
ax_gap = plt.axes((0.1, 0.10, 0.8, 0.03), facecolor=axcolor)
ax_dur = plt.axes((0.1, 0.05, 0.8, 0.03), facecolor=axcolor)

s_cut = Slider(ax_cut, 'Cutoff %', 50, 99, valinit=init['cutoff_pct'], valstep=1)
s_rat = Slider(ax_rat, 'Low % ratio', 0.3, 1.0, valinit=init['hyst_ratio'], valstep=0.05)
s_win = Slider(ax_win, 'SG window', 5, 51, valinit=init['sg_window'], valstep=2)
s_poly = Slider(ax_poly, 'SG polyord', 1, 7, valinit=init['sg_poly'], valstep=2)
s_gap = Slider(ax_gap, 'Max gap (ms)', 0, 2000, valinit=init['max_gap'], valstep=100)
s_dur = Slider(ax_dur, 'Min dur (ms)', 0, 2000, valinit=init['min_dur'], valstep=100)

def update(_val):
    cp = s_cut.val
    hr = s_rat.val
    sw = int(s_win.val)
    sp = int(s_poly.val)
    mg = s_gap.val
    md = s_dur.val

    smooth, hi, lo, mask, _events = detect_events(dff, t_ms, cp, cp*hr, sw, sp, mg, md)
    l_smooth.set_ydata(smooth)
    l_hi.set_ydata([hi, hi])
    l_lo.set_ydata([lo, lo])
    l_mask.set_ydata(mask.astype(int))
    fig.canvas.draw_idle()

for slider in [s_cut, s_rat, s_win, s_poly, s_gap, s_dur]:
    slider.on_changed(update)

reset_ax = plt.axes((0.8, 0.90, 0.1, 0.04))
button = Button(reset_ax, 'Reset', color=axcolor, hovercolor='0.975')
button.on_clicked(lambda _: [s_cut.reset(), s_rat.reset(), s_win.reset(),
                                s_poly.reset(), s_gap.reset(), s_dur.reset()])

plt.show()