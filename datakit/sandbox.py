#%% IMPORTS

from pathlib import Path
import pandas as pd
import numpy as np
import bokeh

#%%  LOAD DATA FROM THE H5 FILE
from datakit.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
dataset =pd.read_hdf( r'C:\dev\hfsa-analysis\250506_HFSA_data.h5', key='HFSA')

#%%
dataset[('meso_meta', 'TimeReceivedByCore')] = dataset[('meso_meta', 'TimeReceivedByCore')].apply(
    lambda arr: pd.to_datetime(arr)
)

dataset[('pupil_meta', 'TimeReceivedByCore')] = dataset[('pupil_meta', 'TimeReceivedByCore')].apply(
    lambda arr: pd.to_datetime(arr)
)
'''
Subject    Session
STREHAB02  01         DatetimeIndex(['2025-03-24 16:57:24.456272', '...
STREHAB03  01         DatetimeIndex(['2025-03-25 17:34:53.303604', '...
'''

times = dataset[('meso_meta', 'TimeReceivedByCore')]
dataset.time_index_meso = times.apply(
    lambda ts: np.round((ts - ts[0]).total_seconds() * 1000.0, 3)
)
# convert each pandas.Index in time_index_pupil to a numpy array of floats
dataset.time_index_meso = dataset.time_index_meso.apply(
    lambda idx: np.asarray(idx, dtype=float)
)

# Compute elapsed time in ms from the first pupil timestamp in each session, skipping the NaT entr
times = dataset[('pupil_meta', 'TimeReceivedByCore')]

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

dataset.time_index_pupil = times.apply(compute_time_index)

# convert each pandas.Index in time_index_pupil to a numpy array of floats
dataset.time_index_pupil = dataset.time_index_pupil.apply(
    lambda idx: np.asarray(idx, dtype=float)
)
'''
# Compute elapsed time in milliseconds from the first timestamp in each session

Subject    Session
STREHAB02  01         Index([        0.0,      19.897,      39.986, ...
STREHAB03  01         Index([        0.0,      19.943,      39.922, ...

'''

dataset.duration = dataset.time_index_meso.iloc[-1] - dataset.time_index_meso.iloc[0]

#%% PLOT AN INTERACTIVE SINGLE SESSION FLUORESCENCE TRACE

from bokeh.plotting import figure, show
from bokeh.io import output_notebook, curdoc
from datetime import datetime

curdoc().theme = "dark_minimal"
meso_trace = database.processed.meso.meso_tiff
meso_times = database.processed.meso.TimeReceivedByCore

output_notebook()

# Extract the trace for the specific MultiIndex
trace = meso_trace.loc[('STREHAB02', '10')][1:]
times = meso_times.loc[('STREHAB02', '10')][1:]

# Convert times to datetime objects
times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in times]

# Calculate elapsed time in seconds from the start
elapsed_time = [(t - times[0]).total_seconds() for t in times]

# Create a Bokeh figure
p = figure(title="Trace for ('STREHAB02', '10')", 
           x_axis_label='Time (seconds)', 
           y_axis_label='Trace Value', 
           width=950, height=400,
        #    background_fill_color=None,
        #    border_fill_color=None,
           )

p.line(elapsed_time, trace, line_width=.8, color="limegreen", alpha=0.8)

# Show the plot
show(p)

#%% GET DURATION UTIL FUNCTION

def get_duration(data, column_key=None, time_format='ms', row=-1):
    """
    Computes the duration of a timeseries array.
    Accepts input from:
        a Pandas.DataFrame, Pandas.Series, or an individual numpy.ndarray. 
    The duration is calculated as the difference between the last and first element of the array, 
    then converted to seconds.

    Parameters:
        data (pd.DataFrame, pd.Series, or np.ndarray): Input data containing an array.
        column_key (str or None): If data is a DataFrame, the column containing the array.
                                  Ignored for Series or numpy arrays.
        time_format (str): The unit of the original time values.
                           Supported values are 'us' (microseconds), 'ms' (milliseconds), and 'sec' (seconds).
        row (int): The row index to use when data is a Series of arrays (default: -1, i.e., the last element).

    Returns:
        float: Duration in seconds.
    """
    conversion = {'us': 1e-6, 'ms': 1e-3, 'sec': 1}
    if time_format not in conversion:
        raise ValueError(f"Unsupported time_format: {time_format}. Choose from {list(conversion.keys())}.")

    # If data is a DataFrame, extract the specified column.
    if isinstance(data, pd.DataFrame):
        if column_key is None:
            raise ValueError("column_key must be provided when data is a DataFrame.")
        series = data[column_key]
        # Obtain the array from the specified row.
        time_array = series.iloc[row] if hasattr(series, 'iloc') else series[row]
    # If data is a Series, check if its elements are scalars (i.e., a single sample per index)
    elif isinstance(data, pd.Series):
        if data.apply(lambda x: np.isscalar(x)).all():
            time_array = data.values
        else:
            time_array = data.iloc[row] if hasattr(data, 'iloc') else data[row]
    # If data is a numpy array, use it directly.
    elif isinstance(data, np.ndarray):
        time_array = data
    else:
        raise TypeError("Unsupported type for data. Expected a DataFrame, Series, or numpy array.")
    elif isinstance(data, pd.DatetimeIndex):
        # Handle a DatetimeIndex directly
        time_array = data.values
    
    
    # Ensure time_array is a numpy array
    if not isinstance(time_array, np.ndarray):
        time_array = np.asarray(time_array)
    duration = (time_array[-1] - time_array[0]) * conversion[time_format]
    
    if duration < 0:
        raise ValueError("Duration is negative! Ensure timestamps are sorted.")
    return duration

# Example usage on a DataFrame:
# duration_sec = get_duration(data.processed.meso, column_key='runner_time_ms', time_format='ms')
# print(round(duration_sec))

# Example usage on a Series:
# series_data = data.processed.meso.runner_time_ms
# duration_sec = get_duration(series_data, time_format='ms')
# print(round(duration_sec))

#%% WRAP-AROUND DETECTION AND CLEANING FOR TEENSY TIMESTAMPS
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

# --- PIPELINE V3: Standardized Encoder Cleaning & Interpolation ---

def clean_encoder_trace(row, micros_per_sec=1e6):
    """
    Clean encoder trace using meso.runner_time_ms duration as reference.
    Logs session-level metadata to ('Processed', ...) and ('Analysis', ...) columns.
    """
    ts = np.asarray(row[('encoder', 'timestamp')])
    speed = np.asarray(row[('encoder', 'speed')])
    distance = np.asarray(row[('encoder', 'distance')])

    # Reference duration from pre-converted meso_meta TimeReceivedByCore
    times = row[('meso_meta', 'TimeReceivedByCore')]
    runner_time_ms = (times - times[0]).total_seconds() * 1000.0

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
    times = row[('meso_meta', 'TimeReceivedByCore')]
    runner_ms = (times - times[0]).total_seconds() * 1000.0

    ts_ms = ts.astype(float) / 1000.0
    runner_ms = np.asarray(runner_ms, dtype=float)

    interp_speed = np.interp(runner_ms, ts_ms, speed)
    return pd.Series({('Analysis', 'interp_encoder_speed'): interp_speed})


# --- apply transformations ---

# 1. Clean encoder traces
encoder_clean = dataset.apply(clean_encoder_trace, axis=1)

# 2. Merge cleaned output
data = dataset.copy(deep=True).join(encoder_clean)

# 3. Interpolate speed
interp_speed_df = data.apply(interpolate_speed_to_runner_time, axis=1)
data = data.join(interp_speed_df)

# 4. Validation
if ('Analysis', 'interp_encoder_speed') in data.columns:
    sample = data[('Analysis', 'interp_encoder_speed')].iloc[0]
    runner_time = data.iloc[0][('meso_meta', 'TimeReceivedByCore')]
    expected_length = len(runner_time) if hasattr(runner_time, '__len__') else 1
    interp_length = len(sample) if hasattr(sample, '__len__') else 1
    print("Interpolation column exists.")
    print("Expected length:", expected_length)
    print("Interpolated speed length:", interp_length)
    print("Validation passed: Interpolated speed column is correct.")
else:
    print("Validation failed: Interpolated speed column is missing.")

# %% VALIDATE CLEANED ENCODER SPEED AND DISTANCE WITH MATPLOTLIB VISUALIZATION
import matplotlib.pyplot as plt

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

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Speed Plot
    axs[0].plot(raw_ts_s, raw_speed, label='Raw Speed', color='gray', alpha=0.5)
    axs[0].plot(clean_ts_s, clean_speed, label='Cleaned Speed', color='blue')
    axs[0].set_ylabel('Speed (mm/s)')
    axs[0].legend()
    axs[0].set_title(f'Encoder Speed & Distance: {subject}, Session {session}')
    axs[0].grid(True)

    # Distance Plot
    axs[1].plot(raw_ts_s, raw_dist, label='Raw Distance', color='gray', alpha=0.5)
    axs[1].plot(clean_ts_s, clean_dist, label='Cleaned Distance', color='green')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Distance (mm)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Get all unique (Subject, Session) pairs
session_tuples = data.index.unique()

# Loop and plot for each
for subject, session in session_tuples:
    plot_encoder_speed_and_distance(data, subject, session)


# %% LOCOMOTION STATISTICS

def detect_locomotion_bouts(ts, speed, distance, 
                            speed_thresh=2, 
                            min_pause_ms=1000, 
                            min_bout_duration_ms=3000, 
                            micro_distance_thresh=50):
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

    is_moving = speed > speed_thresh
    bout_mask = np.zeros_like(is_moving, dtype=bool)

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
    avg_speed = np.mean(speed)

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


locomotion_stats = data.apply(compute_locomotion_per_session, axis=1)
data = data.join(locomotion_stats)

#%% CALCULATE LOCOMOTION STATISTICS LONGFORM
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

data.bouts = get_locomotion_bouts_longform(data)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.histplot(data.bouts, x='duration', hue='is_micro', bins=30, kde=True)
plt.xlabel('Bout Duration (s)')
plt.title('Distribution of Locomotion Bout Durations')
plt.show()


#%% CORRELATE LOCOMOTION WITH MESO SIGNAL
def correlate_locomotion_with_meso(data, speed_thresh=5):
    """
    Computes correlation between interpolated locomotion and meso_tiff signal per session.
    Returns a DataFrame with correlation stats per (Subject, Session).
    """
    records = []

    for (subject, session), row in data.iterrows():
        try:
            # Extract meso signal
            runner_time = dataset.time_index_meso[subject, session]
            meso_signal = row[('meso', 'meso_tiff')]

            # Encoder
            ts = row[('Processed', 'encoder_timestamp')] / 1000.0  # µs → ms
            speed = row[('Processed', 'encoder_speed')]

            # Interpolate locomotion to runner_time_ms timebase
            locomotion_binary = (speed > speed_thresh).astype(int)
            locomotion_interp = np.interp(runner_time, ts, locomotion_binary)

            # Correlation
            corr = np.corrcoef(locomotion_interp, meso_signal)[0, 1]

            records.append({
                'Subject': subject,
                'Session': session,
                'Correlation_Locomotion_Meso': corr,
                'MeanMeso': np.mean(meso_signal),
                'StdMeso': np.std(meso_signal),
                'PercentMoving': np.mean(locomotion_interp) * 100
            })

        except Exception as e:
            print(f"[WARN] Skipping ({subject}, {session}): {e}")
            continue

    df_corr = pd.DataFrame.from_records(records)
    df_corr.set_index(['Subject', 'Session'], inplace=True)
    
    return df_corr

df_corr = correlate_locomotion_with_meso(data)
# Join correlation results back into MultiIndexed structure
data.update(df_corr)  # if the index matches exactly


# %% Interactive plotting with bokeh for future applications (Broken prototype)
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Select
from bokeh.layouts import column
from bokeh.io import output_notebook
output_notebook()  # For Jupyter/IPython environments

def get_bokeh_source_for_session(data, subject, session, speed_thresh=5):
    row = data.loc[(subject, session)]
    
    meso = row[('meso', 'meso_tiff')]
    runner_time = row[('meso', 'runner_time_ms')] / 1000  # ms → sec

    ts = row[('Processed', 'encoder_timestamp')] / 1000  # µs → sec
    speed = row[('Processed', 'encoder_speed')]
    locomotion_binary = (speed > speed_thresh).astype(int)
    locomotion_interp = np.interp(runner_time, ts, locomotion_binary)

    source = ColumnDataSource(data={
        'time': runner_time,
        'meso': meso,
        'locomotion': locomotion_interp
    })

    return source

def create_locomotion_meso_plot(source, subject, session):
    p = figure(title=f"Meso vs Locomotion: {subject} {session}",
               x_axis_label='Time (s)', height=300, width=800,
               tools="pan,wheel_zoom,reset,box_zoom")

    # Meso trace
    p.line('time', 'meso', source=source, line_color='navy', legend_label='Meso Signal')

    # Locomotion state
    p.line('time', 'locomotion', source=source, line_color='green', legend_label='Locomotion (Binary)', alpha=0.5)

    p.legend.location = "top_left"
    p.add_tools(HoverTool(tooltips=[("Time", "@time{s}"), ("Meso", "@meso"), ("Locomotion", "@locomotion")]))
    return p

def bokeh_interactive_plot(data):
    sessions = [f"{s[0]} | {s[1]}" for s in data.index]

    dropdown = Select(title="Subject | Session", value=sessions[0], options=sessions)
    source = get_bokeh_source_for_session(data, *sessions[0].split(" | "))
    plot = create_locomotion_meso_plot(source, *sessions[0].split(" | "))

    def update(attr, old, new):
        subject, session = new.split(" | ")
        new_source = get_bokeh_source_for_session(data, subject, session)
        source.data = new_source.data
        plot.title.text = f"Meso vs Locomotion: {subject} {session}"

    dropdown.on_change('value', update)
    return column(dropdown, plot)

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Select

# Generate components
sessions = [f"{s[0]} | {s[1]}" for s in data.index]
dropdown = Select(title="Subject | Session", value=sessions[0], options=sessions)
subject0, session0 = sessions[0].split(" | ")
source = get_bokeh_source_for_session(data, subject0, session0)
plot = create_locomotion_meso_plot(source, subject0, session0)

def update(attr, old, new):
    subject, session = new.split(" | ")
    new_source = get_bokeh_source_for_session(data, subject, session)
    source.data = new_source.data
    plot.title.text = f"Meso vs Locomotion: {subject} {session}"

dropdown.on_change('value', update)

# Add layout to current Bokeh document
curdoc().add_root(column(dropdown, plot))

layout = bokeh_interactive_plot(data)
show(layout)

# %% PLOT SESSION WITH LOCOMOTION BOUTS
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, BoxAnnotation
from bokeh.layouts import column
import numpy as np

def plot_session_with_locomotion_bouts(data, subject, session, speed_thresh=5):
    row = data.loc[(subject, session)]
    
    # Extract meso data
    meso = row[('meso', 'meso_tiff')]
    # Convert datetime array to elapsed time in seconds
    times = row[('meso_meta', 'TimeReceivedByCore')]
    runner_time = (times - times[0]).total_seconds()

    # Encoder: locomotion
    ts = row[('Processed', 'encoder_timestamp')] / 1000  # µs → sec
    speed = row[('Processed', 'encoder_speed')]
    locomotion_binary = (speed > speed_thresh).astype(int)
    locomotion_interp = np.interp(runner_time, ts, locomotion_binary)

    # Build source
    source = ColumnDataSource(data={
        'time': runner_time[1:],
        'meso': meso[1:],
        'locomotion': locomotion_interp[1:]
    })

    # Initialize plot with y-axis scaled from 6000 to 10000
    p = figure(title=f"Mean Cortical Astrocyte Calcium Fluorescence | {subject} Day: {session}",
               x_axis_label="Time (s)", y_range=(6000, 10000), 
               height=350, 
               width=900,
               tools="pan,wheel_zoom,box_zoom,reset")

    p.line('time', 'meso', 
           source=source, 
           color="green", 
           legend_label="Calcium Signal",)
    # p.line('time', 'locomotion', 
    #        source=source, 
    #        color="green", 
    #        alpha=0.4, 
    #        legend_label="Locomotion")

    # Add locomotion bouts as shaded spans
    bouts = row.get(('Analysis', 'LocomotionBouts'), [])
    for bout in bouts:
        if not bout['is_micro']:  # skip micro movements for now
            span = BoxAnnotation(left=bout['start_time'], right=bout['end_time'],
                                 fill_alpha=0.1, fill_color='green')
            p.add_layout(span)

    p.legend.location = "top_left"
    p.legend.background_fill_color = "gray"
    p.legend.background_fill_alpha = 0.8
    p.legend.label_text_color = "black"
    p.toolbar.autohide = True

    show(p)
    
# for subject, session in session_tuples:
#     plot_session_with_locomotion_bouts(data, subject, session)

plot_session_with_locomotion_bouts(data, "STREHAB02", "02")

# %% PRODUCE SUMMARY STATISTICS
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
    return df_summary

df_summary = summarize_session(data)

# %% CONSOLIDATE ALL SESSION PLOTS INTO A DASHBOARD WITH FLUORESCENCE, DISTANCE, AND STATS

from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource, BoxAnnotation, Div
from bokeh.layouts import column, row
from bokeh.io import output_file
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn

# you already have df_summary from the code above
# df_summary.index = MultiIndex of (Subject, Session)
# data is your master DataFrame with processed columns

def make_session_figure(data, subject, session, speed_thresh=5):
    """ Mean fluorescence + locomotion bouts shading. """
    rowd = data.loc[(subject, session)]
    meso = np.asarray(rowd[('meso', 'meso_tiff')])
    times = rowd[('meso_meta', 'TimeReceivedByCore')]
    runner_time = (times - times[0]).total_seconds() * 1000.0  # elapsed ms

    ts = np.asarray(rowd[('Processed', 'encoder_timestamp')]) / 1000.0  # µs → s
    speed = np.asarray(rowd[('Processed', 'encoder_speed')])
    locomotion = np.interp(runner_time,
                           ts,
                           (speed > speed_thresh).astype(int))

    src = ColumnDataSource({
        't': runner_time[1:],
        'meso': meso[1:],
        'loc': locomotion[1:]
    })

    p = figure(
        title=f"{subject} | Session {session}: Fluorescence",
        x_axis_label='Time (s)',
        y_axis_label='Fluorescence',
        width=900, height=350,
        tools="pan,wheel_zoom,box_zoom,reset"
    )
    p.line('t', 'meso', source=src, color='green', legend_label='Ca²⁺')
    bouts = rowd.get(('Analysis', 'LocomotionBouts'), [])
    for bout in bouts:
        if not bout['is_micro']:
            span = BoxAnnotation(
                left=bout['start_time'],
                right=bout['end_time'],
                fill_alpha=0.1,
                fill_color='green'
            )
            p.add_layout(span)

    p.legend.location = 'top_left'
    p.toolbar.autohide = True
    return p

def make_distance_figure(data, subject, session):
    """ Total distance moved (forward/backward). """
    rowd = data.loc[(subject, session)]
    ts = np.asarray(rowd[('Processed', 'encoder_timestamp')]) / 1e6  # µs → s
    dist = np.asarray(rowd[('Processed', 'encoder_distance')])

    p = figure(
        title=f"{subject} | Session {session}: Distance",
        x_axis_label='Time (s)',
        y_axis_label='Distance (mm)',
        width=500, height=350,
        tools="pan,wheel_zoom,box_zoom,reset"
    )
    p.line(ts, dist, line_color='navy')
    p.toolbar.autohide = True
    return p

def make_stats_table(data, subject, session):
    """A simple Bokeh DataTable with session summary stats from data.Analysis,
    rounding all numeric values to 2 decimal places."""
    # extract only the 'Analysis' columns for this (subject, session)
    stats = data['Analysis'].loc[(subject, session)]
    # build a small DataFrame for the table, formatting numbers to 2 decimals
    df = pd.DataFrame({
        "Stat": stats.index.astype(str),
        "Value": [
            f"{v:.2f}" if isinstance(v, (int, float, np.floating)) else v
            for v in stats.values
        ]
    })
    source = ColumnDataSource(df)
    columns = [
        TableColumn(field="Stat", title="Stat"),
        TableColumn(field="Value", title="Value")
    ]
    table = DataTable(
        source=source,
        columns=columns,
        fit_columns=True,
        width=300,
        height=250,
        index_position=None
    )
    return table

# build one row per session with 3 panels horizontally
session_rows = []
for subject, session in data.index:
    f1 = make_session_figure(data, subject, session)
    f2 = make_distance_figure(data, subject, session)
    f3 = make_stats_table(data, subject, session)
    session_rows.append(row(f1, f2, f3))

# stack each session-row vertically into the dashboard
dashboard = column(*session_rows)

# write out a single HTML file
output_file("all_sessions_dashboard.html", title="All Sessions Dashboard")
save(dashboard)

#%%
from bokeh.io import output_notebook, show
from bokeh.layouts import column, row
from bokeh.models import Select

output_notebook()

# build session keys and default
session_keys = [f"{subj}|{sess}" for subj, sess in data.index]
selector = Select(title="Subject|Session", value=session_keys[0], options=session_keys)

# initial figures
subj0, sess0 = session_keys[0].split("|")
fig1 = make_session_figure(data, subj0, sess0)
fig2 = make_distance_figure(data, subj0, sess0)
tab3 = make_stats_table(data, subj0, sess0)

# container
panels = row(fig1, fig2, tab3)
dashboard = column(selector, panels)

# callback to update on change
def update_session(attr, old, new):
    subj, sess = new.split("|")
    new1 = make_session_figure(data, subj, sess)
    new2 = make_distance_figure(data, subj, sess)
    new3 = make_stats_table(data, subj, sess)
    dashboard.children[1] = row(new1, new2, new3)

selector.on_change('value', update_session)

show(dashboard)

# %%
dashboard.output_backend = "svg"

from bokeh.io import export_svgs
import svglib.svglib as svglib
from reportlab.graphics import renderPDF
import numpy as np

# 1) export the dashboard to SVG
export_svgs(dashboard, filename="all_sessions_dashboard.svg")

# 2) register the font that Bokeh uses by default
svglib.register_font('helvetica', '/Library/Fonts/Helvetica.ttf')

# 3) read the SVG and render to PDF
drawing = svglib.svg2rlg("all_sessions_dashboard.svg")
renderPDF.drawToFile(drawing, "all_sessions_dashboard.pdf")


#%% PLOT ΔF/F₀ TRACE & KERNEL DENSITY ESTIMATE FOR STREHAB02 | 02

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Extract fluorescence trace and compute ΔF/F₀
f = data.meso.meso_tiff.loc[("STREHAB02", "09")][1:]
f_base = np.percentile(f, 5)
dff = (f - f_base) / f_base
dff_vals = np.asarray(dff)

# Kernel density estimation
kde = gaussian_kde(f)
x_grid = np.linspace(f.min(), f.max(), 1000)
density = kde(x_grid)

# Plot density
plt.figure(figsize=(8, 4))
plt.plot(x_grid, density, color='navy', lw=2)
# Mark key percentiles
for pct in [5, 25, 50, 75, 95]:
    v = np.percentile(f, pct)
    plt.axvline(v, linestyle='--', label=f'{pct}th pct')
plt.xlabel("ΔF/F₀")
plt.ylabel("Estimated Density")
plt.title("Kernel Density Estimate of ΔF/F₀ (STREHAB02 Session 02)")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# pull out the meso_tiff column
meso = data[('meso','meso_tiff')]

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

# Ensure fonts are embedded as outlines in the SVG
rcParams['svg.fonttype'] = 'none'

# Loop over all (Subject, Session) pairs
for subject, session in data.index.unique():
    # get time in seconds and fluorescence trace (skip the first frame)
    runner_ms = np.asarray(data.loc[(subject, session), ('meso', 'runner_time_ms')])[1:]
    times_s = runner_ms / 1000.0

    f = np.asarray(data.loc[(subject, session), ('meso', 'meso_tiff')])[1:]
    f_base = np.percentile(f, 5)
    dff = (f - f_base) / f_base

    # get locomotion bouts
    bouts = data.loc[(subject, session), ('Analysis', 'LocomotionBouts')]

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
# %%
