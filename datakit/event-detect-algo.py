#!/usr/bin/env python3
"""
Standalone GUI to explore ΔF/F event‐detection parameters
on two real datasets using PyQt6 + PyQtGraph.

Author: jgronemeyer
"""

import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


with pd.HDFStore("datakit/250506_HFSA_data_rs2.h5", mode="r") as store:
    df_meso = store['meso']
    df_meta = store['meso_meta']

# Combine along columns (axis=1) using index alignment
dataset = pd.concat([df_meso, df_meta], axis=1)


def set_timestamps_to_datetime(dataset):
    """
    Convert timestamp columns to datetime format.
    
    ```python
    Subject    Session
    STREHAB02  '01'         DatetimeIndex(['2025-03-24 16:57:24.456272', '...
    STREHAB03  '01'         DatetimeIndex(['2025-03-25 17:34:53.303604', '...
    ```

    """
    dataset[('TimeReceivedByCore')] = dataset[('TimeReceivedByCore')].apply(
        lambda arr: pd.to_datetime(arr)
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
    set_timestamps_to_datetime(dataset)
        
    # Compute elapsed time in milliseconds from the first timestamp in each session
    times = dataset[('TimeReceivedByCore')]
    dataset.time_index_meso = times.apply(
        lambda ts: np.round((ts - ts[0]).total_seconds() * 1000.0, 3)
    )
    
    # Convert each pandas.Index in time_index_meso to a numpy array of floats
    dataset.time_index_meso = dataset.time_index_meso.apply(
        lambda idx: np.asarray(idx, dtype=float)
    )

make_time_attributes(dataset)

# Compute ΔF/F₀ for all sessions and stash under ('Analysis','meso_dff')
def compute_dff_all_sessions(data, percentile=5):
    def _dff(f_arr):
        arr = np.asarray(f_arr)#[1:]      # drop first frame
        f0  = np.percentile(arr, percentile)
        return (arr - f0) / f0           # ΔF/F₀

    data[('Analysis','meso_dff')] = data[('meso_tiff')].apply(_dff)
    return data

# Run it once
if ('Analysis','meso_dff') not in dataset.columns:
    dataset = compute_dff_all_sessions(dataset)

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QComboBox
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg


def smooth_trace(
    trace: np.ndarray,
    window_length: int,
    polyorder: int
) -> np.ndarray:
    """
    Apply Savitzky–Golay filter to smooth a ΔF/F trace.

    Ensures window_length is odd.

    Parameters
    ----------
    trace
        Raw ΔF/F trace.
    window_length
        Length of filter window (must be odd).
    polyorder
        Polynomial order for filter.

    Returns
    -------
    smoothed : ndarray[float]
        Smoothed trace.
    """
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(trace, window_length, polyorder)


def compute_hysteresis_thresholds(
    trace: np.ndarray,
    high_percentile: float,
    low_percentile: float
) -> tuple[float, float]:
    """
    Compute high and low thresholds for hysteresis.

    Parameters
    ----------
    trace
        Smoothed ΔF/F trace.
    high_percentile
        Upper threshold percentile (0–100).
    low_percentile
        Lower threshold percentile (0–100).

    Returns
    -------
    high_th, low_th : float
        Numeric threshold values.
    """
    high_th = np.percentile(trace, high_percentile)
    low_th = np.percentile(trace, low_percentile)
    return high_th, low_th


def apply_hysteresis_mask(
    trace: np.ndarray,
    high_th: float,
    low_th: float
) -> np.ndarray:
    """
    Generate boolean mask of candidate events via hysteresis.

    Starts event when trace ≥ high_th, ends when ≤ low_th.

    Parameters
    ----------
    trace
        Smoothed ΔF/F trace.
    high_th
        High threshold.
    low_th
        Low threshold.

    Returns
    -------
    mask : ndarray[bool]
        True during event intervals.
    """
    mask = np.zeros_like(trace, dtype=bool)
    in_event = False
    start_idx = 0
    for i, value in enumerate(trace):
        if not in_event and value >= high_th:
            in_event = True
            start_idx = i
        elif in_event and value <= low_th:
            mask[start_idx:i] = True
            in_event = False
    if in_event:
        mask[start_idx:] = True
    return mask


def fill_short_gaps(
    mask: np.ndarray,
    t_ms: np.ndarray,
    max_gap_ms: float
) -> np.ndarray:
    """
    Close gaps in mask shorter than max_gap_ms.

    Converts time‐based gap into frame count.

    Parameters
    ----------
    mask
        Initial boolean mask.
    t_ms
        Time vector (ms).
    max_gap_ms
        Maximum gap length to fill (ms).

    Returns
    -------
    filled_mask : ndarray[bool]
        Mask with short gaps filled.
    """
    dt = np.diff(t_ms, prepend=t_ms[0])
    max_gap_frames = int(max_gap_ms / np.median(dt))
    inv = ~mask
    edges = np.diff(inv.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    filled = mask.copy()
    for s, e in zip(starts, ends):
        if (e - s) <= max_gap_frames:
            filled[s:e] = True
    return filled


def extract_events(
    mask: np.ndarray,
    t_ms: np.ndarray,
    min_duration_ms: float
) -> tuple[list[tuple[int,int]], list[float]]:
    """
    Extract event start/end indices and compute durations.

    Parameters
    ----------
    mask
        Final boolean mask of events.
    t_ms
        Time vector (ms).
    min_duration_ms
        Minimum event duration (ms).

    Returns
    -------
    events : list of (start_idx, end_idx)
        Index pairs for each event.
    durations_s : list of float
        Event durations in seconds.
    """
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    events: list[tuple[int,int]] = []
    durations_s: list[float] = []
    for s, e in zip(starts, ends):
        dur_ms = t_ms[e-1] - t_ms[s]
        if dur_ms >= min_duration_ms:
            events.append((s, e))
            durations_s.append(dur_ms / 1000.0)
    return events, durations_s


def detect_events(
    dff: np.ndarray,
    t_ms: np.ndarray,
    high_pct: float,
    low_pct: float,
    sg_window: int,
    sg_poly: int,
    max_gap_ms: float,
    min_dur_ms: float
) -> tuple[np.ndarray, float, float, np.ndarray, list[tuple[int,int]], list[float]]:
    """
    Full pipeline for ΔF/F event detection.

    Parameters
    ----------
    dff
        Raw ΔF/F trace.
    t_ms
        Time vector in ms.
    high_pct
        Percentile for high hysteresis threshold.
    low_pct
        Percentile for low hysteresis threshold.
    sg_window
        Savitzky–Golay window length.
    sg_poly
        Savitzky–Golay poly order.
    max_gap_ms
        Fill gaps shorter than this (ms).
    min_dur_ms
        Require events ≥ this duration (ms).

    Returns
    -------
    smooth_trace
        Smoothed ΔF/F.
    high_th
        High threshold value.
    low_th
        Low threshold value.
    mask
        Boolean mask of detected events.
    events
        List of (start_idx, end_idx).
    durations_s
        List of durations (s).
    """
    # 1) Smooth
    smooth = smooth_trace(dff, sg_window, sg_poly)

    # 2) Thresholds
    high_th, low_th = compute_hysteresis_thresholds(smooth, high_pct, low_pct)

    # 3) Hysteresis mask
    raw_mask = apply_hysteresis_mask(smooth, high_th, low_th)

    # 4) Gap filling
    filled_mask = fill_short_gaps(raw_mask, t_ms, max_gap_ms)

    # 5) Event extraction
    events, durations_s = extract_events(filled_mask, t_ms, min_dur_ms)

    return smooth, high_th, low_th, filled_mask, events, durations_s


class DffExplorer(QWidget):
    """
    PyQt6 GUI for tuning ΔF/F event‐detection parameters.

    - Select between sessions from the HDF dataset.
    - Adjust thresholds, SG filter, gap & duration.
    - View raw & smoothed traces, thresholds, and shaded events.
    - Status bar shows event count & durations.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ΔF/F Event Explorer")
        # Use HDF-loaded dataset
        self.dataset = dataset
        # List of available (Subject, Session) tuples
        self.sessions = list(self.dataset.index.unique())
        # Set initial selection
        self.current_subject, self.current_session = self.sessions[0]
        # Compute initial trace
        self._load_trace()
        self._build_ui()
        self._update_plot()

    def _load_trace(self):
        """Load runner time and ΔF/F trace for current selection."""
        row = self.dataset.loc[(self.current_subject, self.current_session)]

        # time vector in ms (already computed in dataset.time_index_meso)
        self.t_ms = np.asarray(
            self.dataset.time_index_meso[(self.current_subject, self.current_session)]
        )

        # ΔF/F₀ from the Analysis column
        self.dff  = np.asarray(row[('Analysis','meso_dff')])

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Dataset selector
        combo = QComboBox()
        combo.addItems([f"{s}|{sess}" for s, sess in self.sessions])
        combo.currentTextChanged.connect(self._change_dataset)
        layout.addWidget(QLabel("Select dataset:"))
        layout.addWidget(combo)

        # Plot area
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot)

        # Plot items
        self._raw_line    = self.plot.plot(self.t_ms[1:], self.dff[1:], pen=pg.mkPen('g', width=0.5))
        self._smooth_line = self.plot.plot([], [], pen=pg.mkPen('w', width=2))
        self._high_line   = pg.InfiniteLine(angle=0, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine))
        self._low_line    = pg.InfiniteLine(angle=0, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine))
        self.plot.addItem(self._high_line)
        self.plot.addItem(self._low_line)
        self._mask_line   = self.plot.plot([], [], pen=pg.mkPen('b'))
        self._regions     = []
        # Status
        self.status = QLabel()
        layout.addWidget(self.status)

        # Parameter sliders
        param_specs = {
            'High pct': (90, 50, 99, 1),
            'Low pct':  (75, 10, 90, 1),
            'SG window': (11, 5, 200, 2),
            'SG poly':   (3, 1, 7, 2),
            'Max gap ms': (500, 0, 2000, 100),
            'Min dur ms': (500, 0, 2000, 100),
        }
        self.sliders = {}
        for label, (val, mn, mx, st) in param_specs.items():
            row = QHBoxLayout()
            lbl = QLabel(f"{label}: {val}")
            sld = QSlider(Qt.Orientation.Horizontal)
            factor = 1 / st
            sld.setMinimum(int(mn * factor))
            sld.setMaximum(int(mx * factor))
            sld.setValue(int(val * factor))
            sld.valueChanged.connect(self._update_plot)
            row.addWidget(lbl)
            row.addWidget(sld)
            layout.addLayout(row)
            self.sliders[label] = (sld, lbl, st)

    def _change_dataset(self, text: str):
        """Switch to a different session tuple from HDF."""
        subj, sess = text.split('|')
        self.current_subject = subj
        self.current_session = sess
        self._load_trace()
        self._raw_line.setData(self.t_ms, self.dff)
        self._update_plot()

    def _update_plot(self, _=None):
        """Re-run detection and update all plot elements."""
        # Read slider values
        vals = {lbl: sld.value() * step for lbl, (sld, _, step) in self.sliders.items()}
        for lbl, (_, widget, _) in self.sliders.items():
            v = vals[lbl]
            widget.setText(f"{lbl}: {v:.2f}" if isinstance(v, float) else f"{lbl}: {int(v)}")

        # Map to detect_events args
        smooth, hi, lo, mask, events, durations = detect_events(
            dff=self.dff,
            t_ms=self.t_ms,
            high_pct=vals['High pct'],
            low_pct=vals['Low pct'],
            sg_window=int(vals['SG window']),
            sg_poly=int(vals['SG poly']),
            max_gap_ms=vals['Max gap ms'],
            min_dur_ms=vals['Min dur ms']
        )

        # Clear old regions
        for rg in self._regions:
            self.plot.removeItem(rg)
        self._regions.clear()

        # Shade each event
        for start, end in events:
            # clamp end so we don’t read beyond t_ms
            end_idx = min(end, len(self.t_ms) - 1)
            region = pg.LinearRegionItem(
                values=(self.t_ms[start], self.t_ms[end_idx]),
                brush=(50, 50, 200, 50),
                movable=False
            )
            self.plot.addItem(region)
            self._regions.append(region)

        # Update curves & lines
        self._smooth_line.setData(self.t_ms, smooth)
        self._high_line.setPos(hi)
        self._low_line.setPos(lo)
        self._mask_line.setData(self.t_ms, mask.astype(int))

        # Update status bar
        self.status.setText(
            f"Events: {len(events)}  •  Durations (s): {np.round(durations,2)}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DffExplorer()
    win.resize(900, 650)
    win.show()
    sys.exit(app.exec())