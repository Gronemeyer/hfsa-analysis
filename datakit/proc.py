
import math
import numpy as np
import statistics as st
import pandas as pd



def euclidean_distance(coord1, coord2):
    """Calculate the Euclidean distance between two points."""
    return math.dist(coord1, coord2)


def confidence_filter_coordinates(frames_coords, frames_conf, threshold):
    """
    Vectorized version: batches all frames, extracts coords/conf,
    applies threshold, and returns per‐frame lists.
    """
    # Skip first frame if needed
    coords_stack = np.stack([c[0, :, 0, :] for c in frames_coords[1:]], axis=0)
    conf_stack   = np.stack([f[:, 0, 0]    for f in frames_conf[1:]],   axis=0)
    labels_stack = conf_stack >= threshold

    # Build the output list: each entry is [coords, conf, labels]
    return [
        [coords_stack[i], conf_stack[i], labels_stack[i]]
        for i in range(coords_stack.shape[0])
    ]


def apply_filters(df, speed_col='Speed', clamp_negative=False, threshold=None,
                  smoothing='rolling_median', window_size=10, alpha=0.5):
    """
    Applies optional filtering/smoothing to a speed column in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing speed data.
    speed_col : str
        Name of the column with raw speed values.
    clamp_negative : bool
        If True, speeds < 0 are set to 0.
    threshold : float or None
        A value below which absolute speeds are set to 0. If None, no threshold filter is applied.
    smoothing : str
        The type of smoothing to apply. Options: 'rolling_mean', 'rolling_median', 'ewm', or None for no smoothing.
    window_size : int
        Window size for rolling operations.
    alpha : float
        Smoothing factor for exponential smoothing, between 0 and 1.
        
    Returns
    -------
    pd.DataFrame
        The DataFrame with additional 'Speed_filtered' column.
    """
    df['Speed_filtered'] = df[speed_col]

    # 1. Clamp negative speeds
    if clamp_negative:
        df['Speed_filtered'] = df['Speed_filtered'].clip(lower=0)

    # 2. Threshold near-zero speeds
    if threshold is not None:
        df.loc[df['Speed_filtered'].abs() < threshold, 'Speed_filtered'] = 0

    # 3. Apply smoothing
    if smoothing == 'rolling_mean':
        df['Speed_filtered'] = df['Speed_filtered'].rolling(window=window_size, center=True).mean()
    elif smoothing == 'rolling_median':
        df['Speed_filtered'] = df['Speed_filtered'].rolling(window=window_size, center=True).median()
    elif smoothing == 'ewm':
        df['Speed_filtered'] = df['Speed_filtered'].ewm(alpha=alpha).mean()

    # Fill any NaNs from rolling or ewm at start/end
    df['Speed_filtered'].bfill()
    df['Speed_filtered'].ffill()
    
    return df


def analyze_pupil_data(
    pickle_data: pd.DataFrame,
    confidence_threshold: float = 0.7,
    pixel_to_mm: float = 53.6,
    dpi: int = 300
) -> pd.DataFrame:
    """
    Analyze pupil data from DeepLabCut output.

    This function processes a pandas DataFrame containing per-frame DeepLabCut outputs
    with 'coordinates' and 'confidence' columns, skipping an initial metadata row,
    and computes interpolated pupil diameters in millimetres.

    Steps
    -----
    1. Skip the first (metadata) row.
    2. Extract and convert 'coordinates' and 'confidence' to NumPy arrays.
    3. For each frame:
       - Squeeze arrays and validate dimensions.
       - Mark landmarks with confidence ≥ threshold.
       - Compute Euclidean distances for predefined landmark pairs.
       - Average valid distances as pupil diameter or assign NaN.
    4. Build a pandas Series of diameters, interpolate missing values, convert from pixels to mm.
    5. Reindex to include the metadata index, then drop the initial NaN to align with valid frames.

    Parameters
    ----------
    pickle_data : pandas.DataFrame
        Input DataFrame with an initial metadata row. Must contain:
        - 'coordinates': array-like of shape (n_points, 2) per entry
        - 'confidence': array-like of shape (n_points,) per entry
    threshold : float, optional
        Minimum confidence to include a landmark in diameter computation.
        Default is 0.1.
    pixel_to_mm : float, optional
        Conversion factor from pixels to millimetres.
        Default is 53.6.
    dpi : int, optional
        Dots-per-inch resolution (not used directly).
        Default is 300.

    Returns
    -------
    pandas.DataFrame
        One-column DataFrame ('pupil_diameter_mm') indexed by the input labels
        (excluding the metadata row), containing linearly interpolated
        pupil diameter measurements in millimetres.

    Example
    -------
    Suppose the function returns a DataFrame `result_df`. Its structure would look like:

       frame | pupil_diameter_mm
       ------|------------------
         1   | 1.23
         2   | 1.25
         3   | 1.22
         4   | 1.27
        ...  | ...
    """

    # 1) pull lists, skip metadata row
    coords_list = pickle_data['coordinates'].tolist()[1:]
    conf_list   = pickle_data['confidence'].tolist()[1:]
    
    # Return a warning if no confidence values are above the threshold
    if not any(np.any(np.array(c) >= confidence_threshold) for c in conf_list):
        print(f"[WARNING] {pickle_data.index[0:3]} No confidence values above threshold {confidence_threshold}.")
        
    # 2) to numpy arrays
    coords_arrs = [np.array(c) for c in coords_list]
    conf_arrs   = [np.array(c) for c in conf_list]

    # DEBUG: print first 3 shapes
    # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
    #     print(f"[DEBUG] frame {idx} coords.shape={c.shape}, conf.shape={f.shape}")
        
    # Print the first few values of c and f
    # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
    #     print(f"[DEBUG] frame {idx} coords values:\n{c}")
    #     print(f"[DEBUG] frame {idx} conf values:\n{f}")
        
    # 3) compute mean diameters
    pairs     = [(0, 1), (2, 3), (4, 5), (6, 7)]
    diameters = []
    for i, (coords, conf) in enumerate(zip(coords_arrs, conf_arrs)):
        pts   = np.squeeze(coords)   # expect (n_points, 2)
        cvals = np.squeeze(conf)     # expect (n_points,)
        # DEBUG unexpected shapes
        if pts.ndim != 2 or cvals.ndim != 1:
            print(f"[WARNING] frame {i} unexpected pts.shape={pts.shape}, conf.shape={cvals.shape}")
            diameters.append(np.nan)
            continue
        #print(f"cval type ={type(cvals)}, with values of type {cvals.dtype}\n compared to {type(confidence_threshold)}")
        valid = cvals >= confidence_threshold
        # print("cvals:", cvals)
        # print("threshold:", confidence_threshold)
        # print("mask  :", valid)  
        ds = [
            euclidean_distance(pts[a], pts[b])
            for a, b in pairs
            if a < pts.shape[0] and b < pts.shape[0] and valid[a] and valid[b]
        ]
        diameters.append(st.mean(ds) if ds else np.nan)

    # 4) interpolate & convert to mm, align with original index
    pupil_series = (
        pd.Series(diameters, index=pickle_data.index[1:])
          .interpolate()
          .divide(pixel_to_mm)
    )
    pupil_full = pupil_series.reindex(pickle_data.index)

    # DEBUG
    # print(f"[DEBUG analyze_pupil_data] input index={pickle_data.index}")
    # print(f"[DEBUG analyze_pupil_data] output series head:\n{pupil_full.head()}")

    # 5) return DataFrame without the metadata NaN
    return pd.DataFrame({'pupil_diameter_mm': pupil_full.iloc[1:]})


def unvectorized_process_deeplabcut_pupil_data_unvectorized(
    pickle_data: pd.DataFrame,
    show_plot: bool = False,
    confidence_threshold: float = 0.1,
    pixel_to_mm: float = 53.6
) -> pd.DataFrame:
    """
    Process a DeepLabCut DataFrame containing coordinates and confidence for each frame,
    and compute the pupil diameter per frame.
    
    Parameters
    ----------
    pickle_data : pd.DataFrame
        A DataFrame with a 'coordinates' column and a 'confidence' column for each frame.
    show_plot : bool, optional
        If True, displays a matplotlib plot of pupil diameter (in mm) across frames.
        Defaults to False.
    confidence_threshold : float, optional
        Minimum confidence required to include two landmarks in the diameter calculation.
        Defaults to 0.1.
    pixel_to_mm : float, optional
        Conversion factor from pixels to millimeters. Defaults to 53.6.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with one column ('pupil_diameter_mm') indexed by frame number.
        Frames for which no valid diameter could be calculated will have NaN values.
    """
    
    # 1) Set the raw DataFrame from the input
    raw_df = pickle_data

    # 2) Retrieve the 'coordinates' and 'confidence' columns.
    #    They are assumed to already contain numpy arrays.
    frame_coordinates_array = raw_df['coordinates'].tolist()
    frame_confidence_array = raw_df['confidence'].tolist()
    
    # 3) Filter coordinates by confidence
    labeled_frames = confidence_filter_coordinates(
        frame_coordinates_array,
        frame_confidence_array,
        confidence_threshold
    )
    
    # 4) Calculate mean pupil diameter (in pixels) per frame
    pupil_diameters = []
    for frame_data in labeled_frames:
        coords, conf, labels = frame_data
        frame_diameters = []
        
        # Pairs: (0,1), (2,3), (4,5), (6,7)
        for i in range(0, 7, 2):
            if labels[i] and labels[i+1]:
                diameter_pix = euclidean_distance(coords[i], coords[i+1])
                frame_diameters.append(diameter_pix)
        
        # If multiple diameters exist, use the average; otherwise, set to NaN
        if len(frame_diameters) > 1:
            pupil_diameters.append(st.mean(frame_diameters))
        else:
            pupil_diameters.append(np.nan)
        if len(frame_diameters) > 1:
            pupil_diameters.append(st.mean(frame_diameters))
        else:
            pupil_diameters.append(np.nan)
    
    # 5) Convert diameters to Series and interpolate missing values
    diam_series = pd.Series(pupil_diameters).interpolate()
    
    # 6) Convert from pixels to mm
    diam_series = diam_series / pixel_to_mm
    
    # 7) Optionally plot the results
    #if show_plot:
        # plt.figure(dpi=300)
        # plt.plot(diam_series, color='blue')
        # plt.xlabel('Frame')
        # plt.ylabel('Pupil Diameter (mm)')
        # plt.title('Pupil Diameter Over Frames')
        # plt.show()
    
    # 8) Return a DataFrame with the final diameters
    result_df = pd.DataFrame({'pupil_diameter_mm': diam_series})
    return result_df


def unvectorized_confidence_filter_coordinates(frames_coords, frames_conf, threshold):
    """
    Apply a boolean label to coordinates based on whether 
    their confidence exceeds `threshold`.
    
    Parameters
    ----------
    frames_coords : list
        List of numpy arrays containing pupil coordinates for each frame.
    frames_conf : list
        List of numpy arrays containing confidence values corresponding 
        to the coordinates in `frames_coords`.
    threshold : float
        Confidence cutoff.

    Returns
    -------
    list
        A list of [coords, conf, labels] for each frame, where 'labels' 
        is a list of booleans (True if above threshold, else False).
    """
    thresholded = []
    for coords, conf in zip(frames_coords[1:], frames_conf[1:]):
        frame_coords, frame_conf, frame_labels = [], [], []
        # Each frame has 8 sets of pupil points 
        for i in range(8):
            point = coords[0, i, 0, :]
            cval = conf[i, 0, 0]
            label = (cval >= threshold)
            frame_coords.append(point)
            frame_conf.append(cval)
            frame_labels.append(label)
        thresholded.append([frame_coords, frame_conf, frame_labels])
    return thresholded
