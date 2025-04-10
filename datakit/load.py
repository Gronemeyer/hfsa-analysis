import os
import re
from pathlib import Path

from collections import OrderedDict
from typing import Any, Dict, Optional
import multiprocessing as mp
from datetime import datetime

import pandas as pd
import tifffile
from tqdm import tqdm
import pandas as pd

try:
    import cupy as cp
except ImportError:
    import numpy as np

from datakit.utils.utils import ProvenanceLogger, file_hierarchy
from datakit.config import *



class ExperimentData:
    def __init__(self, source: str | Path, verbose=False, validate=True):
        """
        Initialize the loader. Encapsulates experimental data stored in a MultiIndex pandas DataFrame,
          providing intuitive dot notation for further processing and export.

            - data_dict (dict): Original nested dictionary with experimental file info.
            - data (pd.DataFrame): DataFrame with file paths, indexed by (Subject, Session)
                                    and with MultiIndex columns (raw/processed, category).
        Parameters:
        ----------
        source : str | Path
            Either a directory path containing experiment files or an HDF5 file with pre-structured data.
        verbose : bool
            Whether to print status messages.
        validate : bool
            Whether to validate index alignment between sources.
        """
        self.verbose = verbose
        self.validate = validate
        
        import os
        
        # Initialize containers
        self.provenance = ProvenanceLogger()
        self.sources = OrderedDict()
        self.analysis_steps = OrderedDict()  # global
        self.per_source_analyses = OrderedDict()  # key = source name → list of analysis functions
        self.processed = pd.DataFrame()
        
        # Check if source is a directory or an HDF5 file
        # Import pathlib.Path if not already imported
        
        # Convert source to Path object if it's a string
        if isinstance(source, str):
            source = Path(source)
        
        if source.is_dir():
            # Directory path - use file hierarchy to build data structure from source
            self.experiment_dir = source
            self._dict = file_hierarchy(str(source))  # Ensure string for compatibility
            self.data = self._create_df_from_file_hierarchy(self._dict)
            self._log(f"Initialized from directory: {source}")
        elif source.is_file() and source.suffix.lower() in ('.h5', '.hdf5'):
            # HDF5 file path - load structured data
            self.experiment_dir = source.parent
            self._dict = None
            self.processed = self.from_hdf5(str(source), key='HFSA')  # Ensure string for compatibility
            self.data = self.processed.copy()
            self._log(f"Initialized from HDF5 file: {source}")
        else:
            raise ValueError(f"Source must be either a directory or an HDF5 file, got: {source}")

    def __call__(self):
        """ Makes the ExperimentData instance callable, returning the data DataFrame.
        """
        return self.data
    
    def _log(self, *args):
        # Ensure self._logs exists
        if not hasattr(self, '_logs'):
            self._logs = []
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"[{timestamp}] " + " ".join(str(arg) for arg in args)
        
        self._logs.append(message) # Store the log message
        
        if self.verbose:
            print("[ExperimentData]", *args)
    
    @staticmethod
    def _create_df_from_file_hierarchy(datadict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert a nested file hierarchy dictionary into a DataFrame with:
        - Row MultiIndex: ("Subject", "Session")
        """
        records = {}
        for subject, sessions in datadict.items():
            for session, flat_dict in sessions.items():
                records[(subject, session)] = flat_dict

        df = pd.DataFrame.from_dict(records, orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Subject", "Session"])

        return df
    
    @property
    def progress_summary(self) -> pd.DataFrame:
        """
        Return the progress summary DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame indicating file presence (1 if exists, 0 otherwise).
        """
        # Create the progress summary DataFrame.
        progress_df = self.data.notna().astype(int)
        # progress_df.columns = pd.MultiIndex.from_tuples(
        #     [("progress",) + col for col in progress_df.columns]
        # )
        return progress_df
     
    @property
    def subjects(self) -> pd.Index:
        """
        Return the unique subject identifiers.
        ```python
        self.data.index.get_level_values("Subject").unique()
        ```
        https://pandas.pydata.org/docs/user_guide/advanced.html#reconstructing-the-level-labels

        """
        return self.data.index.get_level_values("Subject").unique()
    
    def get_data_path(self, subject: str, session: str, data_type: str):
        """
        Returns a DataFrame of data for a specific subject and session.
        https://pandas.pydata.org/docs/user_guide/advanced.html#advanced-indexing-with-hierarchical-index

        Args:
            subject (str): Subject identifier.
            session (str): Session identifier.
            type (str): Type of data to retrieve (e.g., "meso_tiff", "pupil_tiff").
        """
        return self.data.loc[(subject, session), data_type]

    def load_tiff(self, subject: str, session: str, data_type: str = 'meso_tiff'):
        filepath = self.data.loc[(subject, session), data_type]
        return tifffile.memmap(filepath)

    # def generate_data_structure(self):
    #     steps = [
    #         ("meso_metadata", lambda col: col.apply(camera_metadata)),
    #         ("pupil_metadata", lambda col: col.apply(camera_metadata)),
    #         ("encoder", lambda col: col.apply(pd.read_csv)),
    #         ("meso_mean", lambda col: parallelize_series(col, compute_mean_trace)),
    #     ]

    #     for key, func in tqdm(steps, desc="Processing data steps", total=len(steps)):
    #         self.data[key] = func(self.data[key])

    def register_series(self, name, file_series, loader_func=pd.read_csv, structured=False):
        """
        Register a file Series with a custom loader function.

        Parameters:
        ----------
        name : str
            Top-level name for the data source
        file_series : pd.Series
            MultiIndex Series with file paths
        loader_func : callable
            Function to load a file into a DataFrame, Series, or array
        structured : bool
            If True, store the entire DataFrame in a single cell (do not decompose into columns)
        """
        entry = {
            'type': 'file_series',
            'data': file_series,
            'loader': loader_func,
            'structured': structured
        }
        self.sources.setdefault(name, []).append(entry)
        self._log(f"Registered file series: {name} ({len(file_series)} items), structured={structured}")

    def register_dataframe(self, name, dataframe):
        """Register a preloaded DataFrame with MultiIndex on rows."""
        entry = {
            'type': 'dataframe',
            'data': dataframe
        }
        self.sources.setdefault(name, []).append(entry)
        self._log(f"Registered in-memory DataFrame: {name} (shape: {dataframe.shape})")
    
    def register_analysis(self, name, function, source=None, description=None, parameters=None):
        """
        Register a named analysis step.

        Args:
            name : str
                Unique identifier for the analysis step.
            function : callable
                Function to compute summary DataFrame (must return MultiIndexed columns).
            source : str or None
                If specified, only apply this function to data from that source.
        
        Example:
        ```python
        loader.register_analysis(
            name='meso_calcium_summary',
            source='meso',
            function=lambda df_meso: compute_trace_summary_stats(
                get_normalized_meso_traces(df_meso, 
                    trace_key='meso_tiff', 
                    time_key='TimeReceivedByCore')))
        ```
        
        """
        entry = {
            'name': name,
            'function': function,
            'function_hash': self.provenance._hash_function(function),
            'source': source,
            'description': description or function.__doc__,
            'parameters': parameters or {},
        }

        if source is None:
            self.analysis_steps[name] = entry
            self._log(f"Registered global analysis: {name}")
        else:
            self.per_source_analyses.setdefault(source, []).append(entry)
            self._log(f"Registered per-source analysis: {name} for source '{source}'")
                    
    def describe_sources(self):
        """
        Print a summary of all registered data sources.
        """
        print("📄 ExperimentData - Registered Sources Summary")
        print("=" * 60)

        for name, configs in self.sources.items():
            print(f"\n📦 Source: {name}")
            
            for config in configs:
                source_type = config['type']
                print(f"  ├─ Type: {source_type}")

                if source_type == 'file_series':
                    file_series = config['data']
                    print(f"  ├─ Entries: {len(file_series)} file(s)")
                    try:
                        sample_df = config['loader'](file_series.iloc[0])
                        if isinstance(sample_df, pd.DataFrame):
                            print(f"  ├─ Columns: {', '.join(sample_df.columns)}")
                            print(f"  ├─ Example shape: {sample_df.shape}")
                            dt_type = self._detect_data_type(sample_df)
                            print(f"  └─ Data Type: {dt_type}")
                    except Exception as e:
                        print(f"  └─ Could not preview data: {e}")

                elif source_type == 'dataframe':
                    df = config['data']
                    print(f"  ├─ Shape: {df.shape}")
                    print(f"  ├─ Index: {df.index.names}")
                    if isinstance(df.columns, pd.MultiIndex):
                        col_list = [f"{a}/{b}" for a, b in df.columns.to_list()]
                    else:
                        col_list = list(df.columns)
                    print(f"  └─ Columns: {', '.join(col_list)}")

                elif source_type == 'series':
                    s = config['data']
                    print(f"  ├─ Length: {len(s)}")
                    dtype = type(s.iloc[0]).__name__
                    print(f"  └─ Value type: {dtype}")

    def save_summary(self, path="data_sources.md"):
        """
        Save a markdown-formatted summary of all registered sources.

        Parameters:
        ----------
        path : str
            Output file path (.md, .txt)
        """
        lines = []
        for name, configs in self.sources.items():
            lines.append(f"## 📦 Source: `{name}`")
            
            for config in configs:
                source_type = config['type']
                lines.append(f"- **Type**: `{source_type}`")

                if source_type == 'file_series':
                    file_series = config['data']
            if source_type == 'file_series':
                file_series = config['data']
                lines.append(f"- **Entries**: {len(file_series)} files")

                try:
                    sample_df = config['loader'](file_series.iloc[0])
                    if isinstance(sample_df, pd.DataFrame):
                        columns = ', '.join(sample_df.columns)
                        lines.append(f"- **Columns**: {columns}")
                        lines.append(f"- **Example shape**: `{sample_df.shape}`")
                        dt_type = self._detect_data_type(sample_df)
                        lines.append(f"- **Data Type**: `{dt_type}`")
                except Exception as e:
                    lines.append(f"- ⚠️ Could not preview data: {e}")

            elif source_type == 'dataframe':
                df = config['data']
                lines.append(f"- **Shape**: `{df.shape}`")
                lines.append(f"- **Index**: `{df.index.names}`")

                if isinstance(df.columns, pd.MultiIndex):
                    col_list = [f"{a}/{b}" for a, b in df.columns.to_list()]
                else:
                    col_list = list(df.columns)

                preview_cols = ', '.join(col_list[:10])
                more_cols = f"... (+{len(col_list)-10} more)" if len(col_list) > 10 else ""
                lines.append(f"- **Columns**: {preview_cols} {more_cols}")

            elif source_type == 'series':
                s = config['data']
                dtype = type(s.iloc[0]).__name__
                lines.append(f"- **Length**: {len(s)}")
                lines.append(f"- **Value Type**: `{dtype}`")

            lines.append("")  # Blank line between sources

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self._log(f"✅ Saved data source summary to: {path}")

    @staticmethod
    def _detect_data_type(df):
        """
        Infer whether a dataframe is 'scalar', 'array', or 'mixed' across its columns.
        """
        if df.shape[0] == 1:
            return "scalar"
        n_unique = df.nunique(dropna=False)
        if n_unique.eq(1).all():
            return "scalar"
        elif n_unique.eq(1).any():
            return "mixed"
        else:
            return "array"

    @staticmethod
    def _process_dataframe(df, data_type):
        row_data = {}
        for col in df.columns:
            series = df[col]
            if series.nunique(dropna=False) == 1:
                row_data[(data_type, col)] = series.iloc[0] # Constant column → store scalar
            else:
                row_data[(data_type, col)] = series.to_numpy() # Variable column → store as array
        return row_data

    @staticmethod
    def _standardize_dataframe(df, name):
        if df.columns.nlevels == 1:
            df.columns = pd.MultiIndex.from_product([[name], df.columns], names=['Source', 'Feature'])
        elif df.columns.nlevels == 2:
            df.columns.set_names(['Source', 'Feature'], inplace=True)
        return df

    def _load_file_series(self, file_series, data_type, loader, structured=False):
        rows = []
        for idx, filepath in file_series.items():
            df = loader(filepath)

            if structured and isinstance(df, pd.DataFrame):
                # Store the entire DataFrame in a single cell
                row = {(data_type, 'dataframe'): df}
            
            elif isinstance(df, pd.DataFrame):
                # Default behavior: decompose into columns
                row = self._process_dataframe(df, data_type)
            
            elif isinstance(df, (pd.Series, np.ndarray)):
                row = {(data_type, 'values'): df}
            
            else:
                raise ValueError(f"Unsupported type: {type(df)} from {filepath}")
            
            rows.append(row)

        result = pd.DataFrame(rows, index=file_series.index)
        result.columns = pd.MultiIndex.from_tuples(result.columns, names=['Source', 'Feature'])
        return result


    def load_all(self):
        all_sources_combined = []

        for name, configs in self.sources.items():
            parts = []

            for config in configs:
                if config['type'] == 'file_series':
                    df = self._load_file_series(
                        config['data'],
                        data_type=name,
                        loader=config['loader'],
                        structured=config.get('structured', False)
                    )
                elif config['type'] == 'dataframe':
                    df = self._standardize_dataframe(config['data'], name)
                elif config['type'] == 'series':
                    df = self._series_to_dataframe(config['data'], name)
                else:
                    raise ValueError(f"Unknown source type: {config['type']}")
                
                parts.append(df)

            # Combine all components under this source
            combined_source_df = pd.concat(parts, axis=1)
            self._log(f"Loaded source: {name}, shape: {combined_source_df.shape}")
            all_sources_combined.append(combined_source_df)

        # Validate and merge all sources
        final_df = pd.concat(all_sources_combined, axis=1)
        
        # Run per-source analyses
        for source, analysis_list in self.per_source_analyses.items():
            # Extract sub-DataFrame for the source
            source_df = final_df[source]
            for config in analysis_list:
                self._log(f"Running analysis: {config['name']} on source: {source}")
                result = config['function'](source_df)

                # Ensure MultiIndex columns
                if not isinstance(result.columns, pd.MultiIndex):
                    result.columns = pd.MultiIndex.from_tuples(result.columns, names=['Source', 'Feature'])

                final_df = final_df.join(result)

        # Run global analyses
        for name, analysis_fn in self.analysis_steps.items():
            self._log(f"Running global analysis: {name}")
            result = analysis_fn(final_df)

            if not isinstance(result.columns, pd.MultiIndex):
                result.columns = pd.MultiIndex.from_tuples(result.columns, names=['Source', 'Feature'])

            final_df = final_df.join(result)

        self._log(f"Final combined shape after analysis: {final_df.shape}")
        self.data = final_df
        return final_df

    def to_hdf5(self, df, path, key="nested_data", compression="blosc"):
        """
        Export nested DataFrame to HDF5 format.

        Args:
            df (pd.DataFrame):  The combined nested DataFrame to save
            path (str):         Output path to HDF5 file
            key (str):          Dataset name within HDF5 file
            compression (str):  Compression backend (ignored in fixed format but kept for future)
        """
        self._log(f"Saving to HDF5 (fixed format): {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with pd.HDFStore(path, mode='w') as store:
            store.put(key, df, format='fixed')
            self._log(f"Saved nested DataFrame to HDF5 under key '{key}'")
            
    def from_hdf5(self, path, key="nested_data"):
        """
        Load nested DataFrame from HDF5 format.

        Args:
            path (str): Path to the HDF5 file
            key (str): Dataset name within HDF5 file

        Returns:
            pd.DataFrame: The loaded nested DataFrame
        """
        self._log(f"Loading from HDF5: {path}")
        with pd.HDFStore(path, mode='r') as store:
            df = store.get(key)
            self._log(f"Loaded nested DataFrame from HDF5 under key '{key}'")
        return df

    def save_analysis_to_json(self, df_all, path, source='Analysis'):
        """
        Export all columns under a given source (e.g., 'Analysis') to a JSON file.
        
        Parameters
        ----------
        df_all : pd.DataFrame
            The combined nested structure
        path : str
            Output path for JSON file
        source : str
            The top-level column source to extract
            
        Example:
        ```python
        loader.save_analysis_to_json(df_all, "outputs/analysis_summary.json", source="Analysis")
        ```
        """
        if source not in df_all.columns.get_level_values('Source'):
            raise ValueError(f"Source '{source}' not found in df_all.")

        df_analysis = df_all[source].copy()
        json_data = df_analysis.reset_index().to_dict(orient='records')

        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)

        self._log(f"✅ Analysis data from source '{source}' saved to {path}")


# DATA_REGISTRY = {
#     "meso_mean": {"loader": compute_mean_trace, "dtype": "np.ndarray"},
#     "meso_tiff": {"loader": read_tiff_stack, "dtype": "np.memmap"},
#     "pupil_tiff": compute_mean_trace,
#     "encoder": pd.read_csv,
#     "meso_metadata": camera_metadata,
#     "pupil_metadata": camera_metadata,
#     "dlc_pupil": pickle_to_df,
#     "session_config": pd.read_csv,
# }

            
def process_dlc_pupil(self, max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Returns a MultiIndex DataFrame of DeepLabCut pupil data, loading files incrementally
    to manage memory usage.
    """
    paths = self.data.loc[:, ("processed", "dlc_pupil")]
    if max_files:
        paths = paths.iloc[:max_files]
    # Initialize with first file
    if len(paths) == 0:
        return pd.DataFrame()
        
    result = pd.DataFrame()
    
    # Process one file at a time
    for idx, filepath in paths.items():
        df = pickle_to_df(filepath)
        # Add MultiIndex using current path keys
        df_indexed = pd.DataFrame(df, index=pd.MultiIndex.from_tuples([idx]))
        result = pd.concat([result, df_indexed])
        # Force garbage collection after each file
        del df
        import gc
        gc.collect()
        
    return result

def pickle_to_df(pickle_path) -> pd.DataFrame:
    """
    Load a DeepLabCut output pickle file and return raw data in a pandas DataFrame.
    """ 
    df = pd.DataFrame(pd.read_pickle(pickle_path))
    return df

def pupil_means(pickle_path) -> pd.DataFrame:
    from wrangling.transform import process_deeplabcut_pupil_data
    process_deeplabcut_pupil_data(pickle_to_df(pickle_path))
    
def camera_metadata(metadata_path) -> pd.DataFrame:
    # Handle case where metadata_path is not a valid path (e.g., NaN or float)
    if not isinstance(metadata_path, str) or pd.isna(metadata_path):
        return pd.DataFrame()  # Return empty DataFrame
    
    try:
        import json
        
        # Load the JSON Data
        with open(metadata_path, 'r') as file:
            data = json.load(file)
        
        p0_data = data['p0'] #p0 is a list of the frames at Position 0 \
                                #(artifact of hardware sequencing in MMCore)
        df = pd.DataFrame(p0_data) # dataframe it

        # Expand 'camera_metadata' into separate columns
        camera_metadata_df = pd.json_normalize(df['camera_metadata'])
        df = df.join(camera_metadata_df)
        # Only drop columns that exist in the dataframe to avoid exceptions
        columns_to_drop = ['camera_metadata', 'property_values', 'version', 'format', 'ROI-X-start',
                        'ROI-Y-start', 'mda_event', 'Height', 'Width', 'camera_device', 'pixel_size_um', 
                        'images_remaining_in_buffer', 'PixelType', 'hardware_triggered']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=existing_columns, inplace=True)
        return df
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        return pd.DataFrame()  # Return empty DataFrame if any error occurs

def read_session_notes(notes_path) -> pd.DataFrame:
    """
    Load a session notes file as a list of lines (individual notes taken during the experiment session).
    
    Notes are a single structured sequence (timestamp + note) → a time-series.
    """
    with open(notes_path, 'r') as file:
        notes = file.readlines()
    # Parse lines into timestamps and notes
    timestamps = []
    note_texts = []

    for line in notes:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Try to extract timestamp and note using regex
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\s*(.*)', line.strip())
        if match:
            timestamp_str, note_text = match.groups()
            try:
                # Convert to datetime object and append
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                timestamps.append(timestamp)
                note_texts.append(note_text)
            except ValueError:
                # If datetime conversion fails, skip this line
                continue

    # Create DataFrame from parsed data
    notes_df = pd.DataFrame({'timestamp': timestamps, 'note': note_texts})
    notes_df.index = pd.to_datetime(notes_df['timestamp'])
    return notes_df

def read_session_config(config_path) -> pd.DataFrame:
    """
    Load a session configuration file and return a DataFrame.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(config_path)
    df.index = df.iloc[:, 0]  # Set the first column as the index
    df = df.T.drop(['Parameter'])  # Transpose and drop the first column
    df.drop(columns=['subject', 'session', 'start_on_trigger', 'trial_duration', 'psychopy_filename', 'protocol'], inplace=True)  #TODO: Make this dynamic
    return df

def csv_to_df(csv_path) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame.
    """
    dataframe = pd.read_csv(csv_path)
    return dataframe

