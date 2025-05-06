import datetime
import json
import os
import hashlib
import time
from functools import wraps
import inspect
from collections import defaultdict
from typing import Any, Dict, Tuple
import multiprocessing as mp
from pathlib import Path

import tifffile

from datakit import np, pd, mp
from datakit.config import SUBJECT_REGEX, SESSION_REGEX, GLOB_PATTERNS

class DatasetProxy:
    def __init__(self, df):
        self._df = df
        self._base = ['columns']
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            self._allowed = self._base + list(dict.fromkeys(lvl0))
        else:
            self._allowed = self._base + list(df.columns)

    @property
    def df(self):
        "🔓 raw pandas DataFrame if you really need it"
        return self._df

    def __getattr__(self, name):
        # 1) expose your small set
        if name in self._allowed:
            sub = self._df[name]
            return DatasetProxy(sub) if isinstance(sub, pd.DataFrame) else sub

        # 2) fallback to pandas (hidden from dir())
        return getattr(self._df, name)

    def __dir__(self):
        # only show your curated attrs
        return self._allowed + ['df']

def log_analysis(name, provenance_logger, source=None, description=None, parameters=None):
    """
    Decorator that runs an analysis function and logs provenance.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            shape = getattr(result, "shape", None)
            provenance_logger.log_analysis(
                name=name,
                function=func,
                source=source,
                description=description,
                parameters=parameters,
                runtime=duration,
                result_shape=shape
            )
            return result
        return wrapper
    return decorator

class ProvenanceLogger:
    def __init__(self):
        self.log = {
            "generated_on": datetime.datetime.now().isoformat(),
            "global_analyses": [],
            "per_source_analyses": {}
        }

    def _hash_function(self, func):
        try:
            source = inspect.getsource(func)
        except Exception:
            source = repr(func)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def log_analysis(self, name, function, source=None, description=None, parameters=None, runtime=None, result_shape=None):
        entry = {
            "name": name,
            "description": description or function.__doc__,
            "parameters": parameters or {},
            "function_hash": self._hash_function(function),
            "runtime_seconds": runtime,
            "result_shape": result_shape
        }

        if source is None:
            self.log["global_analyses"].append(entry)
        else:
            self.log["per_source_analyses"].setdefault(source, []).append(entry)

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2)
        print(f"✅ Saved provenance to JSON: {path}")

    def to_markdown(self, path):
        lines = ["# 📘 Analysis Provenance Log\n"]
        lines.append(f"*Generated on*: `{self.log['generated_on']}`\n")

        lines.append("## 🌐 Global Analyses\n")
        if self.log["global_analyses"]:
            for entry in self.log["global_analyses"]:
                lines += self._entry_to_md(entry)
        else:
            lines.append("*No global analyses registered.*\n")

        lines.append("## 🔍 Per-Source Analyses\n")
        if self.log["per_source_analyses"]:
            for source, entries in self.log["per_source_analyses"].items():
                lines.append(f"### 📦 Source: `{source}`")
                for entry in entries:
                    lines += self._entry_to_md(entry)
        else:
            lines.append("*No per-source analyses registered.*")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"📄 Markdown documentation written to: {path}")

    def _entry_to_md(self, entry):
        lines = [
            f"#### 🔸 {entry['name']}",
            f"- **Description**: {entry.get('description', 'N/A')}",
            f"- **Function Hash**: `{entry.get('function_hash', '')}`",
        ]
        if entry.get('parameters'):
            lines.append(f"- **Parameters**:")
            for k, v in entry['parameters'].items():
                lines.append(f"  - `{k}`: `{v}`")
        if entry.get('runtime_seconds') is not None:
            lines.append(f"- **Runtime**: {entry['runtime_seconds']:.3f} s")
        if entry.get('result_shape'):
            lines.append(f"- **Result Shape**: `{entry['result_shape']}`")
        lines.append("")
        return lines
    

def set_nested_value(d: Dict[str, Any], keys: Tuple[str, ...], value: Any) -> None:
    """
    Recursively set a nested value in a dictionary using a tuple of keys.

    Args:
        d (dict): The dictionary in which to set the value.
        keys (tuple): A tuple of keys representing the nested path.
        value (Any): The value to set.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def file_hierarchy(root_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a file hierarchy using pathlib's match method with support for deeper hierarchies.
    Subject and session IDs are extracted from the entire file path string. Files that do not 
    contain both "sub-" and "ses-" are excluded.

    Args:
        root_dir (str): Root directory containing experiment data.

    Returns:
        dict: A nested dictionary organized by subject and session. The keys for each file
              are now tuple keys (e.g. ("raw", "meso_tiff")) as defined in the new GLOB_PATTERNS.
    """
    db: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(dict))
    root = Path(root_dir)

    # Iterate recursively over all files
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        path_str = str(file_path)

        # Extract subject and session IDs from the full file path string.
        subject_match = SUBJECT_REGEX.search(path_str)
        session_match = SESSION_REGEX.search(path_str)
        if not (subject_match and session_match):
            continue  # Exclude files without both identifiers

        subject = subject_match.group(1)
        session = session_match.group(1)

        # Match file using updated glob patterns which may return a string.
        # Wrap it as a tuple to avoid iterating over characters.
        for glob_pattern, dest in GLOB_PATTERNS.items():
            if file_path.match(f"*{glob_pattern}"):
                key_tuple = dest if isinstance(dest, tuple) else (dest,)
                set_nested_value(db[subject][session], key_tuple, path_str)
                break  # Only process the first matching pattern

    return db

# Helper functions for mapping functions to pandas Series in parallel
def apply_func_on_series_chunk(chunk: pd.Series, func: callable) -> pd.Series:
    return chunk.apply(func)

def parallelize_series(series: pd.Series, func: callable, num_processes: int = None) -> pd.Series:
    if num_processes is None:
        num_processes = mp.cpu_count()
    series_split = np.array_split(series, num_processes)
    with mp.Pool(num_processes) as pool:
        results = pool.starmap(apply_func_on_series_chunk, [(chunk, func) for chunk in series_split])
    return pd.concat(results)

def compute_mean_trace(tiff_path):
    # Check if cupy is available
    if 'cp' in globals():
        return cp.array([cp.mean(frame) for frame in tifffile.memmap(tiff_path)])
    else:
        return np.array([np.mean(frame) for frame in tifffile.memmap(tiff_path)])