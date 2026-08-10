"""
Utility functions for the ML framework.
"""

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger


def generate_checksum(data: Any) -> str:
    """
    Generate a SHA-256 checksum for any data.
    
    Args:
        data: Data to hash (will be pickled if not bytes/string)
        
    Returns:
        Hex digest of the SHA-256 hash
    """
    if isinstance(data, (bytes, str)):
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    # Pickle the data and hash it
    pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(pickled).hexdigest()


def generate_model_version() -> str:
    """
    Generate a unique model version string.
    
    Returns:
        Version string: vYYYYMMDD_HHMMSS_<hash>
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    time_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"v{timestamp}_{time_hash}"


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save data as JSON.
    
    Args:
        data: Dictionary to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle numpy/pandas types
    def convert_to_serializable(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return obj
    
    serializable_data = json.loads(
        json.dumps(data, default=convert_to_serializable)
    )
    
    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def load_json(path: Union[str, Path]) -> dict[str, Any]:
    """Load JSON data from a file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, path: Union[str, Path]) -> None:
    """Save data as pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load pickle data from file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    with open(path, 'rb') as f:
        return pickle.load(f)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
    column_types: Optional[dict[str, type]] = None
) -> bool:
    """
    Validate a DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        column_types: Dict mapping column names to expected types
        
    Returns:
        True if valid, raises ValidationError if invalid
    """
    from .exceptions import ValidationError
    
    if df.empty:
        raise ValidationError("DataFrame is empty")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValidationError(f"Missing required columns: {missing}")
    
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                if not isinstance(df[col].dtype, expected_type):
                    try:
                        # Attempt conversion
                        df[col] = df[col].astype(expected_type)
                    except:
                        raise ValidationError(
                            f"Column {col} cannot be converted to {expected_type}"
                        )
    
    return True


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """
    Log metrics with optional prefix.
    
    Args:
        metrics: Dictionary of metric name to value
        prefix: Optional prefix for log messages
    """
    for name, value in metrics.items():
        logger.info(f"{prefix}{name}: {value:.4f}")


def get_memory_usage() -> dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024),
        'percent': process.memory_percent()
    }


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        self.elapsed_time = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.operation_name} completed in {self.elapsed_time:.2f} seconds")
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed_time is None:
            if self.start_time:
                return (datetime.now() - self.start_time).total_seconds()
            return 0.0
        return self.elapsed_time