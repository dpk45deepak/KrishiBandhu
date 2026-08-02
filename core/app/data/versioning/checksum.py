"""
Checksum generation and validation for versioned entities.
"""

import hashlib
import mmap
from pathlib import Path
from typing import Union, Dict, Any, Optional
import json
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

from .models import ChecksumInfo
from .exceptions import ChecksumMismatchError, UnsupportedFormatError


class ChecksumGenerator:
    """Generates and validates checksums for various data formats."""

    CHUNK_SIZE = 8192  # 8KB chunks for file hashing

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm

    def generate_file_checksum(
        self,
        file_path: Union[str, Path],
        algorithms: list = None
    ) -> ChecksumInfo:
        """
        Generate checksums for a file.

        Args:
            file_path: Path to the file
            algorithms: List of hash algorithms to use

        Returns:
            ChecksumInfo with generated checksums
        """
        if algorithms is None:
            algorithms = ["sha256", "md5"]

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        hashers = {}
        for algo in algorithms:
            hashers[algo] = hashlib.new(algo)

        file_size = file_path.stat().st_size

        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                for chunk in iter(lambda: mmap_obj.read(self.CHUNK_SIZE), b''):
                    for algo, hasher in hashers.items():
                        hasher.update(chunk)

        checksum_info = ChecksumInfo(
            sha256=hashers.get('sha256', hashlib.sha256(b'').hexdigest()).hexdigest(),
            md5=hashers.get('md5', hashlib.md5(b'').hexdigest()).hexdigest(),
            file_size=file_size,
            created_at=datetime.utcnow()
        )

        logger.debug(f"Generated checksums for {file_path}: {checksum_info}")
        return checksum_info

    def generate_dataframe_checksum(
        self,
        df: pd.DataFrame,
        algorithms: list = None
    ) -> Dict[str, str]:
        """
        Generate checksums for a pandas DataFrame.

        Args:
            df: DataFrame to hash
            algorithms: List of hash algorithms to use

        Returns:
            Dictionary of algorithm -> hexdigest
        """
        if algorithms is None:
            algorithms = ["sha256", "md5"]

        # Create a deterministic representation
        # Sort columns and index for consistency
        df_sorted = df.sort_index().sort_index(axis=1)

        # Convert to bytes
        data_bytes = df_sorted.to_csv(index=False).encode('utf-8')

        hashes = {}
        for algo in algorithms:
            hasher = hashlib.new(algo)
            hasher.update(data_bytes)
            hashes[algo] = hasher.hexdigest()

        return hashes

    def generate_numpy_checksum(
        self,
        array: np.ndarray,
        algorithms: list = None
    ) -> Dict[str, str]:
        """
        Generate checksums for a numpy array.

        Args:
            array: NumPy array to hash
            algorithms: List of hash algorithms to use

        Returns:
            Dictionary of algorithm -> hexdigest
        """
        if algorithms is None:
            algorithms = ["sha256", "md5"]

        # Ensure deterministic representation
        data_bytes = array.tobytes()
        hashes = {}

        for algo in algorithms:
            hasher = hashlib.new(algo)
            hasher.update(data_bytes)
            hashes[algo] = hasher.hexdigest()

        return hashes

    def generate_dict_checksum(
        self,
        data: Dict[str, Any],
        algorithms: list = None
    ) -> Dict[str, str]:
        """
        Generate checksums for a dictionary.

        Args:
            data: Dictionary to hash
            algorithms: List of hash algorithms to use

        Returns:
            Dictionary of algorithm -> hexdigest
        """
        if algorithms is None:
            algorithms = ["sha256", "md5"]

        # Sort keys for consistency
        sorted_data = {k: data[k] for k in sorted(data.keys())}
        data_json = json.dumps(sorted_data, sort_keys=True, default=str)
        data_bytes = data_json.encode('utf-8')

        hashes = {}
        for algo in algorithms:
            hasher = hashlib.new(algo)
            hasher.update(data_bytes)
            hashes[algo] = hasher.hexdigest()

        return hashes

    def generate_dataset_fingerprint(
        self,
        df: pd.DataFrame,
        include_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive fingerprint for a dataset.

        Args:
            df: DataFrame to fingerprint
            include_stats: Include statistical information

        Returns:
            Dictionary with dataset fingerprint
        """
        fingerprint = {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'column_names': sorted(df.columns.tolist()),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            'checksums': self.generate_dataframe_checksum(df)
        }

        if include_stats:
            fingerprint['statistics'] = {
                'null_counts': df.isnull().sum().to_dict(),
                'unique_counts': df.nunique().to_dict(),
                'numeric_stats': {
                    col: {
                        'min': float(df[col].min()) if not df[col].empty else None,
                        'max': float(df[col].max()) if not df[col].empty else None,
                        'mean': float(df[col].mean()) if not df[col].empty else None,
                        'std': float(df[col].std()) if not df[col].empty else None
                    }
                    for col in df.select_dtypes(include=[np.number]).columns
                }
            }

        return fingerprint

    def verify_checksum(
        self,
        file_path: Union[str, Path],
        expected_checksum: ChecksumInfo
    ) -> bool:
        """
        Verify a file against an expected checksum.

        Args:
            file_path: Path to the file
            expected_checksum: Expected checksum information

        Returns:
            True if checksums match, False otherwise

        Raises:
            ChecksumMismatchError: If checksums don't match
        """
        actual_checksum = self.generate_file_checksum(
            file_path,
            algorithms=['sha256', 'md5']
        )

        match = (
            actual_checksum.sha256 == expected_checksum.sha256 and
            actual_checksum.md5 == expected_checksum.md5 and
            actual_checksum.file_size == expected_checksum.file_size
        )

        if not match:
            logger.error(f"Checksum mismatch for {file_path}")
            raise ChecksumMismatchError(
                f"Checksum mismatch for {file_path}. "
                f"Expected SHA256: {expected_checksum.sha256}, "
                f"Actual SHA256: {actual_checksum.sha256}"
            )

        logger.debug(f"Checksum verification passed for {file_path}")
        return True

    def detect_duplicate_dataset(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> bool:
        """
        Detect if two datasets are duplicates based on content.

        Args:
            df1: First DataFrame
            df2: Second DataFrame

        Returns:
            True if datasets are duplicates, False otherwise
        """
        # Use checksums for quick comparison
        checksum1 = self.generate_dataframe_checksum(df1)
        checksum2 = self.generate_dataframe_checksum(df2)

        # Compare SHA256
        if checksum1['sha256'] == checksum2['sha256']:
            logger.info("Duplicate dataset detected via SHA256 match")
            return True

        # Additional structural checks
        if len(df1) != len(df2) or len(df1.columns) != len(df2.columns):
            return False

        if set(df1.columns) != set(df2.columns):
            return False

        # Deep comparison if structures match but checksums don't
        # (could be due to different ordering)
        df1_sorted = df1.sort_index().sort_index(axis=1)
        df2_sorted = df2.sort_index().sort_index(axis=1)

        return df1_sorted.equals(df2_sorted)

    def detect_corrupted_file(self, file_path: Union[str, Path]) -> bool:
        """
        Detect if a file is corrupted.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file appears corrupted, False otherwise
        """
        file_path = Path(file_path)

        try:
            # Try reading the file
            with open(file_path, 'rb') as f:
                # Read the first few bytes to check if it's a valid file
                header = f.read(1024)

                # Check for null bytes at start (could indicate corruption)
                if b'\x00' * 10 in header:
                    logger.warning(f"Potential corruption detected in {file_path}")
                    return True

                # For Parquet files, check magic bytes
                if file_path.suffix in ['.parquet', '.pq']:
                    if not header.startswith(b'PAR1'):
                        logger.warning(f"Invalid Parquet magic bytes in {file_path}")
                        return True

                # For CSV files, check if it starts with valid characters
                if file_path.suffix in ['.csv', '.tsv']:
                    try:
                        header.decode('utf-8')
                    except UnicodeDecodeError:
                        logger.warning(f"Invalid UTF-8 encoding in {file_path}")
                        return True

            # Try calculating checksum (will raise error if file is corrupt)
            self.generate_file_checksum(file_path)
            return False

        except Exception as e:
            logger.error(f"Error checking file integrity: {e}")
            return True