"""Shared fixtures for the validation test suite."""


import pandas as pd
import pytest

from app.data.validation.schema import ValidationSchema, build_schema_from_columns


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a valid agricultural dataset fixture."""
    return pd.DataFrame(
        {
            "crop_name": ["Rice", "Wheat", "Maize", "Rice"],
            "state": ["Punjab", "Haryana", "UP", "Punjab"],
            "district": ["Ludhiana", "Karnal", "Varanasi", "Patiala"],
            "temperature": [28.5, 22.0, 30.1, 27.0],
            "humidity": [65.0, 55.0, 70.0, 60.0],
            "rainfall": [800.0, 500.0, 900.0, 750.0],
            "nitrogen": [120.0, 90.0, 110.0, 130.0],
            "phosphorus": [60.0, 40.0, 55.0, 65.0],
            "potassium": [140.0, 100.0, 120.0, 150.0],
            "yield": [4.5, 3.2, 5.1, 4.1],
            "planting_date": pd.to_datetime(
                ["2024-06-01", "2024-11-15", "2024-06-20", "2024-06-10"]
            ),
        }
    )


@pytest.fixture
def invalid_df() -> pd.DataFrame:
    """Return a DataFrame with multiple validation failures."""
    return pd.DataFrame(
        {
            "crop_name": ["Rice", "", "Maize", "Rice"],
            "state": ["Punjab", "123", "UP", "Punjab"],
            "district": ["Ludhiana", None, "Varanasi", "Patiala"],
            "temperature": [28.5, 65.0, 30.1, 27.0],
            "humidity": [65.0, 120.0, 70.0, 60.0],
            "rainfall": [800.0, -5.0, 900.0, 750.0],
            "nitrogen": [120.0, 350.0, 110.0, 130.0],
            "phosphorus": [60.0, 200.0, 55.0, 65.0],
            "potassium": [140.0, 400.0, 120.0, 150.0],
            "yield": [4.5, 3.2, -1.0, 4.1],
            "planting_date": pd.to_datetime(
                ["2024-06-01", "2026-12-31", "2024-06-20", "2024-06-10"]
            ),
        }
    )


@pytest.fixture
def crop_schema() -> ValidationSchema:
    """Return the canonical crop validation schema."""
    return build_schema_from_columns(
        {
            "crop_name": {"dtype": "str", "nullable": False, "min_length": 1, "max_length": 100},
            "state": {"dtype": "str", "nullable": False, "min_length": 2, "max_length": 50},
            "district": {"dtype": "str", "nullable": False, "min_length": 2, "max_length": 50},
            "temperature": {"dtype": "float", "min_value": -20, "max_value": 60},
            "humidity": {"dtype": "float", "min_value": 0, "max_value": 100},
            "rainfall": {"dtype": "float", "min_value": 0},
            "nitrogen": {"dtype": "float", "min_value": 0, "max_value": 300, "required": False},
            "phosphorus": {"dtype": "float", "min_value": 0, "max_value": 150, "required": False},
            "potassium": {"dtype": "float", "min_value": 0, "max_value": 300, "required": False},
            "yield": {"dtype": "float", "min_value": 0},
            "planting_date": {"dtype": "datetime", "required": False},
        },
        name="crop_test_schema",
    )


@pytest.fixture
def schema_dict() -> dict:
    """Return a minimal inline schema dict."""
    return {
        "name": "inline_schema",
        "columns": {
            "temperature": {
                "name": "temperature",
                "dtype": "float",
                "min_value": -20,
                "max_value": 60,
            },
            "humidity": {"name": "humidity", "dtype": "float", "min_value": 0, "max_value": 100},
            "crop_name": {
                "name": "crop_name",
                "dtype": "str",
                "nullable": False,
                "min_length": 1,
            },
        },
    }
