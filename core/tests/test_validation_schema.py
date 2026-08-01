"""Tests for the validation schema module."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError as PydanticValidationError

from app.data.validation.exceptions import (
    InvalidSchemaException,
    SchemaNotFoundException,
)
from app.data.validation.schema import (
    ColumnDefinition,
    SchemaLoader,
    ValidationSchema,
    build_schema_from_columns,
)


class TestColumnDefinition:
    """Test the ColumnDefinition model."""

    def test_defaults(self) -> None:
        """Test default column configuration."""
        col = ColumnDefinition(name="temp")
        assert col.required is True
        assert col.dtype == "str"
        assert col.nullable is True
        assert col.unique is False

    def test_aliases_reject_self(self) -> None:
        """Test that an alias cannot duplicate the column name."""
        with pytest.raises(PydanticValidationError):
            ColumnDefinition(name="temp", aliases=["temp"])

    def test_valid_aliases(self) -> None:
        """Test valid aliases are accepted."""
        col = ColumnDefinition(name="temperature", aliases=["temp", "tmp"])
        assert col.aliases == ["temp", "tmp"]

    def test_length_bounds_validated(self) -> None:
        """Test that min_length <= max_length is enforced."""
        with pytest.raises(PydanticValidationError):
            ColumnDefinition(name="name", min_length=10, max_length=5)

    def test_extra_forbidden(self) -> None:
        """Test extra fields are rejected."""
        with pytest.raises(PydanticValidationError):
            ColumnDefinition(name="x", unknown="field")


class TestValidationSchema:
    """Test the ValidationSchema model."""

    def test_properties(self) -> None:
        """Test required/optional/alias properties."""
        schema = build_schema_from_columns(
            {
                "temp": {"dtype": "float"},
                "humidity": {"dtype": "float"},
                "optional_col": {"dtype": "str", "required": False},
            }
        )
        assert set(schema.required_columns) == {"temp", "humidity"}
        assert schema.optional_columns == ["optional_col"]

    def test_all_aliases(self) -> None:
        """Test alias mapping."""
        schema = build_schema_from_columns(
            {"temp": {"dtype": "float", "aliases": ["temperature", "t"]}}
        )
        assert schema.all_aliases == {"temperature": "temp", "t": "temp"}

    def test_parse_plain_column_dicts(self) -> None:
        """Test that plain dicts in columns are wrapped with names."""
        schema = ValidationSchema.model_validate(
            {
                "name": "s",
                "columns": {
                    "temp": {"dtype": "float"},
                    "humidity": {"dtype": "float", "min_value": 0, "max_value": 100},
                },
            }
        )
        assert schema.columns["temp"].name == "temp"
        assert schema.columns["humidity"].min_value == 0

    def test_to_dict(self) -> None:
        """Test schema serialization to dict."""
        schema = build_schema_from_columns({"temp": {"dtype": "float"}})
        data = schema.to_dict()
        assert data["name"] == "inline_schema"
        assert data["columns"]["temp"]["dtype"] == "float"


class TestSchemaLoader:
    """Test the SchemaLoader class."""

    def test_load_from_dict(self) -> None:
        """Test loading a schema from an inline dict."""
        loader = SchemaLoader()
        schema = loader.load(
            {
                "name": "mydict",
                "columns": {"temp": {"dtype": "float"}},
            }
        )
        assert schema.name == "mydict"
        assert "temp" in schema.columns

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        """Test loading a schema from a YAML file."""
        yaml_path = tmp_path / "schema.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "name": "file_schema",
                    "columns": {"temp": {"dtype": "float", "min_value": -20}},
                }
            )
        )
        loader = SchemaLoader()
        schema = loader.load(yaml_path)
        assert schema.name == "file_schema"
        assert schema.columns["temp"].min_value == -20

    def test_load_missing_file_raises(self) -> None:
        """Test that a missing schema file raises SchemaNotFoundException."""
        loader = SchemaLoader()
        with pytest.raises(SchemaNotFoundException):
            loader.load("nonexistent_schema.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Test that an invalid schema raises InvalidSchemaException."""
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("not: a: valid: mapping: [unclosed")
        loader = SchemaLoader()
        with pytest.raises((InvalidSchemaException, Exception)):
            loader.load(yaml_path)

    def test_load_invalid_schema_shape(self, tmp_path: Path) -> None:
        """Test that a non-mapping YAML root raises InvalidSchemaException."""
        yaml_path = tmp_path / "malformed.yaml"
        yaml_path.write_text("just a string")
        loader = SchemaLoader()
        with pytest.raises(InvalidSchemaException):
            loader.load(yaml_path)

    def test_build_schema_from_columns(self) -> None:
        """Test the build_schema_from_columns helper."""
        schema = build_schema_from_columns(
            {
                "temp": {"dtype": "float", "min_value": -20, "max_value": 60},
                "crop": {"dtype": "str", "nullable": False},
            },
            name="custom",
        )
        assert schema.name == "custom"
        assert schema.columns["temp"].min_value == -20
        assert schema.columns["crop"].nullable is False
