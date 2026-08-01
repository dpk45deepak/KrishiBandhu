"""Reusable, configurable schema definitions for the validation framework.

Schemas are expressed as YAML dictionaries and parsed into strongly-typed
Pydantic models. A schema defines required/optional columns, aliases,
expected data types, nullability, uniqueness, allowed values, regex
patterns, numeric bounds, and length constraints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.data.validation.exceptions import (
    InvalidSchemaException,
    SchemaNotFoundException,
)
from app.utils.path_utils import resolve_path

DataType = Literal["int", "float", "str", "bool", "datetime", "object", "category"]


class ColumnDefinition(BaseModel):
    """Configuration for a single column in a validation schema.

    Attributes:
        name: Canonical column name.
        required: Whether the column must exist and be non-null.
        aliases: Alternative names that map to this column.
        dtype: Expected pandas-compatible data type.
        nullable: Whether null values are permitted.
        unique: Whether values in this column must be unique.
        allowed_values: Optional list of permitted values.
        regex: Optional regex pattern the values must match.
        min_value: Inclusive lower numeric bound.
        max_value: Inclusive upper numeric bound.
        min_length: Minimum string length.
        max_length: Maximum string length.
        constant_value: If set, the column must contain exactly this value.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    required: bool = True
    aliases: list[str] = Field(default_factory=list)
    dtype: DataType = "str"
    nullable: bool = True
    unique: bool = False
    allowed_values: list[Any] | None = None
    regex: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    constant_value: Any | None = None

    @field_validator("aliases")
    @classmethod
    def aliases_not_self(cls, value: list[str], info: Any) -> list[str]:
        """Prevent an alias from colliding with the canonical name."""
        name = info.data.get("name")
        if name and name in value:
            raise ValueError(f"Alias '{name}' duplicates the column name")
        return value

    @model_validator(mode="after")
    def check_length_bounds(self) -> ColumnDefinition:
        """Ensure min_length <= max_length when both are provided."""
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError(
                f"Column '{self.name}': min_length ({self.min_length}) must be "
                f"<= max_length ({self.max_length})"
            )
        return self


class BusinessRuleDefinition(BaseModel):
    """Configuration for a custom, agri-domain business rule.

    Attributes:
        name: Unique rule name.
        description: Human-readable description.
        function: Importable dotted path to a callable
            ``(DataFrame, dict, ValidationRule) -> ValidationResult``.
        parameters: Arbitrary rule parameters injected at runtime.
        severity: Severity when the rule fails.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    function: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    severity: str = "error"


class ValidationSchema(BaseModel):
    """A complete validation schema for an agricultural dataset.

    Attributes:
        name: Schema identifier (e.g. 'crop_schema').
        version: Schema version string.
        description: Optional description of the schema.
        columns: Mapping of column name to :class:`ColumnDefinition`.
        business_rules: Optional custom cross-field business rules.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "default_schema"
    version: str = "1.0.0"
    description: str = ""
    columns: dict[str, ColumnDefinition] = Field(default_factory=dict)
    business_rules: list[BusinessRuleDefinition] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def parse_column_definitions(cls, value: Any) -> Any:
        """Accept plain dicts in the columns mapping and wrap them."""
        if isinstance(value, dict) and "columns" in value:
            columns = value["columns"]
            if isinstance(columns, dict):
                parsed: dict[str, Any] = {}
                for col_name, col_def in columns.items():
                    if isinstance(col_def, dict):
                        parsed[col_name] = {"name": col_name, **col_def}
                    else:
                        parsed[col_name] = col_def
                value = {**value, "columns": parsed}
        return value

    @property
    def required_columns(self) -> list[str]:
        """Return the list of required column names."""
        return [name for name, col in self.columns.items() if col.required]

    @property
    def optional_columns(self) -> list[str]:
        """Return the list of optional column names."""
        return [name for name, col in self.columns.items() if not col.required]

    @property
    def all_aliases(self) -> dict[str, str]:
        """Return a mapping of every alias to its canonical column name."""
        mapping: dict[str, str] = {}
        for col_name, col in self.columns.items():
            for alias in col.aliases:
                mapping[alias] = col_name
        return mapping

    def to_dict(self) -> dict[str, Any]:
        """Serialize the schema to a plain JSON-compatible dict."""
        return self.model_dump(mode="json")


class SchemaLoader:
    """Loads and parses validation schemas from YAML or dict sources.

    This class is intentionally framework-agnostic: it only knows how to
    transform raw configuration into a validated :class:`ValidationSchema`.
    """

    _DEFAULT_SCHEMA_DIRS: ClassVar[list[str]] = [
        "configs/schemas",
        "configs",
        "schemas",
    ]

    def __init__(self, schema_dir: str | Path | None = None) -> None:
        """Initialize the schema loader.

        Args:
            schema_dir: Optional directory to search for schema files.
        """
        self.schema_dir = schema_dir
        logger.debug(f"SchemaLoader initialized, schema_dir={schema_dir}")

    def load(self, schema: str | Path | dict[str, Any]) -> ValidationSchema:
        """Load a schema from a YAML file path or a dictionary.

        Args:
            schema: Path to a YAML file or an inline schema dictionary.

        Returns:
            A validated :class:`ValidationSchema`.

        Raises:
            SchemaNotFoundException: If the file does not exist.
            InvalidSchemaException: If the schema fails Pydantic validation.
        """
        if isinstance(schema, dict):
            logger.debug("Loading schema from inline dictionary")
            return self._from_dict(schema)

        path = Path(schema)
        resolved = self._resolve_schema_path(path)
        try:
            with open(resolved, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise SchemaNotFoundException(
                f"Schema file not found: {resolved}", details={"path": str(resolved)}
            ) from exc

        if not isinstance(raw, dict):
            raise InvalidSchemaException(
                f"Schema file {resolved} must contain a YAML mapping",
                details={"path": str(resolved)},
            )

        logger.info(f"Loaded schema '{raw.get('name', 'unnamed')}' from {resolved}")
        return self._from_dict(raw)

    def _resolve_schema_path(self, path: Path) -> Path:
        """Resolve a schema path, searching default directories and extensions.

        If the supplied path has no recognized YAML suffix, ``.yaml`` and
        ``.yml`` variants are also searched, enabling callers to pass a bare
        schema name (e.g. ``crop_schema``) that resolves to ``crop_schema.yaml``.
        """
        if path.is_absolute() and path.exists():
            return path.resolve()

        name_variants: list[str] = [path.name]
        if path.suffix.lower() not in (".yaml", ".yml"):
            name_variants.extend([f"{path.name}.yaml", f"{path.name}.yml"])

        candidates: list[Path] = []
        if not path.exists() and self.schema_dir is not None:
            for variant in name_variants:
                candidates.append(resolve_path(self.schema_dir) / variant)
        for base in self._DEFAULT_SCHEMA_DIRS:
            for variant in name_variants:
                candidates.append(resolve_path(Path(base) / variant))
        candidates.append(resolve_path(path))

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return resolve_path(path)

    def _from_dict(self, raw: dict[str, Any]) -> ValidationSchema:
        """Convert a raw dict into a validated ValidationSchema."""
        try:
            schema = ValidationSchema.model_validate(raw)
        except Exception as exc:
            raise InvalidSchemaException(
                f"Invalid schema definition: {exc}", details={"schema": raw}
            ) from exc
        logger.info(
            f"Parsed schema '{schema.name}' v{schema.version} with "
            f"{len(schema.columns)} columns and {len(schema.business_rules)} business rules"
        )
        return schema


def build_schema_from_columns(
    column_configs: dict[str, dict[str, Any]],
    name: str = "inline_schema",
    version: str = "1.0.0",
) -> ValidationSchema:
    """Build a ValidationSchema directly from a column configuration dict.

    Args:
        column_configs: Mapping of column name to ColumnDefinition kwargs.
        name: Schema name.
        version: Schema version.

    Returns:
        A validated :class:`ValidationSchema`.
    """
    logger.debug(f"Building inline schema '{name}' from {len(column_configs)} columns")
    return SchemaLoader()._from_dict(
        {
            "name": name,
            "version": version,
            "columns": {
                col_name: {"name": col_name, **(cfg or {})}
                for col_name, cfg in column_configs.items()
            },
        }
    )
