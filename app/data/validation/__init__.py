"""AgriMind AI Data Validation Framework.

This package provides an enterprise-grade validation layer that executes
after data ingestion and before cleaning. It supports schema-driven
validation, custom agricultural business rules, fail-fast / strict modes,
and rich report generation.
"""

from app.data.validation.exceptions import (
    BusinessRuleViolation,
    DuplicateColumnException,
    InvalidSchemaException,
    MissingColumnException,
    RuleRegistrationError,
    SchemaNotFoundException,
    ValidationException,
)
from app.data.validation.models import (
    ColumnStatistic,
    RuleStatistic,
    ValidationError,
    ValidationReport,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
    ValidationSummary,
)
from app.data.validation.report import ValidationReportGenerator
from app.data.validation.rules import (
    DEFAULT_REGISTRY,
    AllowedValuesRule,
    BaseRule,
    BusinessRule,
    ConstantColumnRule,
    CrossColumnRule,
    DataTypeRule,
    DuplicateRule,
    MaxLengthRule,
    MinLengthRule,
    NullValueRule,
    RangeRule,
    RegexRule,
    RequiredColumnRule,
    RuleRegistry,
    UniqueRule,
    build_rules_from_schema,
)
from app.data.validation.schema import (
    BusinessRuleDefinition,
    ColumnDefinition,
    SchemaLoader,
    ValidationSchema,
    build_schema_from_columns,
)
from app.data.validation.validator import ValidationEngine, ValidationEngineConfig

__all__ = [
    # Exceptions
    "BusinessRuleViolation",
    "DuplicateColumnException",
    "InvalidSchemaException",
    "MissingColumnException",
    "RuleRegistrationError",
    "SchemaNotFoundException",
    "ValidationException",
    # Models
    "ColumnStatistic",
    "RuleStatistic",
    "ValidationError",
    "ValidationReport",
    "ValidationResult",
    "ValidationRule",
    "ValidationSeverity",
    "ValidationSummary",
    # Rules
    "AllowedValuesRule",
    "BaseRule",
    "BusinessRule",
    "ConstantColumnRule",
    "CrossColumnRule",
    "DEFAULT_REGISTRY",
    "DataTypeRule",
    "DuplicateRule",
    "MaxLengthRule",
    "MinLengthRule",
    "NullValueRule",
    "RangeRule",
    "RegexRule",
    "RequiredColumnRule",
    "RuleRegistry",
    "UniqueRule",
    "build_rules_from_schema",
    # Schema
    "BusinessRuleDefinition",
    "ColumnDefinition",
    "SchemaLoader",
    "ValidationSchema",
    "build_schema_from_columns",
    # Engine + reports
    "ValidationEngine",
    "ValidationEngineConfig",
    "ValidationReportGenerator",
]
