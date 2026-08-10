"""Tests for individual validation rules."""

import pandas as pd
import pytest

from app.data.validation.exceptions import (
    MissingColumnException,
    RuleRegistrationError,
)
from app.data.validation.models import ValidationSeverity
from app.data.validation.rules import (
    AllowedValuesRule,
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
    crop_name_not_empty,
    date_not_in_future,
    district_not_null,
    humidity_range,
    nitrogen_range,
    phosphorus_range,
    potassium_range,
    rainfall_non_negative,
    state_not_numeric,
    temperature_range,
    yield_non_negative,
)


class TestRequiredColumnRule:
    """Test the RequiredColumnRule."""

    def test_passes_when_present(self) -> None:
        """Test rule passes when all required columns exist."""
        rule = RequiredColumnRule(name="req", required_columns=["a", "b"])
        result = rule.validate(pd.DataFrame({"a": [1], "b": [2]}))
        assert result.passed is True
        assert rule.name == "req"
        assert result.rule_type == "required_column"

    def test_fails_when_missing(self) -> None:
        """Test rule fails when a required column is absent."""
        rule = RequiredColumnRule(name="req", required_columns=["a", "c"])
        result = rule.validate(pd.DataFrame({"a": [1], "b": [2]}))
        assert result.passed is False
        assert result.errors[0].details["missing_columns"] == ["c"]

    def test_from_column_parameter(self) -> None:
        """Test required column is inferred from the column kwarg."""
        rule = RequiredColumnRule(name="req", column="temp")
        result = rule.validate(pd.DataFrame({"temp": [1]}))
        assert result.passed is True


class TestDataTypeRule:
    """Test the DataTypeRule."""

    def test_float_dtype_pass(self) -> None:
        """Test valid float values pass."""
        rule = DataTypeRule(name="dt", column="x", dtype="float")
        result = rule.validate(pd.DataFrame({"x": [1.5, 2.5, 3.0]}))
        assert result.passed is True

    def test_float_dtype_fail_on_string(self) -> None:
        """Test non-numeric strings fail."""
        rule = DataTypeRule(name="dt", column="x", dtype="float")
        result = rule.validate(pd.DataFrame({"x": [1.5, "abc", 3.0]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [1]

    def test_int_dtype_fail_on_decimal(self) -> None:
        """Test decimal values fail int dtype."""
        rule = DataTypeRule(name="dt", column="x", dtype="int")
        result = rule.validate(pd.DataFrame({"x": [1.0, 2.5]}))
        assert result.passed is False

    def test_str_dtype(self) -> None:
        """Test string dtype check."""
        rule = DataTypeRule(name="dt", column="x", dtype="str")
        result = rule.validate(pd.DataFrame({"x": ["a", "b", 42]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [2]

    def test_datetime_dtype(self) -> None:
        """Test datetime dtype check."""
        rule = DataTypeRule(name="dt", column="x", dtype="datetime")
        df = pd.DataFrame({"x": ["2024-01-01", "not-a-date"]})
        result = rule.validate(df)
        assert result.passed is False

    def test_bool_dtype(self) -> None:
        """Test bool dtype check."""
        rule = DataTypeRule(name="dt", column="x", dtype="bool")
        df = pd.DataFrame({"x": [True, False, "yes"]})
        result = rule.validate(df)
        assert result.passed is False

    def test_missing_column_raises(self) -> None:
        """Test a missing column raises MissingColumnException."""
        rule = DataTypeRule(name="dt", column="nope", dtype="float")
        with pytest.raises(MissingColumnException):
            rule.validate(pd.DataFrame({"x": [1.0]}))

    def test_nulls_ignored(self) -> None:
        """Test null values are not flagged by dtype check."""
        rule = DataTypeRule(name="dt", column="x", dtype="float")
        result = rule.validate(pd.DataFrame({"x": [1.5, None, 2.0]}))
        assert result.passed is True


class TestNullValueRule:
    """Test the NullValueRule."""

    def test_no_nulls_passes(self) -> None:
        """Test rule passes when no nulls exist."""
        rule = NullValueRule(name="null", column="x", nullable=False)
        result = rule.validate(pd.DataFrame({"x": [1, 2, 3]}))
        assert result.passed is True

    def test_fails_when_not_nullable(self) -> None:
        """Test fails when nulls exist on a not-nullable column."""
        rule = NullValueRule(name="null", column="x", nullable=False)
        result = rule.validate(pd.DataFrame({"x": [1, None, 3]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [1]

    def test_passes_when_nullable_within_ratio(self) -> None:
        """Test passes when nullable and ratio within threshold."""
        rule = NullValueRule(name="null", column="x", nullable=True, max_null_ratio=0.5)
        result = rule.validate(pd.DataFrame({"x": [1, None, 3]}))
        assert result.passed is True

    def test_fails_when_null_ratio_exceeds(self) -> None:
        """Test fails when null ratio exceeds threshold."""
        rule = NullValueRule(name="null", column="x", nullable=True, max_null_ratio=0.25)
        result = rule.validate(pd.DataFrame({"x": [1, None, None]}))
        assert result.passed is False

    def test_details_report_null_info(self) -> None:
        """Test details include null count and ratio."""
        rule = NullValueRule(name="null", column="x", nullable=False)
        result = rule.validate(pd.DataFrame({"x": [1, None, 3]}))
        assert result.details["null_count"] == 1


class TestDuplicateRule:
    """Test the DuplicateRule."""

    def test_no_duplicates_passes(self) -> None:
        """Test passes when no duplicate rows exist."""
        rule = DuplicateRule(name="dup")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert rule.validate(df).passed is True

    def test_duplicates_detected(self) -> None:
        """Test duplicate rows are flagged."""
        rule = DuplicateRule(name="dup")
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
        result = rule.validate(df)
        assert result.passed is False
        assert result.errors[0].row_indices == [2]

    def test_subset_duplicates(self) -> None:
        """Test duplicate detection over a subset of columns."""
        rule = DuplicateRule(name="dup", subset=["a"])
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        result = rule.validate(df)
        assert result.passed is False
        assert result.errors[0].row_indices == [1]

    def test_max_duplicates_tolerance(self) -> None:
        """Test tolerance for a small number of duplicates."""
        rule = DuplicateRule(name="dup", max_duplicates=1)
        df = pd.DataFrame({"a": [1, 1, 2]})
        result = rule.validate(df)
        assert result.passed is True


class TestUniqueRule:
    """Test the UniqueRule."""

    def test_unique_column_passes(self) -> None:
        """Test unique-valued columns pass."""
        rule = UniqueRule(name="uniq", column="id")
        result = rule.validate(pd.DataFrame({"id": [1, 2, 3]}))
        assert result.passed is True

    def test_duplicates_flagged(self) -> None:
        """Test duplicate values are flagged."""
        rule = UniqueRule(name="uniq", column="id")
        result = rule.validate(pd.DataFrame({"id": [1, 2, 1]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [2]


class TestRangeRule:
    """Test the RangeRule."""

    def test_within_range_passes(self) -> None:
        """Test values within range pass."""
        rule = RangeRule(name="range", column="x", min_value=0, max_value=100)
        result = rule.validate(pd.DataFrame({"x": [0, 50, 100]}))
        assert result.passed is True

    def test_below_min_fails(self) -> None:
        """Test values below min fail."""
        rule = RangeRule(name="range", column="x", min_value=0)
        result = rule.validate(pd.DataFrame({"x": [-5, 10]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [0]

    def test_above_max_fails(self) -> None:
        """Test values above max fail."""
        rule = RangeRule(name="range", column="x", max_value=100)
        result = rule.validate(pd.DataFrame({"x": [10, 150]}))
        assert result.passed is False

    def test_nulls_ignored(self) -> None:
        """Test null values are not flagged."""
        rule = RangeRule(name="range", column="x", min_value=0, max_value=100)
        result = rule.validate(pd.DataFrame({"x": [10, None, 50]}))
        assert result.passed is True

    def test_parameters_override_constructor(self) -> None:
        """Test parameters override constructor args."""
        rule = RangeRule(
            name="range",
            column="x",
            min_value=0,
            max_value=50,
            parameters={"min": 10, "max": 20},
        )
        result = rule.validate(pd.DataFrame({"x": [15, 30]}))
        assert result.passed is False


class TestRegexRule:
    """Test the RegexRule."""

    def test_matching_values_pass(self) -> None:
        """Test values matching the pattern pass."""
        rule = RegexRule(name="re", column="x", pattern=r"^[A-Z]{2}$")
        result = rule.validate(pd.DataFrame({"x": ["AB", "CD"]}))
        assert result.passed is True

    def test_non_matching_fail(self) -> None:
        """Test non-matching values fail."""
        rule = RegexRule(name="re", column="x", pattern=r"^[A-Z]{2}$")
        result = rule.validate(pd.DataFrame({"x": ["AB", "abc"]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [1]

    def test_missing_pattern_raises(self) -> None:
        """Test missing pattern raises RuleRegistrationError."""
        rule = RegexRule(name="re", column="x")
        with pytest.raises(RuleRegistrationError):
            rule.validate(pd.DataFrame({"x": ["AB"]}))


class TestAllowedValuesRule:
    """Test the AllowedValuesRule."""

    def test_allowed_values_pass(self) -> None:
        """Test values in the allowlist pass."""
        rule = AllowedValuesRule(name="allowed", column="x", allowed_values=["a", "b", "c"])
        result = rule.validate(pd.DataFrame({"x": ["a", "b", "c"]}))
        assert result.passed is True

    def test_disallowed_values_fail(self) -> None:
        """Test values outside the allowlist fail."""
        rule = AllowedValuesRule(name="allowed", column="x", allowed_values=["a", "b"])
        result = rule.validate(pd.DataFrame({"x": ["a", "z"]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [1]

    def test_nulls_ignored(self) -> None:
        """Test null values are not flagged."""
        rule = AllowedValuesRule(name="allowed", column="x", allowed_values=["a"])
        result = rule.validate(pd.DataFrame({"x": ["a", None]}))
        assert result.passed is True


class TestMinLengthRule:
    """Test the MinLengthRule."""

    def test_sufficient_length_passes(self) -> None:
        """Test strings meeting the min length pass."""
        rule = MinLengthRule(name="min", column="x", min_length=3)
        result = rule.validate(pd.DataFrame({"x": ["abc", "hello"]}))
        assert result.passed is True

    def test_short_strings_fail(self) -> None:
        """Test strings below the min length fail."""
        rule = MinLengthRule(name="min", column="x", min_length=3)
        result = rule.validate(pd.DataFrame({"x": ["ab", "hello"]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [0]


class TestMaxLengthRule:
    """Test the MaxLengthRule."""

    def test_bounded_length_passes(self) -> None:
        """Test strings within the max length pass."""
        rule = MaxLengthRule(name="max", column="x", max_length=5)
        result = rule.validate(pd.DataFrame({"x": ["abc", "hello"]}))
        assert result.passed is True

    def test_long_strings_fail(self) -> None:
        """Test strings exceeding the max length fail."""
        rule = MaxLengthRule(name="max", column="x", max_length=5)
        result = rule.validate(pd.DataFrame({"x": ["abcdef", "hi"]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [0]


class TestConstantColumnRule:
    """Test the ConstantColumnRule."""

    def test_constant_column_passes(self) -> None:
        """Test a constant column passes."""
        rule = ConstantColumnRule(name="const", column="x")
        result = rule.validate(pd.DataFrame({"x": ["a", "a", "a"]}))
        assert result.passed is True

    def test_variable_column_fails(self) -> None:
        """Test a non-constant column fails."""
        rule = ConstantColumnRule(name="const", column="x")
        result = rule.validate(pd.DataFrame({"x": ["a", "b", "a"]}))
        assert result.passed is False

    def test_expected_constant_value(self) -> None:
        """Test a column matching the expected constant passes."""
        rule = ConstantColumnRule(name="const", column="x", constant_value="fixed")
        result = rule.validate(pd.DataFrame({"x": ["fixed", "fixed"]}))
        assert result.passed is True

    def test_expected_constant_mismatch(self) -> None:
        """Test a column diverging from the expected constant fails."""
        rule = ConstantColumnRule(name="const", column="x", constant_value="fixed")
        result = rule.validate(pd.DataFrame({"x": ["fixed", "other"]}))
        assert result.passed is False


class TestCrossColumnRule:
    """Test the CrossColumnRule."""

    def test_gt_comparison(self) -> None:
        """Test a greater-than cross-column comparison."""
        rule = CrossColumnRule(name="cross", column_a="a", column_b="b", comparison="gt")
        df = pd.DataFrame({"a": [10, 5, 8], "b": [5, 10, 3]})
        result = rule.validate(df)
        assert result.passed is False
        assert result.errors[0].row_indices == [1]

    def test_eq_comparison(self) -> None:
        """Test an equality cross-column comparison."""
        rule = CrossColumnRule(name="cross", column_a="a", column_b="b", comparison="eq")
        df = pd.DataFrame({"a": [1, 2], "b": [1, 3]})
        result = rule.validate(df)
        assert result.passed is False

    def test_invalid_comparison_raises(self) -> None:
        """Test an unsupported comparison raises RuleRegistrationError."""
        rule = CrossColumnRule(name="cross", column_a="a", column_b="b", comparison="??")
        with pytest.raises(RuleRegistrationError):
            rule.validate(pd.DataFrame({"a": [1], "b": [2]}))

    def test_missing_columns_raises(self) -> None:
        """Test missing column refs raise RuleRegistrationError."""
        rule = CrossColumnRule(name="cross", comparison="gt")
        with pytest.raises(RuleRegistrationError):
            rule.validate(pd.DataFrame({"a": [1], "b": [2]}))

    def test_custom_function(self) -> None:
        """Test a custom cross-column function is invoked."""

        def custom(df, parameters, rule):
            return rule.pass_result(df, {"custom": True})

        rule = CrossColumnRule(name="custom", function=custom)
        result = rule.validate(pd.DataFrame({"a": [1]}))
        assert result.passed is True
        assert result.details["custom"] is True


class TestBusinessRule:
    """Test the BusinessRule wrapper."""

    def test_wraps_callable(self) -> None:
        """Test a callable business function is executed."""

        def check(df, parameters, rule):
            return rule.pass_result(df)

        rule = BusinessRule(name="biz", function=check)
        result = rule.validate(pd.DataFrame({"a": [1]}))
        assert result.passed is True

    def test_function_path_import(self) -> None:
        """Test a business rule imports from a dotted path."""
        rule = BusinessRule(
            name="temp",
            function_path="app.data.validation.rules.temperature_range",
            column="temperature",
        )
        df = pd.DataFrame({"temperature": [25.0, 70.0]})
        result = rule.validate(df)
        assert result.passed is False
        assert result.errors[0].row_indices == [1]

    def test_missing_function_raises(self) -> None:
        """Test constructing without a function raises RuleRegistrationError."""
        with pytest.raises(RuleRegistrationError):
            BusinessRule(name="bad")


class TestAgriBusinessRules:
    """Test the built-in agricultural business rule functions."""

    def test_temperature_range(self) -> None:
        """Test temperature bounds."""
        rule = BusinessRule(name="temp", function=temperature_range, column="temperature")
        result = rule.validate(pd.DataFrame({"temperature": [25.0, 70.0, -30.0]}))
        assert result.passed is False
        assert result.errors[0].row_indices == [1, 2]

    def test_humidity_range(self) -> None:
        """Test humidity bounds."""
        rule = BusinessRule(name="hum", function=humidity_range, column="humidity")
        result = rule.validate(pd.DataFrame({"humidity": [50.0, 120.0]}))
        assert result.passed is False

    def test_rainfall_non_negative(self) -> None:
        """Test rainfall must be non-negative."""
        rule = BusinessRule(name="rain", function=rainfall_non_negative, column="rainfall")
        result = rule.validate(pd.DataFrame({"rainfall": [10.0, -5.0]}))
        assert result.passed is False

    def test_nitrogen_range(self) -> None:
        """Test nitrogen bounds."""
        rule = BusinessRule(name="n", function=nitrogen_range, column="nitrogen")
        result = rule.validate(pd.DataFrame({"nitrogen": [100.0, 350.0]}))
        assert result.passed is False

    def test_phosphorus_range(self) -> None:
        """Test phosphorus bounds."""
        rule = BusinessRule(name="p", function=phosphorus_range, column="phosphorus")
        result = rule.validate(pd.DataFrame({"phosphorus": [50.0, 200.0]}))
        assert result.passed is False

    def test_potassium_range(self) -> None:
        """Test potassium bounds."""
        rule = BusinessRule(name="k", function=potassium_range, column="potassium")
        result = rule.validate(pd.DataFrame({"potassium": [100.0, 400.0]}))
        assert result.passed is False

    def test_yield_non_negative(self) -> None:
        """Test yield must be non-negative."""
        rule = BusinessRule(name="y", function=yield_non_negative, column="yield")
        result = rule.validate(pd.DataFrame({"yield": [4.5, -1.0]}))
        assert result.passed is False

    def test_crop_name_not_empty(self) -> None:
        """Test crop name cannot be empty."""
        rule = BusinessRule(name="crop", function=crop_name_not_empty, column="crop_name")
        result = rule.validate(pd.DataFrame({"crop_name": ["Rice", "   ", None]}))
        assert result.passed is False

    def test_state_not_numeric(self) -> None:
        """Test state cannot be numeric."""
        rule = BusinessRule(name="state", function=state_not_numeric, column="state")
        result = rule.validate(pd.DataFrame({"state": ["Punjab", "123"]}))
        assert result.passed is False

    def test_district_not_null(self) -> None:
        """Test district cannot be null."""
        rule = BusinessRule(name="district", function=district_not_null, column="district")
        result = rule.validate(pd.DataFrame({"district": ["Ludhiana", None]}))
        assert result.passed is False

    def test_date_not_in_future(self) -> None:
        """Test date cannot be in the future."""
        rule = BusinessRule(name="date", function=date_not_in_future, column="planting_date")
        df = pd.DataFrame({"planting_date": pd.to_datetime(["2024-01-01", "2030-01-01"])})
        result = rule.validate(df)
        assert result.passed is False
        assert result.errors[0].row_indices == [1]


class TestRuleRegistry:
    """Test the RuleRegistry."""

    def test_register_and_create(self) -> None:
        """Test registering and creating a rule."""
        registry = RuleRegistry()
        registry.register("range", RangeRule)
        rule = registry.create("range", name="r", column="x", min_value=0)
        assert isinstance(rule, RangeRule)

    def test_register_non_callable_raises(self) -> None:
        """Test registering a non-callable raises RuleRegistrationError."""
        registry = RuleRegistry()
        with pytest.raises(RuleRegistrationError):
            registry.register("bad", "not-callable")

    def test_create_unregistered_raises(self) -> None:
        """Test creating an unregistered rule raises."""
        registry = RuleRegistry()
        with pytest.raises(RuleRegistrationError):
            registry.create("unknown_type")

    def test_has_and_select(self) -> None:
        """Test has and select."""
        registry = RuleRegistry()
        registry.register("range", RangeRule)
        assert registry.has("range") is True
        assert registry.has("nope") is False
        assert registry.select(["range", "nope"]) == ["range"]

    def test_unregister(self) -> None:
        """Test unregistering a rule."""
        registry = RuleRegistry()
        registry.register("range", RangeRule)
        registry.unregister("range")
        assert registry.has("range") is False

    def test_rule_types(self) -> None:
        """Test listing rule types."""
        registry = RuleRegistry()
        registry.register("b", RangeRule)
        registry.register("a", UniqueRule)
        assert registry.rule_types() == ["a", "b"]

    def test_default_registry_contains_builtins(self) -> None:
        """Test the default registry contains all expected rule types."""
        from app.data.validation.rules import DEFAULT_REGISTRY

        for rt in [
            "required_column",
            "dtype",
            "null_value",
            "duplicate",
            "unique",
            "range",
            "regex",
            "allowed_values",
            "min_length",
            "max_length",
            "constant_column",
            "business",
            "cross_column",
            "business.temperature",
            "business.humidity",
        ]:
            assert DEFAULT_REGISTRY.has(rt), f"Missing rule type: {rt}"


class TestBaseRule:
    """Test base rule helpers."""

    def test_make_error(self) -> None:
        """Test make_error produces a structured ValidationError."""
        rule = RangeRule(name="r", column="x", severity="warning")
        error = rule.make_error("boom", row_indices=[0, 1], details={"a": 1})
        assert error.rule_name == "r"
        assert error.severity == ValidationSeverity.WARNING
        assert error.details == {"a": 1}

    def test_severity_coercion(self) -> None:
        """Test string severities are coerced."""
        rule = RangeRule(name="r", severity="critical")
        assert rule.severity == ValidationSeverity.CRITICAL
