"""Tests for data cleaning module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from app.data.cleaning import (
    DataCleaner,
    CleaningPipeline,
    CleaningConfig,
    CleaningMetadata,
    MissingValueStrategy,
    OutlierStrategy,
    MissingValueHandler,
    OutlierHandler,
    DuplicateHandler,
    ColumnNameStandardizer,
    UnitConverter,
    DataTypeConverter,
    TextCleaner,
    CleaningReportGenerator,
)


class TestMissingValueHandler:
    """Tests for MissingValueHandler."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'categorical': ['a', 'b', 'c', np.nan, 'e'],
            'mixed': [1, 'b', np.nan, 4, 'e']
        })
    
    def test_drop_row_strategy(self):
        handler = MissingValueHandler(strategy=MissingValueStrategy.DROP_ROW)
        result = handler.apply(self.data)
        assert len(result) == 3  # Dropped rows with missing values
    
    def test_mean_strategy(self):
        handler = MissingValueHandler(strategy=MissingValueStrategy.MEAN)
        result = handler.apply(self.data, column='numeric')
        assert result['numeric'].isna().sum() == 0
        assert result['numeric'].iloc[2] == 3.0  # Mean of [1,2,4,5]
    
    def test_median_strategy(self):
        handler = MissingValueHandler(strategy=MissingValueStrategy.MEDIAN)
        result = handler.apply(self.data, column='numeric')
        assert result['numeric'].isna().sum() == 0
        assert result['numeric'].iloc[2] == 3.0  # Median of [1,2,4,5]
    
    def test_constant_strategy(self):
        handler = MissingValueHandler(
            strategy=MissingValueStrategy.CONSTANT,
            constant_value=0
        )
        result = handler.apply(self.data, column='numeric')
        assert result['numeric'].isna().sum() == 0
        assert result['numeric'].iloc[2] == 0
    
    def test_forward_fill_strategy(self):
        handler = MissingValueHandler(strategy=MissingValueStrategy.FORWARD_FILL)
        result = handler.apply(self.data, column='numeric')
        assert result['numeric'].isna().sum() == 0
        assert result['numeric'].iloc[2] == 2  # Forward fill from previous value
    
    def test_backward_fill_strategy(self):
        handler = MissingValueHandler(strategy=MissingValueStrategy.BACKWARD_FILL)
        result = handler.apply(self.data, column='numeric')
        assert result['numeric'].isna().sum() == 0
        assert result['numeric'].iloc[2] == 4  # Backward fill from next value
    
    def test_interpolate_strategy(self):
        handler = MissingValueHandler(strategy=MissingValueStrategy.INTERPOLATE)
        result = handler.apply(self.data, column='numeric')
        assert result['numeric'].isna().sum() == 0
        assert result['numeric'].iloc[2] == 3.0  # Linear interpolation


class TestOutlierHandler:
    """Tests for OutlierHandler."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'with_outliers': [1, 2, 3, 100, 5, 6, 7, 8, 9, 10],
        })
    
    def test_iqr_outlier_detection(self):
        handler = OutlierHandler(strategy=OutlierStrategy.IQR)
        result = handler.apply(self.data, column='with_outliers')
        # Should detect and handle the outlier (100)
        assert len(result) == 10  # Keep method by default
    
    def test_remove_outliers(self):
        handler = OutlierHandler(strategy=OutlierStrategy.REMOVE)
        result = handler.apply(self.data, column='with_outliers')
        assert len(result) < 10  # Should remove the outlier
    
    def test_clip_outliers(self):
        handler = OutlierHandler(strategy=OutlierStrategy.CLIP)
        result = handler.apply(self.data, column='with_outliers')
        # Outlier should be clipped to within bounds
        assert result['with_outliers'].max() < 20
    
    def test_winsorize_outliers(self):
        handler = OutlierHandler(strategy=OutlierStrategy.WINSORIZE)
        result = handler.apply(self.data, column='with_outliers')
        # Outlier should be winsorized
        assert result['with_outliers'].max() < 20


class TestDuplicateHandler:
    """Tests for DuplicateHandler."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'id': [1, 2, 3, 1, 2, 4],
            'value': ['a', 'b', 'c', 'a', 'b', 'd']
        })
    
    def test_remove_duplicates(self):
        handler = DuplicateHandler()
        result = handler.apply(self.data)
        assert len(result) == 4  # Removed duplicates (id=1,2)
        assert result['id'].tolist() == [1, 2, 3, 4]
    
    def test_keep_first(self):
        handler = DuplicateHandler(keep='first')
        result = handler.apply(self.data)
        # First occurrence of each duplicate is kept
        assert len(result) == 4
        assert result.iloc[0]['id'] == 1
    
    def test_keep_last(self):
        handler = DuplicateHandler(keep='last')
        result = handler.apply(self.data)
        # Last occurrence of each duplicate is kept
        assert len(result) == 4
    
    def test_subset_duplicates(self):
        handler = DuplicateHandler(subset=['value'])
        result = handler.apply(self.data)
        # Removes duplicates based on 'value' column only
        assert len(result) == 4  # a appears twice, so one removed


class TestColumnNameStandardizer:
    """Tests for ColumnNameStandardizer."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'Temp (°C)': [1, 2, 3],
            'Rainfall (mm)': [4, 5, 6],
            'Crop Yield': [7, 8, 9]
        })
    
    def test_standardize_column_names(self):
        standardizer = ColumnNameStandardizer(
            case='lower',
            replace_spaces='_',
            remove_special=True
        )
        result = standardizer.transform(self.data)
        expected_columns = ['temp_c', 'rainfall_mm', 'crop_yield']
        assert result.columns.tolist() == expected_columns
    
    def test_case_normalization(self):
        standardizer = ColumnNameStandardizer(case='upper')
        result = standardizer.transform(self.data)
        assert all(col.isupper() for col in result.columns)
    
    def test_alias_mapping(self):
        alias_mapping = {
            'temperature': ['Temp (°C)', 'temp_c'],
            'precipitation': ['Rainfall (mm)']
        }
        standardizer = ColumnNameStandardizer(
            alias_mapping=alias_mapping,
            case='lower'
        )
        result = standardizer.transform(self.data)
        assert 'temperature' in result.columns
        assert 'precipitation' in result.columns


class TestUnitConverter:
    """Tests for UnitConverter."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'temperature_c': [0, 25, 100],
            'length_m': [1, 2, 3],
            'weight_kg': [1, 2, 3]
        })
    
    def test_temperature_conversion(self):
        converter = UnitConverter(
            conversion_mappings={
                'temperature_c': [
                    {'from_unit': 'celsius', 'to_unit': 'fahrenheit'}
                ]
            }
        )
        result = converter.transform(self.data)
        # 0°C = 32°F, 25°C = 77°F, 100°C = 212°F
        expected = [32, 77, 212]
        assert result['temperature_c'].tolist() == expected
    
    def test_length_conversion(self):
        converter = UnitConverter(
            conversion_mappings={
                'length_m': [
                    {'from_unit': 'meter', 'to_unit': 'cm'}
                ]
            }
        )
        result = converter.transform(self.data)
        expected = [100, 200, 300]
        assert result['length_m'].tolist() == expected
    
    def test_weight_conversion(self):
        converter = UnitConverter(
            conversion_mappings={
                'weight_kg': [
                    {'from_unit': 'kg', 'to_unit': 'gram'}
                ]
            }
        )
        result = converter.transform(self.data)
        expected = [1000, 2000, 3000]
        assert result['weight_kg'].tolist() == expected


class TestDataTypeConverter:
    """Tests for DataTypeConverter."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'numeric_string': ['1', '2', '3'],
            'date_string': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'mixed': ['1', '2', '3.5']
        })
    
    def test_numeric_conversion(self):
        converter = DataTypeConverter(convert_numeric=True)
        result = converter.transform(self.data)
        assert pd.api.types.is_numeric_dtype(result['numeric_string'])
    
    def test_date_conversion(self):
        converter = DataTypeConverter(convert_dates=True)
        result = converter.transform(self.data)
        assert pd.api.types.is_datetime64_any_dtype(result['date_string'])
    
    def test_custom_mapping(self):
        converter = DataTypeConverter(
            custom_mappings={
                'numeric_string': 'float64',
                'mixed': 'float64'
            }
        )
        result = converter.transform(self.data)
        assert result['numeric_string'].dtype == 'float64'
        assert result['mixed'].dtype == 'float64'


class TestTextCleaner:
    """Tests for TextCleaner."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'text': [
                '  Hello World!  ',
                '  Extra   Spaces  ',
                'Special@#$Characters',
                None,
                '  '
            ]
        })
    
    def test_trim_text(self):
        cleaner = TextCleaner(trim=True, remove_whitespace=False)
        result = cleaner.transform(self.data)
        assert result['text'].iloc[0] == 'Hello World!'
    
    def test_remove_whitespace(self):
        cleaner = TextCleaner(trim=False, remove_whitespace=True)
        result = cleaner.transform(self.data)
        assert result['text'].iloc[1] == 'Extra Spaces'
    
    def test_case_normalization(self):
        cleaner = TextCleaner(case_normalization='lower')
        result = cleaner.transform(self.data)
        assert result['text'].iloc[2].islower()
    
    def test_remove_special_characters(self):
        cleaner = TextCleaner(remove_special_chars=True)
        result = cleaner.transform(self.data)
        assert '#' not in result['text'].iloc[2]
        assert '$' not in result['text'].iloc[2]
    
    def test_null_string_conversion(self):
        cleaner = TextCleaner(null_string_conversion=True)
        data = pd.DataFrame({'text': ['nan', 'None', 'NULL', 'null', '']})
        result = cleaner.transform(data)
        assert result['text'].isna().sum() == 5


class TestDataCleaner:
    """Tests for DataCleaner."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'Temp (°C)': [25, np.nan, 30, 1000, 28],
            'Rainfall (mm)': [10, 20, None, 30, 15],
            'Soil pH': [6.5, 7.0, 6.8, 6.9, 7.1],
            'Crop': ['Wheat', 'Corn', 'Soybean', 'Wheat', 'Corn'],
        })
        self.config = CleaningConfig()
    
    def test_clean_basic(self):
        cleaner = DataCleaner(self.config)
        result = cleaner.clean(self.data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert cleaner.metadata is not None
    
    def test_clean_with_missing_values(self):
        config = CleaningConfig()
        config.missing_values = {
            'Temp (°C)': {
                'enabled': True,
                'strategy': 'mean',
                'params': {}
            },
            'Rainfall (mm)': {
                'enabled': True,
                'strategy': 'median',
                'params': {}
            }
        }
        cleaner = DataCleaner(config)
        result = cleaner.clean(self.data)
        assert result['Temp (°C)'].isna().sum() == 0
        assert result['Rainfall (mm)'].isna().sum() == 0
    
    def test_clean_with_outliers(self):
        config = CleaningConfig()
        config.outlier_handling = {
            'Temp (°C)': {
                'enabled': True,
                'strategy': 'clip',
                'params': {}
            }
        }
        cleaner = DataCleaner(config)
        result = cleaner.clean(self.data)
        # The outlier 1000 should be clipped
        assert result['Temp (°C)'].max() < 50
    
    def test_clean_with_column_standardization(self):
        config = CleaningConfig()
        config.column_standardization = {
            'enabled': True,
            'params': {'case': 'lower', 'replace_spaces': '_'}
        }
        cleaner = DataCleaner(config)
        result = cleaner.clean(self.data)
        expected_columns = ['temp_c', 'rainfall_mm', 'soil_ph', 'crop']
        assert result.columns.tolist() == expected_columns
    
    def test_metadata_generation(self):
        cleaner = DataCleaner(self.config)
        cleaner.clean(self.data)
        metadata = cleaner.get_metadata()
        assert metadata is not None
        assert metadata.rows_before == len(self.data)
        assert metadata.rows_after == len(self.data)  # No removal
        assert len(metadata.columns_before) == len(self.data.columns)


class TestCleaningPipeline:
    """Tests for CleaningPipeline."""
    
    def setup_method(self):
        """Set up test data."""
        self.data1 = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        self.data2 = pd.DataFrame({
            'x': [7, 8, 9],
            'y': [10, 11, 12]
        })
    
    def test_clean_single(self):
        pipeline = CleaningPipeline()
        result = pipeline.clean_single(self.data1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_clean_multiple(self):
        pipeline = CleaningPipeline()
        results = pipeline.clean_multiple([self.data1, self.data2])
        assert len(results) == 2
        assert isinstance(results[0], pd.DataFrame)
        assert isinstance(results[1], pd.DataFrame)
    
    def test_clean_sequential(self):
        pipeline = CleaningPipeline(parallel=False)
        results = pipeline.clean_multiple([self.data1, self.data2])
        assert len(results) == 2
        assert pipeline.results is not None
    
    def test_get_summary(self):
        pipeline = CleaningPipeline()
        pipeline.clean_single(self.data1)
        summary = pipeline.get_summary()
        assert 'total_datasets' in summary
        assert summary['total_datasets'] == 1


class TestCleaningReportGenerator:
    """Tests for CleaningReportGenerator."""
    
    def setup_method(self):
        """Set up test data."""
        self.metadata = CleaningMetadata(
            dataset_name="test_dataset",
            source_path="test.csv",
            output_path="interim/test_cleaned.csv",
            rows_before=100,
            rows_after=95,
            columns_before=['a', 'b', 'c'],
            columns_after=['a', 'b'],
            operations=[],
            missing_values_fixed={'a': 5},
            duplicates_removed=2,
            outliers_handled={'b': 3},
            datatype_changes={'c': {'old': 'object', 'new': 'float64'}},
            start_time=datetime.now(),
            end_time=datetime.now(),
            execution_time_seconds=1.5,
            config=CleaningConfig(),
            validation_status="completed",
            warnings=[],
            errors=[]
        )
    
    def test_generate_markdown_report(self):
        generator = CleaningReportGenerator()
        report = generator.generate_report(self.metadata)
        assert isinstance(report, str)
        assert "test_dataset" in report
        assert "100" in report  # rows_before
        assert "95" in report   # rows_after
    
    def test_generate_html_report(self):
        generator = CleaningReportGenerator()
        report = generator.generate_html_report(self.metadata)
        assert isinstance(report, str)
        assert "<html" in report
        assert "test_dataset" in report
    
    def test_generate_json_report(self):
        generator = CleaningReportGenerator()
        report = generator.generate_json_report(self.metadata)
        assert isinstance(report, str)
        assert '"dataset_name": "test_dataset"' in report
    
    def test_save_report(self, tmp_path):
        generator = CleaningReportGenerator()
        saved_paths = generator.save_report(
            self.metadata,
            output_dir=tmp_path,
            formats=["md", "html", "json"]
        )
        assert "md" in saved_paths
        assert "html" in saved_paths
        assert "json" in saved_paths
        assert saved_paths["md"].exists()
        assert saved_paths["html"].exists()
        assert saved_paths["json"].exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.data.cleaning"])