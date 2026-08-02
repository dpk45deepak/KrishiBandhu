"""Tests for standardization framework."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile

from app.data.transformation.standardizer import Standardizer
from app.data.transformation.registry import RegistryManager
from app.data.transformation.metadata import MetadataGenerator
from app.data.transformation.converters import UnitConverter, CategoryNormalizer
from app.data.transformation.mapper import AliasMapper


class TestStandardizationFramework:
    """Test suite for standardization framework."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample agricultural data."""
        return pd.DataFrame({
            'Temp': [25, 28, 30, 22, 26],
            'Rainfall': [120, 85, 95, 140, 110],
            'Crop': ['Rice', 'Wheat', 'Rice', 'Maize', 'Paddy'],
            'Yield_kg_ha': [4500, 3800, 5200, 4100, 4700],
            'Soil_type': ['Clay', 'Sandy', 'Loam', 'Clayey', 'Silt'],
            'Date': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05']
        })
    
    @pytest.fixture
    def temp_schema_dir(self):
        """Create temporary schema directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "schemas"
            schema_path.mkdir()
            
            # Create test schema
            schema_content = """
            schema_name: test
            schema_version: 1.0.0
            description: Test schema
            source: test
            owner: test
            license: test
            
            columns:
              - name: temperature
                data_type: float
                unit: celsius
                unit_category: temperature
                required: true
              - name: rainfall
                data_type: float
                unit: mm
                unit_category: length
                required: true
              - name: crop_type
                data_type: category
                required: true
              - name: yield
                data_type: float
                unit: kg/ha
                required: true
                is_target: true
              - name: soil_type
                data_type: category
                required: false
              - name: date
                data_type: date
                required: true
            """
            
            with open(schema_path / "test.yaml", 'w') as f:
                f.write(schema_content)
            
            yield str(schema_path)
    
    def test_alias_mapper(self):
        """Test column alias mapping."""
        mapper = AliasMapper()
        
        # Test mapping
        assert mapper.map_column('temp') == 'temperature'
        assert mapper.map_column('TEMP') == 'temperature'
        assert mapper.map_column('rainfall') == 'rainfall'
        assert mapper.map_column('AnnualRainfall') == 'rainfall'
        assert mapper.map_column('humidity') == 'humidity'
        
        # Test unknown column
        unknown = mapper.map_column('unknown_column')
        assert unknown == 'unknown_column'
    
    def test_unit_converter(self):
        """Test unit conversions."""
        converter = UnitConverter()
        
        # Test length conversions
        result = converter.convert(1000, 'mm', 'meter', 'length')
        assert result == 1.0
        
        result = converter.convert(2.5, 'meter', 'centimeter', 'length')
        assert result == 250.0
        
        # Test area conversions
        result = converter.convert(10, 'hectare', 'acre', 'area')
        assert abs(result - 24.7105) < 0.001
        
        # Test weight conversions
        result = converter.convert(1000, 'kg', 'ton', 'weight')
        assert result == 1.0
        
        # Test temperature conversions
        result = converter.convert(32, 'fahrenheit', 'celsius', 'temperature')
        assert result == 0.0
        
        result = converter.convert(0, 'celsius', 'fahrenheit', 'temperature')
        assert result == 32.0
    
    def test_category_normalizer(self):
        """Test category normalization."""
        normalizer = CategoryNormalizer()
        
        # Test crop names
        assert normalizer.normalize('rice', 'crop') == 'Rice'
        assert normalizer.normalize('Paddy', 'crop') == 'Rice'
        assert normalizer.normalize('maize', 'crop') == 'Corn'
        assert normalizer.normalize('soy', 'crop') == 'Soybean'
        
        # Test soil types
        assert normalizer.normalize('clay', 'soil') == 'Clay'
        assert normalizer.normalize('loam', 'soil') == 'Loamy'
        assert normalizer.normalize('sandy', 'soil') == 'Sandy'
    
    def test_standardizer_basic(self, sample_data, temp_schema_dir):
        """Test basic standardization functionality."""
        standardizer = Standardizer(temp_schema_dir)
        
        df, report = standardizer.standardize(sample_data, 'test')
        
        # Check columns were renamed
        assert 'temperature' in df.columns
        assert 'rainfall' in df.columns
        assert 'crop_type' in df.columns
        assert 'yield' in df.columns
        assert 'soil_type' in df.columns
        assert 'date' in df.columns
        
        # Check column mapping
        assert 'Temp' in report.column_mappings
        assert report.column_mappings['Temp'] == 'temperature'
    
    def test_metadata_generation(self, sample_data):
        """Test metadata generation."""
        metadata_gen = MetadataGenerator()
        
        metadata = metadata_gen.generate_metadata(
            df=sample_data,
            source='test',
            schema_name='test'
        )
        
        assert metadata['rows'] == 5
        assert metadata['columns'] == 6
        assert 'schema_name' in metadata
        assert 'quality' in metadata
        assert 'completeness' in metadata['quality']
        assert 'uniqueness' in metadata['quality']
    
    def test_registry_operations(self, sample_data, temp_schema_dir):
        """Test registry operations."""
        # Create temporary registry
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry"
            registry = RegistryManager(str(registry_path))
            
            # Create metadata generator
            metadata_gen = MetadataGenerator(registry)
            
            # Create registry entry
            entry = metadata_gen.create_registry_entry(
                df=sample_data,
                file_path='test.csv',
                schema_name='test',
                source='test'
            )
            
            # Register dataset
            dataset_id = registry.register_dataset(entry)
            assert dataset_id is not None
            
            # List datasets
            datasets = registry.list_datasets()
            assert len(datasets) == 1
            assert datasets[0]['dataset_id'] == dataset_id
            
            # Get dataset
            retrieved = registry.get_dataset(dataset_id)
            assert retrieved is not None
            assert retrieved.rows == 5
            assert retrieved.columns == 6
    
    def test_error_handling(self, temp_schema_dir):
        """Test error handling for invalid data."""
        standardizer = Standardizer(temp_schema_dir)
        
        # Test with missing required column
        invalid_df = pd.DataFrame({
            'Temp': [25, 28, 30],
            'Rainfall': [120, 85, 95],
            # Missing crop_type
            'Yield_kg_ha': [4500, 3800, 5200]
        })
        
        df, report = standardizer.standardize(invalid_df, 'test')
        
        # Check that missing columns were added
        assert 'crop_type' in df.columns
        assert df['crop_type'].isna().all()
    
    def test_batch_processing(self, sample_data, temp_schema_dir):
        """Test batch processing capabilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            
            for i in range(3):
                df = sample_data.copy()
                df['Temp'] = df['Temp'] + i
                df.to_csv(data_dir / f"test_{i}.csv", index=False)
            
            # Process batch
            standardizer = Standardizer(temp_schema_dir)
            
            for file in data_dir.glob("*.csv"):
                output_path = data_dir / f"standardized_{file.stem}.csv"
                report = standardizer.standardize_file(
                    input_path=str(file),
                    output_path=str(output_path),
                    schema_name='test',
                    source=file.stem
                )
                
                assert output_path.exists()
                assert report.total_rows == 5


if __name__ == "__main__":
    pytest.main([__file__, '-v'])