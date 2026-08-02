# tests/feature_engineering/test_feature_generator.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from app.data.feature_engineering.feature_generator import FeatureGenerator
from app.data.feature_engineering.feature_registry import FeatureRegistry
from app.data.feature_engineering.models import (
    FeatureDefinition, FeatureType, EncodingType, ScalingType
)


class TestFeatureGenerator:
    """Test suite for FeatureGenerator."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'temperature': np.random.normal(25, 5, 100),
            'rainfall': np.random.exponential(50, 100),
            'humidity': np.random.uniform(30, 90, 100),
            'n': np.random.normal(30, 10, 100),
            'p': np.random.normal(15, 5, 100),
            'k': np.random.normal(20, 7, 100),
            'ph': np.random.uniform(5.5, 7.5, 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'crop_type': np.random.choice(['wheat', 'corn', 'soybean'], 100),
            'region': np.random.choice(['north', 'south', 'east', 'west'], 100)
        })
    
    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Create temporary feature registry."""
        registry_path = tmp_path / 'registry'
        return FeatureRegistry(registry_path)
    
    @pytest.fixture
    def feature_generator(self, temp_registry):
        """Create feature generator with temporary registry."""
        return FeatureGenerator(temp_registry)
    
    def test_generate_numerical_feature(self, feature_generator, sample_dataframe):
        """Test numerical feature generation."""
        definition = FeatureDefinition(
            name='temp_rain_interaction',
            description='Temperature and rainfall interaction',
            feature_type=FeatureType.NUMERICAL,
            source_columns=['temperature', 'rainfall'],
            transformation='multiplication'
        )
        
        result = feature_generator._generate_numerical_feature(
            sample_dataframe, definition
        )
        
        assert result is not None
        assert len(result) == len(sample_dataframe)
        assert isinstance(result, pd.Series)
    
    def test_generate_categorical_feature(self, feature_generator, sample_dataframe):
        """Test categorical feature generation."""
        definition = FeatureDefinition(
            name='crop_region',
            description='Crop and region combination',
            feature_type=FeatureType.CATEGORICAL,
            source_columns=['crop_type', 'region'],
            transformation='concatenation'
        )
        
        result = feature_generator._generate_categorical_feature(
            sample_dataframe, definition
        )
        
        assert result is not None
        assert len(result) == len(sample_dataframe)
        assert result.dtype.name == 'category'
    
    def test_generate_date_feature(self, feature_generator, sample_dataframe):
        """Test date feature generation."""
        definition = FeatureDefinition(
            name='day_of_year',
            description='Day of year from date',
            feature_type=FeatureType.DATE,
            source_columns=['date'],
            transformation='day_of_year',
            parameters={'component': 'day_of_year'}
        )
        
        result = feature_generator._generate_date_feature(
            sample_dataframe, definition
        )
        
        assert result is not None
        assert len(result) == len(sample_dataframe)
        assert result.min() >= 1
        assert result.max() <= 365
    
    def test_generate_agricultural_features(self, feature_generator, sample_dataframe):
        """Test agricultural feature generation."""
        # Test climate index
        definition = FeatureDefinition(
            name='climate_index',
            description='Climate index',
            feature_type=FeatureType.DOMAIN,
            source_columns=['temperature', 'rainfall'],
            transformation='climate_index',
            parameters={
                'agricultural_type': 'climate_index',
                'temperature_col': 'temperature',
                'rainfall_col': 'rainfall'
            }
        )
        
        result = feature_generator._generate_agricultural_feature(
            sample_dataframe, definition
        )
        
        assert result is not None
        assert len(result) == len(sample_dataframe)
        assert result.min() >= 0
        assert result.max() <= 100
    
    def test_generate_soil_fertility(self, feature_generator, sample_dataframe):
        """Test soil fertility calculation."""
        definition = FeatureDefinition(
            name='soil_fertility',
            description='Soil fertility score',
            feature_type=FeatureType.DOMAIN,
            source_columns=['n', 'p', 'k', 'ph'],
            transformation='soil_fertility',
            parameters={
                'agricultural_type': 'soil_fertility',
                'components': ['n', 'p', 'k', 'ph'],
                'weights': [0.4, 0.3, 0.2, 0.1]
            }
        )
        
        result = feature_generator._generate_agricultural_feature(
            sample_dataframe, definition
        )
        
        assert result is not None
        assert len(result) == len(sample_dataframe)
        assert result.min() >= 0
        assert result.max() <= 100
    
    def test_feature_registration(self, feature_generator, sample_dataframe):
        """Test feature registration in registry."""
        definitions = [
            FeatureDefinition(
                name='climate_index_test',
                description='Test climate index',
                feature_type=FeatureType.DOMAIN,
                source_columns=['temperature', 'rainfall'],
                transformation='climate_index',
                parameters={
                    'agricultural_type': 'climate_index',
                    'temperature_col': 'temperature',
                    'rainfall_col': 'rainfall'
                }
            )
        ]
        
        result_df = feature_generator.generate_features(
            sample_dataframe,
            definitions,
            owner='test_user',
            version='1.0.0'
        )
        
        # Check that feature was added to dataframe
        assert 'climate_index_test' in result_df.columns
        
        # Check that feature was registered
        metadata = feature_generator.registry.get_feature('climate_index_test')
        assert metadata is not None
        assert metadata.feature_name == 'climate_index_test'
        assert metadata.owner == 'test_user'
        assert metadata.version == '1.0.0'
    
    def test_get_feature_set(self, feature_generator, sample_dataframe):
        """Test retrieving feature set."""
        definitions = [
            FeatureDefinition(
                name='feature_1',
                description='Feature 1',
                feature_type=FeatureType.NUMERICAL,
                source_columns=['temperature'],
                transformation='identity'
            ),
            FeatureDefinition(
                name='feature_2',
                description='Feature 2',
                feature_type=FeatureType.NUMERICAL,
                source_columns=['rainfall'],
                transformation='identity'
            )
        ]
        
        feature_generator.generate_features(
            sample_dataframe, definitions,
            owner='test_user', version='1.0.0'
        )
        
        feature_set = feature_generator.get_feature_set(
            ['feature_1', 'feature_2']
        )
        
        assert feature_set is not None
        assert len(feature_set.features) == 2
        assert feature_set.version == '1.0.0'
    
    def test_feature_versioning(self, feature_generator, sample_dataframe):
        """Test feature versioning."""
        definition = FeatureDefinition(
            name='versioned_feature',
            description='Versioned feature',
            feature_type=FeatureType.NUMERICAL,
            source_columns=['temperature'],
            transformation='identity'
        )
        
        # Generate first version
        feature_generator.generate_features(
            sample_dataframe, [definition],
            owner='test_user', version='1.0.0'
        )
        
        # Generate second version
        feature_generator.generate_features(
            sample_dataframe, [definition],
            owner='test_user', version='2.0.0'
        )
        
        # Check both versions exist
        v1 = feature_generator.registry.get_feature('versioned_feature', '1.0.0')
        v2 = feature_generator.registry.get_feature('versioned_feature', '2.0.0')
        
        assert v1.version == '1.0.0'
        assert v2.version == '2.0.0'
    
    @pytest.mark.parametrize("feature_type", [
        FeatureType.NUMERICAL,
        FeatureType.CATEGORICAL,
        FeatureType.DATE,
        FeatureType.DOMAIN
    ])
    def test_feature_type_handling(self, feature_generator, sample_dataframe, feature_type):
        """Test handling of different feature types."""
        definition = FeatureDefinition(
            name=f'test_{feature_type}',
            description=f'Test {feature_type} feature',
            feature_type=feature_type,
            source_columns=['temperature'],
            transformation='identity'
        )
        
        if feature_type == FeatureType.DOMAIN:
            definition.parameters = {
                'agricultural_type': 'climate_index',
                'temperature_col': 'temperature',
                'rainfall_col': 'rainfall'
            }
        
        result = feature_generator._generate_feature_by_type(
            sample_dataframe, definition
        )
        
        if result is not None:
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_dataframe)