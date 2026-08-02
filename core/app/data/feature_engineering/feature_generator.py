# app/data/feature_engineering/feature_generator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
from loguru import logger

from app.data.feature_engineering.models import (
    FeatureDefinition, FeatureMetadata, FeatureType, FeatureSet
)
from app.data.feature_engineering.exceptions import FeatureGenerationError
from app.data.feature_engineering.feature_registry import FeatureRegistry
from app.data.feature_engineering.feature_transformer import FeatureTransformer
from app.data.feature_engineering.encoders import FeatureEncoder
from app.data.feature_engineering.aggregations import FeatureAggregator
from app.data.feature_engineering.interaction import FeatureInteractionGenerator
from app.data.feature_engineering.statistics import StatisticalFeatureGenerator


class FeatureGenerator:
    """
    Core feature generator for creating engineered features.
    
    Orchestrates feature creation, transformation, encoding, and registration.
    """
    
    def __init__(
        self,
        registry: FeatureRegistry,
        transformer: Optional[FeatureTransformer] = None,
        encoder: Optional[FeatureEncoder] = None,
        aggregator: Optional[FeatureAggregator] = None,
        interaction_generator: Optional[FeatureInteractionGenerator] = None,
        statistic_generator: Optional[StatisticalFeatureGenerator] = None
    ):
        self.registry = registry
        self.transformer = transformer or FeatureTransformer()
        self.encoder = encoder or FeatureEncoder()
        self.aggregator = aggregator or FeatureAggregator()
        self.interaction_generator = interaction_generator or FeatureInteractionGenerator()
        self.statistic_generator = statistic_generator or StatisticalFeatureGenerator()
        
        self.generated_features: Dict[str, pd.Series] = {}
        self.feature_metadata: Dict[str, FeatureMetadata] = {}
    
    def generate_features(
        self,
        df: pd.DataFrame,
        feature_definitions: List[FeatureDefinition],
        owner: str = "system",
        version: str = "1.0.0"
    ) -> pd.DataFrame:
        """
        Generate features based on definitions.
        
        Args:
            df: Input dataframe
            feature_definitions: List of feature definitions
            owner: Owner of the features
            version: Version for the generated features
            
        Returns:
            Dataframe with generated features
        """
        try:
            result_df = df.copy()
            
            for definition in feature_definitions:
                logger.info(f"Generating feature: {definition.name}")
                
                # Generate feature based on type
                feature_data = self._generate_feature_by_type(
                    df, definition
                )
                
                if feature_data is not None:
                    # Add to result
                    result_df[definition.name] = feature_data
                    
                    # Create metadata
                    metadata = self._create_metadata(
                        definition, 
                        feature_data,
                        owner,
                        version
                    )
                    
                    # Register feature
                    self.registry.register_feature(metadata)
                    
                    # Store for later use
                    self.generated_features[definition.name] = feature_data
                    self.feature_metadata[definition.name] = metadata
            
            logger.info(f"Generated {len(feature_definitions)} features")
            return result_df
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to generate features: {e}")
    
    def _generate_feature_by_type(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> Optional[pd.Series]:
        """Generate feature based on its type."""
        
        generator_map = {
            FeatureType.NUMERICAL: self._generate_numerical_feature,
            FeatureType.CATEGORICAL: self._generate_categorical_feature,
            FeatureType.DATE: self._generate_date_feature,
            FeatureType.TIME: self._generate_time_feature,
            FeatureType.INTERACTION: self._generate_interaction_feature,
            FeatureType.AGGREGATED: self._generate_aggregated_feature,
            FeatureType.ROLLING: self._generate_rolling_feature,
            FeatureType.STATISTICAL: self._generate_statistical_feature,
            FeatureType.POLYNOMIAL: self._generate_polynomial_feature,
            FeatureType.DOMAIN: self._generate_domain_feature
        }
        
        generator = generator_map.get(definition.feature_type)
        if generator:
            return generator(df, definition)
        else:
            raise ValueError(f"Unsupported feature type: {definition.feature_type}")
    
    def _generate_numerical_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate numerical feature."""
        if len(definition.source_columns) == 1:
            # Simple numerical feature
            col = definition.source_columns[0]
            if col in df.columns:
                return df[col]
        else:
            # Combination of numerical columns
            result = df[definition.source_columns[0]]
            for col in definition.source_columns[1:]:
                if col in df.columns:
                    result = result + df[col]
            return result
        return None
    
    def _generate_categorical_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate categorical feature."""
        if len(definition.source_columns) == 1:
            col = definition.source_columns[0]
            if col in df.columns:
                return df[col].astype('category')
        else:
            # Combine categorical columns
            result = df[definition.source_columns[0]].astype(str)
            for col in definition.source_columns[1:]:
                if col in df.columns:
                    result = result + "_" + df[col].astype(str)
            return result.astype('category')
        return None
    
    def _generate_date_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate date-based feature."""
        col = definition.source_columns[0]
        if col not in df.columns:
            return None
        
        date_col = pd.to_datetime(df[col])
        
        # Extract date components
        if definition.parameters.get('component') == 'year':
            return date_col.dt.year
        elif definition.parameters.get('component') == 'month':
            return date_col.dt.month
        elif definition.parameters.get('component') == 'day':
            return date_col.dt.day
        elif definition.parameters.get('component') == 'day_of_week':
            return date_col.dt.dayofweek
        elif definition.parameters.get('component') == 'quarter':
            return date_col.dt.quarter
        elif definition.parameters.get('component') == 'day_of_year':
            return date_col.dt.dayofyear
        else:
            return date_col.dt.dayofyear
    
    def _generate_time_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate time-based feature."""
        col = definition.source_columns[0]
        if col not in df.columns:
            return None
        
        time_col = pd.to_datetime(df[col])
        
        # Extract time components
        if definition.parameters.get('component') == 'hour':
            return time_col.dt.hour
        elif definition.parameters.get('component') == 'minute':
            return time_col.dt.minute
        elif definition.parameters.get('component') == 'second':
            return time_col.dt.second
        else:
            # Time of day in seconds
            return (time_col.dt.hour * 3600 + 
                   time_col.dt.minute * 60 + 
                   time_col.dt.second)
    
    def _generate_interaction_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate interaction feature."""
        if len(definition.source_columns) == 2:
            col1, col2 = definition.source_columns
            if col1 in df.columns and col2 in df.columns:
                return self.interaction_generator.create_interaction(
                    df[col1], df[col2],
                    definition.parameters.get('type', 'multiplicative')
                )
        return None
    
    def _generate_aggregated_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate aggregated feature."""
        group_by = definition.parameters.get('group_by')
        agg_col = definition.source_columns[0]
        agg_func = definition.parameters.get('function', 'mean')
        
        if group_by and agg_col in df.columns and group_by in df.columns:
            return self.aggregator.aggregate_by_group(
                df, group_by, agg_col, agg_func
            )
        return None
    
    def _generate_rolling_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate rolling feature."""
        col = definition.source_columns[0]
        window = definition.parameters.get('window', 7)
        function = definition.parameters.get('function', 'mean')
        
        if col in df.columns:
            return self.aggregator.rolling_aggregation(
                df[col], window, function
            )
        return None
    
    def _generate_statistical_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate statistical feature."""
        col = definition.source_columns[0]
        stat_type = definition.parameters.get('statistic', 'mean')
        
        if col in df.columns:
            return self.statistic_generator.compute_statistic(
                df[col], stat_type
            )
        return None
    
    def _generate_polynomial_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate polynomial feature."""
        col = definition.source_columns[0]
        degree = definition.parameters.get('degree', 2)
        
        if col in df.columns:
            return df[col] ** degree
        return None
    
    def _generate_domain_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate domain-specific agricultural feature."""
        params = definition.parameters
        return self._generate_agricultural_feature(df, definition)
    
    def _generate_agricultural_feature(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Generate agriculture-specific features."""
        feature_type = definition.parameters.get('agricultural_type')
        
        if feature_type == 'climate_index':
            return self._calculate_climate_index(df, definition)
        elif feature_type == 'heat_index':
            return self._calculate_heat_index(df, definition)
        elif feature_type == 'growing_degree_days':
            return self._calculate_growing_degree_days(df, definition)
        elif feature_type == 'soil_fertility':
            return self._calculate_soil_fertility(df, definition)
        elif feature_type == 'npk_balance':
            return self._calculate_npk_balance(df, definition)
        elif feature_type == 'water_requirement':
            return self._calculate_water_requirement(df, definition)
        elif feature_type == 'crop_suitability':
            return self._calculate_crop_suitability(df, definition)
        elif feature_type == 'risk_score':
            return self._calculate_risk_score(df, definition)
        else:
            return None
    
    def _calculate_climate_index(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Calculate climate index from temperature and rainfall."""
        temp_col = definition.parameters.get('temperature_col')
        rain_col = definition.parameters.get('rainfall_col')
        
        if temp_col in df.columns and rain_col in df.columns:
            # Normalize temperature and rainfall
            temp_norm = (df[temp_col] - df[temp_col].min()) / (df[temp_col].max() - df[temp_col].min())
            rain_norm = (df[rain_col] - df[rain_col].min()) / (df[rain_col].max() - df[rain_col].min())
            
            # Weighted climate index
            temp_weight = definition.parameters.get('temp_weight', 0.6)
            rain_weight = definition.parameters.get('rain_weight', 0.4)
            
            return (temp_norm * temp_weight + rain_norm * rain_weight) * 100
        return None
    
    def _calculate_heat_index(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Calculate heat index."""
        temp_col = definition.parameters.get('temperature_col')
        humidity_col = definition.parameters.get('humidity_col')
        
        if temp_col in df.columns and humidity_col in df.columns:
            # Simplified heat index calculation
            T = df[temp_col]
            R = df[humidity_col]
            
            # Heat index formula (simplified)
            HI = (0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094)))
            return HI
        return None
    
    def _calculate_growing_degree_days(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Calculate growing degree days."""
        temp_col = definition.parameters.get('temperature_col')
        base_temp = definition.parameters.get('base_temp', 10)
        upper_temp = definition.parameters.get('upper_temp', 30)
        
        if temp_col in df.columns:
            T = df[temp_col]
            # GDD = ((Tmax + Tmin)/2) - Tbase
            # Using daily temperature as approximation
            GDD = T - base_temp
            GDD = GDD.clip(lower=0, upper=upper_temp - base_temp)
            return GDD
        return None
    
    def _calculate_soil_fertility(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Calculate soil fertility score."""
        components = definition.parameters.get('components', [])
        weights = definition.parameters.get('weights', [])
        
        if len(components) == len(weights):
            fertility_score = pd.Series(0, index=df.index)
            for comp, weight in zip(components, weights):
                if comp in df.columns:
                    # Normalize component
                    comp_norm = (df[comp] - df[comp].min()) / (df[comp].max() - df[comp].min())
                    fertility_score += comp_norm * weight
            
            # Scale to 0-100
            fertility_score = (fertility_score / sum(weights)) * 100
            return fertility_score
        return None
    
    def _calculate_npk_balance(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Calculate NPK balance index."""
        n_col = definition.parameters.get('n_col')
        p_col = definition.parameters.get('p_col')
        k_col = definition.parameters.get('k_col')
        
        if all(col in df.columns for col in [n_col, p_col, k_col]):
            # Ideal NPK ratio for crops is approximately 4-2-1
            n = df[n_col]
            p = df[p_col] * 2  # Scale P to match N
            k = df[k_col] * 4  # Scale K to match N
            
            # Calculate balance score
            balance = np.exp(-((n - p)**2 + (n - k)**2 + (p - k)**2) / 1000)
            return balance * 100
        return None
    
    def _calculate_water_requirement(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Calculate water requirement based on evapotranspiration."""
        temp_col = definition.parameters.get('temperature_col')
        humidity_col = definition.parameters.get('humidity_col')
        wind_col = definition.parameters.get('wind_col', None)
        
        if temp_col in df.columns and humidity_col in df.columns:
            # Simplified water requirement calculation
            T = df[temp_col]
            H = df[humidity_col]
            
            # Reference evapotranspiration (simplified)
            ET = 0.0023 * (T + 17.8) * np.sqrt(max(0, 100 - H))
            
            # Crop coefficient adjustment
            crop_coeff = definition.parameters.get('crop_coeff', 1.0)
            
            return ET * crop_coeff
        return None
    
    def _calculate_crop_suitability(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Calculate crop suitability score."""
        factors = definition.parameters.get('factors', [])
        weights = definition.parameters.get('weights', [])
        optimal_values = definition.parameters.get('optimal_values', [])
        
        if len(factors) == len(weights) == len(optimal_values):
            suitability = pd.Series(0, index=df.index)
            
            for factor, weight, optimal in zip(factors, weights, optimal_values):
                if factor in df.columns:
                    # Calculate deviation from optimal
                    deviation = np.abs(df[factor] - optimal) / (df[factor].max() - df[factor].min())
                    score = 100 * (1 - deviation)
                    suitability += score * weight
            
            return suitability / sum(weights)
        return None
    
    def _calculate_risk_score(
        self,
        df: pd.DataFrame,
        definition: FeatureDefinition
    ) -> pd.Series:
        """Calculate agricultural risk score."""
        risk_factors = definition.parameters.get('risk_factors', {})
        
        risk_score = pd.Series(0, index=df.index)
        
        for factor, config in risk_factors.items():
            if factor in df.columns:
                # Normalize risk factor
                normalized = (df[factor] - df[factor].min()) / (df[factor].max() - df[factor].min())
                
                # Apply weight and direction
                weight = config.get('weight', 1.0)
                direction = config.get('direction', 'positive')  # positive means risk increases with value
                
                if direction == 'positive':
                    risk_score += normalized * weight
                else:
                    risk_score += (1 - normalized) * weight
        
        # Scale to 0-100
        if risk_score.max() > 0:
            risk_score = (risk_score / risk_score.max()) * 100
        
        return risk_score
    
    def _create_metadata(
        self,
        definition: FeatureDefinition,
        feature_data: pd.Series,
        owner: str,
        version: str
    ) -> FeatureMetadata:
        """Create feature metadata."""
        return FeatureMetadata(
            feature_name=definition.name,
            description=definition.description,
            formula=definition.transformation,
            data_type=str(feature_data.dtype),
            feature_type=definition.feature_type,
            owner=owner,
            version=version,
            source_columns=definition.source_columns,
            dependencies=definition.dependencies,
            validation_rules=definition.validation_rules,
            tags=definition.parameters.get('tags', [])
        )
    
    def get_feature_set(
        self,
        feature_names: List[str],
        version: Optional[str] = None
    ) -> FeatureSet:
        """Get a feature set from the registry."""
        return self.registry.get_feature_set(feature_names, version)
    
    def save_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        output_path: str
    ) -> None:
        """Save generated features to disk."""
        features_df = df[feature_names].copy()
        features_df.to_parquet(output_path)
        logger.info(f"Saved {len(feature_names)} features to {output_path}")