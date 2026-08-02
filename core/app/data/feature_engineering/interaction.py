# app/data/feature_engineering/interaction.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from itertools import combinations
from loguru import logger

from app.data.feature_engineering.exceptions import FeatureGenerationError


class FeatureInteractionGenerator:
    """
    Generate interaction features between variables.
    
    Supports multiplicative, additive, polynomial, and custom interactions.
    """
    
    def __init__(self):
        self.interactions: List[Dict] = []
    
    def create_interaction(
        self,
        col1: pd.Series,
        col2: pd.Series,
        interaction_type: str = 'multiplicative',
        **kwargs
    ) -> pd.Series:
        """
        Create interaction feature between two columns.
        
        Args:
            col1: First feature series
            col2: Second feature series
            interaction_type: Type of interaction
            **kwargs: Additional parameters
            
        Returns:
            Interaction feature series
        """
        try:
            if interaction_type == 'multiplicative':
                return col1 * col2
            
            elif interaction_type == 'additive':
                return col1 + col2
            
            elif interaction_type == 'ratio':
                # Add small epsilon to avoid division by zero
                epsilon = 1e-10
                return col1 / (col2 + epsilon)
            
            elif interaction_type == 'difference':
                return col1 - col2
            
            elif interaction_type == 'polynomial':
                degree = kwargs.get('degree', 2)
                result = col1
                for _ in range(1, degree):
                    result = result * col2
                return result
            
            elif interaction_type == 'custom':
                func = kwargs.get('func')
                if func is None:
                    raise ValueError("Custom interaction requires a function")
                return func(col1, col2)
            
            else:
                raise ValueError(f"Unsupported interaction type: {interaction_type}")
                
        except Exception as e:
            raise FeatureGenerationError(f"Failed to create interaction: {e}")
    
    def generate_interactions(
        self,
        df: pd.DataFrame,
        columns: List[str],
        interaction_type: str = 'multiplicative',
        max_features: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate all pairwise interactions for specified columns.
        
        Args:
            df: Input dataframe
            columns: Columns to generate interactions for
            interaction_type: Type of interaction to generate
            max_features: Maximum number of interaction features to generate
            **kwargs: Additional parameters
            
        Returns:
            Dataframe with interaction features
        """
        result_df = df.copy()
        
        # Generate all pairwise combinations
        pairs = list(combinations(columns, 2))
        
        if max_features and len(pairs) > max_features:
            # Select most correlated pairs (simplified approach)
            correlations = []
            for col1, col2 in pairs:
                if col1 in df.columns and col2 in df.columns:
                    corr = df[col1].corr(df[col2])
                    correlations.append((corr, col1, col2))
            
            # Sort by absolute correlation and take top max_features
            correlations.sort(key=lambda x: abs(x[0]), reverse=True)
            pairs = [(c1, c2) for _, c1, c2 in correlations[:max_features]]
        
        # Generate interactions
        for col1, col2 in pairs:
            if col1 in df.columns and col2 in df.columns:
                interaction = self.create_interaction(
                    df[col1], df[col2],
                    interaction_type,
                    **kwargs
                )
                
                # Create meaningful feature name
                feature_name = f"{col1}_{interaction_type}_{col2}"
                result_df[feature_name] = interaction
                
                self.interactions.append({
                    'feature': feature_name,
                    'col1': col1,
                    'col2': col2,
                    'type': interaction_type,
                    'parameters': kwargs
                })
        
        logger.info(f"Generated {len(pairs)} interaction features")
        return result_df
    
    def generate_domain_interactions(
        self,
        df: pd.DataFrame,
        interaction_configs: List[Dict]
    ) -> pd.DataFrame:
        """
        Generate specific domain interactions based on configuration.
        
        Args:
            df: Input dataframe
            interaction_configs: List of interaction configurations
            
        Returns:
            Dataframe with domain interaction features
        """
        result_df = df.copy()
        
        for config in interaction_configs:
            col1 = config.get('col1')
            col2 = config.get('col2')
            interaction_type = config.get('type', 'multiplicative')
            feature_name = config.get('feature_name')
            
            if col1 in df.columns and col2 in df.columns:
                interaction = self.create_interaction(
                    df[col1], df[col2],
                    interaction_type,
                    **config.get('parameters', {})
                )
                
                if feature_name is None:
                    feature_name = f"{col1}_{interaction_type}_{col2}"
                
                result_df[feature_name] = interaction
                
                self.interactions.append({
                    'feature': feature_name,
                    'col1': col1,
                    'col2': col2,
                    'type': interaction_type,
                    'parameters': config.get('parameters', {})
                })
        
        return result_df
    
    def generate_agricultural_interactions(
        self,
        df: pd.DataFrame,
        temp_col: str = 'temperature',
        rainfall_col: str = 'rainfall',
        humidity_col: str = 'humidity',
        n_col: str = 'n',
        p_col: str = 'p',
        k_col: str = 'k',
        ph_col: str = 'ph'
    ) -> pd.DataFrame:
        """
        Generate agricultural-specific interaction features.
        
        Args:
            df: Input dataframe
            temp_col: Temperature column name
            rainfall_col: Rainfall column name
            humidity_col: Humidity column name
            n_col: Nitrogen column name
            p_col: Phosphorus column name
            k_col: Potassium column name
            ph_col: pH column name
            
        Returns:
            Dataframe with agricultural interaction features
        """
        result_df = df.copy()
        
        # Climate interactions
        if temp_col in df.columns and rainfall_col in df.columns:
            result_df['temp_rainfall_interaction'] = df[temp_col] * df[rainfall_col]
        
        if temp_col in df.columns and humidity_col in df.columns:
            result_df['temp_humidity_interaction'] = df[temp_col] * df[humidity_col]
        
        # NPK interactions
        if all(col in df.columns for col in [n_col, p_col, k_col]):
            result_df['npk_product'] = df[n_col] * df[p_col] * df[k_col]
            result_df['npk_ratio'] = df[n_col] / (df[p_col] + 1e-10)
            
            # Balanced NPK score (ideal ratio 4:2:1)
            n_weight = 4 / (4 + 2 + 1)
            p_weight = 2 / (4 + 2 + 1)
            k_weight = 1 / (4 + 2 + 1)
            
            result_df['npk_balance'] = (
                df[n_col] * n_weight + df[p_col] * p_weight + df[k_col] * k_weight
            )
        
        # Soil interactions
        if n_col in df.columns and ph_col in df.columns:
            result_df['n_ph_interaction'] = df[n_col] * df[ph_col]
        
        if p_col in df.columns and ph_col in df.columns:
            result_df['p_ph_interaction'] = df[p_col] * df[ph_col]
        
        if k_col in df.columns and ph_col in df.columns:
            result_df['k_ph_interaction'] = df[k_col] * df[ph_col]
        
        # Temperature-Humidity-Precipitation interaction
        if all(col in df.columns for col in [temp_col, humidity_col, rainfall_col]):
            result_df['thp_interaction'] = df[temp_col] * df[humidity_col] * df[rainfall_col]
        
        # Log interactions
        if n_col in df.columns and p_col in df.columns:
            result_df['n_p_ratio_log'] = np.log(df[n_col] / (df[p_col] + 1e-10))
        
        if p_col in df.columns and k_col in df.columns:
            result_df['p_k_ratio_log'] = np.log(df[p_col] / (df[k_col] + 1e-10))
        
        # Weighted climate interaction
        if temp_col in df.columns and rainfall_col in df.columns and humidity_col in df.columns:
            result_df['climate_interaction'] = (
                0.4 * df[temp_col] + 
                0.3 * df[rainfall_col] + 
                0.3 * df[humidity_col]
            )
        
        logger.info("Generated agricultural interaction features")
        return result_df
    
    def get_interaction_summary(self) -> pd.DataFrame:
        """Get summary of generated interactions."""
        if not self.interactions:
            return pd.DataFrame()
        
        return pd.DataFrame(self.interactions)
    
    def save_interactions(
        self,
        df: pd.DataFrame,
        interaction_features: List[str],
        output_path: str
    ) -> None:
        """Save interaction features to disk."""
        if interaction_features:
            interaction_df = df[interaction_features]
            interaction_df.to_parquet(output_path)
            logger.info(f"Saved {len(interaction_features)} interaction features to {output_path}")