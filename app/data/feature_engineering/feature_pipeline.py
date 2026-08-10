# app/data/feature_engineering/feature_pipeline.py
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
from loguru import logger

from app.data.feature_engineering.models import (
    FeatureDefinition, FeatureMetadata, FeatureSet, FeatureType,
    EncodingType, ScalingType, SelectionMethod
)
from app.data.feature_engineering.feature_registry import FeatureRegistry
from app.data.feature_engineering.feature_generator import FeatureGenerator
from app.data.feature_engineering.feature_selector import FeatureSelector
from app.data.feature_engineering.feature_transformer import FeatureTransformer
from app.data.feature_engineering.encoders import FeatureEncoder
from app.data.feature_engineering.aggregations import FeatureAggregator
from app.data.feature_engineering.interaction import FeatureInteractionGenerator
from app.data.feature_engineering.statistics import StatisticalFeatureGenerator
from app.data.feature_engineering.report import FeatureReport
from app.data.feature_engineering.exceptions import FeatureGenerationError


class FeaturePipeline:
    """
    Enterprise feature engineering pipeline orchestrator.
    
    Coordinates the entire feature engineering workflow from raw data to feature store.
    """
    
    def __init__(
        self,
        config_path: Path,
        feature_store_path: Path
    ):
        self.config_path = Path(config_path)
        self.feature_store_path = Path(feature_store_path)
        self.config = self._load_config()
        
        # Initialize components
        self.registry = FeatureRegistry(feature_store_path / 'registry')
        self.transformer = FeatureTransformer()
        self.encoder = FeatureEncoder()
        self.aggregator = FeatureAggregator()
        self.interaction_generator = FeatureInteractionGenerator()
        self.statistic_generator = StatisticalFeatureGenerator()
        self.selector = FeatureSelector()
        self.generator = FeatureGenerator(
            registry=self.registry,
            transformer=self.transformer,
            encoder=self.encoder,
            aggregator=self.aggregator,
            interaction_generator=self.interaction_generator,
            statistic_generator=self.statistic_generator
        )
        self.report = FeatureReport(
            feature_store_path / 'reports' / 'features'
        )
        
        self.pipeline_metadata: Dict[str, Any] = {}
    
    def _load_config(self) -> Dict:
        """Load feature engineering configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default feature engineering configuration."""
        return {
            'feature_selection': {
                'enabled': True,
                'method': 'variance_threshold',
                'threshold': 0.0
            },
            'scaling': {
                'enabled': True,
                'method': 'standard',
                'columns': []
            },
            'encoding': {
                'enabled': True,
                'method': 'one_hot',
                'max_categories': 20
            },
            'interactions': {
                'enabled': True,
                'methods': ['multiplicative']
            },
            'aggregations': {
                'enabled': True,
                'window': 7,
                'functions': ['mean', 'std', 'min', 'max']
            },
            'store_backend': {
                'offline': True,
                'online': False,
                'compression': 'gzip'
            },
            'versioning': {
                'enabled': True,
                'auto_increment': True
            }
        }
    
    def run_pipeline(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        feature_definitions: Optional[List[FeatureDefinition]] = None,
        owner: str = 'system',
        version: str = '1.0.0'
    ) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Input dataframe
            dataset_name: Name of the dataset
            feature_definitions: List of feature definitions
            owner: Owner of the features
            version: Version of the features
            
        Returns:
            Dataframe with engineered features
        """
        try:
            logger.info(f"Starting feature engineering pipeline for {dataset_name}")
            start_time = datetime.now()
            
            # 1. Generate basic features
            if feature_definitions:
                df_with_features = self.generator.generate_features(
                    df, feature_definitions, owner, version
                )
            else:
                df_with_features = df.copy()
            
            # 2. Apply transformations
            df_with_features = self._apply_transformations(df_with_features)
            
            # 3. Apply encodings
            df_with_features = self._apply_encodings(df_with_features)
            
            # 4. Generate interactions
            df_with_features = self._generate_interactions(df_with_features)
            
            # 5. Generate aggregations
            df_with_features = self._generate_aggregations(df_with_features)
            
            # 6. Select features
            df_with_features = self._select_features(df_with_features)
            
            # 7. Create feature set
            feature_set = self._create_feature_set(
                df_with_features, dataset_name, version
            )
            
            # 8. Store features
            self._store_features(df_with_features, dataset_name, version)
            
            # 9. Generate reports
            self._generate_reports(df_with_features, dataset_name, version)
            
            # 10. Update pipeline metadata
            self.pipeline_metadata = {
                'dataset_name': dataset_name,
                'version': version,
                'owner': owner,
                'start_time': start_time,
                'end_time': datetime.now(),
                'n_features': len(feature_set.features),
                'n_samples': len(df_with_features),
                'config': self.config
            }
            
            logger.info(f"Feature engineering pipeline completed for {dataset_name}")
            return df_with_features
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise FeatureGenerationError(f"Feature pipeline failed: {e}")
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply configured transformations."""
        if not self.config.get('scaling', {}).get('enabled', False):
            return df
        
        scaling_config = self.config['scaling']
        columns = scaling_config.get('columns', [])
        method = scaling_config.get('method', 'standard')
        
        if columns:
            return self.transformer.scale_features(
                df, columns, ScalingType(method)
            )
        return df
    
    def _apply_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply configured encodings."""
        if not self.config.get('encoding', {}).get('enabled', False):
            return df
        
        encoding_config = self.config['encoding']
        max_categories = encoding_config.get('max_categories', 20)
        method = encoding_config.get('method', 'one_hot')
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if df[col].nunique() <= max_categories]
        
        if categorical_cols:
            return self.encoder.encode_features(
                df, categorical_cols, EncodingType(method)
            )
        return df
    
    def _generate_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features."""
        if not self.config.get('interactions', {}).get('enabled', False):
            return df
        
        interaction_config = self.config['interactions']
        methods = interaction_config.get('methods', ['multiplicative'])
        
        # Generate interactions for numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numerical_cols) >= 2:
            for method in methods:
                df = self.interaction_generator.generate_interactions(
                    df, numerical_cols, method
                )
        
        return df
    
    def _generate_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate aggregation features."""
        if not self.config.get('aggregations', {}).get('enabled', False):
            return df
        
        agg_config = self.config['aggregations']
        window = agg_config.get('window', 7)
        functions = agg_config.get('functions', ['mean', 'std'])
        
        # Generate rolling aggregations for numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        for col in numerical_cols:
            for func in functions:
                if col in df.columns:
                    try:
                        result = self.aggregator.rolling_aggregation(
                            df[col], window, func
                        )
                        df[f'{col}_{func}_rolling_{window}'] = result
                    except Exception as e:
                        logger.warning(f"Failed to generate {func} rolling for {col}: {e}")
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features using configured method."""
        if not self.config.get('feature_selection', {}).get('enabled', False):
            return df
        
        selection_config = self.config['feature_selection']
        method = selection_config.get('method', 'variance_threshold')
        
        # Convert string to enum
        method_enum = SelectionMethod(method)
        
        # Select features
        return self.selector.select_features(
            df, method=method_enum, **selection_config
        )
    
    def _create_feature_set(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        version: str
    ) -> FeatureSet:
        """Create a feature set from the processed dataframe."""
        features = []
        
        for col in df.columns:
            feature_metadata = self.registry.get_feature(col, version)
            features.append(feature_metadata)
        
        return FeatureSet(
            dataset_name=dataset_name,
            features=features,
            version=version,
            schema=df.dtypes.to_dict(),
            statistics=df.describe().to_dict(),
            checksum=self._generate_feature_set_checksum(df)
        )
    
    def _store_features(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        version: str
    ) -> None:
        """Store features in the feature store."""
        store_path = self.feature_store_path / 'offline' / dataset_name
        store_path.mkdir(parents=True, exist_ok=True)
        
        # Store features
        features_path = store_path / f'features_v{version}.parquet'
        df.to_parquet(features_path, compression='gzip')
        
        # Store metadata
        metadata_path = store_path / f'metadata_v{version}.json'
        metadata = {
            'dataset_name': dataset_name,
            'version': version,
            'n_features': len(df.columns),
            'n_samples': len(df),
            'created_at': datetime.now().isoformat(),
            'dtypes': df.dtypes.to_dict(),
            'pipeline_metadata': self.pipeline_metadata
        }
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Stored features in {store_path}")
    
    def _generate_reports(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        version: str
    ) -> None:
        """Generate feature engineering reports."""
        self.report.generate_feature_summary(df, dataset_name, version)
        self.report.generate_correlation_report(df, dataset_name, version)
        self.report.generate_distribution_report(df, dataset_name, version)
        self.report.generate_feature_metadata_report(
            self.registry, dataset_name, version
        )
    
    def _generate_feature_set_checksum(self, df: pd.DataFrame) -> str:
        """Generate checksum for feature set."""
        # Generate checksum from data
        data_str = str(df.values.tobytes())
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def get_feature(self, feature_name: str, version: Optional[str] = None) -> FeatureMetadata:
        """Get feature metadata from registry."""
        return self.registry.get_feature(feature_name, version)
    
    def list_features(
        self,
        feature_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all available features."""
        return self.registry.list_features(feature_type)
    
    def search_features(self, query: str) -> List[Dict[str, Any]]:
        """Search for features."""
        return self.registry.search_features(query)
    
    def get_pipeline_metadata(self) -> Dict[str, Any]:
        """Get pipeline execution metadata."""
        return self.pipeline_metadata
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        return self.selector.get_feature_importance()