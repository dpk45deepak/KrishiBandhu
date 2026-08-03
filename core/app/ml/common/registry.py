"""
Model registry for tracking and managing trained models.
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import sqlite3
from loguru import logger

from .models import BaseMLModel, ModelMetadata
from .persistence import ModelPersistence, ModelVersioning
from .exceptions import RegistryError, ModelNotFoundError
from .utils import ensure_directory, save_json, load_json, get_timestamp, generate_checksum


@dataclass
class RegistryEntry:
    """
    Entry in the model registry.
    """
    model_id: str
    model_name: str
    model_type: str  # 'classification' or 'regression'
    version: str
    model_path: str
    metadata_path: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_dataset: Optional[str] = None
    feature_version: Optional[str] = None
    training_timestamp: Optional[str] = None
    checksum: Optional[str] = None
    status: str = 'active'  # 'active', 'archived', 'deprecated'
    tags: List[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_metadata(
        cls,
        metadata: ModelMetadata,
        model_path: str,
        metadata_path: str,
        **kwargs
    ) -> 'RegistryEntry':
        """Create registry entry from model metadata."""
        return cls(
            model_id=metadata.model_id,
            model_name=metadata.model_name,
            model_type=metadata.model_type,
            version=metadata.model_version,
            model_path=model_path,
            metadata_path=metadata_path,
            metrics=metadata.metrics,
            hyperparameters=metadata.hyperparameters,
            training_dataset=metadata.training_dataset_version,
            feature_version=metadata.feature_version,
            training_timestamp=metadata.training_timestamp.isoformat(),
            checksum=metadata.checksum,
            **kwargs
        )


class ModelRegistry:
    """
    Central registry for tracking and managing all models.
    Uses SQLite for persistence with optional JSON backup.
    """
    
    def __init__(
        self,
        registry_path: Union[str, Path],
        db_name: str = 'registry.db'
    ):
        self.registry_path = Path(registry_path)
        ensure_directory(self.registry_path)
        
        self.db_path = self.registry_path / db_name
        self._init_database()
        
        # Initialize versioning
        self.versioning = ModelVersioning(self.registry_path / 'models')
        
        logger.info(f"ModelRegistry initialized at {self.registry_path}")
    
    def _init_database(self) -> None:
        """Initialize the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create registry table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS registry (
                        model_id TEXT PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        version TEXT NOT NULL,
                        model_path TEXT NOT NULL,
                        metadata_path TEXT NOT NULL,
                        metrics TEXT NOT NULL,
                        hyperparameters TEXT NOT NULL,
                        training_dataset TEXT,
                        feature_version TEXT,
                        training_timestamp TEXT,
                        checksum TEXT,
                        status TEXT DEFAULT 'active',
                        tags TEXT,
                        description TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                # Create indexes
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_model_name 
                    ON registry(model_name)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_status 
                    ON registry(status)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_created_at 
                    ON registry(created_at)
                ''')
                
                conn.commit()
                
        except Exception as e:
            raise RegistryError(f"Failed to initialize database: {str(e)}") from e
    
    def register_model(
        self,
        model: BaseMLModel,
        model_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        training_dataset: Optional[str] = None,
        feature_version: Optional[str] = None
    ) -> RegistryEntry:
        """
        Register a trained model in the registry.
        
        Args:
            model: Trained model
            model_path: Path to the saved model
            metadata_path: Path to the model metadata
            tags: Tags for the model
            description: Description of the model
            training_dataset: Dataset version used for training
            feature_version: Feature version used for training
            
        Returns:
            Registry entry
        """
        try:
            # Ensure model has metadata
            if not model._metadata:
                raise RegistryError("Model has no metadata. Train the model first.")
            
            # Get metadata
            metadata = model._metadata
            
            # Update metadata with additional info
            if training_dataset:
                metadata.training_dataset_version = training_dataset
            if feature_version:
                metadata.feature_version = feature_version
            
            # Save model if not already saved
            model_path = Path(model_path)
            if not model_path.exists():
                ModelPersistence.save_model(model, model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Get metadata path
            if metadata_path:
                metadata_path = Path(metadata_path)
            else:
                metadata_path = model_path.with_suffix('.meta.json')
            
            # Create registry entry
            entry = RegistryEntry.from_metadata(
                metadata,
                str(model_path),
                str(metadata_path),
                tags=tags or [],
                description=description,
                training_dataset=training_dataset,
                feature_version=feature_version,
                status='active'
            )
            
            # Insert into database
            self._upsert_entry(entry)
            
            # Create version in versioning system
            self.versioning.create_version(
                model,
                version_tag=metadata.model_version,
                registry_entry=entry.to_dict()
            )
            
            logger.info(f"Model registered: {entry.model_id} ({entry.model_name} v{entry.version})")
            
            return entry
            
        except Exception as e:
            raise RegistryError(f"Failed to register model: {str(e)}") from e
    
    def _upsert_entry(self, entry: RegistryEntry) -> None:
        """Insert or update a registry entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                now = get_timestamp()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO registry (
                        model_id, model_name, model_type, version,
                        model_path, metadata_path, metrics, hyperparameters,
                        training_dataset, feature_version, training_timestamp,
                        checksum, status, tags, description, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.model_id,
                    entry.model_name,
                    entry.model_type,
                    entry.version,
                    entry.model_path,
                    entry.metadata_path,
                    json.dumps(entry.metrics),
                    json.dumps(entry.hyperparameters),
                    entry.training_dataset,
                    entry.feature_version,
                    entry.training_timestamp,
                    entry.checksum,
                    entry.status,
                    json.dumps(entry.tags),
                    entry.description,
                    now,
                    now
                ))
                
                conn.commit()
                
        except Exception as e:
            raise RegistryError(f"Failed to upsert registry entry: {str(e)}") from e
    
    def get_model(
        self,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = 'latest'
    ) -> Tuple[BaseMLModel, RegistryEntry]:
        """
        Get a model from the registry.
        
        Args:
            model_id: Model ID (if known)
            model_name: Model name (if looking by name)
            version: Version tag or 'latest'
            
        Returns:
            Tuple of (model, registry_entry)
        """
        try:
            # Get entry
            entry = self.get_entry(model_id, model_name, version)
            
            # Load model
            model_path = Path(entry.model_path)
            
            if not model_path.exists():
                # Try to find model in versioning system
                version_path = self.registry_path / 'models' / entry.model_name / entry.version
                model_path = version_path / 'model.joblib'
                
                if not model_path.exists():
                    raise ModelNotFoundError(f"Model file not found: {model_path}")
            
            # Load the model
            # We need to determine the model class from the metadata
            # This would require a model factory
            from ..classification import (
                RandomForestClassifier,
                XGBoostClassifier,
                LightGBMClassifier,
                CatBoostClassifier,
                LogisticRegressionClassifier,
                DecisionTreeClassifier,
                SVMClassifier,
                ExtraTreesClassifier,
                VotingEnsemble,
                StackingEnsemble
            )
            from ..regression import (
                LinearRegression,
                RandomForestRegressor,
                XGBoostRegressor,
                LightGBMRegressor,
                CatBoostRegressor,
                ElasticNetRegressor,
                RidgeRegressor,
                LassoRegressor,
                SVRRegressor
            )
            
            # Map model names to classes
            MODEL_MAP = {
                # Classification
                'RandomForestClassifier': RandomForestClassifier,
                'XGBoostClassifier': XGBoostClassifier,
                'LightGBMClassifier': LightGBMClassifier,
                'CatBoostClassifier': CatBoostClassifier,
                'LogisticRegression': LogisticRegressionClassifier,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'SVMClassifier': SVMClassifier,
                'ExtraTreesClassifier': ExtraTreesClassifier,
                'VotingEnsemble': VotingEnsemble,
                'StackingEnsemble': StackingEnsemble,
                # Regression
                'LinearRegression': LinearRegression,
                'RandomForestRegressor': RandomForestRegressor,
                'XGBoostRegressor': XGBoostRegressor,
                'LightGBMRegressor': LightGBMRegressor,
                'CatBoostRegressor': CatBoostRegressor,
                'ElasticNet': ElasticNetRegressor,
                'Ridge': RidgeRegressor,
                'Lasso': LassoRegressor,
                'SVR': SVRRegressor
            }
            
            # Get model class
            model_class = MODEL_MAP.get(entry.model_name)
            if not model_class:
                raise RegistryError(f"Unknown model class: {entry.model_name}")
            
            # Load model
            model = ModelPersistence.load_model(
                model_path,
                model_class=model_class
            )
            
            return model, entry
            
        except Exception as e:
            raise RegistryError(f"Failed to get model: {str(e)}") from e
    
    def get_entry(
        self,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = 'latest'
    ) -> RegistryEntry:
        """
        Get a registry entry.
        
        Args:
            model_id: Model ID
            model_name: Model name
            version: Version tag or 'latest'
            
        Returns:
            Registry entry
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if model_id:
                    cursor.execute(
                        'SELECT * FROM registry WHERE model_id = ?',
                        (model_id,)
                    )
                elif model_name and version:
                    if version == 'latest':
                        cursor.execute('''
                            SELECT * FROM registry 
                            WHERE model_name = ? AND status = 'active'
                            ORDER BY created_at DESC LIMIT 1
                        ''', (model_name,))
                    else:
                        cursor.execute('''
                            SELECT * FROM registry 
                            WHERE model_name = ? AND version = ?
                        ''', (model_name, version))
                else:
                    raise ValueError("Either model_id or (model_name, version) must be provided")
                
                row = cursor.fetchone()
                if not row:
                    raise ModelNotFoundError(
                        f"Model not found: {model_id or f'{model_name} v{version}'}"
                    )
                
                # Convert row to dict
                data = dict(row)
                
                # Parse JSON fields
                data['metrics'] = json.loads(data['metrics'])
                data['hyperparameters'] = json.loads(data['hyperparameters'])
                data['tags'] = json.loads(data['tags']) if data['tags'] else []
                
                return RegistryEntry(**data)
                
        except Exception as e:
            raise RegistryError(f"Failed to get registry entry: {str(e)}") from e
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = 'active',
        limit: int = 100,
        offset: int = 0
    ) -> List[RegistryEntry]:
        """
        List models in the registry.
        
        Args:
            model_type: Filter by model type
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of registry entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = 'SELECT * FROM registry WHERE 1=1'
                params = []
                
                if model_type:
                    query += ' AND model_type = ?'
                    params.append(model_type)
                
                if status:
                    query += ' AND status = ?'
                    params.append(status)
                
                query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                
                entries = []
                for row in cursor.fetchall():
                    data = dict(row)
                    data['metrics'] = json.loads(data['metrics'])
                    data['hyperparameters'] = json.loads(data['hyperparameters'])
                    data['tags'] = json.loads(data['tags']) if data['tags'] else []
                    entries.append(RegistryEntry(**data))
                
                return entries
                
        except Exception as e:
            raise RegistryError(f"Failed to list models: {str(e)}") from e
    
    def update_status(
        self,
        model_id: str,
        status: str
    ) -> None:
        """
        Update model status.
        
        Args:
            model_id: Model ID
            status: New status ('active', 'archived', 'deprecated')
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE registry 
                    SET status = ?, updated_at = ?
                    WHERE model_id = ?
                ''', (status, get_timestamp(), model_id))
                
                if cursor.rowcount == 0:
                    raise ModelNotFoundError(f"Model not found: {model_id}")
                
                conn.commit()
                logger.info(f"Updated model {model_id} status to {status}")
                
        except Exception as e:
            raise RegistryError(f"Failed to update status: {str(e)}") from e
    
    def add_tags(self, model_id: str, tags: List[str]) -> None:
        """
        Add tags to a model.
        
        Args:
            model_id: Model ID
            tags: Tags to add
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get existing tags
                cursor.execute('SELECT tags FROM registry WHERE model_id = ?', (model_id,))
                row = cursor.fetchone()
                if not row:
                    raise ModelNotFoundError(f"Model not found: {model_id}")
                
                existing_tags = json.loads(row[0]) if row[0] else []
                all_tags = list(set(existing_tags + tags))
                
                cursor.execute('''
                    UPDATE registry 
                    SET tags = ?, updated_at = ?
                    WHERE model_id = ?
                ''', (json.dumps(all_tags), get_timestamp(), model_id))
                
                conn.commit()
                logger.info(f"Added tags {tags} to model {model_id}")
                
        except Exception as e:
            raise RegistryError(f"Failed to add tags: {str(e)}") from e
    
    def search(
        self,
        query: str,
        limit: int = 50
    ) -> List[RegistryEntry]:
        """
        Search for models.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of registry entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                search_query = f'%{query}%'
                cursor.execute('''
                    SELECT * FROM registry 
                    WHERE model_name LIKE ? 
                       OR model_id LIKE ? 
                       OR description LIKE ?
                    ORDER BY created_at DESC LIMIT ?
                ''', (search_query, search_query, search_query, limit))
                
                entries = []
                for row in cursor.fetchall():
                    data = dict(row)
                    data['metrics'] = json.loads(data['metrics'])
                    data['hyperparameters'] = json.loads(data['hyperparameters'])
                    data['tags'] = json.loads(data['tags']) if data['tags'] else []
                    entries.append(RegistryEntry(**data))
                
                return entries
                
        except Exception as e:
            raise RegistryError(f"Failed to search: {str(e)}") from e
    
    def export_registry(
        self,
        path: Union[str, Path],
        format: str = 'json'
    ) -> None:
        """
        Export the entire registry.
        
        Args:
            path: Export path
            format: Export format ('json', 'csv')
        """
        path = Path(path)
        ensure_directory(path.parent)
        
        try:
            # Get all entries
            entries = self.list_models(status=None, limit=10000)
            
            # Convert to dictionaries
            data = [e.to_dict() for e in entries]
            
            if format == 'json':
                save_json({'registry': data, 'exported_at': get_timestamp()}, path)
            elif format == 'csv':
                import pandas as pd
                df = pd.DataFrame(data)
                df.to_csv(path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported registry to {path}")
            
        except Exception as e:
            raise RegistryError(f"Failed to export registry: {str(e)}") from e
    
    def import_registry(
        self,
        path: Union[str, Path],
        format: str = 'json'
    ) -> int:
        """
        Import registry data.
        
        Args:
            path: Import path
            format: Import format ('json', 'csv')
            
        Returns:
            Number of entries imported
        """
        path = Path(path)
        
        try:
            if format == 'json':
                data = load_json(path)
                entries = data.get('registry', [])
            elif format == 'csv':
                import pandas as pd
                df = pd.read_csv(path)
                entries = df.to_dict('records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            count = 0
            for entry_data in entries:
                # Convert to RegistryEntry
                entry = RegistryEntry(**entry_data)
                self._upsert_entry(entry)
                count += 1
            
            logger.info(f"Imported {count} entries from {path}")
            return count
            
        except Exception as e:
            raise RegistryError(f"Failed to import registry: {str(e)}") from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total models
                cursor.execute('SELECT COUNT(*) FROM registry')
                total = cursor.fetchone()[0]
                
                # By type
                cursor.execute('''
                    SELECT model_type, COUNT(*) 
                    FROM registry 
                    GROUP BY model_type
                ''')
                by_type = dict(cursor.fetchall())
                
                # By status
                cursor.execute('''
                    SELECT status, COUNT(*) 
                    FROM registry 
                    GROUP BY status
                ''')
                by_status = dict(cursor.fetchall())
                
                # Recent models
                cursor.execute('''
                    SELECT model_name, version, created_at 
                    FROM registry 
                    ORDER BY created_at DESC 
                    LIMIT 5
                ''')
                recent = [{'name': row[0], 'version': row[1], 'date': row[2]} 
                         for row in cursor.fetchall()]
                
                return {
                    'total_models': total,
                    'by_type': by_type,
                    'by_status': by_status,
                    'recent_models': recent,
                    'registry_path': str(self.registry_path),
                    'last_updated': get_timestamp()
                }
                
        except Exception as e:
            raise RegistryError(f"Failed to get statistics: {str(e)}") from e