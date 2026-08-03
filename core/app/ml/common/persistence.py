"""
Model persistence and serialization utilities.
"""

from typing import Any, Optional, Union, Dict, Type
from pathlib import Path
import pickle
import json
import joblib
import yaml
from datetime import datetime
import shutil
from loguru import logger

from .exceptions import ModelSaveError, ModelLoadError
from .models import BaseMLModel, ModelMetadata
from .utils import ensure_directory, save_json, load_json, generate_checksum, get_timestamp


class ModelPersistence:
    """
    Handles saving and loading of models with metadata.
    """
    
    # Supported serialization formats
    FORMATS = ['pickle', 'joblib', 'cloudpickle']
    
    @classmethod
    def save_model(
        cls,
        model: BaseMLModel,
        path: Union[str, Path],
        format: str = 'joblib',
        include_metadata: bool = True,
        compress: bool = False
    ) -> Dict[str, Any]:
        """
        Save a model to disk.
        
        Args:
            model: Model to save
            path: Path to save the model
            format: Serialization format ('pickle', 'joblib', 'cloudpickle')
            include_metadata: Whether to save metadata alongside the model
            compress: Whether to compress the model file
            
        Returns:
            Dictionary with save information
        """
        path = Path(path)
        ensure_directory(path.parent)
        
        try:
            # Determine file extension
            ext = '.joblib' if format == 'joblib' else '.pkl'
            if compress and format == 'joblib':
                ext = '.joblib.gz'
            
            model_path = path.with_suffix(ext)
            
            # Save the model
            if format == 'joblib':
                joblib.dump(
                    model.get_model(),
                    model_path,
                    compress=compress
                )
            elif format == 'pickle':
                with open(model_path, 'wb') as f:
                    pickle.dump(model.get_model(), f)
            elif format == 'cloudpickle':
                import cloudpickle
                with open(model_path, 'wb') as f:
                    cloudpickle.dump(model.get_model(), f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Model saved to {model_path}")
            
            # Save metadata
            metadata_info = {}
            if include_metadata and model._metadata:
                metadata_path = path.with_suffix('.meta.json')
                metadata_dict = model._metadata.model_dump()
                metadata_dict['model_path'] = str(model_path)
                metadata_dict['format'] = format
                metadata_dict['compressed'] = compress
                metadata_dict['saved_at'] = get_timestamp()
                
                save_json(metadata_dict, metadata_path)
                metadata_info = metadata_dict
                logger.info(f"Metadata saved to {metadata_path}")
            
            # Calculate file size
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            
            return {
                'model_path': str(model_path),
                'metadata_path': str(metadata_path) if include_metadata else None,
                'format': format,
                'size_mb': file_size,
                'compressed': compress,
                'saved_at': get_timestamp()
            }
            
        except Exception as e:
            raise ModelSaveError(f"Failed to save model: {str(e)}") from e
    
    @classmethod
    def load_model(
        cls,
        path: Union[str, Path],
        model_class: Optional[Type[BaseMLModel]] = None,
        **kwargs
    ) -> BaseMLModel:
        """
        Load a model from disk.
        
        Args:
            path: Path to the model file or metadata file
            model_class: The model class to instantiate (if metadata not available)
            **kwargs: Additional arguments for model instantiation
            
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        try:
            # Check if it's a metadata file
            if path.suffix == '.meta.json':
                metadata = load_json(path)
                model_path = Path(metadata.get('model_path', path.parent / 'model.joblib'))
                format = metadata.get('format', 'joblib')
            else:
                model_path = path
                # Try to find metadata
                metadata_path = path.with_suffix('.meta.json')
                if metadata_path.exists():
                    metadata = load_json(metadata_path)
                    format = metadata.get('format', 'joblib')
                else:
                    # Guess format from extension
                    if path.suffix in ['.pkl', '.pickle']:
                        format = 'pickle'
                    elif path.suffix in ['.joblib', '.gz']:
                        format = 'joblib'
                    else:
                        format = 'joblib'
            
            # Load the model
            if format == 'joblib':
                model_obj = joblib.load(model_path)
            elif format == 'pickle':
                with open(model_path, 'rb') as f:
                    model_obj = pickle.load(f)
            elif format == 'cloudpickle':
                import cloudpickle
                with open(model_path, 'rb') as f:
                    model_obj = cloudpickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Create model wrapper if needed
            if model_class is not None:
                # Extract model name from metadata or path
                model_name = metadata.get('model_name', path.stem) if 'metadata' in locals() else path.stem
                
                # Get hyperparameters
                hyperparams = metadata.get('hyperparameters', {}) if 'metadata' in locals() else kwargs
                
                # Create model instance
                model = model_class(name=model_name, **hyperparams)
                
                # Set the underlying model
                model._model = model_obj
                model._is_fitted = True
                
                # Restore metadata
                if 'metadata' in locals():
                    model._metadata = ModelMetadata(**metadata)
                
                logger.info(f"Model loaded from {model_path}")
                return model
            else:
                # Return the raw model object
                logger.info(f"Raw model loaded from {model_path}")
                return model_obj
                
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}") from e
    
    @classmethod
    def export_model(
        cls,
        model: BaseMLModel,
        path: Union[str, Path],
        format: str = 'onnx'
    ) -> None:
        """
        Export model to different formats.
        
        Args:
            model: Model to export
            path: Export path
            format: Export format ('onnx', 'pmml', 'sklearn')
        """
        path = Path(path)
        ensure_directory(path.parent)
        
        try:
            if format == 'onnx':
                cls._export_onnx(model, path)
            elif format == 'pmml':
                cls._export_pmml(model, path)
            elif format == 'sklearn':
                cls.save_model(model, path, format='joblib')
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            logger.info(f"Model exported to {path} in {format} format")
            
        except Exception as e:
            raise ModelSaveError(f"Failed to export model: {str(e)}") from e
    
    @classmethod
    def _export_onnx(cls, model: BaseMLModel, path: Path) -> None:
        """Export model to ONNX format."""
        try:
            import onnx
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Get the underlying model
            model_obj = model.get_model()
            
            # Determine input shape
            if hasattr(model, '_n_features'):
                n_features = model._n_features
            else:
                n_features = 10  # Default
            
            # Convert to ONNX
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(
                model_obj,
                initial_types=initial_type,
                target_opset=12
            )
            
            # Save ONNX model
            onnx.save(onnx_model, str(path))
            
        except ImportError:
            raise ImportError("skl2onnx not installed. Install with: pip install skl2onnx onnx")
    
    @classmethod
    def _export_pmml(cls, model: BaseMLModel, path: Path) -> None:
        """Export model to PMML format."""
        try:
            from sklearn2pmml import sklearn2pmml
            from sklearn2pmml.pipeline import PMMLPipeline
            
            # Create PMML pipeline
            model_obj = model.get_model()
            pipeline = PMMLPipeline([('model', model_obj)])
            
            # Export to PMML
            sklearn2pmml(pipeline, str(path))
            
        except ImportError:
            raise ImportError("sklearn2pmml not installed. Install with: pip install sklearn2pmml")


class ModelSerializer:
    """
    Handles serialization and deserialization of model objects.
    """
    
    @staticmethod
    def serialize_model(
        model: BaseMLModel,
        include_metadata: bool = True
    ) -> bytes:
        """
        Serialize a model to bytes.
        
        Args:
            model: Model to serialize
            include_metadata: Whether to include metadata
            
        Returns:
            Serialized bytes
        """
        try:
            # Get model data
            model_data = {
                'model': model.get_model(),
                'metadata': model._metadata.model_dump() if include_metadata and model._metadata else None,
                'is_fitted': model.is_fitted(),
                'name': model.name
            }
            
            # Serialize with pickle
            return pickle.dumps(model_data, protocol=pickle.HIGHEST_PROTOCOL)
            
        except Exception as e:
            raise ModelSaveError(f"Failed to serialize model: {str(e)}") from e
    
    @staticmethod
    def deserialize_model(
        data: bytes,
        model_class: Optional[Type[BaseMLModel]] = None,
        **kwargs
    ) -> BaseMLModel:
        """
        Deserialize a model from bytes.
        
        Args:
            data: Serialized bytes
            model_class: Model class to instantiate
            **kwargs: Additional arguments for model instantiation
            
        Returns:
            Deserialized model
        """
        try:
            # Deserialize data
            model_data = pickle.loads(data)
            
            if model_class is None:
                raise ValueError("model_class is required for deserialization")
            
            # Create model instance
            model_name = model_data.get('name', 'deserialized_model')
            hyperparams = kwargs
            
            if model_data.get('metadata'):
                hyperparams.update(model_data['metadata'].get('hyperparameters', {}))
            
            model = model_class(name=model_name, **hyperparams)
            
            # Set model state
            model._model = model_data['model']
            model._is_fitted = model_data.get('is_fitted', True)
            
            if model_data.get('metadata'):
                model._metadata = ModelMetadata(**model_data['metadata'])
            
            return model
            
        except Exception as e:
            raise ModelLoadError(f"Failed to deserialize model: {str(e)}") from e


class ModelVersioning:
    """
    Handles model versioning and lifecycle management.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        ensure_directory(self.base_path)
    
    def create_version(
        self,
        model: BaseMLModel,
        version_tag: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new version of a model.
        
        Args:
            model: Model to version
            version_tag: Optional version tag
            **kwargs: Additional metadata
            
        Returns:
            Version information
        """
        # Generate version tag if not provided
        if version_tag is None:
            version_tag = datetime.now().strftime('v%Y%m%d_%H%M%S')
        
        # Create version directory
        version_path = self.base_path / model.name / version_tag
        ensure_directory(version_path)
        
        # Save model
        model_path = version_path / 'model.joblib'
        save_info = ModelPersistence.save_model(
            model,
            model_path,
            format='joblib',
            include_metadata=True
        )
        
        # Save additional version info
        version_info = {
            'version': version_tag,
            'model_name': model.name,
            'model_type': model._metadata.model_type if model._metadata else 'unknown',
            'created_at': get_timestamp(),
            'version_path': str(version_path),
            'metrics': model._metadata.metrics if model._metadata else {},
            'hyperparameters': model._metadata.hyperparameters if model._metadata else {},
            'checksum': generate_checksum(model),
            **kwargs
        }
        
        # Save version metadata
        version_meta_path = version_path / 'version.json'
        save_json(version_info, version_meta_path)
        
        # Update latest symlink
        latest_path = self.base_path / model.name / 'latest'
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(version_tag, target_is_directory=True)
        
        logger.info(f"Created version {version_tag} for model {model.name}")
        
        return version_info
    
    def get_version(self, model_name: str, version_tag: str = 'latest') -> Dict[str, Any]:
        """
        Get version information.
        
        Args:
            model_name: Name of the model
            version_tag: Version tag or 'latest'
            
        Returns:
            Version information
        """
        version_path = self.base_path / model_name / version_tag
        
        # Resolve latest symlink
        if version_tag == 'latest':
            latest_path = self.base_path / model_name / 'latest'
            if latest_path.exists() and latest_path.is_symlink():
                version_path = latest_path.resolve()
            else:
                raise FileNotFoundError(f"No latest version found for model {model_name}")
        
        # Load version metadata
        version_meta_path = version_path / 'version.json'
        if not version_meta_path.exists():
            raise FileNotFoundError(f"Version metadata not found: {version_meta_path}")
        
        return load_json(version_meta_path)
    
    def list_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version tags
        """
        model_path = self.base_path / model_name
        if not model_path.exists():
            return []
        
        versions = []
        for item in model_path.iterdir():
            if item.is_dir() and item.name != 'latest':
                # Check if it's a valid version
                version_meta = item / 'version.json'
                if version_meta.exists():
                    versions.append(item.name)
        
        return sorted(versions)
    
    def delete_version(self, model_name: str, version_tag: str) -> None:
        """
        Delete a model version.
        
        Args:
            model_name: Name of the model
            version_tag: Version to delete
        """
        version_path = self.base_path / model_name / version_tag
        
        if not version_path.exists():
            raise FileNotFoundError(f"Version {version_tag} not found for model {model_name}")
        
        # Don't delete latest
        latest_path = self.base_path / model_name / 'latest'
        if latest_path.exists() and latest_path.is_symlink():
            if latest_path.resolve() == version_path:
                raise ValueError("Cannot delete the latest version")
        
        # Delete the version
        shutil.rmtree(version_path)
        logger.info(f"Deleted version {version_tag} for model {model_name}")
    
    def promote_version(self, model_name: str, version_tag: str) -> None:
        """
        Promote a version to latest.
        
        Args:
            model_name: Name of the model
            version_tag: Version to promote
        """
        version_path = self.base_path / model_name / version_tag
        if not version_path.exists():
            raise FileNotFoundError(f"Version {version_tag} not found for model {model_name}")
        
        # Update latest symlink
        latest_path = self.base_path / model_name / 'latest'
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(version_tag, target_is_directory=True)
        
        logger.info(f"Promoted version {version_tag} to latest for model {model_name}")