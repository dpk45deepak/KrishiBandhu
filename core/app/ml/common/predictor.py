"""
Predictor module for making predictions with trained models.
"""

from typing import Optional, Union, Dict, Any, List
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from .models import BaseMLModel
from .exceptions import PredictionError
from .persistence import ModelPersistence


class ModelPredictor:
    """
    Unified predictor for making predictions with trained models.
    """
    
    def __init__(self, model: Optional[BaseMLModel] = None):
        """
        Initialize predictor.
        
        Args:
            model: Trained model (optional, can be loaded later)
        """
        self.model = model
        self._is_loaded = model is not None
    
    def load_model(self, path: Union[str, Path]) -> 'ModelPredictor':
        """
        Load a model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Self
        """
        from .persistence import ModelPersistence
        
        # Detect model class from metadata
        metadata_path = Path(path).with_suffix('.meta.json')
        if metadata_path.exists():
            from .registry import ModelRegistry
            registry = ModelRegistry(Path.cwd() / 'models' / 'registry')
            entry = registry.get_entry(model_name=path.stem)
            # Load model with appropriate class
            self.model = ModelPersistence.load_model(path)
        else:
            self.model = ModelPersistence.load_model(path)
        
        self._is_loaded = True
        logger.info(f"Model loaded from {path}")
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
        batch_size: Optional[int] = None,
        return_proba: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            batch_size: Batch size for large datasets
            return_proba: Whether to return probabilities (classification only)
            
        Returns:
            Predictions array or dictionary with predictions and metadata
        """
        if not self._is_loaded or self.model is None:
            raise PredictionError("No model loaded. Call load_model() first.")
        
        try:
            # Convert to numpy array
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            elif isinstance(X, list):
                X_array = np.array(X)
            else:
                X_array = X
            
            # Check if model is fitted
            if not self.model.is_fitted():
                raise PredictionError("Model is not fitted")
            
            # Make predictions
            if batch_size is not None and len(X_array) > batch_size:
                # Batch prediction for large datasets
                predictions = []
                for i in range(0, len(X_array), batch_size):
                    batch = X_array[i:i+batch_size]
                    if return_proba and hasattr(self.model, 'predict_proba'):
                        pred = self.model.predict_proba(batch)
                    else:
                        pred = self.model.predict(batch)
                    predictions.append(pred)
                
                result = np.vstack(predictions) if len(predictions) > 1 else predictions[0]
            else:
                # Single batch
                if return_proba and hasattr(self.model, 'predict_proba'):
                    result = self.model.predict_proba(X_array)
                else:
                    result = self.model.predict(X_array)
            
            # Prepare result with metadata
            result_dict = {
                'predictions': result.tolist() if isinstance(result, np.ndarray) else result,
                'model_name': self.model.name,
                'n_samples': len(X_array),
                'is_fitted': self.model.is_fitted()
            }
            
            # Add class info for classification
            if hasattr(self.model, 'get_n_classes'):
                result_dict['n_classes'] = self.model.get_n_classes()
                result_dict['classes'] = self.model.get_classes().tolist()
            
            # Add probabilities if not already included
            if not return_proba and hasattr(self.model, 'predict_proba'):
                result_dict['probabilities'] = self.model.predict_proba(X_array).tolist()
            
            return result_dict
            
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}") from e
    
    def predict_single(
        self,
        X: Union[np.ndarray, pd.Series, List],
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """
        Predict for a single instance.
        
        Args:
            X: Single instance features
            return_proba: Whether to return probabilities
            
        Returns:
            Dictionary with prediction and metadata
        """
        if isinstance(X, (pd.Series, list)):
            X = np.array(X).reshape(1, -1)
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X = X.reshape(1, -1)
        
        result = self.predict(X, return_proba=return_proba)
        
        # Extract single prediction
        single_result = {
            'prediction': result['predictions'][0] if isinstance(result['predictions'], list) else result['predictions'],
            'model_name': result['model_name']
        }
        
        if 'probabilities' in result:
            single_result['probabilities'] = result['probabilities'][0]
        
        return single_result
    
    def predict_batch(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        batch_size: int = 1000,
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """
        Predict in batches for large datasets.
        
        Args:
            X: Features to predict on
            batch_size: Batch size
            return_proba: Whether to return probabilities
            
        Returns:
            Dictionary with predictions and metadata
        """
        return self.predict(X, batch_size=batch_size, return_proba=return_proba)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from the model.
        
        Returns:
            Feature importance array or None
        """
        if self.model is None:
            raise PredictionError("No model loaded")
        
        if hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance()
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise PredictionError("No model loaded")
        
        info = {
            'name': self.model.name,
            'is_fitted': self.model.is_fitted(),
            'hyperparameters': self.model.get_params()
        }
        
        if hasattr(self.model, 'get_n_classes'):
            info['type'] = 'classification'
            info['n_classes'] = self.model.get_n_classes()
            info['classes'] = self.model.get_classes().tolist()
        else:
            info['type'] = 'regression'
        
        if self.model._metadata:
            info['metadata'] = self.model._metadata.model_dump()
        
        return info