"""
Ensemble classifiers: Voting and Stacking ensembles.
"""

from typing import Any, Optional, Union, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from ..common.models import ClassificationModel
from ..common.exceptions import TrainingError, PredictionError


class VotingEnsemble(ClassificationModel):
    """
    Voting Ensemble Classifier wrapper.
    Combines multiple models using voting (hard/soft).
    """
    
    def __init__(
        self,
        name: str = "VotingEnsemble",
        estimators: Optional[List[Tuple[str, ClassificationModel]]] = None,
        voting: str = 'soft',
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.estimators = estimators or []
        self.voting = voting
        self.weights = weights
        self._additional_kwargs = kwargs
        
        self._model = None
        self._base_models = None
        self._model_names = None
    
    def add_estimator(self, name: str, model: ClassificationModel) -> 'VotingEnsemble':
        """Add an estimator to the ensemble."""
        self.estimators.append((name, model))
        return self
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'VotingEnsemble':
        """Fit the Voting Ensemble."""
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            if not self.estimators:
                raise ValueError("No estimators added to the ensemble")
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Fit each base model
            self._base_models = []
            self._model_names = []
            sklearn_estimators = []
            
            for name, model in self.estimators:
                model.fit(X, y)
                self._base_models.append(model)
                self._model_names.append(name)
                sklearn_estimators.append((name, model.get_model()))
            
            # Get classes from first model
            self._classes = self._base_models[0].get_classes()
            self._n_classes = len(self._classes)
            
            # Create voting classifier
            self._model = VotingClassifier(
                estimators=sklearn_estimators,
                voting=self.voting,
                weights=self.weights,
                **self._additional_kwargs
            )
            
            # Fit the voting classifier (this will refit models)
            # We already fitted the models, so we just need to fit the voting classifier
            # But VotingClassifier expects unfitted models, so we'll pass the fitted ones
            # and use fit to just set them
            self._model.estimators_ = [est[1] for est in sklearn_estimators]
            self._model.named_estimators_ = dict(sklearn_estimators)
            self._model.classes_ = self._classes
            
            # Set fitted attribute to True
            self._model._is_fitted = True
            self._is_fitted = True
            
            return self
            
        except Exception as e:
            raise TrainingError(f"Voting Ensemble training failed: {str(e)}") from e
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise PredictionError("Model not fitted")
        
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self._model.predict(X)
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}") from e
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise PredictionError("Model not fitted")
        
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self._model.predict_proba(X)
        except Exception as e:
            raise PredictionError(f"Probability prediction failed: {str(e)}") from e
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'estimators': [(name, model.get_params()) for name, model in self.estimators],
            'voting': self.voting,
            'weights': self.weights,
            **self._additional_kwargs
        }
    
    def get_model(self) -> VotingClassifier:
        """Get the underlying sklearn model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model
    
    def get_estimators(self) -> List[Tuple[str, ClassificationModel]]:
        """Get base estimators."""
        return self.estimators


class StackingEnsemble(ClassificationModel):
    """
    Stacking Ensemble Classifier wrapper.
    Combines multiple models using a meta-learner.
    """
    
    def __init__(
        self,
        name: str = "StackingEnsemble",
        estimators: Optional[List[Tuple[str, ClassificationModel]]] = None,
        final_estimator: Optional[ClassificationModel] = None,
        cv: int = 5,
        stack_method: str = 'predict_proba',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.estimators = estimators or []
        self.cv = cv
        self.stack_method = stack_method
        self._additional_kwargs = kwargs
        
        # Set default final estimator if not provided
        if final_estimator is None:
            from .logistic import LogisticRegressionClassifier
            self.final_estimator = LogisticRegressionClassifier()
        else:
            self.final_estimator = final_estimator
        
        self._model = None
        self._base_models = None
        self._model_names = None
    
    def add_estimator(self, name: str, model: ClassificationModel) -> 'StackingEnsemble':
        """Add an estimator to the ensemble."""
        self.estimators.append((name, model))
        return self
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'StackingEnsemble':
        """Fit the Stacking Ensemble."""
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            if not self.estimators:
                raise ValueError("No estimators added to the ensemble")
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Fit each base model
            self._base_models = []
            self._model_names = []
            sklearn_estimators = []
            
            for name, model in self.estimators:
                model.fit(X, y)
                self._base_models.append(model)
                self._model_names.append(name)
                sklearn_estimators.append((name, model.get_model()))
            
            # Get classes from first model
            self._classes = self._base_models[0].get_classes()
            self._n_classes = len(self._classes)
            
            # Fit final estimator
            # For stacking, we need to fit the final estimator on the predictions of base models
            X_stack = self._get_stack_features(X)
            self.final_estimator.fit(X_stack, y)
            
            # Create stacking classifier
            self._model = StackingClassifier(
                estimators=sklearn_estimators,
                final_estimator=self.final_estimator.get_model(),
                cv=self.cv,
                stack_method=self.stack_method,
                **self._additional_kwargs
            )
            
            # Set fitted state
            self._model._is_fitted = True
            self._model.classes_ = self._classes
            self._model.estimators_ = [est[1] for est in sklearn_estimators]
            self._model.named_estimators_ = dict(sklearn_estimators)
            self._model.final_estimator_ = self.final_estimator.get_model()
            
            self._is_fitted = True
            
            return self
            
        except Exception as e:
            raise TrainingError(f"Stacking Ensemble training failed: {str(e)}") from e
    
    def _get_stack_features(self, X: np.ndarray) -> np.ndarray:
        """Get stack features from base models."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        predictions = []
        for model in self._base_models:
            if self.stack_method == 'predict_proba':
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).reshape(-1, 1)
            predictions.append(pred)
        
        return np.hstack(predictions)
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise PredictionError("Model not fitted")
        
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # Get stack features
            X_stack = self._get_stack_features(X)
            
            # Predict with final estimator
            return self.final_estimator.predict(X_stack)
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}") from e
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise PredictionError("Model not fitted")
        
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # Get stack features
            X_stack = self._get_stack_features(X)
            
            # Predict with final estimator
            return self.final_estimator.predict_proba(X_stack)
        except Exception as e:
            raise PredictionError(f"Probability prediction failed: {str(e)}") from e

    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'estimators': [(name, model.get_params()) for name, model in self.estimators],
            'final_estimator': self.final_estimator.get_params(),
            'cv': self.cv,
            'stack_method': self.stack_method,
            **self._additional_kwargs
        }

    def get_model(self) -> StackingClassifier:
        """Get the underlying sklearn model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model
    
    def get_estimators(self) -> List[Tuple[str, ClassificationModel]]:
        """Get base estimators."""
        return self.estimators
    
    def get_final_estimator(self) -> ClassificationModel:
        """Get the final estimator."""
        return self.final_estimator