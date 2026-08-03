"""
Generic trainer for ML models.
"""

from typing import Optional, Union, Any, Tuple, Dict, List, Type
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

from .exceptions import TrainingError
from .models import BaseMLModel, ModelMetadata
from .datasets import DatasetLoader
from .callbacks import (
    CallbackManager,
    TrainingLogger,
    EarlyStopping,
    ModelCheckpoint,
    ProgressBar,
    MetricsTracker,
    Callback
)
from .utils import (
    Timer,
    generate_model_version,
    generate_checksum,
    ensure_directory,
    save_json,
    log_metrics
)


class TrainingConfig:
    """
    Configuration for model training.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.01,
        validation_split: float = 0.1,
        shuffle: bool = True,
        random_seed: int = 42,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_monitor: str = 'val_loss',
        checkpoint: bool = False,
        checkpoint_path: Optional[Path] = None,
        save_best_only: bool = True,
        verbose: bool = True,
        use_progress_bar: bool = True
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.checkpoint = checkpoint
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.use_progress_bar = use_progress_bar
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'shuffle': self.shuffle,
            'random_seed': self.random_seed,
            'early_stopping': self.early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_monitor': self.early_stopping_monitor,
            'checkpoint': self.checkpoint,
            'save_best_only': self.save_best_only,
            'verbose': self.verbose,
            'use_progress_bar': self.use_progress_bar
        }


class ModelTrainer:
    """
    Generic trainer for ML models.
    Handles training loop, validation, and callbacks.
    """
    
    def __init__(
        self,
        model: BaseMLModel,
        config: TrainingConfig,
        callbacks: Optional[List[Callback]] = None
    ):
        self.model = model
        self.config = config
        self.callback_manager = CallbackManager(callbacks or [])
        self.metrics_tracker = MetricsTracker()
        self.training_history: Dict[str, List[float]] = {}
        self.best_model: Optional[BaseMLModel] = None
        self.best_score: Optional[float] = None
        
        # Set up default callbacks
        self._setup_default_callbacks()
    
    def _setup_default_callbacks(self) -> None:
        """Setup default callbacks based on configuration."""
        # Add metrics tracker
        self.callback_manager.add_callback(self.metrics_tracker)
        
        # Add training logger
        logger_callback = TrainingLogger(
            log_every=1,
            log_file=None  # Could be configured
        )
        self.callback_manager.add_callback(logger_callback)
        self._logger_callback = logger_callback
        
        # Add progress bar
        if self.config.use_progress_bar:
            self.callback_manager.add_callback(
                ProgressBar(self.config.epochs)
            )
        
        # Add early stopping
        if self.config.early_stopping:
            self.callback_manager.add_callback(
                EarlyStopping(
                    monitor=self.config.early_stopping_monitor,
                    patience=self.config.early_stopping_patience
                )
            )
        
        # Add model checkpoint
        if self.config.checkpoint and self.config.checkpoint_path:
            self.callback_manager.add_callback(
                ModelCheckpoint(
                    save_path=self.config.checkpoint_path,
                    monitor=self.config.early_stopping_monitor,
                    save_best_only=self.config.save_best_only
                )
            )
    
    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'ModelTrainer':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Self
        """
        try:
            # Prepare data
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(y_train, pd.Series):
                y_train = y_train.values
            
            if X_val is not None and isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if y_val is not None and isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            # Use validation split if validation data not provided
            has_validation = X_val is not None and y_val is not None
            if not has_validation and self.config.validation_split > 0:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train,
                    test_size=self.config.validation_split,
                    random_state=self.config.random_seed,
                    shuffle=self.config.shuffle
                )
                has_validation = True
            
            # Set random seed
            if self.config.random_seed:
                np.random.seed(self.config.random_seed)
            
            # Start training
            self.callback_manager.on_train_begin(
                model=self.model,
                config=self.config
            )
            
            # Training loop
            for epoch in range(self.config.epochs):
                self.callback_manager.on_epoch_begin(
                    epoch=epoch,
                    model=self.model
                )
                
                # Train for one epoch
                train_metrics = self._train_epoch(
                    X_train, y_train, epoch
                )
                
                # Validate if validation data available
                val_metrics = {}
                if has_validation:
                    val_metrics = self._validate_epoch(
                        X_val, y_val
                    )
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Track metrics
                self.callback_manager.on_epoch_end(
                    epoch=epoch,
                    metrics=metrics,
                    model=self.model
                )
                
                # Check for early stopping
                early_stop_callbacks = [
                    cb for cb in self.callback_manager.callbacks
                    if isinstance(cb, EarlyStopping) and cb.should_stop
                ]
                if early_stop_callbacks:
                    self.callback_manager.on_early_stopping(
                        epoch=epoch,
                        model=self.model
                    )
                    break
            
            # End training
            self.callback_manager.on_train_end(
                model=self.model,
                history=self._logger_callback.get_history()
            )
            
            # Store training history
            self.training_history = self.metrics_tracker.get_history()
            
            # Set metadata
            self._set_model_metadata()
            
            return self
            
        except Exception as e:
            raise TrainingError(f"Training failed: {str(e)}") from e
    
    def _train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        Override this method for custom training loops.
        
        Args:
            X: Training features
            y: Training target
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        # Default implementation: fit the model
        # For most scikit-learn models, we just fit once
        if epoch == 0:
            with Timer("Model fitting"):
                self.model.fit(X, y)
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            
            # Import metrics here to avoid circular imports
            from ..evaluation.classification_metrics import calculate_classification_metrics
            from ..evaluation.regression_metrics import calculate_regression_metrics
            
            # Determine metric type
            if hasattr(self.model, 'get_n_classes'):
                # Classification
                metrics = calculate_classification_metrics(y, y_pred)
            else:
                # Regression
                metrics = calculate_regression_metrics(y, y_pred)
            
            return metrics
        
        # For subsequent epochs, just return empty metrics
        return {}
    
    def _validate_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            X: Validation features
            y: Validation target
            
        Returns:
            Validation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Import metrics here to avoid circular imports
        from ..evaluation.classification_metrics import calculate_classification_metrics
        from ..evaluation.regression_metrics import calculate_regression_metrics
        
        # Determine metric type
        if hasattr(self.model, 'get_n_classes'):
            # Classification
            metrics = calculate_classification_metrics(y, y_pred)
        else:
            # Regression
            metrics = calculate_regression_metrics(y, y_pred)
        
        # Add prefix for validation metrics
        return {f"val_{k}": v for k, v in metrics.items()}
    
    def _set_model_metadata(self) -> None:
        """Set metadata for the trained model."""
        # Generate model version
        model_version = generate_model_version()
        
        # Get model parameters
        params = self.model.get_params()
        
        # Determine model type
        if hasattr(self.model, 'get_n_classes'):
            model_type = 'classification'
            n_classes = self.model.get_n_classes()
        else:
            model_type = 'regression'
            n_classes = None
        
        # Get best metrics
        best_metrics = self.metrics_tracker.best_metrics
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=f"{self.model.name}_{model_version}",
            model_version=model_version,
            model_name=self.model.name,
            model_type=model_type,
            training_timestamp=datetime.now(),
            training_dataset_version=None,  # Set by caller
            feature_version=None,  # Set by caller
            training_time_seconds=None,  # Set by caller
            n_samples=None,  # Set by caller
            n_features=None,  # Set by caller
            n_classes=n_classes,
            metrics=best_metrics,
            hyperparameters=params,
            checksum=generate_checksum(self.model),
            model_size_mb=None,  # Set by caller
            additional_info={
                'training_config': self.config.to_dict(),
                'epochs_completed': len(self.training_history.get('loss', [])) if 'loss' in self.training_history else 0
            }
        )
        
        self.model.set_metadata(metadata)
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.training_history
    
    def get_best_model(self) -> Optional[BaseMLModel]:
        """Get the best model from training."""
        # If early stopping was used, get the best model
        checkpoint_callbacks = [
            cb for cb in self.callback_manager.callbacks
            if isinstance(cb, ModelCheckpoint)
        ]
        
        if checkpoint_callbacks:
            best_checkpoint = checkpoint_callbacks[0].get_best_checkpoint()
            if best_checkpoint and best_checkpoint.exists():
                from .utils import load_pickle
                return load_pickle(best_checkpoint)
        
        # Otherwise return the current model
        return self.model
    
    def save_model(self, path: Path) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        from .persistence import ModelPersistence
        ModelPersistence.save_model(self.model, path)
    
    @classmethod
    def from_dataset(
        cls,
        model: BaseMLModel,
        dataset_loader: DatasetLoader,
        config: Optional[TrainingConfig] = None,
        callbacks: Optional[List[Callback]] = None
    ) -> 'ModelTrainer':
        """
        Create a trainer and train using a dataset loader.
        
        Args:
            model: Model to train
            dataset_loader: Dataset loader with preprocessed data
            config: Training configuration
            callbacks: List of callbacks
            
        Returns:
            Trained model trainer
        """
        if config is None:
            config = TrainingConfig()
        
        trainer = cls(model, config, callbacks)
        
        # Get data from dataset loader
        X, y = dataset_loader.get_data()
        
        # Split data
        X_train, y_train, X_val, y_val = dataset_loader.split()
        
        # Train
        trainer.train(X_train, y_train, X_val, y_val)
        
        # Update metadata with dataset info
        if trainer.model._metadata:
            metadata = trainer.model._metadata
            metadata.n_samples = dataset_loader.n_samples
            metadata.n_features = dataset_loader.n_features
            metadata.training_dataset_version = "1.0"  # Could be set from config
        
        return trainer