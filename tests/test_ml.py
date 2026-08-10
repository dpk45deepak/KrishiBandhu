"""
Unit tests for the ML framework.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from sklearn.datasets import make_classification, make_regression

from app.ml.common.datasets import DatasetConfig, DatasetLoader
from app.ml.common.models import BaseMLModel
from app.ml.common.trainer import ModelTrainer, TrainingConfig
from app.ml.common.registry import ModelRegistry
from app.ml.common.persistence import ModelPersistence
from app.ml.common.callbacks import (
    EarlyStopping,
    TrainingLogger,
    ModelCheckpoint,
    MetricsTracker
)
from app.ml.classification import RandomForestClassifier, XGBoostClassifier
from app.ml.regression import RandomForestRegressor, LinearRegression
from app.ml.evaluation import ClassificationMetrics, RegressionMetrics, ModelEvaluator
from app.ml.tuning import GridSearchTuner, RandomSearchTuner


class TestDatasetLoader:
    """Test DatasetLoader functionality."""
    
    def test_load_csv(self, tmp_path):
        """Test loading CSV file."""
        # Create test data
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        csv_path = tmp_path / 'test.csv'
        df.to_csv(csv_path, index=False)
        
        # Load dataset
        config = DatasetConfig(
            file_path=csv_path,
            target_column='target'
        )
        loader = DatasetLoader(config)
        loader.load()
        
        assert loader._data is not None
        assert len(loader._data) == 100
        assert len(loader._data.columns) == 3
    
    def test_preprocess(self, tmp_path):
        """Test preprocessing."""
        # Create test data with missing values
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': ['a', 'b', 'c', np.nan, 'e'],
            'target': [0, 1, 0, 1, 0]
        })
        csv_path = tmp_path / 'test.csv'
        df.to_csv(csv_path, index=False)
        
        config = DatasetConfig(
            file_path=csv_path,
            target_column='target',
            handle_missing='drop'
        )
        loader = DatasetLoader(config)
        loader.load().validate().preprocess()
        
        X, y = loader.get_data()
        assert len(X) > 0
        assert len(y) > 0
    
    def test_split(self, tmp_path):
        """Test train/test split."""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        csv_path = tmp_path / 'test.csv'
        df.to_csv(csv_path, index=False)
        
        config = DatasetConfig(
            file_path=csv_path,
            target_column='target',
            test_size=0.2
        )
        loader = DatasetLoader(config)
        loader.load().validate().preprocess()
        
        X_train, y_train, X_test, y_test = loader.split()
        
        assert len(X_train) + len(X_test) == 100
        assert len(X_train) == 80
        assert len(X_test) == 20


class TestClassificationModels:
    """Test classification models."""
    
    def test_random_forest(self):
        """Test Random Forest classifier."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
        
        model = RandomForestClassifier(n_estimators=10, max_depth=5)
        model.fit(X, y)
        
        assert model.is_fitted()
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert hasattr(model, 'predict_proba')
        
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
    
    def test_xgboost(self):
        """Test XGBoost classifier."""
        pytest.importorskip("xgboost")
        
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
        
        model = XGBoostClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        assert model.is_fitted()
        predictions = model.predict(X)
        assert len(predictions) == len(y)


class TestRegressionModels:
    """Test regression models."""
    
    def test_linear_regression(self):
        """Test Linear Regression."""
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
        
        model = LinearRegression()
        model.fit(X, y)
        
        assert model.is_fitted()
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        
        # Check coefficients
        coefs = model.get_coefficients()
        assert len(coefs) == 10
    
    def test_random_forest_regressor(self):
        """Test Random Forest regressor."""
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
        
        model = RandomForestRegressor(n_estimators=10, max_depth=5)
        model.fit(X, y)
        
        assert model.is_fitted()
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        
        # Check feature importance
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == 10


class TestTraining:
    """Test training functionality."""
    
    def test_trainer(self):
        """Test ModelTrainer."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
        
        model = RandomForestClassifier(n_estimators=10)
        config = TrainingConfig(epochs=10, verbose=False)
        trainer = ModelTrainer(model, config)
        
        trainer.train(X, y)
        
        assert model.is_fitted()
        assert trainer.get_history() is not None
    
    def test_early_stopping(self):
        """Test early stopping callback."""
        class MockModel:
            def __init__(self):
                self.epoch = 0
                self._is_fitted = True
            
            def fit(self, X, y):
                self.epoch += 1
                return self
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min'
        )
        
        # Simulate training with early stopping
        for epoch in range(10):
            metrics = {
                'val_loss': 1.0 - epoch * 0.1  # Decreasing loss
            }
            early_stopping.on_epoch_end(epoch, metrics)
            
            if early_stopping.should_stop:
                assert epoch >= 3
                break


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_classification_metrics(self):
        """Test classification metrics."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8], [0.9, 0.1]])
        
        metrics = ClassificationMetrics.calculate(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
    
    def test_regression_metrics(self):
        """Test regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = RegressionMetrics.calculate(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
    
    def test_evaluator(self):
        """Test ModelEvaluator."""
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2)
        
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(model, X, y)
        
        assert 'metrics' in results
        assert 'predictions' in results
        assert len(results['predictions']) == len(y)


class TestPersistence:
    """Test model persistence."""
    
    def test_save_load(self, tmp_path):
        """Test saving and loading models."""
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2)
        
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        
        model_path = tmp_path / 'model.joblib'
        
        # Save
        save_info = ModelPersistence.save_model(model, model_path)
        assert save_info['model_path'] == str(model_path)
        
        # Load
        loaded_model = ModelPersistence.load_model(model_path)
        assert loaded_model.is_fitted()
        
        # Check predictions match
        pred1 = model.predict(X)
        pred2 = loaded_model.predict(X)
        assert np.array_equal(pred1, pred2)


class TestRegistry:
    """Test model registry."""
    
    def test_registry_operations(self, tmp_path):
        """Test registry operations."""
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2)
        
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        
        # Create registry
        registry = ModelRegistry(tmp_path / 'registry')
        
        # Register model
        model_path = tmp_path / 'model.joblib'
        ModelPersistence.save_model(model, model_path)
        
        entry = registry.register_model(
            model,
            model_path,
            tags=['test', 'rf'],
            description='Test model'
        )
        
        assert entry.model_id is not None
        assert entry.model_name == 'RandomForestClassifier'
        
        # List models
        entries = registry.list_models()
        assert len(entries) > 0
        
        # Get model
        loaded_model, loaded_entry = registry.get_model(model_id=entry.model_id)
        assert loaded_model.is_fitted()


class TestTuning:
    """Test hyperparameter tuning."""
    
    def test_grid_search(self):
        """Test grid search tuning."""
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2)
        
        param_space = {
            'n_estimators': [5, 10],
            'max_depth': [3, 5]
        }
        
        tuner = GridSearchTuner(
            RandomForestClassifier,
            param_space,
            n_trials=4,
            verbose=False
        )
        
        result = tuner.tune(X, y, cv=3)
        
        assert result.best_params is not None
        assert result.best_score is not None
        assert len(result.all_scores) == 4
    
    def test_random_search(self):
        """Test random search tuning."""
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2)
        
        param_space = {
            'n_estimators': {'type': 'int', 'low': 5, 'high': 20},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10}
        }
        
        tuner = RandomSearchTuner(
            RandomForestClassifier,
            param_space,
            n_trials=5,
            verbose=False
        )
        
        result = tuner.tune(X, y, cv=3)
        
        assert result.best_params is not None
        assert result.best_score is not None


def test_model_metadata():
    """Test model metadata."""
    X, y = make_classification(n_samples=50, n_features=5, n_classes=2)
    
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    
    metadata = model.get_metadata()
    assert metadata is not None
    assert metadata.model_name == 'RandomForestClassifier'
    assert metadata.model_type == 'classification'
    assert metadata.metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app/ml", "--cov-report=term"])