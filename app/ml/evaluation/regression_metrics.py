"""
Regression metrics for model evaluation.
"""

from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
    max_error,
    mean_squared_log_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
)
from scipy.stats import pearsonr, spearmanr
from loguru import logger

from ..common.exceptions import EvaluationError


class RegressionMetrics:
    """
    Comprehensive regression metrics calculator.
    """
    
    @staticmethod
    def calculate(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        multioutput: str = 'uniform_average'
    ) -> Dict[str, float]:
        """
        Calculate all regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            multioutput: Aggregation method for multioutput
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Convert to numpy arrays
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            if isinstance(y_pred, pd.Series):
                y_pred = y_pred.values
            
            # Basic metrics
            metrics = {}
            
            # MAE
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            # MSE
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            
            # RMSE
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # R²
            metrics['r2'] = r2_score(y_true, y_pred, multioutput=multioutput)
            
            # MAPE
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            except:
                metrics['mape'] = np.nan
            
            # Median Absolute Error
            metrics['median_ae'] = median_absolute_error(y_true, y_pred)
            
            # Explained Variance
            metrics['explained_variance'] = explained_variance_score(
                y_true, y_pred, multioutput=multioutput
            )
            
            # Max Error
            metrics['max_error'] = max_error(y_true, y_pred)
            
            # Mean Squared Log Error (only for positive values)
            if np.all(y_true > 0) and np.all(y_pred > 0):
                try:
                    metrics['msle'] = mean_squared_log_error(y_true, y_pred)
                    metrics['rmsle'] = np.sqrt(metrics['msle'])
                except:
                    pass
            
            # Poisson Deviance (only for non-negative values)
            if np.all(y_true >= 0) and np.all(y_pred >= 0):
                try:
                    metrics['poisson_deviance'] = mean_poisson_deviance(y_true, y_pred)
                except:
                    pass
            
            # Gamma Deviance (only for positive values)
            if np.all(y_true > 0) and np.all(y_pred > 0):
                try:
                    metrics['gamma_deviance'] = mean_gamma_deviance(y_true, y_pred)
                except:
                    pass
            
            # Correlation metrics
            try:
                pearson_corr, _ = pearsonr(y_true, y_pred)
                metrics['pearson_correlation'] = pearson_corr
            except:
                metrics['pearson_correlation'] = np.nan
            
            try:
                spearman_corr, _ = spearmanr(y_true, y_pred)
                metrics['spearman_correlation'] = spearman_corr
            except:
                metrics['spearman_correlation'] = np.nan
            
            # Additional useful metrics
            residuals = y_true - y_pred
            metrics['residuals_mean'] = np.mean(residuals)
            metrics['residuals_std'] = np.std(residuals)
            metrics['residuals_skew'] = pd.Series(residuals).skew()
            metrics['residuals_kurtosis'] = pd.Series(residuals).kurtosis()
            
            # Relative metrics
            if np.mean(y_true) > 0:
                metrics['relative_absolute_error'] = metrics['mae'] / np.mean(y_true)
                metrics['relative_squared_error'] = metrics['mse'] / (np.mean(y_true) ** 2)
            
            return metrics
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate regression metrics: {str(e)}") from e
    
    @staticmethod
    def get_residuals(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Calculate residuals and related statistics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with residuals and statistics
        """
        try:
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            if isinstance(y_pred, pd.Series):
                y_pred = y_pred.values
            
            residuals = y_true - y_pred
            standardized_residuals = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
            
            return {
                'residuals': residuals,
                'standardized_residuals': standardized_residuals,
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'percentile_25': np.percentile(residuals, 25),
                'percentile_50': np.percentile(residuals, 50),
                'percentile_75': np.percentile(residuals, 75),
            }
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate residuals: {str(e)}") from e
    
    @staticmethod
    def summary(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Generate a complete summary of regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Complete summary dictionary
        """
        metrics = RegressionMetrics.calculate(y_true, y_pred)
        residuals_info = RegressionMetrics.get_residuals(y_true, y_pred)
        
        summary = {
            'metrics': metrics,
            'residuals': residuals_info,
            'n_samples': len(y_true),
            'mean_true': np.mean(y_true),
            'mean_pred': np.mean(y_pred),
            'std_true': np.std(y_true),
            'std_pred': np.std(y_pred),
            'min_true': np.min(y_true),
            'max_true': np.max(y_true),
            'min_pred': np.min(y_pred),
            'max_pred': np.max(y_pred),
        }
        
        return summary
    
    @staticmethod
    def check_assumptions(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Check regression assumptions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with assumption checks
        """
        residuals = y_true - y_pred
        n = len(residuals)
        
        # Normality test (Shapiro-Wilk)
        from scipy import stats
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        # Independence test (Durbin-Watson)
        dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
        
        # Homoscedasticity (Breusch-Pagan test)
        # Simplified version using correlation between residuals and predictions
        bp_corr = np.corrcoef(np.abs(residuals), y_pred)[0, 1]
        
        # Linearity check (correlation between actual and predicted)
        linearity_corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        return {
            'shapiro_wilk_statistic': shapiro_stat,
            'shapiro_wilk_p_value': shapiro_p,
            'normality_check': 'Normal' if shapiro_p > 0.05 else 'Non-normal',
            'durbin_watson': dw,
            'independence_check': 'Independent' if 1.5 < dw < 2.5 else 'Possible autocorrelation',
            'breusch_pagan_correlation': bp_corr,
            'homoscedasticity_check': 'Homoscedastic' if abs(bp_corr) < 0.3 else 'Heteroscedastic',
            'linearity_correlation': linearity_corr,
            'linearity_check': 'Linear' if linearity_corr > 0.7 else 'Potential non-linearity',
            'n_samples': n,
        }