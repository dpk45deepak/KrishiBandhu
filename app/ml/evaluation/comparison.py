"""
Model comparison utilities.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from ..common.models import BaseMLModel
from .evaluator import ModelEvaluator
from .reports import ReportGenerator


class ModelComparator:
    """
    Compare multiple models.
    """
    
    def __init__(self):
        """Initialize model comparator."""
        self.results = {}
        logger.info("ModelComparator initialized")
    
    def compare_models(
        self,
        models: List[Tuple[BaseMLModel, str]],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            models: List of (model, name) tuples
            X_test: Test features
            y_test: Test targets
            **kwargs: Additional arguments for evaluator
            
        Returns:
            Comparison results
        """
        try:
            evaluator = ModelEvaluator()
            results = {}
            
            for model, name in models:
                logger.info(f"Evaluating {name}...")
                eval_results = evaluator.evaluate(
                    model,
                    X_test,
                    y_test,
                    **kwargs
                )
                results[name] = eval_results
            
            self.results = results
            
            # Calculate ranking
            comparison = self._compare_results(results)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {str(e)}")
            raise
    
    def _compare_results(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare evaluation results.
        
        Args:
            results: Dictionary mapping model names to evaluation results
            
        Returns:
            Comparison results
        """
        model_names = list(results.keys())
        all_metrics = {}
        
        # Collect all metrics
        for model_name, eval_results in results.items():
            metrics = eval_results.get('metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in all_metrics:
                        all_metrics[metric] = {}
                    all_metrics[metric][model_name] = value
        
        # Calculate rankings
        rankings = {}
        for metric, values in all_metrics.items():
            # Determine if higher is better
            higher_better = not any(m in metric.lower() for m in ['loss', 'error', 'mae', 'mse', 'rmse'])
            
            sorted_models = sorted(
                values.items(),
                key=lambda x: x[1],
                reverse=higher_better
            )
            
            rankings[metric] = {
                'best': sorted_models[0] if sorted_models else None,
                'worst': sorted_models[-1] if sorted_models else None,
                'ranking': [name for name, _ in sorted_models]
            }
        
        return {
            'models': model_names,
            'metrics': all_metrics,
            'rankings': rankings,
            'best_model': self._find_best_model(rankings, model_names)
        }
    
    def _find_best_model(
        self,
        rankings: Dict[str, Dict[str, Any]],
        model_names: List[str]
    ) -> Tuple[str, float]:
        """
        Find the best overall model.
        
        Args:
            rankings: Rankings dictionary
            model_names: List of model names
            
        Returns:
            Tuple of (best_model_name, score)
        """
        # Simple scoring: count how many times each model is best
        scores = {name: 0 for name in model_names}
        
        for metric, ranking in rankings.items():
            if ranking['best']:
                best_name = ranking['best'][0]
                scores[best_name] += 1
        
        best_model = max(scores.items(), key=lambda x: x[1])
        return best_model
    
    def generate_report(
        self,
        output_dir: Path,
        comparison_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Generate a comparison report.
        
        Args:
            output_dir: Output directory
            comparison_results: Comparison results (uses self.results if None)
        """
        results = comparison_results or self.results
        
        if not results:
            raise ValueError("No results to compare")
        
        report_gen = ReportGenerator(output_dir)
        report_gen.generate_comparison_report(results)
        
        logger.info(f"Comparison report generated at {output_dir}")
    
    def get_best_model(self) -> Tuple[str, BaseMLModel]:
        """
        Get the best model from comparison.
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.results:
            raise ValueError("No comparison results available")
        
        # Find best model based on accuracy/F1/other metrics
        best_score = -np.inf
        best_name = None
        best_model = None
        
        for name, eval_results in self.results.items():
            metrics = eval_results.get('metrics', {})
            
            # Try to use F1 or accuracy
            score = metrics.get('f1_weighted', metrics.get('accuracy', 0))
            
            if score > best_score:
                best_score = score
                best_name = name
                # Get the model from context (would need to store it)
                # For now, return name only
                best_model = None
        
        return best_name, best_model