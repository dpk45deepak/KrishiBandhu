"""
Grid Search tuner implementation.
"""

from typing import Any, Optional, Dict, List, Union
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from loguru import logger

from .tuner import BaseTuner, TuningResult
from ..common.exceptions import TuningError


class GridSearchTuner(BaseTuner):
    """
    Grid Search tuner that exhaustively searches the parameter space.
    """
    
    def __init__(
        self,
        model_class: Any,
        param_space: Dict[str, List[Any]],
        n_trials: int = None,  # Not used for grid search
        metric: str = 'accuracy',
        direction: str = 'maximize',
        random_state: int = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(
            model_class,
            param_space,
            n_trials=1,  # Will be set from grid size
            metric=metric,
            direction=direction,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )
        
        # Calculate total combinations
        self._total_combinations = 1
        for values in param_space.values():
            self._total_combinations *= len(values)
        
        self.n_trials = self._total_combinations
        logger.info(f"Grid search will try {self._total_combinations} combinations")
    
    def tune(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        cv: int = 5,
        **kwargs
    ) -> TuningResult:
        """
        Perform grid search.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            cv: Number of CV folds
            
        Returns:
            TuningResult object
        """
        try:
            # Convert to numpy arrays
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(y_train, pd.Series):
                y_train = y_train.values
            if X_val is not None and isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if y_val is not None and isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            # Generate all parameter combinations
            param_names = list(self.param_space.keys())
            param_values = list(self.param_space.values())
            
            # Use tqdm for progress
            iterator = tqdm(
                product(*param_values),
                total=self._total_combinations,
                desc="Grid Search",
                disable=not self.verbose
            )
            
            best_score = -np.inf if self.direction == 'maximize' else np.inf
            best_params = None
            best_model = None
            all_scores = []
            all_params = []
            
            for values in iterator:
                params = dict(zip(param_names, values))
                
                # Evaluate model
                try:
                    score = self._evaluate_model(
                        params,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        cv
                    )
                    
                    all_scores.append(score)
                    all_params.append(params)
                    
                    # Update best
                    is_better = (self.direction == 'maximize' and score > best_score) or \
                               (self.direction == 'minimize' and score < best_score)
                    
                    if is_better:
                        best_score = score
                        best_params = params
                        
                        # Create best model
                        best_model = self.model_class(**params)
                        if X_val is not None and y_val is not None:
                            best_model.fit(X_train, y_train)
                        else:
                            # Fit on all data for best model
                            best_model.fit(X_train, y_train)
                        
                        if self.verbose:
                            logger.info(f"New best score: {best_score:.4f} with params: {params}")
                
                except Exception as e:
                    logger.warning(f"Failed with params {params}: {e}")
                    all_scores.append(-np.inf if self.direction == 'maximize' else np.inf)
                    all_params.append(params)
            
            # Store results
            self._best_params = best_params
            self._best_score = best_score
            self._best_model = best_model
            self._all_scores = all_scores
            self._all_params = all_params
            
            result = TuningResult(
                best_params=best_params,
                best_score=best_score,
                best_model=best_model,
                all_scores=all_scores,
                all_params=all_params,
                n_trials=len(all_scores),
                study_name="grid_search"
            )
            
            logger.info(f"Grid search completed. Best score: {best_score:.4f}")
            
            return result
            
        except Exception as e:
            raise TuningError(f"Grid search failed: {str(e)}") from e