"""
Random Search tuner implementation.
"""

from typing import Any, Optional, Dict, List, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from .tuner import BaseTuner, TuningResult
from ..common.exceptions import TuningError


class RandomSearchTuner(BaseTuner):
    """
    Random Search tuner that randomly samples the parameter space.
    """
    
    def __init__(
        self,
        model_class: Any,
        param_space: Dict[str, Dict[str, Any]],
        n_trials: int = 100,
        metric: str = 'accuracy',
        direction: str = 'maximize',
        random_state: int = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(
            model_class,
            param_space,
            n_trials=n_trials,
            metric=metric,
            direction=direction,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )
        
        self._rng = np.random.RandomState(random_state)
        
        # Parse parameter space
        self._param_distributions = {}
        for param_name, param_config in param_space.items():
            param_type = param_config.get('type', 'choice')
            
            if param_type == 'choice':
                self._param_distributions[param_name] = {
                    'type': 'choice',
                    'values': param_config['values']
                }
            elif param_type == 'uniform':
                self._param_distributions[param_name] = {
                    'type': 'uniform',
                    'low': param_config['low'],
                    'high': param_config['high']
                }
            elif param_type == 'loguniform':
                self._param_distributions[param_name] = {
                    'type': 'loguniform',
                    'low': param_config['low'],
                    'high': param_config['high']
                }
            elif param_type == 'int':
                self._param_distributions[param_name] = {
                    'type': 'int',
                    'low': param_config['low'],
                    'high': param_config['high']
                }
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        logger.info(f"Random search will try {n_trials} combinations")
    
    def _sample_params(self) -> Dict[str, Any]:
        """Sample a random set of parameters."""
        params = {}
        for param_name, distribution in self._param_distributions.items():
            if distribution['type'] == 'choice':
                params[param_name] = self._rng.choice(distribution['values'])
            elif distribution['type'] == 'uniform':
                params[param_name] = self._rng.uniform(
                    distribution['low'],
                    distribution['high']
                )
            elif distribution['type'] == 'loguniform':
                params[param_name] = np.exp(
                    self._rng.uniform(
                        np.log(distribution['low']),
                        np.log(distribution['high'])
                    )
                )
            elif distribution['type'] == 'int':
                params[param_name] = self._rng.randint(
                    distribution['low'],
                    distribution['high'] + 1
                )
        return params
    
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
        Perform random search.
        
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
            
            best_score = -np.inf if self.direction == 'maximize' else np.inf
            best_params = None
            best_model = None
            all_scores = []
            all_params = []
            
            iterator = tqdm(
                range(self.n_trials),
                desc="Random Search",
                disable=not self.verbose
            )
            
            for trial in iterator:
                # Sample parameters
                params = self._sample_params()
                
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
                study_name="random_search"
            )
            
            logger.info(f"Random search completed. Best score: {best_score:.4f}")
            
            return result
            
        except Exception as e:
            raise TuningError(f"Random search failed: {str(e)}") from e