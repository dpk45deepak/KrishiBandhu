"""
Bayesian Optimization tuner implementation.
"""

from typing import Any, Optional, Dict, List, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from .tuner import BaseTuner, TuningResult
from ..common.exceptions import TuningError


class BayesianTuner(BaseTuner):
    """
    Bayesian Optimization tuner using scikit-optimize.
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
        
        # Parse parameter space for scikit-optimize
        self._param_space = []
        self._param_names = []
        self._param_configs = {}
        
        for param_name, param_config in param_space.items():
            param_type = param_config.get('type', 'choice')
            self._param_names.append(param_name)
            self._param_configs[param_name] = param_config
            
            if param_type == 'choice':
                from skopt.space import Categorical
                self._param_space.append(
                    Categorical(param_config['values'], name=param_name)
                )
            elif param_type in ['uniform', 'loguniform']:
                from skopt.space import Real
                low = param_config['low']
                high = param_config['high']
                prior = 'log-uniform' if param_type == 'loguniform' else 'uniform'
                self._param_space.append(
                    Real(low, high, prior=prior, name=param_name)
                )
            elif param_type == 'int':
                from skopt.space import Integer
                self._param_space.append(
                    Integer(
                        param_config['low'],
                        param_config['high'],
                        name=param_name
                    )
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        logger.info(f"Bayesian optimization with {n_trials} trials")
    
    def _params_to_dict(self, params: List[Any]) -> Dict[str, Any]:
        """Convert list of parameters to dictionary."""
        return dict(zip(self._param_names, params))
    
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
        Perform Bayesian optimization.
        
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
            from skopt import gp_minimize
            from skopt.utils import use_named_args
            
            # Convert to numpy arrays
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(y_train, pd.Series):
                y_train = y_train.values
            if X_val is not None and isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if y_val is not None and isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            # Objective function for scikit-optimize
            @use_named_args(self._param_space)
            def objective(**params):
                # Convert params to dict
                param_dict = {k: v for k, v in params.items()}
                
                # Evaluate model
                try:
                    score = self._evaluate_model(
                        param_dict,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        cv
                    )
                    
                    # Store for later
                    self._all_params.append(param_dict)
                    self._all_scores.append(score)
                    
                    # Return negative for minimization
                    return -score if self.direction == 'maximize' else score
                    
                except Exception as e:
                    logger.warning(f"Failed with params {param_dict}: {e}")
                    self._all_params.append(param_dict)
                    self._all_scores.append(np.inf if self.direction == 'maximize' else -np.inf)
                    return np.inf if self.direction == 'maximize' else -np.inf
            
            # Initialize best score
            best_score = -np.inf if self.direction == 'maximize' else np.inf
            best_params = None
            best_model = None
            self._all_scores = []
            self._all_params = []
            
            # Run optimization
            result = gp_minimize(
                objective,
                self._param_space,
                n_calls=self.n_trials,
                random_state=self.random_state,
                verbose=self.verbose
            )
            
            # Get best results
            best_score = -result.fun if self.direction == 'maximize' else result.fun
            best_params = self._params_to_dict(result.x)
            
            # Create best model
            best_model = self.model_class(**best_params)
            if X_val is not None and y_val is not None:
                best_model.fit(X_train, y_train)
            else:
                best_model.fit(X_train, y_train)
            
            # Store results
            self._best_params = best_params
            self._best_score = best_score
            self._best_model = best_model
            
            result_obj = TuningResult(
                best_params=best_params,
                best_score=best_score,
                best_model=best_model,
                all_scores=self._all_scores,
                all_params=self._all_params,
                n_trials=len(self._all_scores),
                study_name="bayesian_optimization",
                additional_info={
                    'iterations': result.x_iters,
                    'function_values': result.func_vals.tolist() if hasattr(result.func_vals, 'tolist') else result.func_vals
                }
            )
            
            logger.info(f"Bayesian optimization completed. Best score: {best_score:.4f}")
            
            return result_obj
            
        except ImportError:
            raise ImportError("scikit-optimize not installed. Install with: pip install scikit-optimize")
        except Exception as e:
            raise TuningError(f"Bayesian optimization failed: {str(e)}") from e