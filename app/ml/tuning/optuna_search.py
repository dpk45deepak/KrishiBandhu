"""
Optuna-based hyperparameter tuner.
"""

from typing import Any, Optional, Dict, List, Union, Callable
import numpy as np
import pandas as pd
from loguru import logger

from .tuner import BaseTuner, TuningResult
from ..common.exceptions import TuningError


class OptunaTuner(BaseTuner):
    """
    Optuna-based tuner for advanced hyperparameter optimization.
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
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
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
        
        self.study_name = study_name or f"optuna_study_{np.random.randint(10000)}"
        self.storage = storage
        self._study = None
        
        # Parameter sampling functions
        self._param_samplers = {}
        for param_name, param_config in param_space.items():
            param_type = param_config.get('type', 'choice')
            
            if param_type == 'choice':
                self._param_samplers[param_name] = {
                    'type': 'choice',
                    'values': param_config['values']
                }
            elif param_type == 'uniform':
                self._param_samplers[param_name] = {
                    'type': 'uniform',
                    'low': param_config['low'],
                    'high': param_config['high']
                }
            elif param_type == 'loguniform':
                self._param_samplers[param_name] = {
                    'type': 'loguniform',
                    'low': param_config['low'],
                    'high': param_config['high']
                }
            elif param_type == 'int':
                self._param_samplers[param_name] = {
                    'type': 'int',
                    'low': param_config['low'],
                    'high': param_config['high']
                }
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        logger.info(f"Optuna study '{self.study_name}' with {n_trials} trials")
    
    def _objective(
        self,
        trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        cv: int
    ) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            cv: Number of CV folds
            
        Returns:
            Score to optimize
        """
        # Sample parameters
        params = {}
        for param_name, sampler in self._param_samplers.items():
            if sampler['type'] == 'choice':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    sampler['values']
                )
            elif sampler['type'] == 'uniform':
                params[param_name] = trial.suggest_float(
                    param_name,
                    sampler['low'],
                    sampler['high']
                )
            elif sampler['type'] == 'loguniform':
                params[param_name] = trial.suggest_float(
                    param_name,
                    sampler['low'],
                    sampler['high'],
                    log=True
                )
            elif sampler['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    sampler['low'],
                    sampler['high']
                )
        
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
            
            # Store params for later
            trial.set_user_attr('params', params)
            
            return score
            
        except Exception as e:
            logger.warning(f"Trial failed with params {params}: {e}")
            return -np.inf if self.direction == 'maximize' else np.inf
    
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
        Perform Optuna optimization.
        
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
            import optuna
            
            # Convert to numpy arrays
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(y_train, pd.Series):
                y_train = y_train.values
            if X_val is not None and isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if y_val is not None and isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            # Create study
            if self.storage:
                self._study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    direction=self.direction,
                    load_if_exists=True
                )
            else:
                self._study = optuna.create_study(
                    study_name=self.study_name,
                    direction=self.direction
                )
            
            # Optimize
            self._study.optimize(
                lambda trial: self._objective(
                    trial,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    cv
                ),
                n_trials=self.n_trials,
                show_progress_bar=self.verbose
            )
            
            # Get best results
            best_trial = self._study.best_trial
            best_params = best_trial.params
            best_score = best_trial.value
            
            # Create best model
            best_model = self.model_class(**best_params)
            if X_val is not None and y_val is not None:
                best_model.fit(X_train, y_train)
            else:
                best_model.fit(X_train, y_train)
            
            # Get all results
            all_scores = []
            all_params = []
            for trial in self._study.trials:
                if trial.value is not None:
                    all_scores.append(trial.value)
                    params = trial.params
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
                study_name=self.study_name,
                additional_info={
                    'best_trial_number': best_trial.number,
                    'study_statistics': self._study.best_trial.__dict__
                }
            )
            
            logger.info(f"Optuna completed. Best score: {best_score:.4f}")
            
            return result
            
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")
        except Exception as e:
            raise TuningError(f"Optuna optimization failed: {str(e)}") from e
    
    def get_importance(self) -> Dict[str, float]:
        """Get parameter importance from Optuna study."""
        if self._study is None:
            raise TuningError("Study not performed yet")
        
        try:
            import optuna
            importance = optuna.importance.get_param_importances(self._study)
            return {k: v for k, v in importance.items()}
        except:
            return {}
    
    def plot_optimization_history(self, save_path: Optional[Path] = None) -> None:
        """Plot optimization history."""
        if self._study is None:
            raise TuningError("Study not performed yet")
        
        try:
            import optuna
            
            fig = optuna.visualization.plot_optimization_history(self._study)
            if save_path:
                fig.write_html(str(save_path))
                logger.info(f"Optimization history saved to {save_path}")
            return fig
            
        except ImportError:
            raise ImportError("Plotly not installed. Install with: pip install plotly")