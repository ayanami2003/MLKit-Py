from typing import Dict, Any, Optional, Callable, List
from general.structures.component_config import ComponentConfig
from general.base_classes.model_base import BaseModel
import itertools
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from src.data_lifecycle.splitting.cross_validation.stratified_kfold import StratifiedKFold

class GridSearchTuner(BaseModel):
    """
    Performs exhaustive search over a specified grid of hyperparameters to find the best configuration
    for a given model and dataset.

    This class implements grid search, a hyperparameter tuning strategy that evaluates all possible
    combinations of provided parameter values. It systematically works through multiple combinations
    of parameter values to determine which combination yields the best model performance according to
    a specified scoring function.

    Attributes:
        param_grid (Dict[str, List[Any]]): Dictionary where keys are parameter names and values are
            lists of parameter settings to try.
        scoring_func (Callable): Function that evaluates model performance. Should take (model, X, y)
            and return a float score.
        cv_folds (int): Number of cross-validation folds to use for evaluation.
        n_jobs (int): Number of parallel jobs to run. -1 means using all processors.
        best_params_ (Optional[Dict[str, Any]]): Best parameter combination found during search.
        best_score_ (Optional[float]): Best score achieved during search.

    Example:
        >>> tuner = GridSearchTuner(
        ...     param_grid={'alpha': [0.1, 1.0, 10.0]},
        ...     scoring_func=lambda model, X, y: model.score(X, y),
        ...     cv_folds=5
        ... )
        >>> tuner.fit(X_train, y_train)
        >>> print(tuner.best_params_)
    """

    def __init__(self, param_grid: Dict[str, List[Any]], scoring_func: Callable[[Any, Any, Any], float], cv_folds: int=5, n_jobs: int=1, name: Optional[str]=None):
        """
        Initialize the GridSearchTuner.

        Args:
            param_grid: Dictionary with parameters names as keys and lists of parameter settings
                to try as values.
            scoring_func: Function that evaluates model performance. Takes (model, X, y) and
                returns a float score.
            cv_folds: Number of cross-validation folds for evaluation.
            n_jobs: Number of parallel jobs to run. -1 means using all processors.
            name: Optional name for the tuner instance.
        """
        super().__init__(name)
        self.param_grid = param_grid
        self.scoring_func = scoring_func
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self._best_model: Optional[BaseModel] = None
        self._validate_param_grid()

    def _validate_param_grid(self) -> None:
        """Validate the parameter grid."""
        if not isinstance(self.param_grid, dict):
            raise TypeError('param_grid must be a dictionary')
        for (key, value) in self.param_grid.items():
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"Parameter values for '{key}' must be a list or tuple")

    def _evaluate_params(self, params: Dict[str, Any], X: Any, y: Any) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate a single parameter combination using cross-validation.

        Args:
            params: Parameter combination to evaluate.
            X: Training data features.
            y: Training data targets.

        Returns:
            Tuple of (params, mean_score) for the parameter combination.
        """
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds)
            scores = []
            for (train_idx, val_idx) in cv.split(X, y):
                if hasattr(X, 'iloc'):
                    (X_train_fold, X_val_fold) = (X.iloc[train_idx], X.iloc[val_idx])
                    (y_train_fold, y_val_fold) = (y.iloc[train_idx], y.iloc[val_idx])
                else:
                    (X_train_fold, X_val_fold) = (X[train_idx], X[val_idx])
                    (y_train_fold, y_val_fold) = (y[train_idx], y[val_idx])
                model = self.build_model(**params)
                model.fit(X_train_fold, y_train_fold)
                score = self.scoring_func(model, X_val_fold, y_val_fold)
                scores.append(score)
            mean_score = float(np.mean(scores))
            return (params, mean_score)
        except Exception as e:
            logging.warning(f'Failed to evaluate parameters {params}: {e}')
            return (params, -np.inf)

    def fit(self, X: Any, y: Any=None, **kwargs) -> 'GridSearchTuner':
        """
        Perform grid search to find the best hyperparameter combination.

        Args:
            X: Training data features.
            y: Training data targets.
            **kwargs: Additional parameters for fitting.

        Returns:
            GridSearchTuner: Self instance for method chaining.
        """
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = [dict(zip(param_names, v)) for v in itertools.product(*param_values)]
        best_score = -np.inf
        best_params = None
        if self.n_jobs == 1:
            for params in param_combinations:
                (_, score) = self._evaluate_params(params, X, y)
                if score > best_score:
                    best_score = score
                    best_params = params
        else:
            n_workers = self.n_jobs if self.n_jobs > 0 else None
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(self._evaluate_params, params, X, y) for params in param_combinations]
                for future in as_completed(futures):
                    (params, score) = future.result()
                    if score > best_score:
                        best_score = score
                        best_params = params
        self.best_params_ = best_params
        self.best_score_ = best_score if best_score != -np.inf else None
        if self.best_params_ is not None:
            self._best_model = self.build_model(**self.best_params_)
            self._best_model.fit(X, y)
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Make predictions using the best model found during grid search.

        Args:
            X: Data to make predictions on.
            **kwargs: Additional parameters for prediction.

        Returns:
            Predictions from the best model.
        """
        if self._best_model is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        return self._best_model.predict(X, **kwargs)

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate the best model found during grid search on test data.

        Args:
            X: Test data features.
            y: Test data targets.
            **kwargs: Additional parameters for scoring.

        Returns:
            float: Score of the best model on test data.
        """
        if self._best_model is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        return self.scoring_func(self._best_model, X, y)

    def get_best_configuration(self) -> Optional[ComponentConfig]:
        """
        Retrieve the best hyperparameter configuration found during search.

        Returns:
            ComponentConfig: Configuration object with best parameters, or None if search hasn't been performed.
        """
        if self.best_params_ is None:
            return None
        return ComponentConfig(parameters=self.best_params_.copy())