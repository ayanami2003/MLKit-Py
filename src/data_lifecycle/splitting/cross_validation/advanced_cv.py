from general.structures.data_batch import DataBatch
from general.structures.component_config import ComponentConfig
from typing import Iterator, Tuple, Optional, List, Any, Dict
from general.base_classes.pipeline_base import PipelineStep
from itertools import product
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import check_scoring
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import KFold


# ...(code omitted)...


class NestedCrossValidation(PipelineStep):
    """
    Implements nested cross-validation for unbiased model evaluation with hyperparameter tuning.
    
    This strategy separates model evaluation (outer loop) from hyperparameter optimization (inner loop)
    to provide an unbiased performance estimate. The outer loop evaluates generalization performance
    while the inner loop selects optimal hyperparameters for each training subset.
    
    Attributes
    ----------
    outer_cv : object
        Cross-validation strategy for outer loop (e.g., KFold, StratifiedKFold)
    inner_cv : object
        Cross-validation strategy for inner loop (e.g., KFold, StratifiedKFold)
    param_grid : dict
        Dictionary of parameters to search over
    scoring : str or callable
        Scoring metric for hyperparameter optimization
    n_jobs : int
        Number of parallel jobs for hyperparameter search
    random_state : Optional[int]
        Random state for reproducibility
    """

    def __init__(self, estimator, param_grid, outer_cv=None, inner_cv=None, scoring=None, n_jobs=1, random_state=None):
        """
        Initialize nested cross-validation component.
        
        Parameters
        ----------
        estimator : object
            Base estimator to be tuned and evaluated
        param_grid : dict or list of dicts
            Hyperparameter grid to search over
        outer_cv : object, default=None
            Outer cross-validation strategy. If None, uses KFold with n_splits=5
        inner_cv : object, default=None
            Inner cross-validation strategy. If None, uses KFold with n_splits=3
        scoring : str or callable, default=None
            Scoring metric for hyperparameter tuning
        n_jobs : int, default=1
            Number of jobs to run in parallel for hyperparameter search
        random_state : int, optional
            Random state for reproducibility
            
        Raises
        ------
        ValueError
            If estimator is None or param_grid is empty
        """
        super().__init__(name='NestedCrossValidation')
        if estimator is None:
            raise ValueError('estimator cannot be None')
        if not param_grid:
            raise ValueError('param_grid cannot be empty')
        self.estimator = estimator
        self.param_grid = param_grid
        self.outer_cv = outer_cv or KFold(n_splits=5, shuffle=True, random_state=random_state)
        self.inner_cv = inner_cv or KFold(n_splits=3, shuffle=True, random_state=random_state)
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _inner_loop(self, train_data: DataBatch, test_data: DataBatch) -> Tuple[DataBatch, DataBatch, Dict[str, Any]]:
        """
        Execute inner loop of nested cross-validation for hyperparameter tuning.
        
        Trains multiple models with different hyperparameters on the training data
        and selects the best configuration based on validation performance.
        
        Parameters
        ----------
        train_data : DataBatch
            Training data for hyperparameter tuning
        test_data : DataBatch
            Test data for final evaluation (unused in inner loop)
            
        Returns
        -------
        tuple[DataBatch, DataBatch, dict]
            Tuple containing (train_data, test_data, best_parameters)
        """
        param_combinations = list(ParameterGrid(self.param_grid))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        scores = Parallel(n_jobs=self.n_jobs)((delayed(self._evaluate_params)(clone(self.estimator), params, train_data, scorer) for params in param_combinations))
        best_idx = np.argmax(scores)
        best_params = param_combinations[best_idx]
        return (train_data, test_data, best_params)

    def _evaluate_params(self, estimator, params: Dict[str, Any], data: DataBatch, scorer) -> float:
        """
        Evaluate a specific parameter combination using cross-validation.
        
        Parameters
        ----------
        estimator : object
            Estimator to evaluate
        params : dict
            Hyperparameters to set on the estimator
        data : DataBatch
            Data to evaluate on
        scorer : callable
            Scoring function
            
        Returns
        -------
        float
            Mean cross-validation score for the parameter combination
        """
        estimator.set_params(**params)
        cv_scores = []
        for (train_idx, val_idx) in self.inner_cv.split(data.X):
            X_train = data.X.iloc[train_idx] if hasattr(data.X, 'iloc') else data.X[train_idx]
            y_train = data.y.iloc[train_idx] if hasattr(data.y, 'iloc') else data.y[train_idx]
            X_val = data.X.iloc[val_idx] if hasattr(data.X, 'iloc') else data.X[val_idx]
            y_val = data.y.iloc[val_idx] if hasattr(data.y, 'iloc') else data.y[val_idx]
            estimator.fit(X_train, y_train)
            score = scorer(estimator, X_val, y_val)
            cv_scores.append(score)
        return np.mean(cv_scores)

    def execute(self, data: DataBatch, **kwargs) -> Iterator[Tuple[DataBatch, DataBatch, Dict[str, Any]]]:
        """
        Execute nested cross-validation process.
        
        For each outer fold:
        1. Split data into outer train/test sets
        2. Run inner loop to find best hyperparameters on outer train set
        3. Yield outer train/test batches with best parameters
        
        Parameters
        ----------
        data : DataBatch
            Input data to perform nested cross-validation on
        **kwargs : dict
            Additional parameters (ignored)
            
        Yields
        ------
        tuple[DataBatch, DataBatch, dict]
            Tuple containing (outer_train_batch, outer_test_batch, best_parameters)
            for each outer fold
        """
        for (train_idx, test_idx) in self.outer_cv.split(data.X):
            X_train = data.X.iloc[train_idx] if hasattr(data.X, 'iloc') else data.X[train_idx]
            y_train = data.y.iloc[train_idx] if hasattr(data.y, 'iloc') else data.y[train_idx]
            X_test = data.X.iloc[test_idx] if hasattr(data.X, 'iloc') else data.X[test_idx]
            y_test = data.y.iloc[test_idx] if hasattr(data.y, 'iloc') else data.y[test_idx]
            train_batch = DataBatch(X_train, y_train, name=f'{data.name}_outer_train')
            test_batch = DataBatch(X_test, y_test, name=f'{data.name}_outer_test')
            (_, _, best_params) = self._inner_loop(train_batch, test_batch)
            yield (train_batch, test_batch, best_params)