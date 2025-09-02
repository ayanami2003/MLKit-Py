from typing import Dict, Any, Iterator, Optional, Union, List
from general.structures.feature_set import FeatureSet
from general.base_classes.model_base import BaseModel
import numpy as np
from copy import deepcopy
import random


# ...(code omitted)...


class RandomSearchTuner:
    """
    A hyperparameter tuning strategy that randomly samples from a parameter distribution.
    
    This class implements random search for hyperparameter optimization by sampling
    candidate parameter combinations from specified distributions. It supports both
    discrete and continuous parameter spaces and can be used with any model that
    follows the BaseModel interface.
    
    Attributes:
        param_distributions (Dict[str, Any]): Dictionary mapping parameter names to their sampling distributions.
        n_iter (int): Number of parameter settings sampled.
        random_state (Optional[int]): Random seed for reproducibility.
        scoring (Optional[str]): Scoring metric to optimize.
    """

    def __init__(self, param_distributions: Dict[str, Any], n_iter: int=10, random_state: Optional[int]=None, scoring: Optional[str]=None):
        """
        Initialize the RandomSearchTuner.
        
        Args:
            param_distributions: Dictionary where keys are parameter names and values
                are distributions to sample from (e.g., lists, scipy.stats distributions).
            n_iter: Number of parameter settings that are sampled.
            random_state: Controls the randomness of the sampling.
            scoring: Strategy to evaluate the cross-validated scores on the test set.
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.scoring = scoring
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []
        self._best_model = None

    def _sample_params(self) -> Dict[str, Any]:
        """
        Sample a parameter combination from the distributions.
        
        Returns:
            Dict[str, Any]: A dictionary of sampled parameters.
        """
        params = {}
        for (param_name, distribution) in self.param_distributions.items():
            if hasattr(distribution, 'rvs'):
                params[param_name] = distribution.rvs(random_state=self.random_state)
            elif isinstance(distribution, list):
                if self.random_state is not None:
                    local_random = random.Random(self.random_state + hash(param_name) % (2 ** 31 - 1))
                    params[param_name] = local_random.choice(distribution)
                else:
                    params[param_name] = random.choice(distribution)
            else:
                params[param_name] = distribution
        return params

    def fit(self, model: BaseModel, X: Any, y: Optional[Any]=None) -> 'RandomSearchTuner':
        """
        Run fit with all sets of parameters.
        
        Args:
            model: A model object that implements the BaseModel interface.
            X: Training data features.
            y: Training data targets.
            
        Returns:
            RandomSearchTuner: Self instance for method chaining.
        """
        best_score = None
        best_params = None
        best_model = None
        cv_results = []
        for i in range(self.n_iter):
            if self.random_state is not None:
                np.random.seed(self.random_state + i)
                random.seed(self.random_state + i)
            params = self._sample_params()
            model_clone = deepcopy(model)
            for (param_name, param_value) in params.items():
                if hasattr(model_clone, param_name):
                    setattr(model_clone, param_name, param_value)
            model_clone.fit(X, y)
            score = model_clone.score(X, y)
            cv_results.append({'params': params.copy(), 'score': score})
            if best_score is None or score > best_score:
                best_score = score
                best_params = params.copy()
                best_model = deepcopy(model_clone)
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = cv_results
        self._best_model = best_model
        return self

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the best parameters found during the search.
        
        Returns:
            Dict[str, Any]: Dictionary of best parameter values.
        """
        if self.best_params_ is None:
            raise RuntimeError('RandomSearchTuner has not been fitted yet.')
        return self.best_params_.copy()

    def get_best_score(self) -> Optional[float]:
        """
        Get the score of the best parameters found during the search.
        
        Returns:
            float: Best cross-validated score.
        """
        if self.best_score_ is None:
            raise RuntimeError('RandomSearchTuner has not been fitted yet.')
        return self.best_score_

    def predict(self, X: Any) -> Any:
        """
        Make predictions using the best model found during search.
        
        Args:
            X: Input data for prediction.
            
        Returns:
            Any: Model predictions.
        """
        if self._best_model is None:
            raise RuntimeError('RandomSearchTuner has not been fitted yet.')
        return self._best_model.predict(X)

    def score(self, X: Any, y: Any) -> float:
        """
        Evaluate the best model found during search.
        
        Args:
            X: Test data features.
            y: Test data targets.
            
        Returns:
            float: Model performance score.
        """
        if self._best_model is None:
            raise RuntimeError('RandomSearchTuner has not been fitted yet.')
        return self._best_model.score(X, y)

class GridSearchTuner:
    """
    A hyperparameter tuning strategy that exhaustively searches through a parameter grid.
    
    This class implements grid search for hyperparameter optimization by evaluating
    all possible combinations of provided parameter values. It supports early stopping
    to terminate unpromising configurations early, improving efficiency for expensive evaluations.
    
    Attributes:
        param_grid (Dict[str, List[Any]]): Dictionary mapping parameter names to lists of values to try.
        scoring (Optional[str]): Scoring metric to optimize.
        early_stopping (bool): Whether to enable early stopping during evaluation.
        cv_folds (int): Number of cross-validation folds to use.
    """

    def __init__(self, param_grid: Dict[str, List[Any]], scoring: Optional[str]=None, early_stopping: bool=False, cv_folds: int=5):
        """
        Initialize the GridSearchTuner.
        
        Args:
            param_grid: Dictionary where keys are parameter names and values are lists
                of parameter settings to try.
            scoring: Strategy to evaluate the cross-validated scores on the test set.
            early_stopping: Whether to enable early stopping during model evaluation.
            cv_folds: Number of cross-validation folds to use for evaluation.
        """
        self.param_grid = param_grid
        self.scoring = scoring
        self.early_stopping = early_stopping
        self.cv_folds = cv_folds

    def _generate_param_combinations(self) -> Iterator[Dict[str, Any]]:
        """
        Generate all possible combinations of parameters from the parameter grid.
        
        Yields:
            Dict[str, Any]: A dictionary representing one combination of parameters.
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        def _recursive_generator(index: int, current_combo: Dict[str, Any]):
            if index == len(keys):
                yield current_combo.copy()
                return
            key = keys[index]
            for value in values[index]:
                current_combo[key] = value
                yield from _recursive_generator(index + 1, current_combo)
        yield from _recursive_generator(0, {})

    def _evaluate_single_parameter_set(self, model: BaseModel, params: Dict[str, Any], X: FeatureSet, y: Optional[Union[List, Any]]) -> float:
        """
        Evaluate a single parameter set using cross-validation.
        
        Args:
            model: Base model to tune.
            params: Parameters to set on the model.
            X: Training features.
            y: Training targets.
            
        Returns:
            float: Average cross-validation score across folds.
        """
        model_copy = deepcopy(model)
        model_copy.set_params(**params)
        n_samples = len(y) if y is not None else X.shape[0]
        indices = list(range(n_samples))
        random.shuffle(indices)
        fold_size = n_samples // self.cv_folds
        scores = []
        for fold in range(self.cv_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < self.cv_folds - 1 else n_samples
            test_indices = indices[start_idx:end_idx]
            train_indices = indices[:start_idx] + indices[end_idx:]
            if y is not None:
                X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else [X[i] for i in train_indices]
                y_train = [y[i] for i in train_indices]
                X_test = X.iloc[test_indices] if hasattr(X, 'iloc') else [X[i] for i in test_indices]
                y_test = [y[i] for i in test_indices]
            else:
                X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else [X[i] for i in train_indices]
                y_train = None
                X_test = X.iloc[test_indices] if hasattr(X, 'iloc') else [X[i] for i in test_indices]
                y_test = None
            model_copy.fit(X_train, y_train)
            if self.scoring == 'accuracy':
                predictions = model_copy.predict(X_test)
                score = np.mean([pred == true for (pred, true) in zip(predictions, y_test)])
            elif self.scoring == 'mse':
                predictions = model_copy.predict(X_test)
                score = -np.mean([(pred - true) ** 2 for (pred, true) in zip(predictions, y_test)])
            else:
                predictions = model_copy.predict(X_test)
                score = np.mean([pred == true for (pred, true) in zip(predictions, y_test)])
            scores.append(score)
            if self.early_stopping and len(scores) > 1:
                if score < np.mean(scores[:-1]) - 2 * np.std(scores[:-1]):
                    break
        return float(np.mean(scores))

    def fit(self, model: BaseModel, X: FeatureSet, y: Optional[Union[List, Any]]=None) -> 'GridSearchTuner':
        """
        Run fit with all sets of parameters in the grid.
        
        Args:
            model: A model object that implements the BaseModel interface.
            X: Training data features.
            y: Training data targets.
            
        Returns:
            GridSearchTuner: Self instance for method chaining.
        """
        best_score = -np.inf
        best_params = None
        for params in self._generate_param_combinations():
            score = self._evaluate_single_parameter_set(model, params, X, y)
            if score > best_score:
                best_score = score
                best_params = params
        self.best_score_ = best_score
        self.best_params_ = best_params
        return self

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found during the grid search.
        
        Returns:
            Dict[str, Any]: Dictionary of best parameter values.
        """
        if not hasattr(self, 'best_params_'):
            raise ValueError('GridSearchTuner has not been fitted yet.')
        return self.best_params_

    def get_best_score(self) -> float:
        """
        Get the score of the best parameters found during the grid search.
        
        Returns:
            float: Best cross-validated score.
        """
        if not hasattr(self, 'best_score_'):
            raise ValueError('GridSearchTuner has not been fitted yet.')
        return self.best_score_