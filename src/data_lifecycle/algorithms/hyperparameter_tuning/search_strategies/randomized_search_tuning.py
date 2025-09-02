from typing import Dict, Any, Optional, Callable, Union
from general.base_classes.model_base import BaseModel
from general.structures.component_config import ComponentConfig
import random
import numpy as np

class RandomizedSearchTuner(BaseModel):
    """
    A hyperparameter tuner that performs randomized search over a defined search space.
    
    This class implements randomized search for hyperparameter optimization by sampling
    configurations from specified distributions and evaluating their performance.
    It's particularly useful when the search space is large and exhaustive search
    is computationally prohibitive.
    
    Attributes:
        search_space (Dict[str, Any]): Dictionary defining the hyperparameter search space
        n_iter (int): Number of parameter settings sampled (default: 10)
        objective_function (Optional[Callable[..., float]]): Function to optimize
        best_config (Optional[ComponentConfig]): Best configuration found
        best_score (Optional[float]): Best score achieved
        random_state (Optional[int]): Random state for reproducibility
    """

    def __init__(self, search_space: Dict[str, Any], n_iter: int=10, objective_function: Optional[Callable[..., float]]=None, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the RandomizedSearchTuner.
        
        Args:
            search_space (Dict[str, Any]): Dictionary where keys are parameter names and values
                define their search space (e.g., distributions or lists of values)
            n_iter (int): Number of parameter settings sampled (default: 10)
            objective_function (Optional[Callable[..., float]]): Function to maximize/minimize
            random_state (Optional[int]): Random state for reproducibility
            name (Optional[str]): Name for the tuner instance
        """
        super().__init__(name)
        self.search_space = search_space
        self.n_iter = n_iter
        self.objective_function = objective_function
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        self.best_config: Optional[ComponentConfig] = None
        self.best_score: Optional[float] = None
        self._best_model = None

    def _sample_params(self) -> Dict[str, Any]:
        """
        Sample a parameter configuration from the search space.
        
        Returns:
            Dict[str, Any]: Sampled parameter configuration
        """
        params = {}
        for (param_name, param_dist) in self.search_space.items():
            if hasattr(param_dist, 'rvs'):
                params[param_name] = param_dist.rvs(random_state=self.random_state)
            elif isinstance(param_dist, list):
                if self.random_state is not None:
                    local_rng = random.Random(self.random_state + hash(param_name) % (2 ** 31 - 1))
                    params[param_name] = local_rng.choice(param_dist)
                else:
                    params[param_name] = random.choice(param_dist)
            else:
                params[param_name] = param_dist
        return params

    def fit(self, X: Any, y: Optional[Any]=None, **kwargs) -> 'RandomizedSearchTuner':
        """
        Perform randomized search for hyperparameter optimization.
        
        Args:
            X (Any): Training data features
            y (Any, optional): Training data targets
            **kwargs: Additional fitting parameters
            
        Returns:
            RandomizedSearchTuner: Self instance for method chaining
        """
        if self.objective_function is None:
            raise ValueError('Objective function must be set before fitting')
        best_score = None
        best_config = None
        best_model = None
        for i in range(self.n_iter):
            if self.random_state is not None:
                random.seed(self.random_state + i)
                np.random.seed(self.random_state + i)
            params = self._sample_params()
            config = ComponentConfig(component_name='random_search_tuner', component_type='hyperparameter_optimization', parameters=params)
            try:
                result = self.objective_function(config, X, y, **kwargs)
                if best_score is None or (hasattr(result, '__getitem__') and result[0] > best_score) or (not hasattr(result, '__getitem__') and result > best_score):
                    if hasattr(result, '__getitem__'):
                        score = result[0]
                        model = result[1] if len(result) > 1 else None
                    else:
                        score = result
                        model = None
                    best_score = score
                    best_config = config
                    best_model = model
            except Exception as e:
                continue
        self.best_config = best_config
        self.best_score = best_score
        self._best_model = best_model
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Make predictions using the best model found during search.
        
        Args:
            X (Any): Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Any: Model predictions
        """
        if self._best_model is None:
            raise ValueError('No best model found. Please run fit() first.')
        if not hasattr(self._best_model, 'predict'):
            raise ValueError("Best model doesn't have a predict method.")
        return self._best_model.predict(X, **kwargs)

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate the best model found during search.
        
        Args:
            X (Any): Test data features
            y (Any): Test data targets
            **kwargs: Additional scoring parameters
            
        Returns:
            float: Model performance score
        """
        if self._best_model is None:
            raise ValueError('No best model found. Please run fit() first.')
        if not hasattr(self._best_model, 'score'):
            raise ValueError("Best model doesn't have a score method.")
        return self._best_model.score(X, y, **kwargs)

    def get_best_configuration(self) -> Optional[ComponentConfig]:
        """
        Retrieve the best hyperparameter configuration found.
        
        Returns:
            Optional[ComponentConfig]: Best configuration or None if search hasn't been performed
        """
        return self.best_config

    def set_objective_function(self, func: Callable[..., float]) -> None:
        """
        Set the objective function to optimize.
        
        Args:
            func (Callable[..., float]): Function to maximize/minimize
        """
        self.objective_function = func