from typing import Dict, Any, Optional, Callable, List
from general.base_classes.model_base import BaseModel
from general.structures.component_config import ComponentConfig
from itertools import product

class ExhaustiveGridSearchOptimizer(BaseModel):
    """
    Perform exhaustive hyperparameter search over a predefined grid of parameter values.
    
    This optimizer systematically evaluates all possible combinations of hyperparameters
    defined in the search space grid. It is suitable for scenarios where the search space
    is small enough to be exhaustively evaluated, and where computational resources permit
    full enumeration of parameter combinations.
    
    The optimizer requires an objective function that takes hyperparameters as input and
    returns a scalar performance metric to maximize (or minimize).
    
    Attributes:
        search_space (Dict[str, List[Any]]): Dictionary mapping parameter names to lists of possible values.
        max_evaluations (int): Maximum number of evaluations to perform (default: no limit).
        objective_function (Optional[Callable[..., float]]): Function to optimize.
        best_config (Optional[ComponentConfig]): Best configuration found during search.
        best_score (Optional[float]): Best score achieved during search.
    """

    def __init__(self, search_space: Dict[str, List[Any]], max_evaluations: Optional[int]=None, objective_function: Optional[Callable[..., float]]=None, name: Optional[str]=None):
        """
        Initialize the Exhaustive Grid Search Optimizer.
        
        Args:
            search_space (Dict[str, List[Any]]): 
                Dictionary where keys are parameter names and values are lists of possible values.
            max_evaluations (Optional[int]): 
                Maximum number of parameter combinations to evaluate. If None, all combinations are evaluated.
            objective_function (Optional[Callable[..., float]]): 
                Function that takes hyperparameters as keyword arguments and returns a scalar score.
            name (Optional[str]): 
                Name identifier for the optimizer instance.
        """
        super().__init__(name=name)
        self.search_space = search_space
        self.max_evaluations = max_evaluations
        self.objective_function = objective_function
        self.best_config: Optional[ComponentConfig] = None
        self.best_score: Optional[float] = None

    def fit(self, X: Any, y: Optional[Any]=None, **kwargs) -> 'ExhaustiveGridSearchOptimizer':
        """
        Execute the exhaustive grid search to find optimal hyperparameters.
        
        This method iterates through all (or up to max_evaluations) combinations of
        hyperparameters in the search space, evaluates the objective function for
        each combination, and keeps track of the best performing configuration.
        
        Args:
            X (Any): 
                Input data for model training/evaluation. Passed to the objective function.
            y (Optional[Any]): 
                Target values for supervised learning tasks. Passed to the objective function.
            **kwargs (dict): 
                Additional parameters passed to the objective function during evaluation.
                
        Returns:
            ExhaustiveGridSearchOptimizer: 
                Self instance for method chaining.
                
        Raises:
            ValueError: 
                If no objective function has been set before calling fit.
            RuntimeError: 
                If the search space is empty or improperly configured.
        """
        if self.objective_function is None:
            raise ValueError('Objective function must be set before calling fit.')
        if not self.search_space:
            raise RuntimeError('Search space is empty or improperly configured.')
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        all_combinations = list(product(*param_values))
        if self.max_evaluations is not None:
            all_combinations = all_combinations[:self.max_evaluations]
        for combination in all_combinations:
            params = dict(zip(param_names, combination))
            try:
                score = self.objective_function(X, y, **params, **kwargs)
            except Exception as e:
                continue
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_config = ComponentConfig(parameters=params)
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Generate predictions using the best model found during grid search.
        
        This method uses the best hyperparameter configuration identified during
        the search process to make predictions on new data.
        
        Args:
            X (Any): 
                Input data for which predictions are to be made.
            **kwargs (dict): 
                Additional parameters for prediction.
                
        Returns:
            Any: 
                Predictions generated by the best model configuration.
                
        Raises:
            RuntimeError: 
                If called before fitting or if no valid model was found.
        """
        if self.best_config is None:
            raise RuntimeError('No best configuration found. Call fit() first.')
        if self.objective_function is None:
            raise RuntimeError('Objective function not available for prediction.')
        return self.objective_function(X, mode='predict', **self.best_config.parameters, **kwargs)

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate the performance of the best model found during grid search.
        
        Computes the score of the best model configuration on the provided test data.
        
        Args:
            X (Any): 
                Test input data.
            y (Any): 
                True target values for evaluation.
            **kwargs (dict): 
                Additional parameters for scoring.
                
        Returns:
            float: 
                Performance score of the best model on the test data.
                
        Raises:
            RuntimeError: 
                If called before fitting or if no valid model was found.
        """
        if self.best_config is None:
            raise RuntimeError('No best configuration found. Call fit() first.')
        if self.objective_function is None:
            raise RuntimeError('Objective function not available for scoring.')
        return self.objective_function(X, y, mode='score', **self.best_config.parameters, **kwargs)

    def get_best_configuration(self) -> Optional[ComponentConfig]:
        """
        Retrieve the best hyperparameter configuration found during search.
        
        Returns:
            Optional[ComponentConfig]: 
                Configuration with the highest (or lowest) objective function value,
                or None if no search has been performed yet.
        """
        return self.best_config

    def set_objective_function(self, func: Callable[..., float]) -> None:
        """
        Set or update the objective function to be optimized.
        
        Args:
            func (Callable[..., float]): 
                Function that takes hyperparameters as keyword arguments and returns
                a scalar performance metric. Higher values indicate better performance
                unless specified otherwise in the function implementation.
        """
        self.objective_function = func