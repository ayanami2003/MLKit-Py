from general.structures.component_config import ComponentConfig
from general.base_classes.model_base import BaseModel
from typing import Any, Callable, Dict, Optional, Union
import numpy as np

class HyperparameterOptimizer(BaseModel):
    """
    Abstract base class for hyperparameter optimization strategies.
    
    This class defines a standardized interface for implementing various
    hyperparameter tuning algorithms such as grid search, random search,
    Bayesian optimization, and evolutionary algorithms. It extends the
    BaseModel interface to integrate seamlessly with the ML pipeline.
    
    The optimizer works with a search space definition and an objective
    function to find the best hyperparameter configuration for a given
    model or algorithm.
    
    Attributes
    ----------
    search_space : Dict[str, Any]
        Definition of the hyperparameter search space
    max_evaluations : int
        Maximum number of evaluations allowed
    best_config : Optional[ComponentConfig]
        Best configuration found during optimization
    best_score : Optional[float]
        Best score achieved during optimization
    """

    def __init__(self, search_space: Dict[str, Any], max_evaluations: int=100, objective_function: Optional[Callable[..., float]]=None, name: Optional[str]=None):
        """
        Initialize the hyperparameter optimizer.
        
        Parameters
        ----------
        search_space : Dict[str, Any]
            Dictionary defining the hyperparameter search space where keys are
            parameter names and values are parameter specifications
        max_evaluations : int, optional
            Maximum number of evaluations to perform (default is 100)
        objective_function : Callable[..., float], optional
            Function to optimize that takes hyperparameters and returns a score
        name : str, optional
            Name identifier for the optimizer
        """
        super().__init__(name)
        self.search_space = search_space
        self.max_evaluations = max_evaluations
        self._objective_function = objective_function
        self.best_config: Optional[ComponentConfig] = None
        self.best_score: Optional[float] = None

    def fit(self, X: Any, y: Optional[Any]=None, **kwargs) -> 'HyperparameterOptimizer':
        """
        Perform hyperparameter optimization.
        
        This method executes the optimization process to find the best
        hyperparameter configuration according to the specified search
        strategy and objective function.
        
        Parameters
        ----------
        X : Any
            Input data for the optimization process (typically model training data)
        y : Any, optional
            Target values for supervised learning tasks
        **kwargs : dict
            Additional parameters for the optimization process
            
        Returns
        -------
        HyperparameterOptimizer
            Self instance for method chaining
        """
        if self._objective_function is None:
            raise ValueError('Objective function must be set before fitting. Use set_objective_function().')
        if not self.search_space:
            raise ValueError('Search space cannot be empty.')
        best_score = float('-inf')
        best_config = None
        for _ in range(self.max_evaluations):
            config_params = {}
            for (param_name, param_spec) in self.search_space.items():
                if isinstance(param_spec, list):
                    config_params[param_name] = np.random.choice(param_spec)
                elif isinstance(param_spec, dict) and 'range' in param_spec:
                    (low, high) = param_spec['range']
                    if param_spec.get('type', 'float') == 'int':
                        config_params[param_name] = np.random.randint(low, high + 1)
                    else:
                        config_params[param_name] = np.random.uniform(low, high)
                else:
                    config_params[param_name] = param_spec
            config = ComponentConfig(component_name='hyperparameter_optimization', component_type='optimizer', parameters=config_params)
            try:
                score = self._objective_function(config_params, X, y, **kwargs)
                if score > best_score:
                    best_score = score
                    best_config = config
            except Exception as e:
                continue
        self.best_config = best_config
        self.best_score = best_score
        self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Not applicable for hyperparameter optimization.
        
        Hyperparameter optimizers don't make predictions in the traditional sense.
        This method raises a NotImplementedError.
        
        Parameters
        ----------
        X : Any
            Input data
            
        Returns
        -------
        Any
            This method always raises NotImplementedError
        """
        raise NotImplementedError('HyperparameterOptimizer does not support prediction')

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate the best configuration found during optimization.
        
        This method evaluates the performance of the best hyperparameter
        configuration found during the optimization process.
        
        Parameters
        ----------
        X : Any
            Test features
        y : Any
            True target values
        **kwargs : dict
            Additional evaluation parameters
            
        Returns
        -------
        float
            Score of the best configuration
        """
        if not self.is_fitted or self.best_config is None:
            raise ValueError("Optimizer has not been fitted yet. Call 'fit' first.")
        if self._objective_function is None:
            raise ValueError('Objective function must be set to evaluate configurations.')
        return self._objective_function(self.best_config.parameters, X, y, **kwargs)

    def get_best_configuration(self) -> Optional[ComponentConfig]:
        """
        Get the best hyperparameter configuration found during optimization.
        
        Returns
        -------
        Optional[ComponentConfig]
            The best configuration found or None if optimization hasn't been performed
        """
        return self.best_config

    def set_objective_function(self, func: Callable[..., float]) -> None:
        """
        Set the objective function to optimize.
        
        Parameters
        ----------
        func : Callable[..., float]
            Function that takes hyperparameters and returns a score to maximize
        """
        self._objective_function = func