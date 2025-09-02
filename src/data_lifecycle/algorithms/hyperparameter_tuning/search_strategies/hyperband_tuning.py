from typing import Dict, Any, Optional, Callable
from general.base_classes.model_base import BaseModel
from general.structures.component_config import ComponentConfig
from typing import Dict, Any, Optional, Callable, List, Tuple
import math
import random
import numpy as np

class HyperbandTuningStrategy(BaseModel):

    def __init__(self, search_space: Dict[str, Any], max_resource: int=81, reduction_factor: float=3.0, brackets: int=4, objective_function: Optional[Callable[..., float]]=None, name: Optional[str]=None):
        """
        Initialize the Hyperband tuning strategy.
        
        Args:
            search_space (Dict[str, Any]): Dictionary defining the hyperparameter search space.
                Keys are parameter names, values define their domains.
            max_resource (int): Maximum resource level (e.g., epochs, iterations) to allocate.
                Defaults to 81.
            reduction_factor (float): Factor by which to reduce configurations in each round.
                Defaults to 3.0.
            brackets (int): Number of brackets to run in parallel. Defaults to 4.
            objective_function (Optional[Callable[..., float]]): Function to optimize.
                Should take hyperparameters and return a scalar score.
            name (Optional[str]): Name identifier for this tuner.
        """
        super().__init__(name)
        self.search_space = search_space
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.brackets = brackets
        self.objective_function = objective_function
        self._best_config: Optional[ComponentConfig] = None
        self._best_score: Optional[float] = None

    def fit(self, X: Any, y: Optional[Any]=None, **kwargs) -> 'HyperbandTuningStrategy':
        """
        Execute the hyperband search strategy to find optimal hyperparameters.
        
        This method runs the full hyperband algorithm, performing successive halving
        across multiple brackets to efficiently identify promising configurations.
        
        Args:
            X (Any): Training data features
            y (Optional[Any]): Training data targets
            **kwargs: Additional parameters for fitting
            
        Returns:
            HyperbandTuningStrategy: Self instance for method chaining
        """
        if self.objective_function is None:
            raise ValueError('Objective function must be set before fitting. Use set_objective_function().')
        if not self.search_space:
            raise ValueError('Search space cannot be empty.')
        max_brackets = int(math.log(self.max_resource) / math.log(self.reduction_factor)) + 1
        actual_brackets = min(self.brackets, max_brackets)
        best_score = float('-inf')
        best_config = None
        for bracket in range(actual_brackets):
            (n_configs, min_resource) = self._get_bracket_resources(bracket, max_brackets)
            if n_configs <= 0:
                continue
            configs = [self._sample_configuration() for _ in range(n_configs)]
            resource = min_resource
            while len(configs) >= 1 and resource <= self.max_resource:
                evaluated_configs = []
                for config in configs:
                    try:
                        score = self._evaluate_configuration(config, resource, X, y, **kwargs)
                        evaluated_configs.append((config, score))
                    except Exception:
                        continue
                if not evaluated_configs:
                    break
                evaluated_configs.sort(key=lambda x: x[1], reverse=True)
                if evaluated_configs and evaluated_configs[0][1] > best_score:
                    best_score = evaluated_configs[0][1]
                    best_config = evaluated_configs[0][0]
                if len(evaluated_configs) == 1:
                    break
                keep_count = max(1, int(len(evaluated_configs) / self.reduction_factor))
                configs = [config for (config, _) in evaluated_configs[:keep_count]]
                resource = int(resource * self.reduction_factor)
        if best_config is not None:
            try:
                final_score = self._evaluate_configuration(best_config, self.max_resource, X, y, **kwargs)
                if final_score > best_score:
                    best_score = final_score
            except Exception:
                pass
            self._best_config = ComponentConfig(component_name='hyperband_tuning', component_type='hyperparameter_optimizer', parameters=best_config)
            self._best_score = best_score
            self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Make predictions using the best found configuration.
        
        Args:
            X (Any): Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Any: Predictions from model with best hyperparameters
        """
        if not self.is_fitted or self._best_config is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        if self.objective_function is None:
            raise ValueError('Objective function must be set to make predictions.')
        raise NotImplementedError('HyperbandTuningStrategy does not directly support prediction. Use the best configuration to train a model for predictions.')

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate the performance of the best configuration.
        
        Args:
            X (Any): Test data features
            y (Any): Test data targets
            **kwargs: Additional evaluation parameters
            
        Returns:
            float: Performance score of the best configuration
        """
        if not self.is_fitted or self._best_config is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        if self.objective_function is None:
            raise ValueError('Objective function must be set to evaluate configurations.')
        return self._evaluate_configuration(self._best_config.parameters, self.max_resource, X, y, **kwargs)

    def get_best_configuration(self) -> Optional[ComponentConfig]:
        """
        Retrieve the best hyperparameter configuration found during search.
        
        Returns:
            Optional[ComponentConfig]: Configuration with highest score, or None if not fitted
        """
        return self._best_config

    def set_objective_function(self, func: Callable[..., float]) -> None:
        """
        Set the objective function to optimize.
        
        Args:
            func (Callable[..., float]): Function that takes hyperparameters and returns score
        """
        self.objective_function = func

    def _get_bracket_resources(self, bracket: int, max_brackets: int) -> Tuple[int, int]:
        """
        Calculate the number of configurations and minimum resource for a bracket.
        
        Args:
            bracket (int): Bracket index
            max_brackets (int): Total number of brackets
            
        Returns:
            Tuple[int, int]: Number of configurations and minimum resource level
        """
        s = max_brackets - 1 - bracket
        min_resource = max(1, int(self.max_resource / self.reduction_factor ** s))
        n_configs = int(np.ceil(self.reduction_factor ** s * (max_brackets / (s + 1))))
        return (n_configs, min_resource)

    def _sample_configuration(self) -> Dict[str, Any]:
        """
        Sample a random configuration from the search space.
        
        Returns:
            Dict[str, Any]: Sampled configuration
        """
        config = {}
        for (param_name, param_spec) in self.search_space.items():
            if isinstance(param_spec, list):
                config[param_name] = random.choice(param_spec)
            elif isinstance(param_spec, dict):
                if 'range' in param_spec:
                    (low, high) = param_spec['range']
                    if param_spec.get('type', 'float') == 'int':
                        config[param_name] = random.randint(low, high)
                    else:
                        config[param_name] = random.uniform(low, high)
                elif 'choices' in param_spec:
                    config[param_name] = random.choice(param_spec['choices'])
                else:
                    config[param_name] = param_spec
            else:
                config[param_name] = param_spec
        return config

    def _evaluate_configuration(self, config: Dict[str, Any], resource: int, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate a configuration with a specific resource allocation.
        
        Args:
            config (Dict[str, Any]): Configuration to evaluate
            resource (int): Amount of resource to allocate
            X (Any): Training data features
            y (Any): Training data targets
            **kwargs: Additional parameters
            
        Returns:
            float: Evaluation score
        """
        if self.objective_function is None:
            raise ValueError('Objective function must be set before evaluation.')
        return self.objective_function(config, X, y, resource=resource, **kwargs)