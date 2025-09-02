from typing import Dict, Any, Optional, Callable, List
from general.base_classes.model_base import BaseModel
from general.structures.component_config import ComponentConfig
import numpy as np

class SuccessiveHalvingTuner(BaseModel):

    def __init__(self, search_space: Dict[str, Any], max_budget: float, eta: int=3, min_budget: float=1.0, n_initial_configs: Optional[int]=None, objective_function: Optional[Callable[..., float]]=None, name: Optional[str]=None):
        """
        Initialize the Successive Halving tuner.
        
        Args:
            search_space: Dictionary defining the hyperparameter search space where keys
                are parameter names and values are parameter specifications.
            max_budget: Maximum resource budget (e.g., iterations, samples) for final
                evaluation of configurations.
            eta: Downsampling rate; proportion of configurations to keep after each
                round (commonly 3 or 4). Higher values lead to more aggressive pruning.
            min_budget: Minimum resource budget for initial evaluations.
            n_initial_configs: Number of initial configurations to start with. If None,
                automatically calculated based on eta and budget ratios.
            objective_function: Function to optimize that takes hyperparameters and
                returns a performance score (higher is better).
            name: Optional name for the tuner instance.
        """
        super().__init__(name)
        self.search_space = search_space
        self.max_budget = max_budget
        self.eta = eta
        self.min_budget = min_budget
        self.n_initial_configs = n_initial_configs
        self.objective_function = objective_function
        self._best_config: Optional[ComponentConfig] = None
        self._performance_history: List[Dict[str, Any]] = []
        self._configs: List[ComponentConfig] = []

    def _sample_configurations(self, n_configs: int) -> List[ComponentConfig]:
        """
        Sample random configurations from the search space.
        
        Args:
            n_configs: Number of configurations to sample.
            
        Returns:
            List of sampled configurations.
        """
        configs = []
        for _ in range(n_configs):
            params = {}
            for (param_name, param_spec) in self.search_space.items():
                if isinstance(param_spec, list):
                    params[param_name] = np.random.choice(param_spec)
                elif isinstance(param_spec, dict) and 'range' in param_spec:
                    (low, high) = param_spec['range']
                    if param_spec.get('type') == 'int':
                        params[param_name] = np.random.randint(low, high + 1)
                    else:
                        params[param_name] = np.random.uniform(low, high)
                elif isinstance(param_spec, dict) and 'choices' in param_spec:
                    params[param_name] = np.random.choice(param_spec['choices'])
                else:
                    params[param_name] = np.random.choice(param_spec) if isinstance(param_spec, (list, tuple)) else param_spec
            config = ComponentConfig(component_name='tuned_model', component_type='hyperparameter_configuration', parameters=params)
            configs.append(config)
        return configs

    def _calculate_n_rounds(self) -> int:
        """Calculate the number of rounds based on budget ratio and eta."""
        budget_ratio = self.max_budget / self.min_budget
        return int(np.ceil(np.log(budget_ratio) / np.log(self.eta)))

    def fit(self, X: Any, y: Optional[Any]=None, **kwargs) -> 'SuccessiveHalvingTuner':
        """
        Execute the successive halving search process.
        
        This method runs the full successive halving procedure, iteratively evaluating
        configurations with increasing budgets and pruning poor performers.
        
        Args:
            X: Input features or training data.
            y: Target values (optional depending on the objective function).
            **kwargs: Additional parameters for fitting.
            
        Returns:
            SuccessiveHalvingTuner: Self instance for method chaining.
        """
        if self.objective_function is None:
            raise ValueError('Objective function must be provided before fitting.')
        if self.n_initial_configs is None:
            n_rounds = self._calculate_n_rounds()
            self.n_initial_configs = int(self.eta ** n_rounds)
        self._configs = self._sample_configurations(self.n_initial_configs)
        current_configs = self._configs.copy()
        round_num = 0
        n_rounds = self._calculate_n_rounds()
        budgets = [self.min_budget * self.eta ** i for i in range(n_rounds)]
        if budgets:
            budgets[-1] = self.max_budget
        while len(current_configs) > 1 and round_num < len(budgets):
            current_budget = budgets[round_num]
            performances = []
            for config in current_configs:
                try:
                    score = self.objective_function(X, y, budget=current_budget, **config.parameters, **kwargs)
                except Exception:
                    score = -np.inf
                performance_record = {'config': config, 'parameters': config.parameters, 'budget': current_budget, 'score': score, 'round': round_num}
                self._performance_history.append(performance_record)
                performances.append(score)
                if self._best_config is None or score > self._best_config.metadata.get('score', -np.inf):
                    self._best_config = config
                    self._best_config.metadata['score'] = score
            n_keep = max(1, int(len(current_configs) // self.eta))
            sorted_indices = np.argsort(performances)[::-1]
            current_configs = [current_configs[i] for i in sorted_indices[:n_keep]]
            round_num += 1
        if current_configs:
            final_budget = self.max_budget
            for config in current_configs:
                try:
                    score = self.objective_function(X, y, budget=final_budget, **config.parameters, **kwargs)
                except Exception:
                    score = -np.inf
                performance_record = {'config': config, 'parameters': config.parameters, 'budget': final_budget, 'score': score, 'round': round_num}
                self._performance_history.append(performance_record)
                if self._best_config is None or score > self._best_config.metadata.get('score', -np.inf):
                    self._best_config = config
                    self._best_config.metadata['score'] = score
        self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Make predictions using the best found configuration.
        
        This method uses the best hyperparameter configuration discovered during
        fitting to make predictions on new data.
        
        Args:
            X: Input features for prediction.
            **kwargs: Additional prediction parameters.
            
        Returns:
            Predictions made by the model with the best hyperparameters.
        """
        if not self.is_fitted:
            raise RuntimeError('Tuner must be fitted before making predictions.')
        if self._best_config is None:
            raise RuntimeError('No valid configuration found during fitting.')
        return self.objective_function(X, mode='predict', **self._best_config.parameters, **kwargs)

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate the performance of the best configuration.
        
        Computes the performance score of the best hyperparameter configuration
        found during the search process.
        
        Args:
            X: Test features.
            y: True target values.
            **kwargs: Additional scoring parameters.
            
        Returns:
            float: Performance score of the best configuration.
        """
        if not self.is_fitted:
            raise RuntimeError('Tuner must be fitted before scoring.')
        if self._best_config is None:
            raise RuntimeError('No valid configuration found during fitting.')
        return self.objective_function(X, y, mode='score', **self._best_config.parameters, **kwargs)

    def get_best_configuration(self) -> Optional[ComponentConfig]:
        """
        Retrieve the best hyperparameter configuration found.
        
        Returns:
            ComponentConfig: Configuration with the highest performance score,
                or None if no search has been completed.
        """
        return self._best_config

    def get_performance_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all evaluated configurations.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing configuration
                parameters, budgets, and performance scores.
        """
        return self._performance_history.copy()