from typing import Dict, Any, Optional, Callable, List
from general.base_classes.model_base import BaseModel
from general.structures.component_config import ComponentConfig
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
from scipy.stats import norm

class BayesianHyperparameterOptimizer(BaseModel):

    def __init__(self, search_space: Dict[str, Any], max_evaluations: int=100, objective_function: Optional[Callable[..., float]]=None, acquisition_function: str='ei', n_initial_points: int=10, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the Bayesian hyperparameter optimizer.

        Args:
            search_space (Dict[str, Any]): Dictionary defining the hyperparameter search space.
                Keys are parameter names, values are tuples/lists representing ranges or choices.
            max_evaluations (int): Maximum number of function evaluations to perform.
            objective_function (Optional[Callable[..., float]]): The function to maximize/minimize.
                Should take hyperparameters as input and return a scalar performance metric.
            acquisition_function (str): Acquisition function to use for selecting next points.
                Options: 'ei' (Expected Improvement), 'pi' (Probability of Improvement), 
                'ucb' (Upper Confidence Bound).
            n_initial_points (int): Number of initial random evaluations before 
                starting Bayesian optimization.
            random_state (Optional[int]): Random seed for reproducible results.
            name (Optional[str]): Name identifier for the optimizer.
        """
        super().__init__(name)
        self.search_space = search_space
        self.max_evaluations = max_evaluations
        self.objective_function = objective_function
        self.acquisition_function = acquisition_function.lower()
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self._history: List[Dict[str, Any]] = []
        self._best_config: Optional[ComponentConfig] = None
        self._best_score: Optional[float] = None
        self._surrogate_model: Optional[GaussianProcessRegressor] = None
        self._param_names: List[str] = list(search_space.keys())
        if self.acquisition_function not in ['ei', 'pi', 'ucb']:
            raise ValueError("Acquisition function must be one of 'ei', 'pi', or 'ucb'")
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _sample_random_config(self) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        config = {}
        for (param_name, param_spec) in self.search_space.items():
            if isinstance(param_spec, dict) and 'type' in param_spec:
                if param_spec['type'] == 'continuous':
                    (low, high) = param_spec['range']
                    config[param_name] = np.random.uniform(low, high)
                elif param_spec['type'] == 'integer':
                    (low, high) = param_spec['range']
                    config[param_name] = np.random.randint(low, high + 1)
                elif param_spec['type'] == 'categorical':
                    config[param_name] = np.random.choice(param_spec['choices'])
                else:
                    raise ValueError(f"Unsupported parameter type: {param_spec['type']}")
            elif isinstance(param_spec, (list, tuple)):
                if len(param_spec) == 2:
                    (low, high) = param_spec
                    if isinstance(low, int) and isinstance(high, int):
                        config[param_name] = np.random.randint(low, high + 1)
                    else:
                        config[param_name] = np.random.uniform(low, high)
                elif len(param_spec) > 2:
                    config[param_name] = np.random.choice(param_spec)
            elif isinstance(param_spec, (int, float)):
                config[param_name] = param_spec
            else:
                raise ValueError(f'Unsupported parameter specification for {param_name}: {param_spec}')
        return config

    def _config_to_array(self, config: Dict[str, Any]) -> np.ndarray:
        """Convert configuration dict to numpy array."""
        values = []
        for param_name in self._param_names:
            param_value = config[param_name]
            param_spec = self.search_space[param_name]
            if isinstance(param_spec, (list, tuple)) and len(param_spec) > 2:
                if param_value in param_spec:
                    param_value = param_spec.index(param_value)
            elif isinstance(param_spec, dict) and 'type' in param_spec and (param_spec['type'] == 'categorical'):
                if param_value in param_spec['choices']:
                    param_value = param_spec['choices'].index(param_value)
            elif isinstance(param_spec, dict) and 'type' in param_spec and (param_spec['type'] == 'integer'):
                param_value = int(param_value)
            elif isinstance(param_spec, (list, tuple)) and len(param_spec) == 2:
                if isinstance(param_spec[0], int) and isinstance(param_spec[1], int):
                    param_value = int(param_value)
            values.append(param_value)
        return np.array(values).reshape(1, -1)

    def _array_to_config(self, array: np.ndarray) -> Dict[str, Any]:
        """Convert numpy array to configuration dict."""
        config = {}
        for (i, param_name) in enumerate(self._param_names):
            param_spec = self.search_space[param_name]
            param_value = array[0, i]
            if isinstance(param_spec, (list, tuple)) and len(param_spec) > 2:
                idx = int(round(param_value))
                idx = max(0, min(idx, len(param_spec) - 1))
                param_value = param_spec[idx]
            elif isinstance(param_spec, dict) and 'type' in param_spec:
                if param_spec['type'] == 'categorical':
                    idx = int(round(param_value))
                    idx = max(0, min(idx, len(param_spec['choices']) - 1))
                    param_value = param_spec['choices'][idx]
                elif param_spec['type'] == 'integer':
                    param_value = int(round(param_value))
            elif isinstance(param_spec, (list, tuple)) and len(param_spec) == 2:
                if isinstance(param_spec[0], int) and isinstance(param_spec[1], int):
                    param_value = int(round(param_value))
            config[param_name] = param_value
        return config

    def _expected_improvement(self, X: np.ndarray, xi: float=0.01) -> np.ndarray:
        """Compute Expected Improvement acquisition function."""
        if self._surrogate_model is None or self._best_score is None:
            return np.zeros((X.shape[0], 1))
        (mu, sigma) = self._surrogate_model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        sigma = np.maximum(sigma, 1e-09)
        imp = mu - self._best_score - xi
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def _probability_of_improvement(self, X: np.ndarray, xi: float=0.01) -> np.ndarray:
        """Compute Probability of Improvement acquisition function."""
        if self._surrogate_model is None or self._best_score is None:
            return np.zeros((X.shape[0], 1))
        (mu, sigma) = self._surrogate_model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        sigma = np.maximum(sigma, 1e-09)
        z = (mu - self._best_score - xi) / sigma
        pi = norm.cdf(z)
        return pi

    def _upper_confidence_bound(self, X: np.ndarray, kappa: float=2.0) -> np.ndarray:
        """Compute Upper Confidence Bound acquisition function."""
        if self._surrogate_model is None:
            return np.zeros((X.shape[0], 1))
        (mu, sigma) = self._surrogate_model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        ucb = mu + kappa * sigma
        return ucb

    def _optimize_acquisition(self, n_candidates: int=1000) -> Dict[str, Any]:
        """Find the next best configuration by optimizing the acquisition function."""
        candidates = []
        for _ in range(n_candidates):
            config = self._sample_random_config()
            candidates.append(self._config_to_array(config))
        X_candidates = np.vstack(candidates)
        if self.acquisition_function == 'ei':
            acq_values = self._expected_improvement(X_candidates)
        elif self.acquisition_function == 'pi':
            acq_values = self._probability_of_improvement(X_candidates)
        elif self.acquisition_function == 'ucb':
            acq_values = self._upper_confidence_bound(X_candidates)
        else:
            raise ValueError(f'Unknown acquisition function: {self.acquisition_function}')
        best_idx = np.argmax(acq_values)
        best_x = X_candidates[best_idx:best_idx + 1, :]
        return self._array_to_config(best_x)

    def fit(self, X: Any, y: Optional[Any]=None, **kwargs) -> 'BayesianHyperparameterOptimizer':
        """
        Perform Bayesian optimization to find optimal hyperparameters.

        This method runs the Bayesian optimization process, iteratively evaluating
        hyperparameter configurations and updating the surrogate model.

        Args:
            X (Any): Input data for model training (if needed by objective function).
            y (Optional[Any]): Target values for model training (if needed).
            **kwargs: Additional parameters for the optimization process.

        Returns:
            BayesianHyperparameterOptimizer: Self instance for method chaining.
        """
        if self.objective_function is None:
            raise ValueError('Objective function must be set before calling fit()')
        kernel = Matern(nu=2.5)
        self._surrogate_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-06, normalize_y=True, n_restarts_optimizer=5, random_state=self.random_state)
        self._history = []
        self._best_config = None
        self._best_score = None
        configs = []
        scores = []
        n_evals = min(self.n_initial_points, self.max_evaluations)
        for i in range(n_evals):
            config = self._sample_random_config()
            try:
                score = self.objective_function(config, X, y, **kwargs)
            except Exception as e:
                warnings.warn(f'Evaluation {i + 1} failed with error: {e}. Using score of -inf.')
                score = -np.inf
            record = {'config': config.copy(), 'score': score, 'iteration': i + 1}
            self._history.append(record)
            if self._best_score is None or score > self._best_score:
                self._best_score = score
                self._best_config = ComponentConfig(**config)
            configs.append(self._config_to_array(config))
            scores.append(score)
        if self.max_evaluations > self.n_initial_points:
            X_train = np.vstack(configs)
            y_train = np.array(scores)
            self._surrogate_model.fit(X_train, y_train)
            for i in range(self.n_initial_points + 1, self.max_evaluations + 1):
                next_config = self._optimize_acquisition()
                try:
                    score = self.objective_function(next_config, X, y, **kwargs)
                except Exception as e:
                    warnings.warn(f'Evaluation {i} failed with error: {e}. Using score of -inf.')
                    score = -np.inf
                record = {'config': next_config.copy(), 'score': score, 'iteration': i}
                self._history.append(record)
                if score > self._best_score:
                    self._best_score = score
                    self._best_config = ComponentConfig(**next_config)
                X_train = np.vstack([X_train, self._config_to_array(next_config)])
                y_train = np.append(y_train, score)
                self._surrogate_model.fit(X_train, y_train)
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Not typically used for optimizers, but included for BaseModel compatibility.

        Args:
            X (Any): Input data.
            **kwargs: Additional parameters.

        Returns:
            Any: This method is not typically used for optimizers.
        """
        return self.get_best_configuration()

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate the performance of the best found hyperparameter configuration.

        Args:
            X (Any): Input data for evaluation.
            y (Any): Target values for evaluation.
            **kwargs: Additional evaluation parameters.

        Returns:
            float: Performance score of the best configuration.
        """
        if self._best_config is None:
            raise ValueError('No optimization has been performed yet.')
        if self.objective_function is None:
            raise ValueError('Objective function must be set before scoring.')
        return self.objective_function(self._best_config.to_dict(), X, y, **kwargs)

    def get_best_configuration(self) -> Optional[ComponentConfig]:
        """
        Retrieve the best hyperparameter configuration found during optimization.

        Returns:
            Optional[ComponentConfig]: The best configuration as a ComponentConfig object,
                or None if no optimization has been performed.
        """
        return self._best_config

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of evaluated hyperparameter configurations and their scores.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing configuration and score history.
        """
        return self._history.copy()

    def set_objective_function(self, func: Callable[..., float]) -> None:
        """
        Set or update the objective function to optimize.

        Args:
            func (Callable[..., float]): The function to maximize/minimize.
                Should take hyperparameters as input and return a scalar performance metric.
        """
        self.objective_function = func