import numpy as np
import random
from typing import Dict, Any, Callable, List, Tuple, Optional, Union
from scipy.stats import uniform, randint, norm
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from general.base_classes.model_base import BaseModel
from general.structures.component_config import ComponentConfig

class TreeStructuredParzenEstimatorOptimizer(BaseModel):

    def __init__(self, search_space: Dict[str, Any], max_evaluations: int=100, gamma: float=0.25, n_startup_jobs: int=20, n_EI_candidates: int=24, objective_function: Optional[Callable[..., float]]=None, name: Optional[str]=None):
        """
        Initialize the Tree-Structured Parzen Estimator optimizer.
        
        Args:
            search_space (Dict[str, Any]): Hyperparameter search space definition.
                Keys are parameter names, values are dictionaries defining:
                - 'type': 'continuous', 'integer', or 'categorical'
                - 'bounds': (min, max) for continuous/integer, list of choices for categorical
                - 'log': Boolean indicating log-scale sampling for continuous parameters
            max_evaluations (int): Maximum number of function evaluations
            gamma (float): Proportion of observations used for building l(x) (good) model
            n_startup_jobs (int): Number of random evaluations before using TPE
            n_EI_candidates (int): Number of candidates for expected improvement calculation
            objective_function (Optional[Callable[..., float]]): Function to maximize/minimize
            name (Optional[str]): Name identifier for the optimizer
        """
        super().__init__(name)
        self.search_space = search_space
        self.max_evaluations = max_evaluations
        self.gamma = gamma
        self.n_startup_jobs = n_startup_jobs
        self.n_EI_candidates = n_EI_candidates
        self.objective_function = objective_function
        self.best_configuration: Optional[ComponentConfig] = None
        self._evaluations: List[Tuple[Dict[str, Any], float]] = []
        self._best_score = float('-inf')

    def _sample_random_configuration(self) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        config = {}
        for (param_name, param_def) in self.search_space.items():
            param_type = param_def['type']
            if param_type == 'categorical':
                config[param_name] = random.choice(param_def['bounds'])
            elif param_type in ['continuous', 'integer']:
                (low, high) = param_def['bounds']
                if param_def.get('log', False) and param_type == 'continuous':
                    (log_low, log_high) = (np.log(low), np.log(high))
                    value = np.exp(np.random.uniform(log_low, log_high))
                    config[param_name] = value
                elif param_type == 'continuous':
                    config[param_name] = np.random.uniform(low, high)
                else:
                    config[param_name] = np.random.randint(low, high + 1)
            else:
                raise ValueError(f'Unknown parameter type: {param_type}')
        return config

    def _build_parzen_estimator(self, configurations: List[Dict[str, Any]], param_name: str, param_def: Dict[str, Any]) -> Callable:
        """
        Build a Parzen estimator for a specific parameter based on observed configurations.
        
        Args:
            configurations: List of configuration dictionaries
            param_name: Name of the parameter
            param_def: Definition of the parameter
            
        Returns:
            A function that computes the probability density/mass at a given value
        """
        param_type = param_def['type']
        values = [config[param_name] for config in configurations]
        if param_type == 'categorical':
            (unique_vals, counts) = np.unique(values, return_counts=True)
            probs = counts / len(values)

            def pdf(x):
                if x in unique_vals:
                    return probs[np.where(unique_vals == x)[0][0]]
                return 1e-12
            return pdf
        elif param_type in ['continuous', 'integer']:
            if len(values) == 1:
                mean_val = values[0]
                std_val = 0.1

                def pdf(x):
                    return norm.pdf(x, loc=mean_val, scale=std_val)
                return pdf
            else:
                std_val = np.std(values)
                n = len(values)
                if std_val > 0:
                    bandwidth = 1.06 * std_val * n ** (-1 / 5)
                else:
                    bandwidth = 0.1
                bandwidth = max(bandwidth, 1e-06)

                def pdf(x):
                    densities = [norm.pdf(x, loc=val, scale=bandwidth) for val in values]
                    return np.mean(densities)
                return pdf
        else:
            raise ValueError(f'Unsupported parameter type: {param_type}')

    def _compute_expected_improvement(self, candidate_configs: List[Dict[str, Any]], l_models: Dict[str, Callable], g_models: Dict[str, Callable]) -> List[float]:
        """
        Compute expected improvement for a set of candidate configurations.
        
        EI(x) = E[max(l(x)/g(x) - psi, 0)] where psi is a threshold
        
        For practical computation, we approximate this as l(x)/g(x) with stabilization.
        """
        ei_values = []
        for config in candidate_configs:
            l_prob = 1.0
            g_prob = 1.0
            for (param_name, param_def) in self.search_space.items():
                if param_name not in config:
                    l_prob *= 1e-12
                    g_prob *= 1e-12
                    continue
                param_value = config[param_name]
                try:
                    l_p = l_models[param_name](param_value)
                    g_p = g_models[param_name](param_value)
                except:
                    l_p = 1e-12
                    g_p = 1e-12
                l_p = max(l_p, 1e-12)
                g_p = max(g_p, 1e-12)
                l_prob *= l_p
                g_prob *= g_p
            if g_prob > 1e-15:
                likelihood_ratio = l_prob / g_prob
                ei = likelihood_ratio
            else:
                ei = l_prob / 1e-12 if l_prob > 0 else 1.0
            ei_values.append(ei)
        return ei_values

    def _sample_from_l_estimator(self, l_models: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Sample a new configuration from the l-estimator (good model).
        
        This implements the tree-structured sampling where each parameter is sampled
        conditional on previously sampled parameters (though we simplify assuming independence).
        """
        config = {}
        for (param_name, param_def) in self.search_space.items():
            if param_def['type'] == 'categorical':
                categories = param_def['bounds']
                probs = [l_models[param_name](cat) for cat in categories]
                probs = np.array(probs)
                probs = np.maximum(probs, 1e-12)
                probs = probs / np.sum(probs)
                config[param_name] = np.random.choice(categories, p=probs)
            else:
                (low, high) = param_def['bounds']
                max_attempts = 100
                best_sample = None
                best_likelihood = -1
                for _ in range(max_attempts):
                    if param_def['type'] == 'continuous':
                        if param_def.get('log', False):
                            (log_low, log_high) = (np.log(low), np.log(high))
                            sample = np.exp(np.random.uniform(log_low, log_high))
                        else:
                            sample = np.random.uniform(low, high)
                    else:
                        sample = np.random.randint(low, high + 1)
                    try:
                        likelihood = l_models[param_name](sample)
                    except:
                        likelihood = 1e-12
                    if likelihood > best_likelihood:
                        best_likelihood = likelihood
                        best_sample = sample
                config[param_name] = best_sample if best_sample is not None else (low + high) / 2
        return config

    def fit(self, X: Any, y: Optional[Any]=None, **kwargs) -> 'TreeStructuredParzenEstimatorOptimizer':
        """
        Execute the TPE optimization process to find optimal hyperparameters.
        
        This method performs the main optimization loop, alternating between:
        1. Random sampling (for startup jobs)
        2. TPE-based sampling (after startup phase)
        3. Model updates based on evaluation results
        
        Args:
            X (Any): Input data for the objective function (if applicable)
            y (Optional[Any]): Target values for the objective function (if applicable)
            **kwargs: Additional parameters for the optimization process
            
        Returns:
            TreeStructuredParzenEstimatorOptimizer: Self instance for method chaining
        """
        if self.objective_function is None:
            raise ValueError('Objective function must be set before calling fit()')
        for eval_idx in range(self.max_evaluations):
            if eval_idx < self.n_startup_jobs:
                config_dict = self._sample_random_configuration()
            else:
                sorted_evals = sorted(self._evaluations, key=lambda x: x[1], reverse=True)
                n_good = max(1, int(len(sorted_evals) * self.gamma))
                good_configs = [cfg for (cfg, _) in sorted_evals[:n_good]]
                bad_configs = [cfg for (cfg, _) in sorted_evals[n_good:]]
                if len(bad_configs) == 0 or len(good_configs) == 0:
                    config_dict = self._sample_random_configuration()
                else:
                    l_models = {}
                    g_models = {}
                    for (param_name, param_def) in self.search_space.items():
                        l_models[param_name] = self._build_parzen_estimator(good_configs, param_name, param_def)
                        g_models[param_name] = self._build_parzen_estimator(bad_configs, param_name, param_def)
                    candidate_configs = []
                    for _ in range(self.n_EI_candidates):
                        candidate_configs.append(self._sample_from_l_estimator(l_models))
                    ei_values = self._compute_expected_improvement(candidate_configs, l_models, g_models)
                    best_candidate_idx = np.argmax(ei_values)
                    config_dict = candidate_configs[best_candidate_idx]
            try:
                score = self.objective_function(**config_dict, X=X, y=y, **kwargs)
            except Exception as e:
                score = float('-inf')
            self._evaluations.append((config_dict, score))
            if score > self._best_score:
                self._best_score = score
                self.best_configuration = ComponentConfig(name='best_tpe_config', parameters=config_dict, metadata={'score': score, 'evaluation_index': eval_idx})
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Predict using the best configuration found during optimization.
        
        Args:
            X (Any): Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Any: Prediction results using the best found configuration
        """
        if self.best_configuration is None:
            default_config = ComponentConfig(parameters={}, metadata={'score': float('-inf')})
            return default_config
        return self.best_configuration

    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate the performance of the best configuration found.
        
        Args:
            X (Any): Input data for evaluation
            y (Any): Target values for evaluation
            **kwargs: Additional scoring parameters
            
        Returns:
            float: Performance score of the best configuration
        """
        if self.best_configuration is None:
            raise ValueError('No best configuration found. Run fit() first.')
        if self.objective_function is None:
            raise ValueError('Objective function must be set to compute score')
        config_params = self.best_configuration.parameters
        try:
            score = self.objective_function(**config_params, X=X, y=y, **kwargs)
        except Exception:
            score = float('-inf')
        return score

    def get_best_configuration(self) -> Optional[ComponentConfig]:
        """
        Retrieve the best hyperparameter configuration found during optimization.
        
        Returns:
            Optional[ComponentConfig]: Best configuration with parameter values and metadata,
                                       or None if optimization hasn't been run yet
        """
        return self.best_configuration

    def set_objective_function(self, func: Callable[..., float]) -> None:
        """
        Set the objective function to be optimized.
        
        Args:
            func (Callable[..., float]): Function to maximize/minimize.
                Should accept hyperparameter values as keyword arguments and return a float.
        """
        self.objective_function = func