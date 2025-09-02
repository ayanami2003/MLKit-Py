from typing import Optional, List, Union
import numpy as np
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch

class DifferentialEvolutionStoppingCriterion(BaseValidator):

    def __init__(self, tolerance: float=1e-06, max_iterations: int=1000, min_population_diversity: float=0.0001, window_size: int=10, name: Optional[str]=None):
        """
        Initialize the differential evolution stopping criterion validator.
        
        Parameters
        ----------
        tolerance : float, optional
            Convergence threshold for objective function improvement (default: 1e-6).
        max_iterations : int, optional
            Maximum iteration count before forced stopping (default: 1000).
        min_population_diversity : float, optional
            Minimum diversity threshold in parameter space (default: 1e-4).
        window_size : int, optional
            Window size for convergence monitoring (default: 10).
        name : Optional[str], optional
            Custom name for the validator instance.
        """
        super().__init__(name=name)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.min_population_diversity = min_population_diversity
        self.window_size = window_size
        self._current_iteration = 0
        self._objective_history: List[float] = []

    def validate(self, data: Union[DataBatch, dict], **kwargs) -> bool:
        """
        Check if the differential evolution process should continue.
        
        Evaluates stopping criteria based on:
        - Iteration count limit
        - Objective function convergence
        - Population diversity thresholds
        
        Parameters
        ----------
        data : Union[DataBatch, dict]
            Evolution state data containing objectives and population information.
            Expected to have keys 'objectives' and 'population'.
        **kwargs : dict
            Additional parameters (unused).
            
        Returns
        -------
        bool
            True if optimization should continue, False if it should stop.
        """
        if isinstance(data, DataBatch):
            if data.metadata is None:
                raise ValueError("DataBatch must contain metadata with 'objectives' and 'population'")
            objectives = data.metadata.get('objectives')
            population = data.metadata.get('population')
        else:
            objectives = data.get('objectives')
            population = data.get('population')
        if objectives is None or population is None:
            raise ValueError("Data must contain 'objectives' and 'population' keys")
        self._current_iteration += 1
        if self._current_iteration >= self.max_iterations:
            return False
        if isinstance(objectives, (list, np.ndarray)):
            current_objective = float(np.min(objectives))
        else:
            current_objective = float(objectives)
        self._objective_history.append(current_objective)
        if len(self._objective_history) >= self.window_size:
            recent_values = self._objective_history[-self.window_size:]
            if len(recent_values) >= 2:
                improvements = [abs(recent_values[i] - recent_values[i - 1]) for i in range(1, len(recent_values))]
                avg_improvement = np.mean(improvements)
                if avg_improvement < self.tolerance:
                    return False
        if len(population) > 1:
            population_array = np.array(population)
            n_individuals = population_array.shape[0]
            total_distance = 0.0
            pair_count = 0
            for i in range(n_individuals):
                for j in range(i + 1, n_individuals):
                    distance = np.linalg.norm(population_array[i] - population_array[j])
                    total_distance += distance
                    pair_count += 1
            avg_diversity = total_distance / pair_count if pair_count > 0 else 0.0
            if avg_diversity < self.min_population_diversity:
                return False
        return True

    def reset_monitoring_state(self) -> None:
        """
        Reset internal monitoring state to initial conditions.
        
        Clears iteration counters and historical data to prepare for a new optimization run.
        """
        self._current_iteration = 0
        self._objective_history = []