import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, List, Callable, Optional
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.mathematical_foundations.optimization_methods.evolutionary_algorithms import GeneticAlgorithmOptimizer, ParticleSwarmOptimizer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier



class MultiObjectiveFeatureSelector(BaseTransformer):

    def __init__(self, n_features_to_select: Optional[int]=None, objectives: List[str]=None, algorithm: str='NSGA2', max_generations: int=100, population_size: int=50, name: Optional[str]=None):
        """
        Initialize the MultiObjectiveFeatureSelector.

        Parameters
        ----------
        n_features_to_select : Optional[int], default=None
            Desired number of features to select. If None, the optimizer determines it.
        objectives : List[str], default=['accuracy', 'feature_count']
            List of objectives to optimize. Supported objectives include:
            - 'accuracy': Classification accuracy
            - 'feature_count': Number of selected features
            - 'redundancy': Average correlation between selected features
            - 'stability': Stability of feature selection across folds
        algorithm : str, default='NSGA2'
            Multi-objective optimization algorithm to use. Options include:
            - 'NSGA2': Non-dominated Sorting Genetic Algorithm II
            - 'MOEA/D': Multi-Objective Evolutionary Algorithm based on Decomposition
            - 'PSO': Particle Swarm Optimization
        max_generations : int, default=100
            Maximum number of generations for the optimization algorithm.
        population_size : int, default=50
            Size of the population for evolutionary algorithms.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.n_features_to_select = n_features_to_select
        self.objectives = objectives or ['accuracy', 'feature_count']
        self.algorithm = algorithm
        self.max_generations = max_generations
        self.population_size = population_size
        supported_objectives = ['accuracy', 'feature_count', 'redundancy', 'stability']
        for obj in self.objectives:
            if obj not in supported_objectives:
                raise ValueError(f"Unsupported objective '{obj}'. Supported objectives are: {supported_objectives}")
        supported_algorithms = ['NSGA2', 'MOEA/D', 'PSO']
        if self.algorithm not in supported_algorithms:
            raise ValueError(f"Unsupported algorithm '{self.algorithm}'. Supported algorithms are: {supported_algorithms}")
        self.is_fitted = False
        self.n_features_ = 0
        self._selected_mask = None
        self._feature_scores = None

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'MultiObjectiveFeatureSelector':
        """
        Fit the feature selector using multi-objective optimization.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Training data containing features.
        y : np.ndarray
            Target values.
        **kwargs : dict
            Additional parameters for fitting.

        Returns
        -------
        MultiObjectiveFeatureSelector
            Fitted feature selector instance.
        """
        if isinstance(data, FeatureSet):
            X = np.array(data.features)
        else:
            X = np.array(data)
        self.n_features_ = X.shape[1]
        self._feature_names = getattr(data, 'feature_names', None)

        def objective_function(solution):
            feature_mask = solution > 0.5
            n_selected = np.sum(feature_mask)
            if n_selected == 0:
                return [np.inf] * len(self.objectives)
            X_selected = X[:, feature_mask]
            objective_values = []
            for obj in self.objectives:
                if obj == 'accuracy':
                    try:
                        clf = KNeighborsClassifier(n_neighbors=5)
                        scores = cross_val_score(clf, X_selected, y, cv=3, scoring='accuracy')
                        objective_values.append(-np.mean(scores))
                    except:
                        objective_values.append(0.0)
                elif obj == 'feature_count':
                    objective_values.append(float(n_selected))
                elif obj == 'redundancy':
                    if n_selected > 1:
                        corr_matrix = np.corrcoef(X_selected, rowvar=False)
                        corr_values = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
                        avg_redundancy = np.mean(np.abs(corr_values))
                        objective_values.append(avg_redundancy)
                    else:
                        objective_values.append(0.0)
                elif obj == 'stability':
                    objective_values.append(1.0 / n_selected)
            return objective_values
        if self.algorithm == 'NSGA2':
            optimizer = GeneticAlgorithmOptimizer(dim=self.n_features_, population_size=self.population_size, max_generations=self.max_generations, crossover_rate=0.8, mutation_rate=0.1, selection_method='tournament', elitism_count=1)
        elif self.algorithm == 'MOEA/D':
            optimizer = ParticleSwarmOptimizer(dim=self.n_features_, num_particles=self.population_size, max_iterations=self.max_generations, inertia_weight=0.729, cognitive_coefficient=1.494, social_coefficient=1.494)
        elif self.algorithm == 'PSO':
            optimizer = ParticleSwarmOptimizer(dim=self.n_features_, num_particles=self.population_size, max_iterations=self.max_generations, inertia_weight=0.729, cognitive_coefficient=1.494, social_coefficient=1.494)

        def single_objective_wrapper(solution):
            obj_values = objective_function(solution)
            return np.sum(obj_values)
        optimizer.fit(single_objective_wrapper)
        best_solution = optimizer.predict(None)
        self._selected_mask = best_solution > 0.5
        self._feature_scores = best_solution.copy()
        self.is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply feature selection to the input data.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters for transformation.

        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with selected features.
        """
        if not self.is_fitted:
            raise ValueError("This MultiObjectiveFeatureSelector instance is not fitted yet. Call 'fit' with appropriate arguments before using this transformer.")
        if isinstance(data, FeatureSet):
            X = np.array(data.features)
            feature_names = getattr(data, 'feature_names', None)
        else:
            X = np.array(data)
            feature_names = None
        X_selected = X[:, self._selected_mask]
        if isinstance(data, FeatureSet):
            selected_feature_names = None
            if feature_names is not None:
                selected_feature_names = [name for (i, name) in enumerate(feature_names) if self._selected_mask[i]]
            result = FeatureSet(features=X_selected.tolist(), feature_names=selected_feature_names, metadata=getattr(data, 'metadata', None))
            return result
        else:
            return X_selected

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Reverse the feature selection transformation (not supported).

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to inverse transform.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Original data format (raises NotImplementedError).
        """
        raise NotImplementedError('Inverse transformation is not supported for MultiObjectiveFeatureSelector.')

    def get_support(self, indices: bool=False) -> Union[np.ndarray, List[bool]]:
        """
        Get a mask or indices of the selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return indices of selected features.
            If False, return a boolean mask.

        Returns
        -------
        Union[np.ndarray, List[bool]]
            Boolean mask or list of indices indicating selected features.
        """
        if not self.is_fitted:
            raise ValueError("This MultiObjectiveFeatureSelector instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if indices:
            return np.where(self._selected_mask)[0]
        else:
            return self._selected_mask.copy()

    def get_feature_scores(self) -> np.ndarray:
        """
        Get scores for each feature from the optimization process.

        Returns
        -------
        np.ndarray
            Array of scores for each feature, where higher values indicate
            more important features according to the optimization objectives.
        """
        if not self.is_fitted:
            return np.zeros(self.n_features_)
        return self._feature_scores.copy()