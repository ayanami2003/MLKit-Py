from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class GeneticKMeansModel(BaseModel):
    """
    A Genetic K-Means clustering algorithm implementation that uses evolutionary computation
    to optimize cluster centroids and assignments.

    This model combines the k-means clustering approach with genetic algorithm principles
    to find optimal cluster centers. It evolves a population of potential solutions over
    generations using selection, crossover, and mutation operators to minimize the
    clustering objective function.

    Attributes:
        n_clusters (int): Number of clusters to form.
        population_size (int): Size of the genetic algorithm population.
        generations (int): Number of generations to evolve.
        mutation_rate (float): Probability of mutation occurring.
        crossover_rate (float): Probability of crossover occurring.
        elite_size (int): Number of best solutions to keep in each generation.
        random_state (Optional[int]): Random seed for reproducibility.
    """

    def __init__(self, n_clusters: int=8, population_size: int=50, generations: int=100, mutation_rate: float=0.1, crossover_rate: float=0.8, elite_size: int=5, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the Genetic K-Means model.

        Args:
            n_clusters (int): Number of clusters to form. Defaults to 8.
            population_size (int): Size of the genetic algorithm population. Defaults to 50.
            generations (int): Number of generations to evolve. Defaults to 100.
            mutation_rate (float): Probability of mutation occurring. Defaults to 0.1.
            crossover_rate (float): Probability of crossover occurring. Defaults to 0.8.
            elite_size (int): Number of best solutions to keep in each generation. Defaults to 5.
            random_state (Optional[int]): Random seed for reproducibility. Defaults to None.
            name (Optional[str]): Name of the model instance. Defaults to class name.
        """
        super().__init__(name=name)
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.random_state = random_state

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'GeneticKMeansModel':
        """
        Fit the genetic k-means model to the data.

        Args:
            X (Union[FeatureSet, np.ndarray]): Training instances to cluster.
            y (Optional[np.ndarray]): Not used, present for API consistency.
            **kwargs: Additional fitting parameters.

        Returns:
            GeneticKMeansModel: Fitted estimator.
        """
        if isinstance(X, FeatureSet):
            X_data = X.features
        elif isinstance(X, np.ndarray):
            X_data = X
        else:
            raise TypeError('X must be either a FeatureSet or numpy array')
        if X_data.ndim != 2:
            raise ValueError('X must be a 2D array')
        (n_samples, n_features) = X_data.shape
        if self.n_clusters > n_samples:
            raise ValueError(f'n_clusters ({self.n_clusters}) cannot be larger than number of samples ({n_samples})')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._population = []
        for _ in range(self.population_size):
            indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
            self._population.append(X_data[indices].copy())
        fitness_scores = []
        for centroids in self._population:
            (_, inertia) = self._assign_clusters(X_data, centroids)
            fitness_scores.append(inertia)
        for generation in range(self.generations):
            sorted_indices = np.argsort(fitness_scores)
            sorted_population = [self._population[i] for i in sorted_indices]
            sorted_fitness = [fitness_scores[i] for i in sorted_indices]
            new_population = sorted_population[:self.elite_size]
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(sorted_population, sorted_fitness)
                parent2 = self._tournament_selection(sorted_population, sorted_fitness)
                if np.random.random() < self.crossover_rate:
                    (child1, child2) = self._crossover(parent1, parent2)
                else:
                    (child1, child2) = (parent1.copy(), parent2.copy())
                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, X_data)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, X_data)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            self._population = new_population[:self.population_size]
            fitness_scores = []
            for centroids in self._population:
                (_, inertia) = self._assign_clusters(X_data, centroids)
                fitness_scores.append(inertia)
        best_idx = np.argmin(fitness_scores)
        self.cluster_centers_ = self._population[best_idx].copy()
        (self.labels_, self.inertia_) = self._assign_clusters(X_data, self.cluster_centers_)
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Args:
            X (Union[FeatureSet, np.ndarray]): New data to predict.
            **kwargs: Additional prediction parameters.

        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        if isinstance(X, FeatureSet):
            X_data = X.features
        elif isinstance(X, np.ndarray):
            X_data = X
        else:
            raise TypeError('X must be either a FeatureSet or numpy array')
        if X_data.ndim != 2:
            raise ValueError('X must be a 2D array')
        (labels, _) = self._assign_clusters(X_data, self.cluster_centers_)
        return labels

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method that combines fit and predict.

        Args:
            X (Union[FeatureSet, np.ndarray]): Training instances to cluster.
            y (Optional[np.ndarray]): Not used, present for API consistency.
            **kwargs: Additional fitting parameters.

        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Calculate the negative inertia (sum of squared distances to centroids) for the model.

        Args:
            X (Union[FeatureSet, np.ndarray]): Data to evaluate.
            y (Optional[np.ndarray]): Not used, present for API consistency.
            **kwargs: Additional evaluation parameters.

        Returns:
            float: Negative inertia (higher is better).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        if isinstance(X, FeatureSet):
            X_data = X.features
        elif isinstance(X, np.ndarray):
            X_data = X
        else:
            raise TypeError('X must be either a FeatureSet or numpy array')
        if X_data.ndim != 2:
            raise ValueError('X must be a 2D array')
        (_, inertia) = self._assign_clusters(X_data, self.cluster_centers_)
        return -inertia

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> tuple:
        """
        Assign data points to clusters based on nearest centroid.
        
        Args:
            X (np.ndarray): Data points of shape (n_samples, n_features)
            centroids (np.ndarray): Cluster centroids of shape (n_clusters, n_features)
            
        Returns:
            tuple: (labels, inertia) where labels is array of cluster indices and 
                   inertia is sum of squared distances to centroids
        """
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        inertia = 0.0
        for i in range(len(X)):
            inertia += np.linalg.norm(X[i] - centroids[labels[i]]) ** 2
        return (labels, inertia)

    def _tournament_selection(self, population: list, fitness_scores: list) -> np.ndarray:
        """
        Perform tournament selection to choose a parent.
        
        Args:
            population (list): List of centroid arrays
            fitness_scores (list): Corresponding fitness scores (inertia values)
            
        Returns:
            np.ndarray: Selected parent (centroids array)
        """
        tournament_size = max(2, len(population) // 5)
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        best_idx_in_tournament = tournament_indices[np.argmin([fitness_scores[i] for i in tournament_indices])]
        return population[best_idx_in_tournament].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """
        Perform crossover between two parents to create offspring.
        
        Args:
            parent1 (np.ndarray): First parent centroids
            parent2 (np.ndarray): Second parent centroids
            
        Returns:
            tuple: Two offspring centroid arrays
        """
        (n_clusters, n_features) = parent1.shape
        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)
        for i in range(n_clusters):
            for j in range(n_features):
                if np.random.random() < 0.5:
                    child1[i, j] = parent1[i, j]
                    child2[i, j] = parent2[i, j]
                else:
                    child1[i, j] = parent2[i, j]
                    child2[i, j] = parent1[i, j]
        return (child1, child2)

    def _mutate(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Apply mutation to a centroids array.
        
        Args:
            centroids (np.ndarray): Centroids to mutate
            X (np.ndarray): Original data points
            
        Returns:
            np.ndarray: Mutated centroids
        """
        mutated_centroids = centroids.copy()
        (n_clusters, n_features) = mutated_centroids.shape
        for i in range(n_clusters):
            if np.random.random() < 0.3:
                random_point_idx = np.random.randint(0, len(X))
                random_point = X[random_point_idx]
                alpha = np.random.random() * 0.2
                mutated_centroids[i] = mutated_centroids[i] + alpha * (random_point - mutated_centroids[i])
        return mutated_centroids