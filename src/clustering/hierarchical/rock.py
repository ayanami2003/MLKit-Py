from typing import Optional, Any
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class ROCKClusteringModel(BaseModel):

    def __init__(self, theta: float=0.5, nbr_average_links: int=5, n_clusters: int=3, convergence_threshold: float=0.0001, max_iterations: int=100, name: Optional[str]=None):
        """
        Initialize ROCK clustering model.
        
        Parameters
        ----------
        theta : float, default=0.5
            Neighborhood radius parameter (0 < theta < 1). Controls the threshold for
            considering two points as neighbors based on their similarity.
        nbr_average_links : int, default=5
            Average number of neighbors for each point. Used to adjust the similarity
            threshold dynamically.
        n_clusters : int, default=3
            Number of clusters to form.
        convergence_threshold : float, default=1e-4
            Convergence threshold for the clustering algorithm.
        max_iterations : int, default=100
            Maximum number of iterations allowed.
        name : str, optional
            Name of the model instance.
        """
        super().__init__(name=name)
        self.theta = theta
        self.nbr_average_links = nbr_average_links
        self.n_clusters = n_clusters
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.labels_ = np.array([])
        self.cluster_centers_ = None
        self.n_features_in_ = 0
        self._training_data = None
        self._clusters = None

    def _compute_jaccard_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute Jaccard similarity between two categorical vectors.
        
        Parameters
        ----------
        x1 : np.ndarray
            First vector.
        x2 : np.ndarray
            Second vector.
            
        Returns
        -------
        float
            Jaccard similarity coefficient.
        """
        set1 = set(x1[x1 != 0]) if len(x1) > 0 else set()
        set2 = set(x2[x2 != 0]) if len(x2) > 0 else set()
        if len(set1) == 0 and len(set2) == 0:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 0.0
        return intersection / union

    def _compute_links(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the link matrix where links[i,j] represents the number of common neighbors
        between points i and j.
        
        Parameters
        ----------
        X : np.ndarray
            Input data array.
            
        Returns
        -------
        np.ndarray
            Link matrix of shape (n_samples, n_samples).
        """
        n_samples = X.shape[0]
        similarity_matrix = np.zeros((n_samples, n_samples))
        links = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                similarity = self._compute_jaccard_similarity(X[i], X[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        neighborhoods = []
        for i in range(n_samples):
            neighbors = np.where(similarity_matrix[i] >= self.theta)[0]
            neighborhoods.append(set(neighbors.tolist()))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                common_neighbors = len(neighborhoods[i].intersection(neighborhoods[j]))
                links[i, j] = common_neighbors
                links[j, i] = common_neighbors
        return links

    def _compute_goodness_measure(self, cluster_i: list, cluster_j: list, links: np.ndarray) -> float:
        """
        Compute the goodness measure for merging two clusters.
        
        Parameters
        ----------
        cluster_i : list
            Indices of points in cluster i.
        cluster_j : list
            Indices of points in cluster j.
        links : np.ndarray
            Link matrix.
            
        Returns
        -------
        float
            Goodness measure for merging the two clusters.
        """
        n_i = len(cluster_i)
        n_j = len(cluster_j)
        link_count = 0
        for i_idx in cluster_i:
            for j_idx in cluster_j:
                if i_idx < links.shape[0] and j_idx < links.shape[1]:
                    link_count += links[i_idx, j_idx]
        denominator = (n_i + n_j) ** (1 + 2 * (1 - self.theta)) - n_i ** (1 + 2 * (1 - self.theta)) - n_j ** (1 + 2 * (1 - self.theta))
        if denominator == 0:
            return 0.0
        goodness = link_count / denominator
        return goodness

    def _compute_link_between_clusters(self, cluster1: list, cluster2: list, data: np.ndarray) -> float:
        """
        Compute the number of links between two clusters.
        
        Parameters
        ----------
        cluster1 : list
            Indices of points in first cluster.
        cluster2 : list
            Indices of points in second cluster.
        data : np.ndarray
            Original training data.
            
        Returns
        -------
        float
            Number of links between the two clusters.
        """
        link_count = 0
        for i_idx in cluster1:
            for j_idx in cluster2:
                if i_idx < len(data) and j_idx < len(data):
                    similarity = self._compute_jaccard_similarity(data[i_idx], data[j_idx])
                    if similarity >= self.theta:
                        link_count += 1
        return link_count

    def fit(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> 'ROCKClusteringModel':
        """
        Compute ROCK clustering.
        
        Parameters
        ----------
        X : FeatureSet
            Training instances to cluster. Must contain categorical data represented
            as integers or strings.
        y : Ignored
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        ROCKClusteringModel
            Fitted estimator.
        """
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        if X.features is None:
            raise ValueError('FeatureSet must contain features')
        data = X.features
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        (n_samples, n_features) = data.shape
        self.n_features_in_ = n_features
        if n_samples < self.n_clusters:
            raise ValueError('Number of samples must be greater than or equal to n_clusters')
        self._training_data = data.copy()
        clusters = [[i] for i in range(n_samples)]
        links = self._compute_links(data)
        iteration = 0
        while len(clusters) > self.n_clusters and iteration < self.max_iterations:
            best_merge = None
            best_goodness = -np.inf
            n_clusters_current = len(clusters)
            for i in range(n_clusters_current):
                for j in range(i + 1, n_clusters_current):
                    if i < len(clusters) and j < len(clusters):
                        goodness = self._compute_goodness_measure(clusters[i], clusters[j], links)
                        if goodness > best_goodness:
                            best_goodness = goodness
                            best_merge = (i, j)
            if best_merge is None or best_goodness <= 0:
                break
            (i, j) = best_merge
            clusters[i].extend(clusters[j])
            del clusters[j]
            if links.shape[0] > 1 and links.shape[1] > 1:
                links = np.delete(links, j, axis=0)
                links = np.delete(links, j, axis=1)
            if len(clusters) > 1:
                new_links = np.zeros((len(clusters), len(clusters)))
                old_idx = 0
                for new_i in range(len(clusters)):
                    if new_i == i:
                        continue
                    old_idx += 1
                for k in range(len(clusters)):
                    if k != i:
                        link_value = self._compute_link_between_clusters(clusters[i], clusters[k], self._training_data)
                        new_links[i, k] = link_value
                        new_links[k, i] = link_value
                old_i_offset = 0
                for new_i in range(len(clusters)):
                    if new_i == i:
                        old_i_offset = 1
                        continue
                    old_i = new_i - old_i_offset
                    old_j_offset = 0
                    for new_j in range(new_i + 1, len(clusters)):
                        if new_j == i:
                            old_j_offset = 1
                            continue
                        old_j = new_j - old_j_offset
                        if old_i < links.shape[0] and old_j < links.shape[1]:
                            new_links[new_i, new_j] = links[old_i, old_j]
                            new_links[new_j, new_i] = links[old_i, old_j]
                links = new_links
            iteration += 1
        self._clusters = clusters
        labels = np.full(n_samples, -1, dtype=int)
        for (cluster_id, cluster_points) in enumerate(clusters):
            for point_idx in cluster_points:
                labels[point_idx] = cluster_id
        self.labels_ = labels
        self.cluster_centers_ = None
        return self

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : FeatureSet
            New data to predict. Must have the same feature structure as the training data.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        np.ndarray
            Index of the cluster each sample belongs to.
        """
        if len(self.labels_) == 0:
            raise ValueError("This ROCKClusteringModel instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        if X.features is None:
            raise ValueError('FeatureSet must contain features')
        data = X.features
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {data.shape[1]} features, but ROCKClusteringModel is expecting {self.n_features_in_} features as input.')
        if self._training_data is None or self._clusters is None:
            raise ValueError('Model state is corrupted. Please re-fit the model.')
        n_samples = data.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            sample = data[i]
            cluster_scores = np.zeros(len(self._clusters))
            for (j, cluster) in enumerate(self._clusters):
                total_links = 0
                for point_idx in cluster:
                    training_sample = self._training_data[point_idx]
                    similarity = self._compute_jaccard_similarity(sample, training_sample)
                    if similarity >= self.theta:
                        total_links += 1
                cluster_scores[j] = total_links
            predictions[i] = np.argmax(cluster_scores)
        return predictions

    def score(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> float:
        """
        Return the negative of the objective function value.
        
        The ROCK algorithm aims to maximize the number of intra-cluster links while
        minimizing inter-cluster links. This method returns the negative value of
        the objective function which can be used for model evaluation.
        
        Parameters
        ----------
        X : FeatureSet
            Data to evaluate. Must have the same feature structure as the training data.
        y : Ignored
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            Negative value of the objective function.
        """
        if len(self.labels_) == 0:
            raise ValueError("This ROCKClusteringModel instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        if X.features is None:
            raise ValueError('FeatureSet must contain features')
        data = X.features
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {data.shape[1]} features, but ROCKClusteringModel is expecting {self.n_features_in_} features as input.')
        links = self._compute_links(self._training_data)
        objective_value = 0.0
        unique_labels = np.unique(self.labels_)
        for label in unique_labels:
            cluster_points = np.where(self.labels_ == label)[0]
            n_points = len(cluster_points)
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    idx_i = cluster_points[i]
                    idx_j = cluster_points[j]
                    if idx_i < links.shape[0] and idx_j < links.shape[1]:
                        objective_value += links[idx_i, idx_j]
        return -objective_value