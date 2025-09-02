from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from typing import Optional, List
from typing import Optional, List, Union
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

class KNearestNeighborsClassifier(BaseModel):

    def __init__(self, n_neighbors: int=5, weights: str='uniform', algorithm: str='auto', leaf_size: int=30, p: int=2, metric: str='minkowski', random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the K-Nearest Neighbors classifier.
        
        Args:
            n_neighbors (int): Number of neighbors to use. Defaults to 5.
            weights (str): Weight function used in prediction. 'uniform' assigns equal
                          weights to all neighbors, 'distance' weights by inverse distance.
                          Defaults to 'uniform'.
            algorithm (str): Algorithm used to compute nearest neighbors. Options are
                           'ball_tree', 'kd_tree', 'brute', or 'auto'. Defaults to 'auto'.
            leaf_size (int): Leaf size for tree-based algorithms. Affects speed and memory.
                            Defaults to 30.
            p (int): Power parameter for Minkowski metric. When p=1, equivalent to Manhattan
                    distance; when p=2, equivalent to Euclidean distance. Defaults to 2.
            metric (str): Distance metric to use. Supported metrics include 'euclidean',
                         'manhattan', 'chebyshev', 'minkowski', etc. Defaults to 'minkowski'.
            random_state (Optional[int]): Random state for reproducible results. Defaults to None.
            name (Optional[str]): Name for the model instance. Defaults to class name.
        """
        super().__init__(name=name)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.random_state = random_state
        if self.weights not in ['uniform', 'distance']:
            raise ValueError("weights must be 'uniform' or 'distance'")
        if self.metric == 'minkowski' and self.p < 1:
            raise ValueError('p must be >= 1 for minkowski metric')

    def _extract_features(self, X):
        """Extract feature array from various input formats."""
        if hasattr(X, 'features'):
            return np.array(X.features)
        elif hasattr(X, 'data'):
            return np.array(X.data)
        else:
            return np.array(X)

    def _extract_targets(self, y):
        """Extract target array from various input formats."""
        if hasattr(y, 'data'):
            return np.array(y.data)
        else:
            return np.array(y)

    def fit(self, X: FeatureSet, y: DataBatch, **kwargs) -> 'KNearestNeighborsClassifier':
        """
        Fit the K-Nearest Neighbors classifier according to the training data.
        
        Args:
            X (FeatureSet): Training data features.
            y (DataBatch): Target values for training data.
            **kwargs: Additional fitting parameters.
            
        Returns:
            KNearestNeighborsClassifier: Fitted classifier instance.
            
        Raises:
            ValueError: If the dimensions of X and y do not match.
        """
        X_array = self._extract_features(X)
        y_array = self._extract_targets(y)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(f'X and y must have the same number of samples. Got {X_array.shape[0]} and {y_array.shape[0]}')
        self._X_train = X_array
        self._y_train = y_array
        self.classes_ = np.unique(y_array)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        return self

    def predict(self, X: FeatureSet, **kwargs) -> List[int]:
        """
        Predict class labels for samples in X.
        
        Args:
            X (FeatureSet): Test samples.
            **kwargs: Additional prediction parameters.
            
        Returns:
            List[int]: Predicted class labels for each sample.
        """
        X_array = self._extract_features(X)
        neighbors_indices = self._find_neighbors(X_array)
        predictions = []
        for (i, neighbor_indices) in enumerate(neighbors_indices):
            neighbor_labels = self._y_train[neighbor_indices]
            if self.weights == 'uniform':
                prediction = Counter(neighbor_labels).most_common(1)[0][0]
            else:
                distances = self._compute_distances(X_array[i:i + 1], self._X_train[neighbor_indices])[0]
                prediction = self._weighted_vote(neighbor_labels, distances)
            predictions.append(int(prediction))
        return predictions

    def predict_proba(self, X: FeatureSet, **kwargs) -> List[List[float]]:
        """
        Return probability estimates for the test data X.
        
        Args:
            X (FeatureSet): Test samples.
            **kwargs: Additional prediction parameters.
            
        Returns:
            List[List[float]]: Probability estimates for each class for each sample.
        """
        X_array = self._extract_features(X)
        neighbors_indices = self._find_neighbors(X_array)
        probabilities = []
        for (i, neighbor_indices) in enumerate(neighbors_indices):
            neighbor_labels = self._y_train[neighbor_indices]
            if self.weights == 'uniform':
                weights = np.ones(len(neighbor_labels))
            else:
                distances = self._compute_distances(X_array[i:i + 1], self._X_train[neighbor_indices])[0]
                weights = self._calculate_distance_weights(distances)
            class_probs = self._compute_class_probabilities(neighbor_labels, weights)
            probabilities.append(class_probs)
        return probabilities

    def score(self, X: FeatureSet, y: DataBatch, **kwargs) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X (FeatureSet): Test samples.
            y (DataBatch): True labels for test samples.
            **kwargs: Additional scoring parameters.
            
        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        y_array = self._extract_targets(y)
        predictions = self.predict(X)
        correct_predictions = np.sum(np.array(predictions) == y_array)
        accuracy = correct_predictions / len(y_array)
        return float(accuracy)

    def _find_neighbors(self, X: np.ndarray) -> np.ndarray:
        """
        Find the k nearest neighbors for each sample in X.
        
        Args:
            X (np.ndarray): Test samples.
            
        Returns:
            np.ndarray: Indices of k nearest neighbors for each sample.
        """
        distances = self._compute_distances(X, self._X_train)
        neighbors_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        return neighbors_indices

    def _compute_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute distances between samples in X and Y.
        
        Args:
            X (np.ndarray): First set of samples.
            Y (np.ndarray): Second set of samples.
            
        Returns:
            np.ndarray: Distance matrix.
        """
        if self.metric == 'minkowski':
            if self.p == 1:
                metric = 'cityblock'
            elif self.p == 2:
                metric = 'euclidean'
            else:
                distances = np.power(np.sum(np.power(np.abs(X[:, np.newaxis] - Y), self.p), axis=2), 1 / self.p)
                return distances
        elif self.metric == 'manhattan':
            metric = 'cityblock'
        else:
            metric = self.metric
        distances = cdist(X, Y, metric=metric)
        return distances

    def _weighted_vote(self, labels: np.ndarray, distances: np.ndarray) -> int:
        """
        Perform weighted voting based on inverse distances.
        
        Args:
            labels (np.ndarray): Labels of neighbors.
            distances (np.ndarray): Distances to neighbors.
            
        Returns:
            int: Predicted label.
        """
        weights = self._calculate_distance_weights(distances)
        class_votes = {}
        for (label, weight) in zip(labels, weights):
            if label in class_votes:
                class_votes[label] += weight
            else:
                class_votes[label] = weight
        return max(class_votes, key=class_votes.get)

    def _calculate_distance_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Calculate weights based on inverse distances.
        
        Args:
            distances (np.ndarray): Distances to neighbors.
            
        Returns:
            np.ndarray: Weights for each neighbor.
        """
        weights = np.where(distances == 0, 1000000000000.0, 1 / np.maximum(distances, 1e-12))
        return weights

    def _compute_class_probabilities(self, labels: np.ndarray, weights: np.ndarray) -> List[float]:
        """
        Compute class probabilities based on neighbor labels and weights.
        
        Args:
            labels (np.ndarray): Labels of neighbors.
            weights (np.ndarray): Weights for each neighbor.
            
        Returns:
            List[float]: Probability for each class.
        """
        if not hasattr(self, 'classes_'):
            self.classes_ = np.unique(labels)
        class_probs = [0.0] * len(self.classes_)
        total_weight = np.sum(weights)
        if total_weight > 0:
            for (label, weight) in zip(labels, weights):
                class_indices = np.where(self.classes_ == label)[0]
                if len(class_indices) > 0:
                    class_index = class_indices[0]
                    class_probs[class_index] += weight
            class_probs = [prob / total_weight for prob in class_probs]
        else:
            class_probs = [1.0 / len(self.classes_)] * len(self.classes_)
        return class_probs