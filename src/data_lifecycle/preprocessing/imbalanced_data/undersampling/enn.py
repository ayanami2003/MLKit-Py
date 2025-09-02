from typing import Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from collections import Counter

class EditedNearestNeighborsCleaner(BaseTransformer):
    """
    A transformer that applies Edited Nearest Neighbors (ENN) cleaning to remove noisy or borderline samples
    from imbalanced datasets. ENN works by removing samples that are misclassified by their k-nearest neighbors,
    helping to clean the dataset and potentially improve model performance.

    This implementation supports both binary and multiclass classification scenarios and allows customization
    of the number of neighbors and editing iterations.

    Attributes
    ----------
    n_neighbors : int, default=3
        Number of neighbors to consider for classification.
    kind_sel : str, default='all'
        Strategy for sample selection. Options are 'all' or 'mode'.
    allow_minority : bool, default=False
        Whether to allow minority classes to be edited.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_neighbors: int=3, kind_sel: str='all', allow_minority: bool=False, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the EditedNearestNeighborsCleaner.

        Parameters
        ----------
        n_neighbors : int, default=3
            Number of neighbors to use for checking sample consistency.
        kind_sel : str, default='all'
            Sample selection strategy ('all' or 'mode').
        allow_minority : bool, default=False
            If True, allows cleaning of minority class samples.
        random_state : Optional[int], default=None
            Random seed for reproducible results.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.allow_minority = allow_minority
        self.random_state = random_state
        if self.n_neighbors <= 0:
            raise ValueError('n_neighbors must be positive')
        if self.kind_sel not in ['all', 'mode']:
            raise ValueError("kind_sel must be either 'all' or 'mode'")
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, data: FeatureSet, **kwargs) -> 'EditedNearestNeighborsCleaner':
        """
        Fit the cleaner to the input data.

        This method analyzes the data to identify which samples should be removed
        based on the ENN algorithm but does not modify the data yet.

        Parameters
        ----------
        data : FeatureSet
            Input feature set with features and labels.
        **kwargs : dict
            Additional fitting parameters (ignored).

        Returns
        -------
        EditedNearestNeighborsCleaner
            Self instance for method chaining.
        """
        X = data.features
        y = data.labels
        if len(X) != len(y):
            raise ValueError('Features and labels must have the same length')
        if len(X) < self.n_neighbors + 1:
            raise ValueError(f'Not enough samples ({len(X)}) for n_neighbors={self.n_neighbors}')
        X_array = np.array(X)
        y_array = np.array(y)
        class_counts = Counter(y_array)
        majority_class = max(class_counts, key=class_counts.get)
        distances = self._compute_distances(X_array)
        knn_indices = []
        for i in range(len(X_array)):
            dists = distances[i]
            sorted_indices = np.argsort(dists)[1:self.n_neighbors + 1]
            knn_indices.append(sorted_indices)
        self.indices_to_remove_ = set()
        for i in range(len(X_array)):
            current_label = y_array[i]
            if not self.allow_minority and current_label != majority_class:
                continue
            neighbor_labels = y_array[knn_indices[i]]
            if self.kind_sel == 'all':
                should_remove = not np.any(neighbor_labels == current_label)
            else:
                most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
                should_remove = most_common_label != current_label
            if should_remove:
                self.indices_to_remove_.add(i)
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply ENN cleaning to remove noisy or mislabeled samples.

        This method removes samples identified during fitting as inconsistent
        with their neighborhood according to the ENN algorithm.

        Parameters
        ----------
        data : FeatureSet
            Input feature set to clean.
        **kwargs : dict
            Additional transformation parameters (ignored).

        Returns
        -------
        FeatureSet
            Cleaned feature set with inconsistent samples removed.
        """
        if not hasattr(self, 'indices_to_remove_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        X = data.features
        y = data.labels
        indices_to_keep = [i for i in range(len(X)) if i not in self.indices_to_remove_]
        filtered_features = [X[i] for i in indices_to_keep]
        filtered_labels = [y[i] for i in indices_to_keep]
        cleaned_data = FeatureSet(features=filtered_features, labels=filtered_labels, feature_names=data.feature_names, label_names=data.label_names)
        return cleaned_data

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for ENN cleaning as information is lost.

        Parameters
        ----------
        data : FeatureSet
            Feature set to inverse transform (ignored).

        **kwargs : dict
            Additional parameters (ignored).

        Returns
        -------
        FeatureSet
            The same data passed as input since inverse transformation is not possible.

        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for ENN cleaning.')

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between all samples.
        
        Parameters
        ----------
        X : np.ndarray
            Array of features.
            
        Returns
        -------
        np.ndarray
            Pairwise distance matrix.
        """
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances