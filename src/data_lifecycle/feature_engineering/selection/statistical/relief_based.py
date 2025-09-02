from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet


# ...(code omitted)...


class ReliefFeatureSelector(BaseTransformer):

    def __init__(self, n_features_to_select: Optional[int]=None, score_threshold: Optional[float]=None, n_neighbors: int=10, sample_size: Optional[Union[int, float]]=None, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_features_to_select = n_features_to_select
        self.score_threshold = score_threshold
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size
        self.random_state = random_state
        self._support_mask = None
        self._scores = None

    def fit(self, data: FeatureSet, y: Union[np.ndarray, list], **kwargs) -> 'ReliefFeatureSelector':
        """
        Compute Relief scores and identify selected features.
        
        Parameters
        ----------
        data : FeatureSet
            Input features with shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Target values (class labels)
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        ReliefFeatureSelector
            Fitted selector instance
            
        Raises
        ------
        ValueError
            If data and target shapes don't match or if neither n_features_to_select nor score_threshold is specified
        """
        if self.n_features_to_select is None and self.score_threshold is None:
            raise ValueError('Either n_features_to_select or score_threshold must be specified')
        if self.n_features_to_select is not None and self.score_threshold is not None:
            raise ValueError('Only one of n_features_to_select or score_threshold can be specified')
        X = data.features
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Number of samples in X ({X.shape[0]}) does not match number of targets ({y.shape[0]})')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        if self.sample_size is None:
            n_samples_to_use = n_samples
        elif isinstance(self.sample_size, float):
            n_samples_to_use = int(self.sample_size * n_samples)
        else:
            n_samples_to_use = min(self.sample_size, n_samples)
        if n_samples_to_use < n_samples:
            indices = np.random.choice(n_samples, n_samples_to_use, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        self._scores = self._compute_relief_scores(X_sample, y_sample)
        if self.n_features_to_select is not None:
            if self.n_features_to_select > len(self._scores):
                raise ValueError(f'n_features_to_select ({self.n_features_to_select}) cannot be larger than number of features ({len(self._scores)})')
            top_indices = np.argpartition(self._scores, -self.n_features_to_select)[-self.n_features_to_select:]
            self._support_mask = np.zeros(len(self._scores), dtype=bool)
            self._support_mask[top_indices] = True
        else:
            self._support_mask = self._scores >= self.score_threshold
        return self

    def _compute_relief_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute Relief scores for features.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
            
        Returns
        -------
        np.ndarray
            Relief scores for each feature
        """
        (n_samples, n_features) = X.shape
        scores = np.zeros(n_features)
        for i in range(n_samples):
            (hit_indices, miss_indices) = self._find_nearest_neighbors(X, y, i)
            for j in range(n_features):
                hit_diff = np.mean(np.abs(X[i, j] - X[hit_indices, j])) if len(hit_indices) > 0 else 0
                miss_diff = np.mean(np.abs(X[i, j] - X[miss_indices, j])) if len(miss_indices) > 0 else 0
                scores[j] -= hit_diff / n_samples
                scores[j] += miss_diff / n_samples
        return scores

    def _find_nearest_neighbors(self, X: np.ndarray, y: np.ndarray, idx: int) -> tuple:
        """
        Find nearest neighbors of the same and different classes.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        idx : int
            Index of reference sample
            
        Returns
        -------
        tuple
            Indices of hit neighbors (same class) and miss neighbors (different class)
        """
        n_samples = X.shape[0]
        distances = np.full(n_samples, np.inf, dtype=float)
        distances[idx] = np.inf
        for i in range(n_samples):
            if i != idx:
                distances[i] = np.sqrt(np.sum((X[idx] - X[i]) ** 2))
        sorted_indices = np.argsort(distances)
        same_class_mask = y == y[idx]
        same_class_mask[idx] = False
        hit_candidates = sorted_indices[same_class_mask[sorted_indices]]
        hit_indices = hit_candidates[:min(len(hit_candidates), self.n_neighbors)]
        diff_class_mask = y != y[idx]
        miss_candidates = sorted_indices[diff_class_mask[sorted_indices]]
        miss_indices = miss_candidates[:min(len(miss_candidates), self.n_neighbors)]
        return (hit_indices, miss_indices)

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform the input data by selecting only the relevant features.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Transformed data with only selected features
            
        Raises
        ------
        RuntimeError
            If transformer has not been fitted yet
        """
        if self._support_mask is None:
            raise RuntimeError('Transformer has not been fitted yet.')
        X_selected = data.features[:, self._support_mask]
        selected_feature_names = [data.feature_names[i] for i in range(len(data.feature_names)) if self._support_mask[i]]
        selected_feature_types = [data.feature_types[i] for i in range(len(data.feature_types)) if self._support_mask[i]] if data.feature_types else None
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform by reconstructing the full feature set.
        Note: This implementation fills unselected features with zeros.
        
        Parameters
        ----------
        data : FeatureSet
            Transformed features to inverse transform
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Reconstructed feature set with all original features
            
        Raises
        ------
        RuntimeError
            If transformer has not been fitted yet
        """
        if self._support_mask is None:
            raise RuntimeError('Transformer has not been fitted yet.')
        n_features_original = len(self._support_mask)
        n_samples = data.features.shape[0]
        X_full = np.zeros((n_samples, n_features_original))
        X_full[:, self._support_mask] = data.features
        feature_names = []
        for i in range(n_features_original):
            if self._support_mask[i]:
                selected_idx = np.where(self._support_mask[:i + 1])[0][-1]
                actual_idx = np.sum(self._support_mask[:i])
                if actual_idx < len(data.feature_names):
                    feature_names.append(data.feature_names[actual_idx])
                else:
                    feature_names.append(f'feature_{i}')
            else:
                feature_names.append(f'feature_{i}')
        feature_types = None
        if data.feature_types:
            feature_types = []
            for i in range(n_features_original):
                if self._support_mask[i]:
                    actual_idx = np.sum(self._support_mask[:i])
                    if actual_idx < len(data.feature_types):
                        feature_types.append(data.feature_types[actual_idx])
                    else:
                        feature_types.append(None)
                else:
                    feature_types.append(None)
        return FeatureSet(features=X_full, feature_names=feature_names, feature_types=feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)

    def get_support(self) -> np.ndarray:
        """
        Get the boolean mask of selected features.
        
        Returns
        -------
        np.ndarray of shape (n_features,)
            Boolean mask indicating selected features
            
        Raises
        ------
        RuntimeError
            If transformer has not been fitted yet
        """
        if self._support_mask is None:
            raise RuntimeError('Transformer has not been fitted yet.')
        return self._support_mask.copy()

    def get_scores(self) -> np.ndarray:
        """
        Get Relief scores for each feature.
        
        Higher scores indicate more relevant features.
        
        Returns
        -------
        np.ndarray of shape (n_features,)
            Relief scores for each feature
            
        Raises
        ------
        RuntimeError
            If transformer has not been fitted yet
        """
        if self._scores is None:
            raise RuntimeError('Transformer has not been fitted yet.')
        return self._scores.copy()