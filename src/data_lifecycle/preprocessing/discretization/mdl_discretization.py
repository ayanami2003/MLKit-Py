from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class MDLDiscretizer(BaseTransformer):

    def __init__(self, feature_indices: Optional[List[int]]=None, max_bins: int=10, min_samples_per_bin: int=1, name: Optional[str]=None):
        super().__init__(name=name)
        self.feature_indices = feature_indices
        self.max_bins = max_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.cut_points_ = []
        self.n_bins_ = []
        self._is_fitted = False

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'MDLDiscretizer':
        """
        Fit the MDL discretizer to the input data.
        
        Computes the optimal cut points for each specified feature by applying
        the MDL principle to find the discretization that minimizes the total
        description length.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data containing continuous features to discretize.
        **kwargs : dict
            Additional fitting parameters (not used).
            
        Returns
        -------
        MDLDiscretizer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If data contains non-numeric features or if fitting fails.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data)
        else:
            X = np.asarray(data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D array')
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('All features must be numeric for discretization')
        (n_samples, n_features) = X.shape
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        for idx in feature_indices:
            if idx < 0 or idx >= n_features:
                raise ValueError(f'Feature index {idx} is out of bounds for data with {n_features} features')
        self.cut_points_ = []
        self.n_bins_ = []
        for i in range(n_features):
            if i in feature_indices:
                feature_data = X[:, i]
                valid_mask = ~np.isnan(feature_data)
                clean_data = feature_data[valid_mask]
                if len(clean_data) == 0:
                    self.cut_points_.append(np.array([]))
                    self.n_bins_.append(1)
                    continue
                sorted_data = np.sort(clean_data)
                cut_points = self._find_optimal_cut_points(sorted_data)
                self.cut_points_.append(cut_points)
                self.n_bins_.append(len(cut_points) + 1)
            else:
                self.cut_points_.append(np.array([]))
                self.n_bins_.append(1)
        self._is_fitted = True
        return self

    def _find_optimal_cut_points(self, sorted_data: np.ndarray) -> np.ndarray:
        """
        Find optimal cut points using the MDL principle.
        
        Parameters
        ----------
        sorted_data : np.ndarray
            1D array of sorted data values.
            
        Returns
        -------
        np.ndarray
            Array of optimal cut points.
        """
        n_samples = len(sorted_data)
        if n_samples < 2 * self.min_samples_per_bin:
            return np.array([])
        if np.all(sorted_data == sorted_data[0]):
            return np.array([])
        best_cut_points = np.array([])
        best_mdl_cost = np.inf
        max_bins_possible = min(self.max_bins, n_samples // self.min_samples_per_bin)
        if max_bins_possible < 2:
            return np.array([])
        for n_bins in range(2, max_bins_possible + 1):
            cut_points = self._find_cut_points_for_bins(sorted_data, n_bins)
            if len(cut_points) == n_bins - 1:
                mdl_cost = self._calculate_mdl_cost(sorted_data, cut_points)
                if mdl_cost < best_mdl_cost:
                    best_mdl_cost = mdl_cost
                    best_cut_points = cut_points
        return best_cut_points

    def _find_cut_points_for_bins(self, sorted_data: np.ndarray, n_bins: int) -> np.ndarray:
        """
        Find cut points that create approximately equal-sized bins.
        
        Parameters
        ----------
        sorted_data : np.ndarray
            1D array of sorted data values.
        n_bins : int
            Desired number of bins.
            
        Returns
        -------
        np.ndarray
            Array of cut points.
        """
        n_samples = len(sorted_data)
        if n_samples < n_bins * self.min_samples_per_bin:
            return np.array([])
        bin_size = n_samples // n_bins
        cut_points = []
        for i in range(1, n_bins):
            pos = i * bin_size
            if pos < self.min_samples_per_bin or pos > n_samples - self.min_samples_per_bin:
                continue
            cut_value = (sorted_data[pos - 1] + sorted_data[pos]) / 2
            cut_points.append(cut_value)
        return np.array(cut_points)

    def _calculate_mdl_cost(self, sorted_data: np.ndarray, cut_points: np.ndarray) -> float:
        """
        Calculate the MDL cost for a given discretization.
        
        Parameters
        ----------
        sorted_data : np.ndarray
            1D array of sorted data values.
        cut_points : np.ndarray
            Array of cut points.
            
        Returns
        -------
        float
            Total MDL cost (model cost + data cost).
        """
        n_samples = len(sorted_data)
        n_bins = len(cut_points) + 1
        model_cost = len(cut_points) * np.log2(n_samples)
        bin_counts = np.zeros(n_bins)
        bin_indices = np.digitize(sorted_data, cut_points)
        for idx in bin_indices:
            bin_counts[idx - 1] += 1
        epsilon = 1e-10
        probabilities = (bin_counts + epsilon) / (n_samples + n_bins * epsilon)
        data_cost = -np.sum(bin_counts * np.log2(probabilities + epsilon))
        return model_cost + data_cost

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Apply MDL discretization to the input data.
        
        Transforms continuous features into discrete bins based on the cut points
        computed during fitting. Each feature is mapped to integer bin indices.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to discretize.
        **kwargs : dict
            Additional transformation parameters (not used).
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Discretized data with features replaced by bin indices.
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or data shape mismatch.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        original_type = type(data).__name__
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data).copy()
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            batch_id = data.batch_id
            labels = data.labels
        else:
            X = np.asarray(data).copy()
            original_type = 'ndarray'
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        if n_features != len(self.n_bins_):
            raise ValueError(f'Data has {n_features} features, but transformer was fitted on data with {len(self.n_bins_)} features')
        X_discrete = X.copy()
        for i in range(n_features):
            if len(self.cut_points_[i]) > 0:
                cut_points = self.cut_points_[i]
                X_discrete[:, i] = np.digitize(X[:, i], cut_points)
            else:
                X_discrete[:, i] = X[:, i]
        if isinstance(data, FeatureSet):
            new_feature_types = feature_types.copy() if feature_types is not None else ['numeric'] * n_features
            for i in range(n_features):
                if len(self.cut_points_[i]) > 0:
                    new_feature_types[i] = 'categorical'
            return FeatureSet(features=X_discrete, feature_names=feature_names, feature_types=new_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif isinstance(data, DataBatch):
            return DataBatch(data=X_discrete, labels=labels, metadata=metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=batch_id)
        else:
            if original_type == 'ndarray' and data.ndim == 1:
                return X_discrete.ravel()
            return X_discrete

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Reverse the discretization transformation.
        
        Maps discretized bin indices back to approximate continuous values by
        using the midpoint of each bin interval.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Discretized data to inverse transform.
        **kwargs : dict
            Additional inverse transformation parameters (not used).
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Data with bin indices converted to approximate continuous values.
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or data shape mismatch.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        original_type = type(data).__name__
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data).copy()
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            batch_id = data.batch_id
            labels = data.labels
        else:
            X = np.asarray(data).copy()
            original_type = 'ndarray'
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        if n_features != len(self.n_bins_):
            raise ValueError(f'Data has {n_features} features, but transformer was fitted on data with {len(self.n_bins_)} features')
        X_continuous = X.copy().astype(float)
        for i in range(n_features):
            if len(self.cut_points_[i]) > 0:
                cut_points = self.cut_points_[i]
                bin_edges = np.concatenate([[-np.inf], cut_points, [np.inf]])
                bin_midpoints = []
                for j in range(len(bin_edges) - 1):
                    left_edge = bin_edges[j]
                    right_edge = bin_edges[j + 1]
                    if np.isinf(left_edge):
                        if len(cut_points) > 1:
                            spacing = cut_points[1] - cut_points[0]
                        else:
                            spacing = 1.0
                        midpoint = cut_points[0] - spacing / 2
                    elif np.isinf(right_edge):
                        if len(cut_points) > 1:
                            spacing = cut_points[-1] - cut_points[-2]
                        else:
                            spacing = 1.0
                        midpoint = cut_points[-1] + spacing / 2
                    else:
                        midpoint = (left_edge + right_edge) / 2
                    bin_midpoints.append(midpoint)
                bin_midpoints = np.array(bin_midpoints)
                for bin_idx in range(len(bin_midpoints)):
                    mask = X[:, i] == bin_idx + 1
                    X_continuous[mask, i] = bin_midpoints[bin_idx]
            else:
                pass
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_continuous, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif isinstance(data, DataBatch):
            return DataBatch(data=X_continuous, labels=labels, metadata=metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=batch_id)
        else:
            if original_type == 'ndarray' and data.ndim == 1:
                return X_continuous.ravel()
            return X_continuous