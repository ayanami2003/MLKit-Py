from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class ChiSquareDiscretizer(BaseTransformer):

    def __init__(self, feature_indices: Optional[List[int]]=None, n_initial_bins: int=10, min_bin_size: float=0.05, max_bins: int=5, name: Optional[str]=None):
        """
        Initialize the ChiSquareDiscretizer.
        
        Parameters
        ----------
        feature_indices : Optional[List[int]], optional
            Indices of features to discretize. If None, all features are processed.
        n_initial_bins : int, default=10
            Number of initial bins to create for finding optimal splits.
        min_bin_size : float, default=0.05
            Minimum proportion of samples required in each bin (between 0 and 1).
        max_bins : int, default=5
            Maximum number of bins to produce after optimization.
        name : Optional[str], optional
            Name for the transformer instance.
        """
        super().__init__(name=name)
        self.feature_indices = feature_indices
        self.n_initial_bins = n_initial_bins
        self.min_bin_size = min_bin_size
        self.max_bins = max_bins
        self.bin_boundaries_ = []

    def fit(self, data: Union[FeatureSet, DataBatch], y: Optional[np.ndarray]=None, **kwargs) -> 'ChiSquareDiscretizer':
        """
        Fit the discretizer to the input data using chi-square optimization.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data containing continuous features to discretize.
            Must contain labels when used with DataBatch.
        y : Optional[np.ndarray], optional
            Target values for supervised discretization. Required when data is FeatureSet.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        ChiSquareDiscretizer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If target values are not provided or data format is invalid.
        """
        if isinstance(data, DataBatch):
            X = np.asarray(data.data)
            if y is None:
                y = data.labels
                if y is None:
                    raise ValueError('Target values (y) must be provided either explicitly or through DataBatch labels')
        elif isinstance(data, FeatureSet):
            X = data.features
            if y is None:
                raise ValueError('Target values (y) must be provided when using FeatureSet')
        else:
            raise ValueError('Data must be either FeatureSet or DataBatch')
        if y is None:
            raise ValueError('Target values are required for supervised discretization')
        y = np.asarray(y)
        if len(y) != X.shape[0]:
            raise ValueError('Number of target values must match number of samples in features')
        n_features = X.shape[1]
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        if any((idx < 0 or idx >= n_features for idx in feature_indices)):
            raise ValueError('Feature indices must be valid indices for the input data')
        self.bin_boundaries_ = []
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            bin_edges = self._compute_chi_square_bins(feature_values, y)
            self.bin_boundaries_.append(bin_edges)
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Apply chi-square based discretization to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to discretize.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Discretized data with same type as input.
            
        Raises
        ------
        ValueError
            If the discretizer has not been fitted or data format is invalid.
        """
        if not hasattr(self, 'bin_boundaries_') or len(self.bin_boundaries_) == 0:
            raise ValueError("Discretizer has not been fitted yet. Call 'fit' first.")
        original_type = type(data)
        if isinstance(data, DataBatch):
            X = np.asarray(data.data).copy()
            feature_indices = self.feature_indices if self.feature_indices is not None else list(range(X.shape[1]))
        elif isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_indices = self.feature_indices if self.feature_indices is not None else list(range(X.shape[1]))
        else:
            raise ValueError('Data must be either FeatureSet or DataBatch')
        for (i, feature_idx) in enumerate(feature_indices):
            if i < len(self.bin_boundaries_):
                bin_indices = np.digitize(X[:, feature_idx], self.bin_boundaries_[i]) - 1
                bin_indices = np.clip(bin_indices, 0, len(self.bin_boundaries_[i]) - 2)
                X[:, feature_idx] = bin_indices
        X = X.astype(int)
        if original_type == DataBatch:
            return DataBatch(data=X, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        else:
            return FeatureSet(features=X, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Inverse transformation is not supported for chi-square discretization.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Discretized data to inverse transform.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Original data format (identity transformation).
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for this method.
        """
        raise NotImplementedError('Inverse transformation is not supported for chi-square discretization.')

    def _compute_chi_square_bins(self, feature_values: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute optimal bin boundaries for a feature using chi-square statistics.
        
        Parameters
        ----------
        feature_values : np.ndarray
            Continuous values of a single feature.
        y : np.ndarray
            Target labels.
            
        Returns
        -------
        np.ndarray
            Array of bin boundaries.
        """
        mask = ~np.isnan(feature_values)
        if not np.any(mask):
            return np.array([0.0, 1.0])
        clean_values = feature_values[mask]
        clean_y = y[mask]
        classes = np.unique(clean_y)
        n_classes = len(classes)
        if n_classes < 2:
            return self._quantile_binning(clean_values, self.max_bins)
        n_samples = len(clean_values)
        min_bin_size_samples = max(1, int(self.min_bin_size * n_samples))
        initial_bins = min(self.n_initial_bins, n_samples)
        quantiles = np.linspace(0, 1, initial_bins + 1)
        initial_boundaries = np.quantile(clean_values, quantiles)
        initial_boundaries = np.unique(initial_boundaries)
        if len(initial_boundaries) <= 2:
            return initial_boundaries
        best_chi2 = -1
        best_boundaries = initial_boundaries
        for n_bins in range(2, min(self.max_bins + 1, len(initial_boundaries))):
            indices = np.linspace(0, len(initial_boundaries) - 1, n_bins + 1).astype(int)
            boundaries = initial_boundaries[indices]
            bin_counts = np.histogram(clean_values, bins=boundaries)[0]
            if np.all(bin_counts >= min_bin_size_samples):
                chi2_stat = self._calculate_chi_square_statistic(clean_values, clean_y, boundaries, classes)
                if chi2_stat > best_chi2:
                    best_chi2 = chi2_stat
                    best_boundaries = boundaries
        if best_chi2 == -1:
            best_boundaries = self._quantile_binning(clean_values, self.max_bins)
        return best_boundaries

    def _quantile_binning(self, values: np.ndarray, n_bins: int) -> np.ndarray:
        """
        Create bin boundaries using quantiles.
        
        Parameters
        ----------
        values : np.ndarray
            Input values.
        n_bins : int
            Number of bins.
            
        Returns
        -------
        np.ndarray
            Bin boundaries.
        """
        quantiles = np.linspace(0, 1, n_bins + 1)
        return np.quantile(values, quantiles)

    def _calculate_chi_square_statistic(self, feature_values: np.ndarray, y: np.ndarray, boundaries: np.ndarray, classes: np.ndarray) -> float:
        """
        Calculate chi-square statistic for given binning.
        
        Parameters
        ----------
        feature_values : np.ndarray
            Feature values.
        y : np.ndarray
            Target labels.
        boundaries : np.ndarray
            Bin boundaries.
        classes : np.ndarray
            Unique class labels.
            
        Returns
        -------
        float
            Chi-square statistic.
        """
        bin_indices = np.digitize(feature_values, boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, len(boundaries) - 2)
        n_bins = len(boundaries) - 1
        n_classes = len(classes)
        contingency_table = np.zeros((n_bins, n_classes))
        for i in range(n_bins):
            for (j, cls) in enumerate(classes):
                contingency_table[i, j] = np.sum((bin_indices == i) & (y == cls))
        row_totals = np.sum(contingency_table, axis=1)
        col_totals = np.sum(contingency_table, axis=0)
        total = np.sum(contingency_table)
        if total == 0:
            return 0.0
        chi2_stat = 0.0
        for i in range(n_bins):
            for j in range(n_classes):
                expected = row_totals[i] * col_totals[j] / total if total > 0 else 0
                if expected > 0:
                    observed = contingency_table[i, j]
                    chi2_stat += (observed - expected) ** 2 / expected
        return chi2_stat