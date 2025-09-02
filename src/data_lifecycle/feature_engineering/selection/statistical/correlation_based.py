from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np


# ...(code omitted)...


class CorrelationThresholdSelector(BaseTransformer):
    """
    A transformer that removes features with high inter-feature correlations.
    
    This selector identifies and removes features that are highly correlated with each other,
    helping to reduce multicollinearity in the feature set. It supports both Pearson and 
    Spearman correlation methods and allows setting a maximum correlation threshold.
    
    Attributes
    ----------
    method : str
        Correlation method to use ('pearson' or 'spearman')
    threshold : float
        Maximum allowed correlation between features
    keep_first : bool
        Whether to keep the first feature encountered in case of high correlation
        
    Methods
    -------
    fit(data) -> CorrelationThresholdSelector
        Compute inter-feature correlations and identify features to remove
    transform(data) -> FeatureSet
        Remove highly correlated features from the dataset
    """

    def __init__(self, method: str='pearson', threshold: float=0.95, keep_first: bool=True, name: Optional[str]=None):
        """
        Initialize the correlation threshold-based feature selector.
        
        Parameters
        ----------
        method : str, default='pearson'
            Correlation method to use ('pearson' or 'spearman')
        threshold : float, default=0.95
            Maximum allowed correlation between features. Features with higher
            correlations will be removed.
        keep_first : bool, default=True
            Whether to keep the first feature encountered in case of high correlation
        name : Optional[str], default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        if method not in ['pearson', 'spearman']:
            raise ValueError("Method must be either 'pearson' or 'spearman'")
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        self.method = method
        self.threshold = threshold
        self.keep_first = keep_first
        self._correlation_matrix = None
        self._selected_features = None

    def fit(self, data: FeatureSet, **kwargs) -> 'CorrelationThresholdSelector':
        """
        Compute inter-feature correlations and identify features to remove.
        
        Parameters
        ----------
        data : FeatureSet
            Input features for correlation computation
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        CorrelationThresholdSelector
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If threshold is not between 0 and 1
        """
        if not 0 <= self.threshold <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        X = data.features
        n_features = X.shape[1]
        if self.method == 'pearson':
            corr_matrix = np.corrcoef(X, rowvar=False)
        else:
            ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, X)
            corr_matrix = np.corrcoef(ranks, rowvar=False)
        np.fill_diagonal(corr_matrix, 1.0)
        self._correlation_matrix = corr_matrix
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)
        high_corr_pairs = np.where(abs_corr > self.threshold)
        to_remove = set()
        processed_pairs = set()
        for (i, j) in zip(high_corr_pairs[0], high_corr_pairs[1]):
            if i == j or (min(i, j), max(i, j)) in processed_pairs:
                continue
            processed_pairs.add((min(i, j), max(i, j)))
            if self.keep_first:
                to_remove.add(j)
            else:
                to_remove.add(i)
        self._selected_features = np.array([i not in to_remove for i in range(n_features)])
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Remove highly correlated features from the dataset.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed feature set with highly correlated features removed
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        """
        if self._selected_features is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'transform'.")
        selected_features = data.features[:, self._selected_features]
        selected_feature_names = None
        if data.feature_names:
            selected_feature_names = [name for (i, name) in enumerate(data.feature_names) if self._selected_features[i]]
        selected_feature_types = None
        if data.feature_types:
            selected_feature_types = [ftype for (i, ftype) in enumerate(data.feature_types) if self._selected_features[i]]
        transformed_data = FeatureSet(features=selected_features, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else {})
        return transformed_data

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for feature selection.
        
        Parameters
        ----------
        data : FeatureSet
            Transformed data to inverse transform
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Original feature set with removed features filled with zeros
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for feature selection
        """
        raise NotImplementedError('Inverse transformation is not supported for feature selection.')

    def get_support(self) -> np.ndarray:
        """
        Get the indices of selected features.
        
        Returns
        -------
        np.ndarray
            Boolean array indicating which features are selected
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        """
        if self._selected_features is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'get_support'.")
        return self._selected_features.copy()

    def get_correlation_matrix(self) -> np.ndarray:
        """
        Get the computed correlation matrix.
        
        Returns
        -------
        np.ndarray
            The correlation matrix of the input features
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        """
        if self._correlation_matrix is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'get_correlation_matrix'.")
        return self._correlation_matrix.copy()