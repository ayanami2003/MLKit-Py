from typing import Optional, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ExplainedVarianceRatioExtractor(BaseTransformer):
    """
    Extract features based on explained variance ratio, typically used for PCA or similar 
    dimensionality reduction techniques. This transformer identifies the most important 
    components that contribute to the variance in the dataset.

    The transformer computes the explained variance ratio for each feature or component 
    and retains those that meet a specified threshold or count. It is primarily intended 
    for use with linear transformations where variance contribution per component is meaningful.

    Attributes:
        threshold (Optional[float]): Minimum explained variance ratio for a feature to be retained.
                                     If None, uses `n_components` instead.
        n_components (Optional[int]): Number of top components to retain based on explained variance.
                                      If None, uses `threshold` instead.
        name (Optional[str]): Optional name for the transformer instance.
    """

    def __init__(self, threshold: Optional[float]=None, n_components: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        if threshold is None and n_components is None:
            raise ValueError("At least one of 'threshold' or 'n_components' must be specified.")
        self.threshold = threshold
        self.n_components = n_components
        self._explained_variance_ratios: Optional[np.ndarray] = None
        self._selected_indices: Optional[List[int]] = None
        self._components_: Optional[np.ndarray] = None
        self._mean_: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'ExplainedVarianceRatioExtractor':
        """
        Fit the extractor to compute explained variance ratios for the input features.

        This method analyzes the variance contribution of each feature in the input dataset
        and stores the computed ratios for later use in transformation.

        Args:
            data (FeatureSet): Input feature set with `features` attribute containing a 2D array.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ExplainedVarianceRatioExtractor: Returns self for method chaining.

        Raises:
            ValueError: If neither `threshold` nor `n_components` is specified.
        """
        X = data.features
        self._mean_ = np.mean(X, axis=0)
        X_centered = X - self._mean_
        cov_matrix = np.cov(X_centered, rowvar=False)
        (eigenvals, eigenvecs) = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        total_variance = np.sum(eigenvals)
        if total_variance == 0:
            self._explained_variance_ratios = np.zeros_like(eigenvals)
        else:
            self._explained_variance_ratios = eigenvals / total_variance
        self._components_ = eigenvecs
        if self.threshold is not None:
            cumulative_variance_ratio = np.cumsum(self._explained_variance_ratios)
            self._selected_indices = np.where(cumulative_variance_ratio <= self.threshold)[0].tolist()
            if len(self._selected_indices) < len(self._explained_variance_ratios):
                next_idx = len(self._selected_indices)
                if next_idx < len(self._explained_variance_ratios):
                    if cumulative_variance_ratio[next_idx] <= self.threshold or len(self._selected_indices) == 0:
                        self._selected_indices.append(next_idx)
        else:
            n_features = min(self.n_components, len(self._explained_variance_ratios))
            self._selected_indices = list(range(n_features))
        self._is_fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform the input data by selecting features based on explained variance ratios.

        Applies the previously computed explained variance thresholds to reduce the feature set,
        returning a new FeatureSet with only the selected features.

        Args:
            data (FeatureSet): Input feature set to transform.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            FeatureSet: Transformed feature set with reduced dimensionality.
        """
        if not self._is_fitted or self._selected_indices is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        X_centered = data.features - self._mean_
        selected_components = self._components_[:, self._selected_indices]
        X_selected = X_centered @ selected_components
        new_feature_names = None
        if data.feature_names is not None:
            new_feature_names = [f'PC{i + 1}' for i in range(len(self._selected_indices))]
        new_feature_types = None
        if data.feature_types is not None:
            new_feature_types = ['continuous'] * len(self._selected_indices)
        return FeatureSet(features=X_selected, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the transformation by restoring the original feature space.

        Since explained variance-based selection is generally not reversible, this method
        raises a NotImplementedError unless overridden in subclasses with specific behavior.

        Args:
            data (FeatureSet): Transformed feature set to invert.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            FeatureSet: Feature set restored to original dimensions (if supported).

        Raises:
            NotImplementedError: Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for variance-based feature selection.')

    def get_explained_variance_ratios(self) -> Optional[np.ndarray]:
        """
        Retrieve the computed explained variance ratios for all features.

        Returns:
            Optional[np.ndarray]: Array of explained variance ratios, or None if not yet fitted.
        """
        return self._explained_variance_ratios if self._is_fitted else None

    def get_selected_feature_indices(self) -> Optional[List[int]]:
        """
        Get indices of features selected based on explained variance criteria.

        Returns:
            Optional[List[int]]: List of selected feature indices, or None if not yet fitted.
        """
        return self._selected_indices if self._is_fitted else None