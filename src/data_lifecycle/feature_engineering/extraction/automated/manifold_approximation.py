from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, List, Union
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

class UniformManifoldApproximationProjection(BaseTransformer):

    def __init__(self, n_components: int=2, n_neighbors: int=15, min_dist: float=0.1, metric: str='euclidean', random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

    def fit(self, data: FeatureSet, **kwargs) -> 'UniformManifoldApproximationProjection':
        """
        Fit the UMAP transformer to the input data.

        Parameters
        ----------
        data : FeatureSet
            Input feature set to fit the transformer on.
        **kwargs : dict
            Additional parameters for fitting.

        Returns
        -------
        UniformManifoldApproximationProjection
            Self instance for method chaining.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance')
        if self.n_components <= 0:
            raise ValueError('n_components must be positive')
        if self.n_neighbors <= 0:
            raise ValueError('n_neighbors must be positive')
        if self.min_dist < 0:
            raise ValueError('min_dist must be non-negative')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._training_data = data.features.copy()
        self.embedding_ = self._compute_umap_embedding(data.features)
        self.feature_names_ = [f'umap_dim_{i}' for i in range(self.n_components)]
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply UMAP transformation to reduce dimensionality.

        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs : dict
            Additional parameters for transformation.

        Returns
        -------
        FeatureSet
            Transformed feature set with reduced dimensions.
        """
        pass

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Attempt to reconstruct the original data from the embedding.

        Note: UMAP does not provide a true inverse transformation, so this
        will typically be an approximation or may raise an exception.

        Parameters
        ----------
        data : FeatureSet
            Low-dimensional feature set to reconstruct.
        **kwargs : dict
            Additional parameters for inverse transformation.

        Returns
        -------
        FeatureSet
            Approximated reconstruction of the original high-dimensional data.
        """
        pass

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Generate feature names for the output embedding dimensions.

        Parameters
        ----------
        input_features : List[str], optional
            Names of input features (not directly used in UMAP but included for API consistency).

        Returns
        -------
        List[str]
            Names of the output embedding dimensions.
        """
        pass