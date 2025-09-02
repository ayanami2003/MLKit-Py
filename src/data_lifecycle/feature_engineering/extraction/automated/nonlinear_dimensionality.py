from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from .isomap_embedding import IsomapEmbeddingExtractor
from .locally_linear_embedding import LocallyLinearEmbeddingTransformer
from .diffusion_maps import DiffusionMapsEmbedding
from .stochastic_neighbor_embedding import StochasticNeighborEmbeddingTransformer
from .manifold_approximation import UniformManifoldApproximationProjection


class NonLinearDimensionalityReducer(BaseTransformer):

    def __init__(self, method: str='tsne', n_components: int=2, random_state: Optional[int]=None, **kwargs):
        """
        Initialize the non-linear dimensionality reducer.
        
        Parameters
        ----------
        method : str, default="tsne"
            The non-linear dimensionality reduction technique to use
        n_components : int, default=2
            Number of dimensions in the output space
        random_state : Optional[int], default=None
            Random seed for reproducible results
        **kwargs : dict
            Additional method-specific parameters
        """
        super().__init__(name=f'{method}_dimensionality_reducer')
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.parameters = kwargs
        self.method_params = kwargs
        self._is_fitted = False

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'NonLinearDimensionalityReducer':
        """
        Fit the non-linear dimensionality reducer to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the reducer on. Should be a FeatureSet or numpy array
            with shape (n_samples, n_features)
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        NonLinearDimensionalityReducer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the input data is invalid or incompatible
        NotImplementedError
            If the specified method is not supported
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.training_data_shape_ = X.shape
        params_with_random = self.parameters.copy()
        if self.random_state is not None:
            params_with_random['random_state'] = self.random_state
        params_without_random = {k: v for (k, v) in self.parameters.items() if k != 'random_state'}
        if self.method == 'isomap':
            self._reducer = IsomapEmbeddingExtractor(n_components=self.n_components, **self.parameters)
        elif self.method == 'lle':
            if self.n_components >= X.shape[1]:
                raise ValueError(f'For LLE, n_components ({self.n_components}) must be less than n_features ({X.shape[1]})')
            self._reducer = LocallyLinearEmbeddingTransformer(n_components=self.n_components, **self.parameters)
        elif self.method == 'diffusion_maps':
            diffusion_params = params_without_random.copy()
            if self.random_state is not None:
                diffusion_params['random_state'] = self.random_state
            self._reducer = DiffusionMapsEmbedding(n_components=self.n_components, **diffusion_params)
        elif self.method == 'tsne':
            tsne_params = params_with_random.copy()
            self._reducer = StochasticNeighborEmbeddingTransformer(n_components=self.n_components, **tsne_params)
        elif self.method == 'umap':
            umap_params = params_with_random.copy()
            self._reducer = UniformManifoldApproximationProjection(n_components=self.n_components, **umap_params)
        else:
            raise NotImplementedError(f"Method '{self.method}' is not yet implemented")
        self._reducer.fit(data, **kwargs)
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply non-linear dimensionality reduction to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Should be a FeatureSet or numpy array
            with shape (n_samples, n_features)
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Transformed data with reduced dimensionality
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or input data is invalid
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            sample_ids = data.sample_ids
        elif isinstance(data, np.ndarray):
            X = data
            sample_ids = None
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.training_data_shape_[1]:
            raise ValueError(f'Input data must have {self.training_data_shape_[1]} features, but got {X.shape[1]}')
        transformed_data = self._reducer.transform(data, **kwargs)
        if isinstance(data, FeatureSet) and hasattr(data, 'sample_ids'):
            if not hasattr(transformed_data, 'sample_ids') or transformed_data.sample_ids is None:
                transformed_data.sample_ids = data.sample_ids
        elif sample_ids is not None and (not hasattr(transformed_data, 'sample_ids') or transformed_data.sample_ids is None):
            transformed_data.sample_ids = sample_ids
        feature_names = None
        if hasattr(self._reducer, 'get_feature_names'):
            try:
                feature_names = self._reducer.get_feature_names()
                if feature_names is not None and len(feature_names) == self.n_components:
                    transformed_data.feature_names = feature_names
                else:
                    feature_names = None
            except:
                feature_names = None
        if feature_names is None:
            if self.method == 'diffusion_maps':
                feature_names = [f'dm_component_{i}' for i in range(self.n_components)]
            elif self.method == 'isomap':
                feature_names = [f'isomap_component_{i}' for i in range(self.n_components)]
            elif self.method == 'lle':
                feature_names = [f'lle_component_{i}' for i in range(self.n_components)]
            elif self.method == 'tsne':
                feature_names = [f'tsne_component_{i}' for i in range(self.n_components)]
            elif self.method == 'umap':
                feature_names = [f'umap_component_{i}' for i in range(self.n_components)]
            else:
                feature_names = [f'{self.method}_component_{i}' for i in range(self.n_components)]
            transformed_data.feature_names = feature_names
        return transformed_data

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Attempt to reconstruct the original data from the reduced representation.
        
        Note: Not all non-linear dimensionality reduction methods support inverse transformation.
        For methods that don't support it, this will raise NotImplementedError.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Low-dimensional data to reconstruct
        **kwargs : dict
            Additional reconstruction parameters
            
        Returns
        -------
        FeatureSet
            Reconstructed high-dimensional data
            
        Raises
        ------
        NotImplementedError
            If the selected method does not support inverse transformation
        ValueError
            If the input data is invalid or transformer is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if self.method in ['tsne']:
            raise NotImplementedError(f"Method '{self.method}' does not support inverse transformation")
        if not hasattr(self._reducer, 'inverse_transform'):
            raise NotImplementedError(f"Method '{self.method}' does not support inverse transformation")
        try:
            result = self._reducer.inverse_transform(data, **kwargs)
            return result
        except (NotImplementedError, AttributeError):
            raise NotImplementedError(f"Method '{self.method}' does not support inverse transformation")

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get names for the reduced features.
        
        Returns
        -------
        Optional[List[str]]
            List of feature names for the reduced dimensions, or None if not available
        """
        if not self._is_fitted:
            return None
        if hasattr(self._reducer, 'get_feature_names'):
            try:
                feature_names = self._reducer.get_feature_names()
                if feature_names is not None and len(feature_names) == self.n_components:
                    return feature_names
            except:
                pass
        if self.method == 'diffusion_maps':
            return [f'dm_component_{i}' for i in range(self.n_components)]
        elif self.method == 'isomap':
            return [f'isomap_component_{i}' for i in range(self.n_components)]
        elif self.method == 'lle':
            return [f'lle_component_{i}' for i in range(self.n_components)]
        elif self.method == 'tsne':
            return [f'tsne_component_{i}' for i in range(self.n_components)]
        elif self.method == 'umap':
            return [f'umap_component_{i}' for i in range(self.n_components)]
        else:
            return [f'{self.method}_component_{i}' for i in range(self.n_components)]