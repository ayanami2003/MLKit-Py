from typing import Optional, List, Union
import numpy as np
from sklearn.manifold import Isomap
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class IsomapEmbeddingExtractor(BaseTransformer):

    def __init__(self, n_components: int=2, n_neighbors: int=5, radius: Optional[float]=None, metric: Union[str, callable]='minkowski', metric_params: Optional[dict]=None, eigen_solver: str='auto', tol: float=0, max_iter: Optional[int]=None, path_method: str='auto', neighbors_algorithm: str='auto', n_jobs: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.metric = metric
        self.metric_params = metric_params
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'IsomapEmbeddingExtractor':
        """
        Fit the Isomap embedding model to the input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to fit the model on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional fitting parameters (ignored).
            
        Returns
        -------
        IsomapEmbeddingExtractor
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the input data is invalid or incompatible.
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
        n_neighbors_param = self.n_neighbors if self.radius is None else None
        radius_param = self.radius
        self._isomap = Isomap(n_components=self.n_components, n_neighbors=n_neighbors_param, radius=radius_param, metric=self.metric, metric_params=self.metric_params, eigen_solver=self.eigen_solver, tol=self.tol, max_iter=self.max_iter, path_method=self.path_method, neighbors_algorithm=self.neighbors_algorithm, n_jobs=self.n_jobs)
        self._isomap.fit(X)
        self.embedding_ = self._isomap.embedding_
        try:
            self.kernel_pca_ = self._isomap.kernel_pca_
        except AttributeError:
            self.kernel_pca_ = None
        self.feature_names_ = [f'isomap_component_{i}' for i in range(self.n_components)]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the Isomap embedding to the input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to transform. Must have the same number of features as the training data.
        **kwargs : dict
            Additional transformation parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed data with reduced dimensionality.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or input data is incompatible.
        """
        if not hasattr(self, '_isomap'):
            raise ValueError("This IsomapEmbeddingExtractor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.training_data_shape_[1]:
            raise ValueError(f'Input data must have {self.training_data_shape_[1]} features, but got {X.shape[1]}')
        transformed_data = self._isomap.transform(X)
        return FeatureSet(features=transformed_data, feature_names=self.feature_names_)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform data back to the original space (not implemented for Isomap).
        
        Isomap embedding is a non-linear transformation that generally cannot
        be inverted exactly. This method raises NotImplementedError.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Embedded data to transform back (will raise NotImplementedError).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            This method always raises NotImplementedError.
            
        Raises
        ------
        NotImplementedError
            Isomap embedding does not support inverse transformation.
        """
        raise NotImplementedError('Isomap embedding does not support inverse transformation.')

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get names of the extracted features.
        
        Parameters
        ----------
        input_features : list of str, optional
            Names of input features (ignored as Isomap creates new feature names).
            
        Returns
        -------
        list of str
            Names of the extracted embedding features.
        """
        if not hasattr(self, 'feature_names_'):
            raise ValueError("This IsomapEmbeddingExtractor instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        return self.feature_names_