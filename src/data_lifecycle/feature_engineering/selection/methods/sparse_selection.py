from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, Union
from sklearn.linear_model import Lasso, ElasticNet

class SparseFeatureSelector(BaseTransformer):

    def __init__(self, method: str='l1_regularization', alpha: float=1.0, l1_ratio: float=0.5, n_features: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the SparseFeatureSelector.
        
        Parameters
        ----------
        method : str, default='l1_regularization'
            The sparse selection method to use. Options include:
            - 'l1_regularization': Uses L1 penalty to induce sparsity
            - 'elastic_net': Combines L1 and L2 penalties
        alpha : float, default=1.0
            Regularization strength. Higher values lead to more sparsity.
        l1_ratio : float, default=0.5
            Mixing parameter for elastic net (0 <= l1_ratio <= 1).
            1 is pure L1, 0 is pure L2.
        n_features : int, optional
            Number of features to select. If specified, overrides alpha.
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_features = n_features
        self.feature_names_in_ = None
        self.selected_features_ = None
        self.selected_indices_ = None
        self.coef_ = None

    def fit(self, data: Union[FeatureSet, DataBatch], y: Optional[np.ndarray]=None, **kwargs) -> 'SparseFeatureSelector':
        """
        Fit the sparse feature selector to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input features to fit the selector on
        y : np.ndarray, optional
            Target values for supervised feature selection
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        SparseFeatureSelector
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data format is unsupported or fitting fails
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self.feature_names_in_ = data.feature_names
            if y is None:
                y = kwargs.get('y', None)
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            self.feature_names_in_ = data.feature_names
            if y is None:
                y = data.labels
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if y is None:
            y = np.var(X, axis=0).reshape(-1, 1) if X.shape[0] > 1 else np.ones(X.shape[1]).reshape(-1, 1)
            if y.shape[1] == 1 and y.shape[0] == X.shape[1]:
                y = np.mean(X, axis=1)
            else:
                y = np.mean(X, axis=1)
        if self.method == 'l1_regularization':
            model = Lasso(alpha=self.alpha, random_state=0)
        elif self.method == 'elastic_net':
            model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=0)
        else:
            raise ValueError(f"Unsupported method: {self.method}. Supported methods: 'l1_regularization', 'elastic_net'")
        model.fit(X, y)
        self.coef_ = model.coef_.copy()
        if self.n_features is not None and self.n_features > 0:
            coef_abs = np.abs(self.coef_)
            self.selected_indices_ = np.argsort(coef_abs)[::-1][:self.n_features]
        else:
            self.selected_indices_ = np.where(self.coef_ != 0)[0]
        if self.feature_names_in_ is None:
            self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Apply sparse feature selection to transform the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input features to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Transformed feature set with only selected features
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or data shape mismatch
        """
        if self.selected_indices_ is None:
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
            feature_types = data.feature_types
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
            sample_ids = None
            metadata = None
            quality_scores = None
            feature_types = None
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if X.shape[1] != len(self.coef_):
            raise ValueError(f'Data has {X.shape[1]} features, but transformer was fitted on {len(self.coef_)} features.')
        if len(self.selected_indices_) == 0:
            selected_X = np.empty((X.shape[0], 0))
            selected_feature_names = []
            selected_feature_types = [] if feature_types is not None else None
        else:
            selected_X = X[:, self.selected_indices_]
            if feature_names is not None:
                selected_feature_names = [feature_names[i] for i in self.selected_indices_]
            else:
                selected_feature_names = [f'feature_{i}' for i in self.selected_indices_]
            if feature_types is not None:
                selected_feature_types = [feature_types[i] for i in self.selected_indices_]
            else:
                selected_feature_types = None
        transformed_data = FeatureSet(features=selected_X, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        return transformed_data

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reverse the transformation (not typically supported for sparse selection).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data to inverse transform
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Feature set with original dimensionality (zeros for removed features)
            
        Raises
        ------
        NotImplementedError
            If inverse transformation is not supported
        """
        raise NotImplementedError('SparseFeatureSelector does not support inverse transformation')

    def get_support(self, indices: bool=False) -> Union[np.ndarray, list]:
        """
        Get a mask or indices of selected features.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, return indices of selected features; 
            if False, return a boolean mask.
            
        Returns
        -------
        Union[np.ndarray, list]
            Boolean mask or list of indices for selected features
        """
        if self.selected_indices_ is None:
            raise ValueError('Transformer has not been fitted yet.')
        if indices:
            return self.selected_indices_.tolist()
        else:
            mask = np.zeros(len(self.coef_), dtype=bool)
            mask[self.selected_indices_] = True
            return mask