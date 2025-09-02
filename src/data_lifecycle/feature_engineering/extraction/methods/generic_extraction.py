import numpy as np
from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import Optional, List, Union, Callable, Dict, Any
from functools import partial
_supported_methods: Dict[str, Callable] = {'mean': lambda X, axis=None: np.mean(X, axis=axis), 'std': lambda X, axis=None: np.std(X, axis=axis), 'var': lambda X, axis=None: np.var(X, axis=axis), 'min': lambda X, axis=None: np.min(X, axis=axis), 'max': lambda X, axis=None: np.max(X, axis=axis), 'median': lambda X, axis=None: np.median(X, axis=axis), 'sum': lambda X, axis=None: np.sum(X, axis=axis), 'ptp': lambda X, axis=None: np.ptp(X, axis=axis)}
_supported_methods: Dict[str, Callable] = {'mean': lambda X, axis=None: np.mean(X, axis=axis), 'std': lambda X, axis=None: np.std(X, axis=axis), 'var': lambda X, axis=None: np.var(X, axis=axis), 'min': lambda X, axis=None: np.min(X, axis=axis), 'max': lambda X, axis=None: np.max(X, axis=axis), 'median': lambda X, axis=None: np.median(X, axis=axis), 'sum': lambda X, axis=None: np.sum(X, axis=axis), 'ptp': lambda X, axis=None: np.ptp(X, axis=axis)}

class GenericFeatureExtractor(BaseTransformer):
    _supported_methods: Dict[str, Callable] = {'mean': lambda X, axis=None: np.mean(X, axis=axis), 'std': lambda X, axis=None: np.std(X, axis=axis), 'var': lambda X, axis=None: np.var(X, axis=axis), 'min': lambda X, axis=None: np.min(X, axis=axis), 'max': lambda X, axis=None: np.max(X, axis=axis), 'median': lambda X, axis=None: np.median(X, axis=axis), 'sum': lambda X, axis=None: np.sum(X, axis=axis), 'ptp': lambda X, axis=None: np.ptp(X, axis=axis)}

    def __init__(self, extraction_methods: List[str], name: Optional[str]=None):
        """
        Initialize the GenericFeatureExtractor.
        
        Parameters
        ----------
        extraction_methods : List[str]
            List of feature extraction method names to apply
        name : Optional[str]
            Name of the transformer instance
            
        Raises
        ------
        ValueError
            If any extraction method is not supported
        """
        super().__init__(name)
        unsupported_methods = set(extraction_methods) - set(self._supported_methods.keys())
        if unsupported_methods:
            raise ValueError(f'Unsupported extraction methods: {unsupported_methods}. Supported methods: {list(self._supported_methods.keys())}')
        self.extraction_methods = extraction_methods
        self.feature_names: Optional[List[str]] = None
        self.is_fitted_ = False
        self._input_feature_count: Optional[int] = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'GenericFeatureExtractor':
        """
        Fit the feature extractor to the input data.
        
        This method validates the input data and prepares the transformer for
        feature extraction by generating appropriate feature names.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the extractor on
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        GenericFeatureExtractor
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
            input_feature_names = data.feature_names
        else:
            X = data
            input_feature_names = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        self._input_feature_count = X.shape[1]
        feature_names = []
        if input_feature_names is not None:
            for method in self.extraction_methods:
                for feat_name in input_feature_names:
                    feature_names.append(f'{method}_of_{feat_name}')
        else:
            temp_feature_names = [f'x{i + 1}' for i in range(X.shape[1])]
            for method in self.extraction_methods:
                for feat_name in temp_feature_names:
                    feature_names.append(f'{method}_of_{feat_name}')
        self.feature_names = feature_names
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply feature extraction to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed data with extracted features
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            sample_ids = data.sample_ids
            metadata = data.metadata or {}
        else:
            X = data
            sample_ids = None
            metadata = {}
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if X.shape[1] != self._input_feature_count:
            raise ValueError(f'Number of features in transform data ({X.shape[1]}) does not match fitted data ({self._input_feature_count})')
        extracted_features_list = []
        for method_name in self.extraction_methods:
            method_func = self._supported_methods[method_name]
            method_results = method_func(X, axis=0)
            extracted_features_list.append(method_results)
        if extracted_features_list:
            transformed_data = np.column_stack(extracted_features_list)
        else:
            transformed_data = np.empty((X.shape[0], 0))
        n_samples_out = transformed_data.shape[0]
        if sample_ids is not None:
            new_sample_ids = [f'extracted_{i}' for i in range(n_samples_out)]
        else:
            new_sample_ids = None
        return FeatureSet(features=transformed_data, feature_names=self.feature_names, feature_types=['numeric'] * len(self.feature_names) if self.feature_names else None, sample_ids=new_sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation if possible.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to invert
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Data in original format (if invertible)
            
        Raises
        ------
        NotImplementedError
            Feature extraction is generally not invertible
        """
        raise NotImplementedError('Generic feature extraction is not invertible')

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get names of extracted features.
        
        Returns
        -------
        Optional[List[str]]
            List of feature names or None if not available
        """
        return self.feature_names