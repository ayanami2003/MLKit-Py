from typing import List, Optional, Dict, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class LabelEncoder(BaseTransformer):

    def __init__(self, handle_unknown: str='error', name: Optional[str]=None):
        super().__init__(name=name)
        if handle_unknown not in ['error', 'ignore']:
            raise ValueError("handle_unknown must be either 'error' or 'ignore'")
        self.handle_unknown = handle_unknown
        self.categories: Dict[int, List[str]] = {}

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'LabelEncoder':
        """
        Learn the categories from the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        LabelEncoder
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data contains unsupported data types
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._input_feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            self._input_feature_names = None
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        n_features = X.shape[1]
        self.categories = {}
        for i in range(n_features):
            column_values = X[:, i]
            seen = set()
            unique_vals = []
            for val in column_values:
                str_val = str(val) if not (isinstance(val, float) and np.isnan(val)) else None
                if str_val is not None and str_val not in seen:
                    seen.add(str_val)
                    unique_vals.append(str_val)
            self.categories[i] = sorted(unique_vals)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the input data using learned encodings.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed data with categorical features encoded as integers
            
        Raises
        ------
        ValueError
            If handle_unknown is 'error' and unknown categories are encountered
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if not self.categories:
            raise ValueError("LabelEncoder has not been fitted yet. Call 'fit' before using this transformer.")
        (n_samples, n_features) = X.shape
        if n_features != len(self.categories):
            raise ValueError(f'Number of features ({n_features}) does not match fitted data ({len(self.categories)})')
        transformed_X = np.full(X.shape, -1, dtype=int)
        for i in range(n_features):
            category_to_int = {cat: idx for (idx, cat) in enumerate(self.categories[i])}
            for j in range(n_samples):
                value = X[j, i]
                if isinstance(value, float) and np.isnan(value):
                    transformed_X[j, i] = -1
                    continue
                str_value = str(value)
                if str_value in category_to_int:
                    transformed_X[j, i] = category_to_int[str_value]
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{str_value}' encountered in column {i}")
                else:
                    transformed_X[j, i] = -1
        if isinstance(data, np.ndarray) and feature_names is None:
            if hasattr(self, '_input_feature_names') and self._input_feature_names is not None:
                feature_names = self._input_feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(n_features)]
        return FeatureSet(features=transformed_X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert encoded integers back to original categories.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Encoded data to convert back to categories
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Data with integers converted back to original categories
            
        Raises
        ------
        ValueError
            If encoded values are outside the expected range
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if not self.categories:
            raise ValueError("LabelEncoder has not been fitted yet. Call 'fit' before using this transformer.")
        (n_samples, n_features) = X.shape
        if n_features != len(self.categories):
            raise ValueError(f'Number of features ({n_features}) does not match fitted data ({len(self.categories)})')
        transformed_X = np.empty_like(X, dtype=object)
        for i in range(n_features):
            categories = self.categories[i]
            n_categories = len(categories)
            for j in range(n_samples):
                value = X[j, i]
                if not isinstance(value, (int, np.integer)) or value == -1:
                    transformed_X[j, i] = np.nan
                    continue
                if value < 0 or value >= n_categories:
                    raise ValueError(f'Encoded value {value} in column {i} is outside the expected range [0, {n_categories - 1}]')
                transformed_X[j, i] = categories[value]
        if isinstance(data, np.ndarray) and feature_names is None:
            if hasattr(self, '_input_feature_names') and self._input_feature_names is not None:
                feature_names = self._input_feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(n_features)]
        return FeatureSet(features=transformed_X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names after encoding.
        
        Parameters
        ----------
        input_features : Optional[List[str]], default=None
            Input feature names. If None, generated names will be used
            
        Returns
        -------
        List[str]
            Output feature names
        """
        if not self.categories:
            raise ValueError("LabelEncoder has not been fitted yet. Call 'fit' before using this method.")
        n_features = len(self.categories)
        if input_features is None:
            if hasattr(self, '_input_feature_names') and self._input_feature_names is not None:
                return self._input_feature_names
            else:
                return [f'feature_{i}' for i in range(n_features)]
        else:
            if len(input_features) != n_features:
                raise ValueError(f'Length of input_features ({len(input_features)}) does not match number of features ({n_features})')
            return input_features