from typing import List, Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ThermometerEncoder(BaseTransformer):

    def __init__(self, handle_unknown: str='error', categories: Union[str, List[List[int]]]='auto', name: Optional[str]=None):
        super().__init__(name=name)
        if handle_unknown not in ['error', 'ignore']:
            raise ValueError("handle_unknown must be either 'error' or 'ignore'")
        self.handle_unknown = handle_unknown
        self.categories = categories

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ThermometerEncoder':
        """
        Fit the thermometer encoder to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to be encoded.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        ThermometerEncoder
            Fitted encoder instance.
            
        Raises
        ------
        ValueError
            If data contains non-integer values or categories are inconsistent.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names_in = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            self._feature_names_in = None
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        self.n_features_in_ = X.shape[1]
        if self.categories == 'auto':
            self._fitted_categories = []
            for i in range(self.n_features_in_):
                unique_vals = np.unique(X[:, i])
                categories = [val for val in unique_vals if not (isinstance(val, float) and np.isnan(val))]
                for val in categories:
                    if not isinstance(val, (int, np.integer)):
                        try:
                            int_val = int(val)
                            if int_val != float(val):
                                raise ValueError(f'All categories must be integers. Found non-integer value: {val}')
                        except (ValueError, TypeError):
                            raise ValueError(f'All categories must be integers. Found non-integer value: {val}')
                categories = sorted([int(val) for val in categories])
                self._fitted_categories.append(np.array(categories))
        else:
            if len(self.categories) != self.n_features_in_:
                raise ValueError(f'Number of category lists ({len(self.categories)}) must match number of features ({self.n_features_in_})')
            self._fitted_categories = [np.array([int(cat) for cat in cats]) for cats in self.categories]
            for cats in self._fitted_categories:
                cats_sorted = np.sort(cats)
                if not np.array_equal(cats, cats_sorted):
                    raise ValueError(f'Each category list must contain sorted integers. Found: {cats.tolist()}')
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Transform the input data using thermometer encoding.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to be transformed.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet or np.ndarray
            Transformed data with thermometer-encoded features.
            
        Raises
        ------
        ValueError
            If handle_unknown is 'error' and unknown categories are encountered.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError('Encoder has not been fitted yet. Call fit() first.')
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
            return_feature_set = True
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
            return_feature_set = False
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        (n_samples, n_features) = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(f'Number of features in transform data ({n_features}) does not match fitted data ({self.n_features_in_})')
        X_int = np.empty_like(X, dtype=object)
        for i in range(n_features):
            for j in range(n_samples):
                val = X[j, i]
                if isinstance(val, float) and np.isnan(val):
                    X_int[j, i] = np.nan
                else:
                    try:
                        X_int[j, i] = int(val)
                    except (ValueError, TypeError):
                        raise ValueError(f'All values must be convertible to integers. Found: {val}')
        encoded_features_list = []
        new_feature_names = []
        valid_rows = np.ones(n_samples, dtype=bool)
        for feature_idx in range(n_features):
            categories = self._fitted_categories[feature_idx]
            n_categories = len(categories)
            n_encoded_cols = max(1, n_categories - 1) if n_categories > 0 else 0
            encoded_cols = np.zeros((n_samples, n_encoded_cols), dtype=int)
            feature_valid_rows = np.ones(n_samples, dtype=bool)
            for sample_idx in range(n_samples):
                val = X_int[sample_idx, feature_idx]
                if isinstance(val, float) and np.isnan(val):
                    continue
                val = int(val)
                if val in categories:
                    pos = np.where(categories == val)[0][0]
                    if pos > 0 and n_encoded_cols > 0:
                        encoded_cols[sample_idx, :min(pos, n_encoded_cols)] = 1
                elif self.handle_unknown == 'error':
                    raise ValueError(f'Unknown category {val} encountered in feature {feature_idx}. Known categories: {categories.tolist()}')
                elif self.handle_unknown == 'ignore':
                    feature_valid_rows[sample_idx] = False
                    encoded_cols[sample_idx, :] = 0
            valid_rows &= feature_valid_rows
            encoded_features_list.append(encoded_cols)
            orig_name = feature_names[feature_idx] if feature_names else f'feature_{feature_idx}'
            for col_idx in range(n_encoded_cols):
                new_feature_names.append(f'{orig_name}_thermometer_{col_idx}')
        if encoded_features_list:
            encoded_features = np.concatenate(encoded_features_list, axis=1, dtype=int)
            if self.handle_unknown == 'ignore':
                pass
        else:
            encoded_features = np.empty((n_samples, 0), dtype=int)
        metadata['encoding_type'] = 'thermometer_encoding'
        metadata['original_shape'] = (n_samples, n_features)
        if feature_names:
            metadata['original_feature_names'] = feature_names
        if return_feature_set:
            return FeatureSet(features=encoded_features, feature_names=new_feature_names, feature_types=['numeric'] * len(new_feature_names), sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return encoded_features

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Convert thermometer-encoded data back to original categorical values.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Thermometer-encoded data to convert back.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet or np.ndarray
            Data with original categorical values restored.
            
        Raises
        ------
        ValueError
            If the encoded data does not conform to valid thermometer encoding patterns.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError('Encoder has not been fitted yet. Call fit() first.')
        if isinstance(data, FeatureSet):
            X = data.features
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
            return_feature_set = True
        elif isinstance(data, np.ndarray):
            X = data
            sample_ids = None
            metadata = {}
            quality_scores = {}
            return_feature_set = False
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        n_samples = X.shape[0]
        if isinstance(data, FeatureSet) and 'original_feature_names' in (data.metadata or {}):
            original_feature_names = data.metadata['original_feature_names']
        elif hasattr(self, '_feature_names_in') and self._feature_names_in is not None:
            original_feature_names = self._feature_names_in
        else:
            original_feature_names = [f'feature_{i}' for i in range(self.n_features_in_)]
        expected_cols_per_feature = [max(1, len(categories) - 1) if len(categories) > 0 else 0 for categories in self._fitted_categories]
        total_expected_cols = sum(expected_cols_per_feature)
        if X.shape[1] != total_expected_cols:
            raise ValueError(f'Number of columns in input data ({X.shape[1]}) does not match expected encoded columns ({total_expected_cols})')
        decoded_features_list = []
        col_offset = 0
        for (feature_idx, n_encoded_cols) in enumerate(expected_cols_per_feature):
            categories = self._fitted_categories[feature_idx]
            if n_encoded_cols == 0:
                decoded_col = np.zeros(n_samples, dtype=int)
                decoded_features_list.append(decoded_col)
            else:
                encoded_feature_cols = X[:, col_offset:col_offset + n_encoded_cols]
                decoded_col = np.zeros(n_samples, dtype=int)
                for sample_idx in range(n_samples):
                    encoded_row = encoded_feature_cols[sample_idx]
                    count = 0
                    for i in range(len(encoded_row)):
                        if encoded_row[i] == 1:
                            count += 1
                        else:
                            break
                    valid_pattern = True
                    for i in range(count, len(encoded_row)):
                        if encoded_row[i] != 0:
                            valid_pattern = False
                            break
                    if not valid_pattern:
                        raise ValueError(f'Invalid thermometer encoding pattern found at sample {sample_idx} for feature {feature_idx}: {encoded_row}')
                    if len(categories) == 1:
                        decoded_value = categories[0]
                    elif count < len(categories):
                        decoded_value = categories[count]
                    else:
                        decoded_value = categories[-1]
                    decoded_col[sample_idx] = decoded_value
                decoded_features_list.append(decoded_col)
            col_offset += n_encoded_cols
        if decoded_features_list:
            decoded_features = np.column_stack(decoded_features_list)
        else:
            decoded_features = np.empty((n_samples, 0), dtype=int)
        if return_feature_set:
            inverted_metadata = metadata.copy()
            if 'encoding_type' in inverted_metadata:
                inverted_metadata.pop('encoding_type')
            if 'original_shape' in inverted_metadata:
                inverted_metadata.pop('original_shape')
            if 'original_feature_names' in inverted_metadata:
                inverted_metadata.pop('original_feature_names')
            return FeatureSet(features=decoded_features, feature_names=original_feature_names, feature_types=['categorical'] * len(original_feature_names), sample_ids=sample_ids, metadata=inverted_metadata, quality_scores=quality_scores)
        else:
            return decoded_features

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names for the thermometer-encoded features.
        
        Parameters
        ----------
        input_features : Optional[List[str]], default=None
            Input feature names. If None, generated names will be used.
            
        Returns
        -------
        List[str]
            Output feature names for all thermometer-encoded columns.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError('Encoder has not been fitted yet. Call fit() first.')
        if input_features is not None:
            if len(input_features) != self.n_features_in_:
                raise ValueError(f'Length of input_features ({len(input_features)}) does not match number of features ({self.n_features_in_})')
            feature_names_in = input_features
        elif hasattr(self, '_feature_names_in') and self._feature_names_in is not None:
            feature_names_in = self._feature_names_in
        else:
            feature_names_in = [f'feature_{i}' for i in range(self.n_features_in_)]
        output_names = []
        for (feature_idx, feature_name) in enumerate(feature_names_in):
            categories = self._fitted_categories[feature_idx]
            n_categories = len(categories)
            n_encoded_cols = max(1, n_categories - 1) if n_categories > 0 else 0
            for col_idx in range(n_encoded_cols):
                output_names.append(f'{feature_name}_thermometer_{col_idx}')
        return output_names