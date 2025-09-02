from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class CyclicalFeatureEncoder(BaseTransformer):

    def __init__(self, periods: dict, feature_names: Optional[List[str]]=None, drop_original: bool=True, name: Optional[str]=None):
        """
        Initialize the CyclicalFeatureEncoder.

        Parameters
        ----------
        periods : dict
            Dictionary mapping feature names to their cycle periods
        feature_names : Optional[List[str]], default=None
            Names of features to encode. If None, all features with specified periods will be encoded
        drop_original : bool, default=True
            Whether to drop the original cyclical features after encoding
        name : Optional[str], default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.periods = periods
        self.feature_names = feature_names
        self.drop_original = drop_original

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'CyclicalFeatureEncoder':
        """
        Fit the encoder to the input data.

        For cyclical encoding, fitting typically just validates that the required features exist
        and have appropriate data types.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing cyclical features to encode
        **kwargs : dict
            Additional parameters (ignored)

        Returns
        -------
        CyclicalFeatureEncoder
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            self._input_is_feature_set = True
            X = data.features
            feature_names = data.feature_names
        else:
            self._input_is_feature_set = False
            X = data
            if self.feature_names is None:
                raise ValueError('feature_names must be provided when fitting with numpy array')
            feature_names = self.feature_names
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        n_features = X.shape[1]
        if feature_names is None:
            if self.feature_names is not None:
                feature_names = self.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(n_features)]
        self._feature_names_in = feature_names
        if isinstance(data, FeatureSet):
            self.feature_names = feature_names
        if self.feature_names is None:
            self._features_to_encode = [f for f in self.periods.keys() if f in feature_names]
        else:
            self._features_to_encode = [f for f in self.feature_names if f in self.periods]
        missing_features = set(self._features_to_encode) - set(feature_names)
        if missing_features:
            raise ValueError(f'The following features were not found in the input data: {missing_features}')
        missing_periods = set(self._features_to_encode) - set(self.periods.keys())
        if missing_periods:
            raise ValueError(f'No periods defined for the following features: {missing_periods}')
        self._feature_indices = []
        for feature in self._features_to_encode:
            try:
                idx = feature_names.index(feature)
                self._feature_indices.append(idx)
            except ValueError:
                raise ValueError(f"Feature '{feature}' not found in input data")
        for idx in self._feature_indices:
            if not np.issubdtype(X[:, idx].dtype, np.number):
                raise ValueError(f'Feature at index {idx} is not numeric and cannot be encoded as cyclical')
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply cyclical encoding to the specified features.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing cyclical features to encode
        **kwargs : dict
            Additional parameters (ignored)

        Returns
        -------
        FeatureSet
            Transformed data with cyclical features encoded as sine/cosine components
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names.copy() if data.feature_names is not None else None
            feature_types = data.feature_types.copy() if data.feature_types is not None else None
        else:
            X = data.copy()
            feature_names = None
            feature_types = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        if not self._features_to_encode:
            return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types)
        new_feature_names = []
        new_feature_types = [] if feature_types is not None else None
        non_encoded_indices = [i for i in range(len(feature_names)) if i not in self._feature_indices]
        for i in non_encoded_indices:
            new_feature_names.append(feature_names[i])
            if feature_types is not None:
                new_feature_types.append(feature_types[i])
        for feature_name in self._features_to_encode:
            new_feature_names.extend([f'{feature_name}_sin', f'{feature_name}_cos'])
            if feature_types is not None:
                new_feature_types.extend(['numeric', 'numeric'])
        if not self.drop_original:
            for i in self._feature_indices:
                new_feature_names.append(feature_names[i])
                if feature_types is not None:
                    new_feature_types.append(feature_types[i])
        n_new_features = len(new_feature_names)
        X_transformed = np.zeros((X.shape[0], n_new_features))
        new_col_idx = 0
        for i in non_encoded_indices:
            X_transformed[:, new_col_idx] = X[:, i]
            new_col_idx += 1
        for feature_idx in self._feature_indices:
            feature_name = feature_names[feature_idx]
            if feature_name in self.periods:
                period = self.periods[feature_name]
                sin_values = np.sin(2 * np.pi * X[:, feature_idx] / period)
                cos_values = np.cos(2 * np.pi * X[:, feature_idx] / period)
                X_transformed[:, new_col_idx] = sin_values
                X_transformed[:, new_col_idx + 1] = cos_values
                new_col_idx += 2
        if not self.drop_original:
            for i in self._feature_indices:
                X_transformed[:, new_col_idx] = X[:, i]
                new_col_idx += 1
        return FeatureSet(features=X_transformed, feature_names=new_feature_names, feature_types=new_feature_types)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation to recover original cyclical features.

        Note: This is an approximation as the arc tangent function has limitations in
        determining the correct quadrant without additional information.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data with encoded cyclical features
        **kwargs : dict
            Additional parameters (ignored)

        Returns
        -------
        FeatureSet
            Data with cyclical features restored (approximate)
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names.copy() if data.feature_names is not None else None
        else:
            X = data.copy()
            feature_names = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        original_feature_names = []
        sin_cos_indices = {}
        for feature in self._features_to_encode:
            sin_name = f'{feature}_sin'
            cos_name = f'{feature}_cos'
            try:
                sin_idx = feature_names.index(sin_name)
                cos_idx = feature_names.index(cos_name)
                sin_cos_indices[feature] = (sin_idx, cos_idx)
                original_feature_names.append(feature)
            except ValueError:
                raise ValueError(f"Could not find encoded features for '{feature}'")
        non_encoded_indices = []
        non_encoded_names = []
        for (i, name) in enumerate(feature_names):
            is_encoded_component = False
            for feature in self._features_to_encode:
                if name == f'{feature}_sin' or name == f'{feature}_cos':
                    is_encoded_component = True
                    break
            if not is_encoded_component:
                non_encoded_indices.append(i)
                non_encoded_names.append(name)
        n_recovered_features = len(non_encoded_indices) + len(self._features_to_encode)
        X_recovered = np.zeros((X.shape[0], n_recovered_features))
        col_idx = 0
        for i in non_encoded_indices:
            X_recovered[:, col_idx] = X[:, i]
            col_idx += 1
        for feature in self._features_to_encode:
            (sin_idx, cos_idx) = sin_cos_indices[feature]
            period = self.periods[feature]
            angles = np.arctan2(X[:, sin_idx], X[:, cos_idx])
            recovered_values = angles / (2 * np.pi) * period
            recovered_values = np.where(recovered_values < 0, recovered_values + period, recovered_values)
            X_recovered[:, col_idx] = recovered_values
            col_idx += 1
        recovered_feature_names = non_encoded_names + self._features_to_encode
        return FeatureSet(features=X_recovered, feature_names=recovered_feature_names)

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get the names of the output features after transformation.

        Parameters
        ----------
        input_features : Optional[List[str]], default=None
            Names of input features. If None, uses fitted feature names

        Returns
        -------
        List[str]
            Names of output features
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        if input_features is None:
            input_features = self._feature_names_in
        new_feature_names = []
        non_encoded_features = [name for name in input_features if name not in self._features_to_encode]
        new_feature_names.extend(non_encoded_features)
        for feature_name in self._features_to_encode:
            new_feature_names.extend([f'{feature_name}_sin', f'{feature_name}_cos'])
        if not self.drop_original:
            encoded_original_features = [name for name in input_features if name in self._features_to_encode]
            new_feature_names.extend(encoded_original_features)
        return new_feature_names