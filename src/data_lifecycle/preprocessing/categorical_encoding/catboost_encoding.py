from typing import Optional, List, Union
import numpy as np
import pandas as pd
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class CatBoostEncoder(BaseTransformer):

    def __init__(self, handle_unknown: str='error', handle_missing: str='error', sigma: float=0.0, name: Optional[str]=None):
        """
        Initialize the CatBoost encoder.
        
        Parameters
        ----------
        handle_unknown : str, default='error'
            Whether to raise an error or ignore unknown categories during transform.
            Options: ['error', 'ignore']
        handle_missing : str, default='error'
            Whether to raise an error or encode missing values.
            Options: ['error', 'return_nan']
        sigma : float, default=0.0
            Amount of noise to add to the encoding for regularization.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.sigma = sigma
        self.encoding_maps_ = {}
        self._fitted = False
        self._feature_names_in = None

    def fit(self, data: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'CatBoostEncoder':
        """
        Fit the CatBoost encoder to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode.
        y : Optional[np.ndarray]
            Target values used for encoding. Required for target-based encoding.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        CatBoostEncoder
            Fitted encoder instance.
            
        Raises
        ------
        ValueError
            If y is not provided or if data and y shapes don't match.
        """
        if y is None:
            raise ValueError('Target values (y) must be provided for CatBoost encoding.')
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names_in = data.feature_names
        else:
            X = data
            self._feature_names_in = [f'feature_{i}' for i in range(X.shape[1])]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._feature_names_in)
        if len(y) != len(X):
            raise ValueError('Data and target lengths must match.')
        self.encoding_maps_ = {}
        self._global_mean = np.mean(y)
        self._global_means = [self._global_mean] * X.shape[1]
        global_mean = self._global_mean
        for (col_idx, col_name) in enumerate(X.columns):
            temp_df = pd.DataFrame({'col': X[col_name], 'target': y})
            grouped = temp_df.groupby('col')['target'].agg(['count', 'mean'])
            prior_count = 1
            encoding_map = {}
            for (category, row) in grouped.iterrows():
                count = row['count']
                mean = row['mean']
                encoding_value = (count * mean + prior_count * global_mean) / (count + prior_count)
                encoding_map[category] = encoding_value
            if X[col_name].isnull().any():
                missing_mask = X[col_name].isnull()
                missing_mean = np.mean(y[missing_mask]) if np.sum(missing_mask) > 0 else global_mean
                encoding_map[np.nan] = missing_mean
            self.encoding_maps_[col_idx] = encoding_map
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical features using the fitted CatBoost encoding.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed data with encoded categorical features.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if not self._fitted:
            raise ValueError("This CatBoostEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = getattr(self, '_feature_names_in', [f'feature_{i}' for i in range(X.shape[1])])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        X_encoded = X.copy()
        for (col_idx, col_name) in enumerate(X.columns):
            if col_idx in self.encoding_maps_:
                encoding_map = self.encoding_maps_[col_idx].copy()
                series_to_encode = X[col_name].copy()
                if self.handle_missing == 'return_nan':
                    missing_mask = series_to_encode.isnull()
                    mapped_values = series_to_encode.map(encoding_map)
                    if missing_mask.any():
                        mapped_values[missing_mask] = np.nan
                    X_encoded[col_name] = mapped_values
                else:
                    X_encoded[col_name] = series_to_encode.map(encoding_map)
                if self.handle_unknown == 'error':
                    if X_encoded[col_name].isna().any():
                        raise ValueError(f'Unknown categories found in column {col_name}.')
                elif self.handle_unknown == 'ignore':
                    pass
                if self.sigma > 0:
                    mask = ~X_encoded[col_name].isna()
                    if mask.any():
                        noise = np.random.normal(0, self.sigma, size=np.sum(mask))
                        X_encoded.loc[mask, col_name] = X_encoded.loc[mask, col_name] + noise
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_encoded.values, feature_names=list(X_encoded.columns), feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        else:
            return FeatureSet(features=X_encoded.values, feature_names=list(X_encoded.columns))

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reverse the encoding transformation (not supported for CatBoost encoding).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Encoded data to reverse.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original data (if possible).
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for CatBoost encoding.')

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names after encoding.
        
        Parameters
        ----------
        input_features : Optional[List[str]]
            Input feature names.
            
        Returns
        -------
        List[str]
            Output feature names.
        """
        if input_features is not None:
            return input_features
        elif self._feature_names_in is not None:
            return self._feature_names_in
        else:
            return []