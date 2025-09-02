from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class EffectCodingEncoder(BaseTransformer):

    def __init__(self, handle_unknown: str='error', categories: Union[str, List[List[str]]]='auto', drop: Optional[Union[str, int]]=None, name: Optional[str]=None):
        super().__init__(name=name)
        if handle_unknown not in ['error', 'ignore']:
            raise ValueError("handle_unknown must be either 'error' or 'ignore'")
        self.handle_unknown = handle_unknown
        self.categories = categories
        self.drop = drop

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'EffectCodingEncoder':
        """
        Fit the effect coding encoder to the input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data containing categorical features to encode
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        EffectCodingEncoder
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data contains unsupported types or invalid configurations
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names_in = data.feature_names
        else:
            X = data
            self._feature_names_in = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if self.categories == 'auto':
            self.categories_ = []
            for i in range(n_features):
                unique_vals = np.unique(X[:, i])
                categories = [str(val) for val in unique_vals if not (isinstance(val, float) and np.isnan(val))]
                self.categories_.append(np.array(sorted(categories)))
        else:
            if len(self.categories) != n_features:
                raise ValueError(f'Number of category lists ({len(self.categories)}) must match number of features ({n_features})')
            self.categories_ = [np.array([str(cat) for cat in cats]) for cats in self.categories]
        self.drop_idx_ = []
        for (i, cats) in enumerate(self.categories_):
            n_cats = len(cats)
            if n_cats <= 1:
                self.drop_idx_.append(None)
                continue
            if self.drop is None or self.drop == 'first':
                drop_idx = 0
            elif self.drop == 'last':
                drop_idx = n_cats - 1
            elif isinstance(self.drop, str):
                try:
                    drop_idx = np.where(cats == self.drop)[0][0]
                except IndexError:
                    raise ValueError(f"Specified drop category '{self.drop}' not found in categories for feature {i}")
            else:
                raise ValueError("drop must be None, 'first', 'last', or a category value")
            self.drop_idx_.append(drop_idx)
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply effect coding transformation to input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Transformed data with effect-coded features
            
        Raises
        ------
        ValueError
            If transformer was not fitted or data is incompatible
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("This EffectCodingEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this transformer.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if n_features != len(self.categories_):
            raise ValueError(f'Number of features in transform data ({n_features}) does not match fitted data ({len(self.categories_)})')
        X_str = np.array([[str(val) if not (isinstance(val, float) and np.isnan(val)) else 'NaN' for val in row] for row in X])
        encoded_columns = []
        self.encoded_feature_names_ = []
        for feat_idx in range(n_features):
            cats = self.categories_[feat_idx]
            drop_idx = self.drop_idx_[feat_idx]
            n_cats = len(cats)
            if n_cats <= 1 or drop_idx is None:
                continue
            for (cat_idx, cat) in enumerate(cats):
                if cat_idx == drop_idx:
                    continue
                col = np.where(X_str[:, feat_idx] == cat, 1, np.where(X_str[:, feat_idx] == cats[drop_idx], -1, 0))
                if self.handle_unknown == 'ignore':
                    unknown_mask = ~np.isin(X_str[:, feat_idx], cats)
                    col[unknown_mask] = 0
                elif self.handle_unknown == 'error':
                    unknown_categories = set(X_str[:, feat_idx]) - set(cats)
                    if unknown_categories and '' not in unknown_categories:
                        raise ValueError(f'Found unknown categories {unknown_categories} in feature {feat_idx} during transform')
                encoded_columns.append(col)
                input_feature_name = feature_names[feat_idx] if feature_names else f'x{feat_idx}'
                self.encoded_feature_names_.append(f'{input_feature_name}_{cat}')
        if encoded_columns:
            X_encoded = np.column_stack(encoded_columns).astype(float)
        else:
            X_encoded = np.empty((n_samples, 0))
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_encoded, feature_names=self.encoded_feature_names_)
        else:
            return FeatureSet(features=X_encoded, feature_names=self.encoded_feature_names_)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Convert effect-coded features back to original categorical values.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Effect-coded data to convert back
        **kwargs : dict
            Additional inverse transformation parameters
            
        Returns
        -------
        FeatureSet
            Data with original categorical values restored
            
        Raises
        ------
        ValueError
            If transformer was not fitted or data is incompatible
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("This EffectCodingEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this transformer.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        (n_samples, n_encoded_features) = X.shape
        n_original_features = len(self.categories_)
        reconstructed_features = []
        encoded_feat_idx = 0
        for orig_feat_idx in range(n_original_features):
            cats = self.categories_[orig_feat_idx]
            drop_idx = self.drop_idx_[orig_feat_idx]
            n_cats = len(cats)
            if n_cats <= 1 or drop_idx is None:
                if n_cats == 1:
                    reconstructed_col = np.full(n_samples, cats[0])
                else:
                    reconstructed_col = np.full(n_samples, 'unknown')
                reconstructed_features.append(reconstructed_col)
                continue
            expected_encoded_feats = n_cats - 1
            if encoded_feat_idx + expected_encoded_feats > n_encoded_features:
                raise ValueError('Not enough features in input for inverse transform')
            feat_encoded = X[:, encoded_feat_idx:encoded_feat_idx + expected_encoded_feats]
            reconstructed_col = np.full(n_samples, cats[drop_idx])
            cat_idx_in_encoded = 0
            for (cat_idx, cat) in enumerate(cats):
                if cat_idx == drop_idx:
                    continue
                mask = feat_encoded[:, cat_idx_in_encoded] == 1
                reconstructed_col[mask] = cat
                cat_idx_in_encoded += 1
            reconstructed_features.append(reconstructed_col)
            encoded_feat_idx += expected_encoded_feats
        if reconstructed_features:
            X_reconstructed = np.column_stack(reconstructed_features)
        else:
            X_reconstructed = np.empty((n_samples, 0))
        if isinstance(data, FeatureSet):
            if self._feature_names_in is not None:
                reconstructed_feature_names = self._feature_names_in
            else:
                reconstructed_feature_names = [f'feature_{i}' for i in range(n_original_features)]
            return FeatureSet(features=X_reconstructed, feature_names=reconstructed_feature_names)
        else:
            return X_reconstructed

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names after effect coding transformation.
        
        Parameters
        ----------
        input_features : list of str, optional
            Input feature names. If None, generated names will be used.
            
        Returns
        -------
        list of str
            Output feature names for the effect-coded features
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("This EffectCodingEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if input_features is not None:
            if len(input_features) != len(self.categories_):
                raise ValueError(f'Length of input_features ({len(input_features)}) does not match number of features ({len(self.categories_)})')
            feature_names_in = input_features
        elif self._feature_names_in is not None:
            feature_names_in = self._feature_names_in
        else:
            feature_names_in = [f'x{i}' for i in range(len(self.categories_))]
        output_names = []
        for (feat_idx, (cats, drop_idx)) in enumerate(zip(self.categories_, self.drop_idx_)):
            if len(cats) <= 1 or drop_idx is None:
                continue
            for (cat_idx, cat) in enumerate(cats):
                if cat_idx == drop_idx:
                    continue
                output_names.append(f'{feature_names_in[feat_idx]}_{cat}')
        return output_names