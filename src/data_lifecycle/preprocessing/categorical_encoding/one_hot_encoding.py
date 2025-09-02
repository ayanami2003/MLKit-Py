from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union
import numpy as np


# ...(code omitted)...


class DummyVariableEncoder(BaseTransformer):
    """
    Encode categorical features as dummy variables (one-hot encoding without multicollinearity).
    
    This transformer creates binary columns for each category but drops one category per feature
    to avoid the dummy variable trap (multicollinearity). It's equivalent to one-hot encoding
    with drop='first' but with clearer semantic intent for statistical modeling contexts.
    
    Attributes
    ----------
    handle_unknown : str, default='error'
        Specifies the behavior when encountering unknown categories during transformation.
        Options: 'error', 'ignore'.
    categories : Union[str, List[List[str]]], default='auto'
        Categories for each feature. 'auto' infers categories from training data.
    encoded_feature_names : List[str]
        Names of the encoded features generated during transformation.
    reference_categories : List[str]
        The dropped category for each feature (used as reference level).
        
    Methods
    -------
    fit : Fits the encoder to the input data.
    transform : Transforms input data using the fitted encoder.
    inverse_transform : Converts encoded data back to original categorical format.
    get_feature_names_out : Gets output feature names for the encoded features.
    """

    def __init__(self, handle_unknown: str='error', categories: Union[str, List[List[str]]]='auto', name: Optional[str]=None):
        """
        Initialize the DummyVariableEncoder.
        
        Parameters
        ----------
        handle_unknown : str, default='error'
            Specifies the behavior when encountering unknown categories.
            Options: 'error', 'ignore'.
        categories : Union[str, List[List[str]]], default='auto'
            Categories for each feature. 'auto' infers categories from training data.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        if handle_unknown not in ['error', 'ignore']:
            raise ValueError("handle_unknown must be either 'error' or 'ignore'")
        self.handle_unknown = handle_unknown
        self.categories = categories
        self._fitted_categories: List[List[str]] = []
        self.encoded_feature_names: List[str] = []
        self.reference_categories: List[str] = []
        self._feature_names: Optional[List[str]] = None
        self._is_fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'DummyVariableEncoder':
        """
        Fit the encoder to the input data by identifying categories for each feature.
        
        For each categorical feature, one category is selected as reference (dropped)
        to prevent multicollinearity in downstream modeling tasks.
        
        Parameters
        ----------
        data : FeatureSet
            Input data containing categorical features to encode. Must have feature_names.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        DummyVariableEncoder
            Fitted encoder instance.
            
        Raises
        ------
        ValueError
            If data does not contain feature names or if categories are inconsistent.
        """
        if data.feature_names is None:
            raise ValueError('FeatureSet must contain feature_names for DummyVariableEncoder')
        self._feature_names = data.feature_names.copy()
        X = data.features
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        if self.categories == 'auto':
            self._fitted_categories = []
            for i in range(n_features):
                unique_vals = np.unique(X[:, i])
                categories = [str(val) for val in unique_vals if not (isinstance(val, float) and np.isnan(val))]
                self._fitted_categories.append(sorted(categories))
        else:
            if len(self.categories) != n_features:
                raise ValueError(f'Number of category lists ({len(self.categories)}) must match number of features ({n_features})')
            self._fitted_categories = [sorted([str(cat) for cat in cats]) for cats in self.categories]
        self.reference_categories = [cats[0] for cats in self._fitted_categories]
        self.encoded_feature_names = []
        for (feature_name, categories, ref_category) in zip(self._feature_names, self._fitted_categories, self.reference_categories):
            for category in categories:
                if category != ref_category:
                    self.encoded_feature_names.append(f'{feature_name}_{category}')
        self._is_fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform input data using the fitted dummy variable encoder.
        
        Creates binary columns for each category except the reference category.
        Unknown categories are handled according to the handle_unknown setting.
        
        Parameters
        ----------
        data : FeatureSet
            Input data to transform. Must have same features as fitted data.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Transformed data with dummy variables. Feature names reflect encoding scheme.
            
        Raises
        ------
        ValueError
            If handle_unknown='error' and unknown categories are encountered.
        """
        if not self._is_fitted:
            raise ValueError('Encoder has not been fitted yet. Call fit() first.')
        X = data.features
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if n_features != len(self._fitted_categories):
            raise ValueError(f'Number of features in transform data ({n_features}) does not match fitted data ({len(self._fitted_categories)})')
        X_str = np.array([[str(val) if not (isinstance(val, float) and np.isnan(val)) else 'nan' for val in row] for row in X])
        n_encoded_features = len(self.encoded_feature_names)
        X_encoded = np.zeros((n_samples, n_encoded_features), dtype=int)
        col_idx = 0
        for (feature_idx, (categories, ref_category)) in enumerate(zip(self._fitted_categories, self.reference_categories)):
            for category in categories:
                if category != ref_category:
                    matches = X_str[:, feature_idx] == category
                    X_encoded[matches, col_idx] = 1
                    if self.handle_unknown == 'error' and (not np.any(matches)) and (category not in X_str[:, feature_idx]):
                        unique_vals = set(X_str[:, feature_idx])
                        known_vals = set(categories)
                        unknown_vals = unique_vals - known_vals
                        if unknown_vals:
                            raise ValueError(f'Unknown category(s) {unknown_vals} encountered in feature {feature_idx}. Known categories: {categories}')
                    col_idx += 1
        return FeatureSet(features=X_encoded, feature_names=self.encoded_feature_names.copy(), feature_types=['numeric'] * n_encoded_features, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Convert dummy variable encoded data back to original categorical format.
        
        Uses the reference categories to reconstruct the original categorical values.
        
        Parameters
        ----------
        data : FeatureSet
            Dummy variable encoded data to convert back.
        **kwargs : dict
            Additional parameters for inverse transformation.
            
        Returns
        -------
        FeatureSet
            Data in original categorical format with reconstructed feature values.
            
        Raises
        ------
        ValueError
            If encoded data dimensions don't match expected feature structure.
        """
        if not self._is_fitted:
            raise ValueError('Encoder has not been fitted yet. Call fit() first.')
        X_encoded = data.features
        if X_encoded.ndim == 1:
            X_encoded = X_encoded.reshape(-1, 1)
        (n_samples, n_encoded_features) = X_encoded.shape
        if n_encoded_features != len(self.encoded_feature_names):
            raise ValueError(f'Number of features in encoded data ({n_encoded_features}) does not match expected number of encoded features ({len(self.encoded_feature_names)})')
        n_original_features = len(self._fitted_categories)
        X_reconstructed = np.empty((n_samples, n_original_features), dtype=object)
        for (i, ref_category) in enumerate(self.reference_categories):
            X_reconstructed[:, i] = ref_category
        col_idx = 0
        for (feature_idx, (categories, ref_category)) in enumerate(zip(self._fitted_categories, self.reference_categories)):
            for category in categories:
                if category != ref_category:
                    active_rows = X_encoded[:, col_idx] == 1
                    X_reconstructed[active_rows, feature_idx] = category
                    col_idx += 1
        return FeatureSet(features=X_reconstructed, feature_names=self._feature_names.copy() if self._feature_names else None, feature_types=['categorical'] * n_original_features, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names for the dummy variable encoded features.
        
        Excludes reference categories from the returned feature names.
        
        Parameters
        ----------
        input_features : Optional[List[str]], default=None
            Names of input features. If None, uses fitted feature names.
            
        Returns
        -------
        List[str]
            Names of the encoded features (excluding reference categories).
        """
        if not self._is_fitted:
            raise ValueError('Encoder has not been fitted yet. Call fit() first.')
        if input_features is not None:
            if len(input_features) != len(self._fitted_categories):
                raise ValueError(f'Length of input_features ({len(input_features)}) does not match number of fitted features ({len(self._fitted_categories)})')
            feature_names = input_features
        else:
            if self._feature_names is None:
                raise ValueError('Feature names not available. Either fit with feature names or provide input_features.')
            feature_names = self._feature_names
        encoded_names = []
        for (feature_name, categories, ref_category) in zip(feature_names, self._fitted_categories, self.reference_categories):
            for category in categories:
                if category != ref_category:
                    encoded_names.append(f'{feature_name}_{category}')
        return encoded_names