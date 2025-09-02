from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class HelmertEncoder(BaseTransformer):

    def __init__(self, handle_unknown: str='error', categories: Union[str, List[List[str]]]='auto', drop: Optional[str]=None, name: Optional[str]=None):
        """
        Initialize the Helmert encoder.
        
        Parameters
        ----------
        handle_unknown : str, default='error'
            Strategy for handling unknown categories during transform.
        categories : Union[str, List[List[str]]], default='auto'
            Categories for each feature. If 'auto', they will be inferred from training data.
        drop : Optional[str], default=None
            Method for dropping categories to reduce multicollinearity.
        name : Optional[str]
            Optional name for the transformer.
        """
        super().__init__(name=name)
        if handle_unknown not in ['error', 'ignore']:
            raise ValueError("handle_unknown must be either 'error' or 'ignore'")
        if drop not in [None, 'first', 'if_binary']:
            raise ValueError("drop must be None, 'first', or 'if_binary'")
        if not (categories == 'auto' or isinstance(categories, list)):
            raise ValueError("categories must be 'auto' or a list of lists")
        self.handle_unknown = handle_unknown
        self.categories = categories
        self.drop = drop
        self._fitted_categories = None
        self._category_mappings = None
        self._feature_names = None
        self._encoded_feature_names = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'HelmertEncoder':
        """
        Fit the Helmert encoder to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to be encoded.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        HelmertEncoder
            Fitted encoder instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        else:
            X = data
            self._feature_names = None
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
        self._category_mappings = []
        self._encoded_feature_names = []
        for (feat_idx, categories) in enumerate(self._fitted_categories):
            n_categories = len(categories)
            drop_category = False
            if self.drop == 'first':
                drop_category = True
            elif self.drop == 'if_binary' and n_categories == 2:
                drop_category = True
            if n_categories == 0:
                mapping = {}
                actual_categories = []
            elif n_categories == 1:
                mapping = {categories[0]: np.array([])}
                actual_categories = []
            else:
                if drop_category:
                    contrast_matrix = self._create_helmert_matrix(n_categories)
                    contrast_matrix = contrast_matrix[1:, :]
                    actual_categories = categories[1:]
                else:
                    contrast_matrix = self._create_helmert_matrix(n_categories)
                    actual_categories = categories
                mapping = {cat: contrast_matrix[i] for (i, cat) in enumerate(categories)}
            self._category_mappings.append(mapping)
            feature_name = f'x{feat_idx}' if self._feature_names is None else self._feature_names[feat_idx]
            if len(actual_categories) > 0:
                for i in range(len(actual_categories)):
                    self._encoded_feature_names.append(f'{feature_name}_Helmert_{i + 1}')
        return self

    def _create_helmert_matrix(self, n_categories: int) -> np.ndarray:
        """
        Create Helmert contrast matrix for n categories.
        
        Parameters
        ----------
        n_categories : int
            Number of categories
            
        Returns
        -------
        np.ndarray
            Helmert contrast matrix of shape (n_categories, n_categories-1)
        """
        if n_categories <= 1:
            return np.array([]).reshape(n_categories, max(0, n_categories - 1))
        matrix = np.zeros((n_categories, n_categories - 1))
        matrix[0, 0] = -1
        for i in range(1, n_categories):
            matrix[i, 0] = 1 / (n_categories - 1)
        for j in range(1, n_categories - 1):
            n_part = j + 1
            for i in range(n_part):
                matrix[i, j] = -1 / n_part
            matrix[n_part, j] = 1
        return matrix

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply Helmert coding to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to be transformed using Helmert coding.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed data with Helmert-encoded features.
        """
        if self._fitted_categories is None:
            raise ValueError("This HelmertEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if n_features != len(self._fitted_categories):
            raise ValueError(f'Number of features in transform data ({n_features}) does not match fitted data ({len(self._fitted_categories)})')
        transformed_features = []
        for feat_idx in range(n_features):
            categories = self._fitted_categories[feat_idx]
            mapping = self._category_mappings[feat_idx]
            if len(categories) <= 1:
                continue
            n_encoded_features = len(list(mapping.values())[0]) if mapping and len(mapping) > 0 else 0
            if n_encoded_features > 0:
                feature_encoded = np.zeros((n_samples, n_encoded_features))
                for sample_idx in range(n_samples):
                    category = str(X[sample_idx, feat_idx])
                    if category in mapping:
                        feature_encoded[sample_idx] = mapping[category]
                    elif self.handle_unknown == 'error':
                        raise ValueError(f"Unknown category '{category}' encountered in feature {feat_idx}. Set handle_unknown='ignore' to ignore unknown categories.")
                    else:
                        feature_encoded[sample_idx] = np.zeros(n_encoded_features)
                transformed_features.append(feature_encoded)
        if transformed_features:
            encoded_features = np.hstack(transformed_features)
        else:
            encoded_features = np.array([]).reshape(n_samples, 0)
        return FeatureSet(features=encoded_features, feature_names=self._encoded_feature_names.copy() if self._encoded_feature_names else None)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert Helmert-coded data back to original categorical values.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Helmert-encoded data to convert back to original form.
        **kwargs : dict
            Additional inverse transformation parameters.
            
        Returns
        -------
        FeatureSet
            Data converted back to original categorical values.
        """
        if self._fitted_categories is None:
            raise ValueError("This HelmertEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        n_samples = X.shape[0]
        original_features = []
        col_idx = 0
        for (feat_idx, mapping) in enumerate(self._category_mappings):
            categories = self._fitted_categories[feat_idx]
            if len(categories) <= 1:
                continue
            if not mapping or len(mapping) == 0:
                continue
            n_encoded_features = len(list(mapping.values())[0])
            if col_idx + n_encoded_features > X.shape[1]:
                raise ValueError('Input data does not match the expected shape for inverse transformation')
            feature_data = X[:, col_idx:col_idx + n_encoded_features]
            col_idx += n_encoded_features
            original_feature = np.full(n_samples, '', dtype=object)
            for sample_idx in range(n_samples):
                encoded_vals = feature_data[sample_idx]
                min_distance = np.inf
                best_category = categories[0]
                for (category, helmert_vals) in mapping.items():
                    distance = np.sum((encoded_vals - helmert_vals) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        best_category = category
                original_feature[sample_idx] = best_category
            original_features.append(original_feature)
        if original_features:
            reconstructed_features = np.column_stack(original_features)
        else:
            reconstructed_features = np.array([]).reshape(n_samples, 0)
        return FeatureSet(features=reconstructed_features, feature_names=self._feature_names.copy() if self._feature_names else None)

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names after Helmert encoding.
        
        Parameters
        ----------
        input_features : Optional[List[str]]
            Input feature names. If None, generated names will be used.
            
        Returns
        -------
        List[str]
            Output feature names after encoding.
        """
        if self._encoded_feature_names is None:
            raise ValueError("This HelmertEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if input_features is not None:
            if len(input_features) != len(self._fitted_categories):
                raise ValueError(f'Length of input_features ({len(input_features)}) does not match number of features ({len(self._fitted_categories)})')
            encoded_names = []
            for (feat_idx, categories) in enumerate(self._fitted_categories):
                drop_category = False
                if self.drop == 'first':
                    drop_category = True
                elif self.drop == 'if_binary' and len(categories) == 2:
                    drop_category = True
                actual_categories = categories[1:] if drop_category else categories
                if len(actual_categories) > 0:
                    for i in range(len(actual_categories)):
                        encoded_names.append(f'{input_features[feat_idx]}_Helmert_{i + 1}')
            return encoded_names
        else:
            return self._encoded_feature_names.copy()