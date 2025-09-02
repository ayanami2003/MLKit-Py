from typing import Optional, List, Union, Dict, Any
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class RareCategoryNormalizer(BaseTransformer):

    def __init__(self, threshold: float=0.01, threshold_type: str='relative', replacement_label: str='Other', handle_unknown: str='ignore', name: Optional[str]=None):
        """
        Initialize the RareCategoryNormalizer.

        Parameters
        ----------
        threshold : float, default=0.01
            Minimum frequency for a category to be considered frequent.
        threshold_type : str, default='relative'
            Interpretation of threshold: 'relative' for proportion, 'absolute' for count.
        replacement_label : str, default='Other'
            Label used to replace rare categories.
        handle_unknown : str, default='ignore'
            Strategy for handling unknown categories during transformation.
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.replacement_label = replacement_label
        self.handle_unknown = handle_unknown
        self._category_mappings = {}
        self._feature_names_in = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'RareCategoryNormalizer':
        """
        Learn which categories are rare based on the training data.

        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data containing categorical features to analyze.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        RareCategoryNormalizer
            Fitted transformer instance.
        """
        if self.threshold_type not in ['relative', 'absolute']:
            raise ValueError("threshold_type must be either 'relative' or 'absolute'")
        if self.handle_unknown not in ['ignore', 'replace']:
            raise ValueError("handle_unknown must be either 'ignore' or 'replace'")
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names_in = data.feature_names
        else:
            X = data
            self._feature_names_in = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        self._category_mappings = {}
        for feature_idx in range(n_features):
            feature_column = X[:, feature_idx]
            feature_str = np.array([str(val) if not (isinstance(val, float) and np.isnan(val)) else 'NaN' for val in feature_column])
            (unique_categories, counts) = np.unique(feature_str, return_counts=True)
            rare_categories = set()
            if self.threshold_type == 'relative':
                frequencies = counts / n_samples
                rare_mask = frequencies < self.threshold
            else:
                rare_mask = counts < self.threshold
            rare_categories = set(unique_categories[rare_mask])
            self._category_mappings[feature_idx] = {'rare_categories': rare_categories, 'replacement_label': self.replacement_label}
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Replace rare categories with the specified replacement label.

        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to transform.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        FeatureSet
            Transformed data with rare categories normalized.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        X_transformed = X.copy().astype(object)
        for feature_idx in range(n_features):
            if feature_idx not in self._category_mappings:
                continue
            feature_column = X[:, feature_idx]
            rare_categories = self._category_mappings[feature_idx]['rare_categories']
            replacement_label = self._category_mappings[feature_idx]['replacement_label']
            feature_str = np.array([str(val) if not (isinstance(val, float) and np.isnan(val)) else 'NaN' for val in feature_column])
            rare_mask = np.isin(feature_str, list(rare_categories))
            X_transformed[rare_mask, feature_idx] = replacement_label
            if self.handle_unknown == 'replace':
                known_categories = set(list(rare_categories) + [replacement_label])
                unknown_mask = ~np.isin(feature_str, list(known_categories))
                X_transformed[unknown_mask, feature_idx] = replacement_label
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_transformed, feature_names=feature_names)
        else:
            return FeatureSet(features=X_transformed, feature_names=feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Attempt to reverse the normalization (not fully reversible).

        Note: This operation is not guaranteed to restore original categories,
        especially for those that were replaced.

        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Transformed data to attempt inversion on.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        FeatureSet
            Data with some categories potentially restored.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_inverted = X.copy()
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_inverted, feature_names=feature_names)
        else:
            return FeatureSet(features=X_inverted, feature_names=feature_names)

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names after transformation.

        Parameters
        ----------
        input_features : list of str, optional
            Input feature names.

        Returns
        -------
        list of str
            Output feature names.
        """
        if input_features is not None:
            return input_features
        elif self._feature_names_in is not None:
            return self._feature_names_in
        else:
            if hasattr(self, '_category_mappings'):
                n_features = len(self._category_mappings)
            else:
                n_features = 0
            return [f'x{i}' for i in range(n_features)]