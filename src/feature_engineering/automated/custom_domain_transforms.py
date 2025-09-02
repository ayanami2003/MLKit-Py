from typing import Any, Dict, List, Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np

class DomainSpecificFeatureEngineer(BaseTransformer):

    def __init__(self, domain_rules: Dict[str, Dict[str, Any]], name: Optional[str]=None):
        """
        Initialize the DomainSpecificFeatureEngineer.
        
        Args:
            domain_rules: A dictionary where keys are feature names and values are dictionaries
                         specifying the transformation type and parameters.
            name: Optional name for the transformer.
        """
        super().__init__(name)
        self._validate_domain_rules(domain_rules)
        self.domain_rules = domain_rules.copy()
        self._fitted_params = {}

    def _validate_domain_rules(self, domain_rules: Dict[str, Dict[str, Any]]) -> None:
        """
        Validate the domain rules structure and supported transformation types.
        
        Args:
            domain_rules: Dictionary of domain rules to validate.
            
        Raises:
            ValueError: If rules are invalid or contain unsupported transformations.
        """
        if not isinstance(domain_rules, dict):
            raise TypeError('domain_rules must be a dictionary')
        supported_transforms = {'log_transform', 'one_hot_encode'}
        for (feature_name, rule) in domain_rules.items():
            if not isinstance(rule, dict):
                raise TypeError(f"Rule for feature '{feature_name}' must be a dictionary")
            if 'type' not in rule or not rule['type']:
                raise ValueError(f"Rule for feature '{feature_name}' must specify a 'type'")
            transform_type = rule['type']
            if transform_type not in supported_transforms:
                raise ValueError(f"Unsupported transformation type '{transform_type}' for feature '{feature_name}'. Supported types: {supported_transforms}")

    def fit(self, data: Union[FeatureSet, Any], **kwargs) -> 'DomainSpecificFeatureEngineer':
        """
        Fit the transformer to the input data.
        
        For domain-specific feature engineering, fitting might involve analyzing the data
        to determine parameters for transformations but typically doesn't require learning.
        
        Args:
            data: Input data to fit the transformer on.
            **kwargs: Additional parameters for fitting.
            
        Returns:
            Self instance for method chaining.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('data must be a FeatureSet instance')
        if data.feature_names is None:
            raise ValueError('FeatureSet must contain feature names')
        self._fitted_params = {}
        for (feature_name, rule) in self.domain_rules.items():
            if feature_name not in data.feature_names:
                raise ValueError(f"Feature '{feature_name}' not found in input data")
            transform_type = rule['type']
            feature_idx = data.feature_names.index(feature_name)
            feature_data = data.features[:, feature_idx]
            if transform_type == 'log_transform':
                if not np.all(feature_data > 0):
                    raise ValueError(f"All values for feature '{feature_name}' must be positive for log transformation")
                self._fitted_params[feature_name] = {'min_value': np.min(feature_data), 'max_value': np.max(feature_data)}
            elif transform_type == 'one_hot_encode':
                unique_values = np.unique(feature_data)
                categories = []
                has_nan = False
                for val in unique_values:
                    if isinstance(val, float) and np.isnan(val):
                        has_nan = True
                    else:
                        categories.append(str(val))
                if has_nan:
                    categories.append('NaN')
                self._fitted_params[feature_name] = {'categories': sorted(categories)}
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, Any], **kwargs) -> FeatureSet:
        """
        Apply domain-specific transformations to generate new features.
        
        Args:
            data: Input data to transform.
            **kwargs: Additional parameters for transformation.
            
        Returns:
            FeatureSet with newly engineered features.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        if not isinstance(data, FeatureSet):
            raise TypeError('data must be a FeatureSet instance')
        if data.feature_names is None:
            raise ValueError('FeatureSet must contain feature names')
        original_features = data.features.copy()
        original_feature_names = data.feature_names.copy() if data.feature_names else []
        original_feature_types = data.feature_types.copy() if data.feature_types else []
        new_features_list = [original_features]
        new_feature_names = original_feature_names.copy()
        new_feature_types = original_feature_types.copy()
        for (feature_name, rule) in self.domain_rules.items():
            if feature_name not in data.feature_names:
                raise ValueError(f"Feature '{feature_name}' not found in input data")
            transform_type = rule['type']
            feature_idx = data.feature_names.index(feature_name)
            feature_data = data.features[:, feature_idx]
            if transform_type == 'log_transform':
                transformed_data = np.log(feature_data.astype(float))
                transformed_data = transformed_data.reshape(-1, 1)
                new_features_list.append(transformed_data)
                new_feature_names.append(f'{feature_name}_log')
                new_feature_types.append('numeric')
            elif transform_type == 'one_hot_encode':
                categories = self._fitted_params[feature_name]['categories']
                n_categories = len(categories)
                if n_categories <= 1:
                    encoded_data = np.ones((len(feature_data), 1))
                    new_features_list.append(encoded_data)
                    new_feature_names.append(f'{feature_name}_{categories[0]}' if categories else f'{feature_name}_encoded')
                    new_feature_types.append('categorical')
                else:
                    encoded_data = np.zeros((len(feature_data), n_categories))
                    feature_data_str = np.array([str(val) if not (isinstance(val, float) and np.isnan(val)) else 'NaN' for val in feature_data], dtype=object)
                    for (i, category) in enumerate(categories):
                        encoded_data[:, i] = (feature_data_str == category).astype(int)
                    new_features_list.append(encoded_data)
                    for category in categories:
                        new_feature_names.append(f'{feature_name}_{category}')
                        new_feature_types.append('categorical')
        if new_features_list:
            final_features = np.concatenate(new_features_list, axis=1)
        else:
            final_features = original_features
        return FeatureSet(features=final_features, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)

    def inverse_transform(self, data: Union[FeatureSet, Any], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation if possible.
        
        Args:
            data: Transformed data to invert.
            **kwargs: Additional parameters.
            
        Returns:
            Original data format if inversion is possible.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        if not isinstance(data, FeatureSet):
            raise TypeError('data must be a FeatureSet instance')
        if data.feature_names is None:
            raise ValueError('FeatureSet must contain feature names')
        original_feature_names = []
        for name in data.feature_names:
            is_transformed = False
            for rule_feature in self.domain_rules.keys():
                if name.startswith(f'{rule_feature}_'):
                    is_transformed = True
                    break
            if not is_transformed:
                original_feature_names.append(name)
        original_indices = []
        for orig_name in original_feature_names:
            try:
                idx = data.feature_names.index(orig_name)
                original_indices.append(idx)
            except ValueError:
                raise ValueError(f"Original feature '{orig_name}' not found in transformed data")
        if not original_indices:
            raise ValueError('Could not identify original features in transformed data')
        original_features = data.features[:, original_indices]
        original_feature_types = None
        if data.feature_types:
            original_feature_types = [data.feature_types[i] for i in original_indices]
        return FeatureSet(features=original_features, feature_names=original_feature_names, feature_types=original_feature_types, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)

    def add_domain_rule(self, feature_name: str, rule: Dict[str, Any]) -> None:
        """
        Add a new domain-specific transformation rule.
        
        Args:
            feature_name: Name of the feature to which the rule applies.
            rule: Dictionary specifying the transformation type and parameters.
        """
        temp_rules = {feature_name: rule}
        self._validate_domain_rules(temp_rules)
        self.domain_rules[feature_name] = rule
        if hasattr(self, '_is_fitted') and self._is_fitted:
            self._is_fitted = False

    def remove_domain_rule(self, feature_name: str) -> bool:
        """
        Remove a domain-specific transformation rule.
        
        Args:
            feature_name: Name of the feature whose rule should be removed.
            
        Returns:
            bool: True if rule was removed, False if rule didn't exist.
        """
        if feature_name in self.domain_rules:
            del self.domain_rules[feature_name]
            if feature_name in self._fitted_params:
                del self._fitted_params[feature_name]
            if hasattr(self, '_is_fitted') and self._is_fitted:
                self._is_fitted = False
            return True
        else:
            return False