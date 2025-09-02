from typing import Optional, Union, Dict, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from itertools import combinations_with_replacement, combinations
from scipy.stats import skew, kurtosis
import copy

class AutomatedFeatureGenerator(BaseTransformer):

    def __init__(self, max_depth: int=3, feature_types: Optional[Dict[str, str]]=None, transformations: Optional[List[str]]=None, max_features: int=1000, preserve_original: bool=True, name: Optional[str]=None):
        """
        Initialize the AutomatedFeatureGenerator.
        
        Parameters
        ----------
        max_depth : int, default=3
            Maximum depth of feature combinations to generate
        feature_types : dict, optional
            Dictionary mapping column names to their types ('numerical' or 'categorical')
        transformations : list of str, optional
            List of transformation types to apply (e.g., 'polynomial', 'log', 'sqrt')
        max_features : int, default=1000
            Maximum number of features to generate
        preserve_original : bool, default=True
            Whether to keep original features in the output
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.max_depth = max_depth
        self.feature_types = feature_types
        self.transformations = transformations
        self.max_features = max_features
        self.preserve_original = preserve_original
        if self.transformations is not None:
            valid_transformations = ['polynomial', 'log', 'sqrt', 'square', 'interaction', 'inverse', 'abs']
            for t in self.transformations:
                if t not in valid_transformations:
                    raise ValueError(f"Invalid transformation '{t}'. Valid options: {valid_transformations}")
        if self.max_depth < 1:
            raise ValueError('max_depth must be at least 1')
        if self.max_features < 1:
            raise ValueError('max_features must be at least 1')

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'AutomatedFeatureGenerator':
        """
        Fit the automated feature generator to the input data.
        
        This method analyzes the input data to determine appropriate transformations
        and feature combinations to generate.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to fit the generator on
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        AutomatedFeatureGenerator
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
            self._feature_types = data.feature_types
        else:
            X = data
            self._feature_names = None
            self._feature_types = None
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self._n_input_features = X.shape[1]
        if self._feature_names is None:
            self._feature_names = [f'x{i}' for i in range(self._n_input_features)]
        if len(self._feature_names) != self._n_input_features:
            raise ValueError(f'Length of feature_names ({len(self._feature_names)}) must match number of features ({self._n_input_features})')
        if self._feature_types is None:
            self._determine_feature_types(X)
            self._feature_types_dict = self._feature_types_auto
        else:
            self._feature_types_dict = {}
            for (i, name) in enumerate(self._feature_names):
                self._feature_types_dict[name] = self._feature_types[i]
        for name in self._feature_names:
            if name in self._feature_types_dict:
                if self._feature_types_dict[name] not in ['numerical', 'categorical']:
                    raise ValueError(f"Feature type for '{name}' must be 'numerical' or 'categorical'")
            else:
                raise ValueError(f"Feature type for '{name}' not provided")
        self._numerical_indices = [i for (i, name) in enumerate(self._feature_names) if self._feature_types_dict[name] == 'numerical']
        self._categorical_indices = [i for (i, name) in enumerate(self._feature_names) if self._feature_types_dict[name] == 'categorical']
        self._validate_transformations_for_features(X)
        return self

    def _determine_feature_types(self, X: np.ndarray):
        """Automatically determine feature types based on data."""
        self._feature_types_auto = {}
        for (i, name) in enumerate(self._feature_names):
            feature_col = X[:, i]
            if feature_col.dtype.kind in ['U', 'S', 'O']:
                self._feature_types_auto[name] = 'categorical'
            else:
                try:
                    float_values = feature_col.astype(float)
                    mask = ~np.isnan(float_values)
                    if not np.any(mask):
                        self._feature_types_auto[name] = 'categorical'
                    elif np.all(mask):
                        if np.allclose(float_values, float_values.astype(int)):
                            unique_count = len(np.unique(float_values))
                            if unique_count <= min(20, max(1, int(len(float_values) * 0.1))):
                                self._feature_types_auto[name] = 'categorical'
                            else:
                                self._feature_types_auto[name] = 'numerical'
                        else:
                            self._feature_types_auto[name] = 'numerical'
                    else:
                        non_nan_values = float_values[mask]
                        if len(non_nan_values) > 0 and np.allclose(non_nan_values, non_nan_values.astype(int)):
                            unique_count = len(np.unique(non_nan_values))
                            if unique_count <= min(20, max(1, int(len(non_nan_values) * 0.1))):
                                self._feature_types_auto[name] = 'categorical'
                            else:
                                self._feature_types_auto[name] = 'numerical'
                        else:
                            self._feature_types_auto[name] = 'numerical'
                except (ValueError, TypeError):
                    self._feature_types_auto[name] = 'categorical'
        self._feature_types = [self._feature_types_auto[name] for name in self._feature_names]

    def _validate_transformations_for_features(self, X: np.ndarray):
        """Validate which transformations can be applied to which features."""
        self._valid_transformations = {}
        transformations_to_use = self.transformations if self.transformations is not None else ['polynomial', 'log', 'sqrt', 'square', 'interaction']
        for (i, name) in enumerate(self._feature_names):
            feature_type = self._feature_types_dict[name]
            feature_col = X[:, i]
            valid_transforms = []
            if feature_type == 'numerical':
                try:
                    numeric_col = feature_col.astype(float)
                    for transform in transformations_to_use:
                        if transform == 'polynomial':
                            valid_transforms.append(transform)
                        elif transform == 'interaction':
                            valid_transforms.append(transform)
                        elif transform == 'log':
                            mask = ~np.isnan(numeric_col)
                            if np.all(numeric_col[mask] > 0):
                                valid_transforms.append(transform)
                        elif transform == 'sqrt':
                            mask = ~np.isnan(numeric_col)
                            if np.all(numeric_col[mask] >= 0):
                                valid_transforms.append(transform)
                        elif transform in ['square', 'abs', 'inverse']:
                            valid_transforms.append(transform)
                except (ValueError, TypeError):
                    if 'interaction' in transformations_to_use:
                        valid_transforms.append('interaction')
            elif feature_type == 'categorical':
                if 'interaction' in transformations_to_use:
                    valid_transforms.append('interaction')
            self._valid_transformations[name] = valid_transforms

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Generate new features from the input data.
        
        Applies the configured automated feature generation techniques to create
        new features based on the fitted parameters.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed data with newly generated features
        """
        if not hasattr(self, '_n_input_features'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self._n_input_features:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted with {self._n_input_features} features.')
        if feature_names is None:
            feature_names = self._feature_names
        (new_features, new_feature_names) = self._generate_features(X)
        if self.preserve_original:
            combined_features = np.hstack([X, new_features])
            combined_names = feature_names + new_feature_names
        else:
            combined_features = new_features
            combined_names = new_feature_names
        combined_types = []
        if self.preserve_original:
            for name in feature_names:
                combined_types.append(self._feature_types_dict[name])
        for _ in new_feature_names:
            combined_types.append('numerical')
        return FeatureSet(features=combined_features, feature_names=combined_names, feature_types=combined_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def _generate_features(self, X: np.ndarray) -> tuple:
        """Generate new features based on the fitted configuration."""
        features_list = []
        names_list = []
        transformations_to_use = self.transformations if self.transformations is not None else ['polynomial', 'log', 'sqrt', 'square', 'interaction']
        for (i, name) in enumerate(self._feature_names):
            if len(names_list) >= self.max_features:
                break
            feature_col = X[:, i]
            feature_type = self._feature_types_dict[name]
            if feature_type == 'numerical':
                try:
                    numeric_col = feature_col.astype(float)
                    for transform in self._valid_transformations.get(name, []):
                        if len(names_list) >= self.max_features:
                            break
                        if transform == 'log':
                            new_feature = np.log(numeric_col + 1e-08)
                            new_name = f'{name}_log'
                        elif transform == 'sqrt':
                            new_feature = np.sqrt(np.abs(numeric_col))
                            new_name = f'{name}_sqrt'
                        elif transform == 'square':
                            new_feature = np.square(numeric_col)
                            new_name = f'{name}_square'
                        elif transform == 'abs':
                            new_feature = np.abs(numeric_col)
                            new_name = f'{name}_abs'
                        elif transform == 'inverse':
                            new_feature = 1.0 / (numeric_col + 1e-08)
                            new_name = f'{name}_inverse'
                        else:
                            continue
                        features_list.append(new_feature.reshape(-1, 1))
                        names_list.append(new_name)
                except (ValueError, TypeError):
                    pass
        if 'polynomial' in transformations_to_use and len(self._numerical_indices) > 0:
            (poly_features, poly_names) = self._generate_polynomial_features(X)
            for (feat, name) in zip(poly_features.T, poly_names):
                if len(names_list) >= self.max_features:
                    break
                features_list.append(feat.reshape(-1, 1))
                names_list.append(name)
        if 'interaction' in transformations_to_use:
            (interaction_features, interaction_names) = self._generate_interaction_features(X)
            for (feat, name) in zip(interaction_features.T, interaction_names):
                if len(names_list) >= self.max_features:
                    break
                features_list.append(feat.reshape(-1, 1))
                names_list.append(name)
        if len(features_list) > self.max_features:
            features_list = features_list[:self.max_features]
            names_list = names_list[:self.max_features]
        if len(features_list) == 0:
            return (np.empty((X.shape[0], 0)), [])
        new_features = np.hstack(features_list)
        self._generated_feature_names = names_list
        return (new_features, names_list)

    def _generate_polynomial_features(self, X: np.ndarray) -> tuple:
        """Generate polynomial features up to max_depth for numerical features."""
        if len(self._numerical_indices) == 0:
            return (np.empty((X.shape[0], 0)), [])
        features_list = []
        names_list = []
        numerical_data = np.zeros((X.shape[0], len(self._numerical_indices)))
        for (i, idx) in enumerate(self._numerical_indices):
            try:
                numerical_data[:, i] = X[:, idx].astype(float)
            except (ValueError, TypeError):
                return (np.empty((X.shape[0], 0)), [])
        transformations_to_use = self.transformations if self.transformations is not None else ['polynomial', 'log', 'sqrt', 'square', 'interaction']
        if 'polynomial' not in transformations_to_use:
            return (np.empty((X.shape[0], 0)), [])
        for d in range(2, min(self.max_depth + 1, 4)):
            for indices in combinations_with_replacement(range(len(self._numerical_indices)), d):
                if len(names_list) >= self.max_features:
                    break
                feature_product = np.ones(X.shape[0])
                name_parts = []
                valid_combination = True
                for idx in indices:
                    original_idx = self._numerical_indices[idx]
                    try:
                        feature_product *= numerical_data[:, idx]
                        name_parts.append(self._feature_names[original_idx])
                    except (ValueError, TypeError):
                        valid_combination = False
                        break
                if valid_combination:
                    features_list.append(feature_product)
                    names_list.append('*'.join(name_parts))
        if len(features_list) == 0:
            return (np.empty((X.shape[0], 0)), [])
        return (np.column_stack(features_list), names_list)

    def _generate_interaction_features(self, X: np.ndarray) -> tuple:
        """Generate interaction features between features."""
        features_list = []
        names_list = []
        transformations_to_use = self.transformations if self.transformations is not None else ['polynomial', 'log', 'sqrt', 'square', 'interaction']
        if 'interaction' not in transformations_to_use:
            return (np.empty((X.shape[0], 0)), [])
        for d in range(2, min(self.max_depth + 1, 3)):
            for indices in combinations(range(self._n_input_features), d):
                if len(names_list) >= self.max_features:
                    break
                name_parts = [self._feature_names[idx] for idx in indices]
                all_numerical = all((self._feature_types_dict[self._feature_names[idx]] == 'numerical' for idx in indices))
                if all_numerical:
                    interaction = np.ones(X.shape[0])
                    valid_interaction = True
                    for idx in indices:
                        try:
                            interaction *= X[:, idx].astype(float)
                        except (ValueError, TypeError):
                            valid_interaction = False
                            break
                    if valid_interaction:
                        features_list.append(interaction)
                        names_list.append(':'.join(name_parts))
                else:
                    pass
        if len(features_list) == 0:
            return (np.empty((X.shape[0], 0)), [])
        return (np.column_stack(features_list), names_list)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Return the data to its original form (without generated features).
        
        Since this transformer adds features rather than modifying existing ones,
        this method will return only the original features if preserve_original=True.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Transformed data with generated features
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        FeatureSet
            Data with generated features removed
        """
        if not hasattr(self, '_n_input_features'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if self.preserve_original:
            original_features = X[:, :self._n_input_features]
            if isinstance(data, FeatureSet):
                result = copy.deepcopy(data)
                result.features = original_features
                if result.feature_names is not None:
                    result.feature_names = result.feature_names[:self._n_input_features]
                if result.feature_types is not None:
                    result.feature_types = result.feature_types[:self._n_input_features]
                return result
            else:
                return FeatureSet(features=original_features, feature_names=self._feature_names if feature_names is not None else None, feature_types=[self._feature_types_dict[name] for name in self._feature_names] if self._feature_names is not None else None, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            original_features = X[:, :self._n_input_features] if X.shape[1] >= self._n_input_features else X
            return FeatureSet(features=original_features, feature_names=self._feature_names if feature_names is not None else None, feature_types=[self._feature_types_dict[name] for name in self._feature_names] if self._feature_names is not None else None, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def get_generated_feature_names(self) -> List[str]:
        """
        Get the names of all generated features.
        
        Returns
        -------
        list of str
            Names of the generated features
        """
        if not hasattr(self, '_n_input_features'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if not hasattr(self, '_generated_feature_names'):
            return []
        return self._generated_feature_names