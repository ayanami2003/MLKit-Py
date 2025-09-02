from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from itertools import combinations_with_replacement, combinations

class PolynomialFeatureGenerator(BaseTransformer):

    def __init__(self, degree: int=2, interaction_only: bool=False, include_bias: bool=True, name: Optional[str]=None):
        """
        Initialize the PolynomialFeatureGenerator.

        Parameters
        ----------
        degree : int, default=2
            The degree of polynomial features to generate.
        interaction_only : bool, default=False
            If True, only interaction features are produced.
        include_bias : bool, default=True
            If True, include a bias column in the output.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.feature_names: Optional[List[str]] = None
        self._input_feature_names: Optional[List[str]] = None
        self._n_input_features: Optional[int] = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'PolynomialFeatureGenerator':
        """
        Fit the transformer to the input data.

        This method computes the necessary information to generate polynomial features,
        including determining feature names for the output.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on. If FeatureSet, uses feature names
            for generating output feature names.
        **kwargs : dict
            Additional parameters (ignored).

        Returns
        -------
        PolynomialFeatureGenerator
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._input_feature_names = data.feature_names
        else:
            X = data
            self._input_feature_names = None
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self._n_input_features = X.shape[1]
        if self._input_feature_names is None:
            self._input_feature_names = [f'x{i}' for i in range(self._n_input_features)]
        self.feature_names = None
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Generate polynomial features from input data.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Must have the same number of features as
            the data used during fitting.
        **kwargs : dict
            Additional parameters (ignored).

        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with polynomial features. If input was FeatureSet,
            returns a new FeatureSet with updated features and feature names.
        """
        if self._n_input_features is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            input_feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
            is_feature_set = True
        else:
            X = data
            input_feature_names = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
            is_feature_set = False
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self._n_input_features:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted with {self._n_input_features} features.')
        (n_samples, n_features) = X.shape
        if n_features == 0:
            if self.include_bias:
                XP = np.ones((n_samples, 1))
                feature_names_out = ['1']
            else:
                XP = np.empty((n_samples, 0))
                feature_names_out = []
        elif self.degree == 0:
            if self.include_bias:
                XP = np.ones((n_samples, 1))
                feature_names_out = ['1']
            else:
                XP = np.empty((n_samples, 0))
                feature_names_out = []
        else:
            if self.interaction_only:
                index_combinations = []
                for d in range(1, self.degree + 1):
                    index_combinations.extend(combinations(range(n_features), d))
            else:
                index_combinations = []
                for d in range(1, self.degree + 1):
                    index_combinations.extend(combinations_with_replacement(range(n_features), d))
            features_list = []
            feature_names_out = []
            if self.include_bias:
                features_list.append(np.ones((n_samples, 1)))
                feature_names_out.append('1')
            for indices in index_combinations:
                feature_product = np.prod(X[:, indices], axis=1, keepdims=True)
                features_list.append(feature_product)
                if len(indices) == 1:
                    name = self._input_feature_names[indices[0]]
                else:
                    name_parts = {}
                    for idx in indices:
                        if idx in name_parts:
                            name_parts[idx] += 1
                        else:
                            name_parts[idx] = 1
                    name_components = []
                    for (idx, count) in sorted(name_parts.items()):
                        if count == 1:
                            name_components.append(self._input_feature_names[idx])
                        else:
                            name_components.append(f'{self._input_feature_names[idx]}^{count}')
                    name = '*'.join(name_components)
                feature_names_out.append(name)
            if features_list:
                XP = np.hstack(features_list)
            else:
                XP = np.empty((n_samples, 0))
        if is_feature_set:
            return FeatureSet(features=XP, feature_names=feature_names_out, feature_types=['numeric'] * XP.shape[1], sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return XP

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation is not supported for polynomial features.

        Polynomial feature generation is a non-invertible transformation since
        information is lost during the expansion process.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data to inverse transform (ignored).
        **kwargs : dict
            Additional parameters (ignored).

        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Same as input data (identity operation).

        Raises
        ------
        NotImplementedError
            Always raised since inverse transformation is not mathematically possible.
        """
        raise NotImplementedError('Polynomial feature generation is a non-invertible transformation.')

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get the names of the generated polynomial features.

        Returns
        -------
        Optional[List[str]]
            List of feature names for the generated polynomial features,
            or None if not yet fitted.
        """
        if self._n_input_features is None:
            return None
        if self.feature_names is None:
            self.feature_names = self._generate_feature_names()
        return self.feature_names

    def _generate_feature_names(self) -> List[str]:
        """
        Generate feature names for the polynomial features.
        
        Returns
        -------
        List[str]
            List of feature names for the generated polynomial features.
        """
        if self._n_input_features == 0:
            if self.include_bias:
                return ['1']
            else:
                return []
        if self.degree == 0:
            if self.include_bias:
                return ['1']
            else:
                return []
        if self.interaction_only:
            index_combinations = []
            for d in range(1, self.degree + 1):
                index_combinations.extend(combinations(range(self._n_input_features), d))
        else:
            index_combinations = []
            for d in range(1, self.degree + 1):
                index_combinations.extend(combinations_with_replacement(range(self._n_input_features), d))
        feature_names_out = []
        if self.include_bias:
            feature_names_out.append('1')
        for indices in index_combinations:
            if len(indices) == 1:
                name = self._input_feature_names[indices[0]]
            else:
                name_parts = {}
                for idx in indices:
                    if idx in name_parts:
                        name_parts[idx] += 1
                    else:
                        name_parts[idx] = 1
                name_components = []
                for (idx, count) in sorted(name_parts.items()):
                    if count == 1:
                        name_components.append(self._input_feature_names[idx])
                    else:
                        name_components.append(f'{self._input_feature_names[idx]}^{count}')
                name = '*'.join(name_components)
            feature_names_out.append(name)
        return feature_names_out