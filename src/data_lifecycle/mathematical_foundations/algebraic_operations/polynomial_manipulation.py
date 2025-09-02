from typing import Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from math import comb

class PolynomialExpander(BaseTransformer):

    def __init__(self, degree: int=2, interaction_only: bool=False, include_bias: bool=True, name: str=None):
        """
        Initialize the PolynomialExpander.
        
        Args:
            degree (int): The maximum degree of polynomial features to generate. Defaults to 2.
            interaction_only (bool): Whether to only produce interaction features (excluding pure polynomial terms).
                                     Defaults to False.
            include_bias (bool): Whether to include a bias column (intercept term) in the output.
                                Defaults to True.
            name (str, optional): Name of the transformer instance. If None, uses class name.
        """
        super().__init__(name=name)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self._n_features_in = None
        self._n_features_out = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'PolynomialExpander':
        """
        Fit the transformer to the input data.
        
        This method calculates the expected output dimensions based on the input data shape
        and polynomial expansion parameters.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit the transformer on.
                                                 If FeatureSet, uses the features attribute.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            PolynomialExpander: Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        self._n_features_in = n_features
        self._n_features_out = 0
        if self.include_bias:
            self._n_features_out += 1
        if self.degree >= 1:
            from math import comb
            if self.interaction_only:
                for d in range(1, min(self.degree, n_features) + 1):
                    if n_features >= d:
                        self._n_features_out += comb(n_features, d)
            else:
                for d in range(1, self.degree + 1):
                    self._n_features_out += comb(n_features + d - 1, d)
        self.n_output_features_ = self._n_features_out
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply polynomial expansion to the input data.
        
        Generates polynomial and interaction features according to the configured parameters.
        The output maintains alignment with the system's FeatureSet structure.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
                                                 If FeatureSet, uses the features attribute.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: Transformed data with expanded polynomial features.
                       Contains updated feature names reflecting the generated terms.
        """
        if not hasattr(self, '_n_features_in') or self._n_features_in is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self._n_features_in:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self._n_features_in} features')
        from itertools import combinations_with_replacement, combinations
        from collections import Counter
        (n_samples, n_features) = X.shape
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(n_features)]
        feature_combinations = []
        combination_names = []
        if self.include_bias:
            feature_combinations.append(())
            combination_names.append('1')
        for d in range(1, self.degree + 1):
            if self.interaction_only:
                combos = combinations(range(n_features), d)
            else:
                combos = combinations_with_replacement(range(n_features), d)
            for combo in combos:
                if self.interaction_only and len(set(combo)) == 1 and (d > 1):
                    continue
                feature_combinations.append(combo)
                if len(combo) == 0:
                    name = '1'
                elif len(combo) == 1:
                    if d == 1:
                        name = feature_names[combo[0]]
                    else:
                        name = f'{feature_names[combo[0]]}^{d}'
                else:
                    counts = Counter(combo)
                    terms = []
                    for (idx, count) in sorted(counts.items()):
                        if count == 1:
                            terms.append(feature_names[idx])
                        else:
                            terms.append(f'{feature_names[idx]}^{count}')
                    name = ' '.join(terms)
                combination_names.append(name)
        expanded_features = []
        for combo in feature_combinations:
            if len(combo) == 0:
                expanded_features.append(np.ones(n_samples))
            else:
                feature_product = np.ones(n_samples)
                for idx in combo:
                    feature_product *= X[:, idx]
                expanded_features.append(feature_product)
        if expanded_features:
            expanded_data = np.column_stack(expanded_features)
        else:
            expanded_data = np.empty((n_samples, 0))
        metadata['polynomial_expansion'] = {'degree': self.degree, 'interaction_only': self.interaction_only, 'include_bias': self.include_bias}
        return FeatureSet(features=expanded_data, feature_names=combination_names, feature_types=['numeric'] * expanded_data.shape[1], sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for polynomial expansion as information is lost during expansion.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Transformed data (ignored).
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: The method raises a NotImplementedError as inversion is not mathematically possible.
            
        Raises:
            NotImplementedError: Always raised since polynomial expansion is not invertible.
        """
        raise NotImplementedError('Polynomial expansion is not invertible.')

    def get_feature_names(self, input_features: List[str]=None) -> List[str]:
        """
        Generate feature names for the expanded polynomial features.
        
        Creates human-readable names for all generated polynomial and interaction terms.
        
        Args:
            input_features (List[str], optional): Names of input features. If None, generates default names.
            
        Returns:
            List[str]: Names of all output features after polynomial expansion.
        """
        if self._n_features_in is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if input_features is None:
            input_features = [f'x{i}' for i in range(self._n_features_in)]
        elif len(input_features) != self._n_features_in:
            raise ValueError(f'input_features has {len(input_features)} elements, but transformer was fitted on {self._n_features_in} features')
        from itertools import combinations_with_replacement, combinations
        from collections import Counter
        combination_names = []
        if self.include_bias:
            combination_names.append('1')
        for d in range(1, self.degree + 1):
            if self.interaction_only:
                combos = combinations(range(self._n_features_in), d)
            else:
                combos = combinations_with_replacement(range(self._n_features_in), d)
            for combo in combos:
                if self.interaction_only and len(set(combo)) == 1 and (d > 1):
                    continue
                if len(combo) == 0:
                    name = '1'
                elif len(combo) == 1:
                    if d == 1:
                        name = input_features[combo[0]]
                    else:
                        name = f'{input_features[combo[0]]}^{d}'
                else:
                    counts = Counter(combo)
                    terms = []
                    for (idx, count) in sorted(counts.items()):
                        if count == 1:
                            terms.append(input_features[idx])
                        else:
                            terms.append(f'{input_features[idx]}^{count}')
                    name = ' '.join(terms)
                combination_names.append(name)
        return combination_names