from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union
import numpy as np
from itertools import combinations_with_replacement, combinations
from copy import deepcopy

class GenericInteractionTransformer(BaseTransformer):

    def __init__(self, degree: int=2, interaction_only: bool=True, include_bias: bool=False, name: Optional[str]=None):
        """
        Initialize the GenericInteractionTransformer.

        Parameters
        ----------
        degree : int, default=2
            The maximum degree of interactions to generate.
            For example, degree=2 will create pairwise interactions.
        interaction_only : bool, default=True
            If True, only interaction terms between different features are created.
            If False, powers of individual features (e.g., x^2) are also included.
        include_bias : bool, default=False
            If True, a bias column (column of ones) is added to the output.
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.interaction_indices_ = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'GenericInteractionTransformer':
        """
        Fit the transformer to the input data.

        This method analyzes the input features to determine how to construct
        interaction terms and prepares the transformation mapping.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on. If FeatureSet, uses feature names
            for more informative output feature naming.
        **kwargs : dict
            Additional parameters for fitting (not used in this implementation)

        Returns
        -------
        GenericInteractionTransformer
            Self instance for method chaining

        Raises
        ------
        ValueError
            If data is empty or has incompatible dimensions
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if n_features == 0:
            raise ValueError('No features in input data')
        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError('Length of feature_names must match number of features')
            self.feature_names_in_ = list(feature_names)
        else:
            self.feature_names_in_ = [f'x{i + 1}' for i in range(n_features)]
        self.interaction_indices_ = []
        self.feature_names_out_ = []
        if self.include_bias:
            self.interaction_indices_.append(())
            self.feature_names_out_.append('1')
        if self.degree >= 1:
            if self.degree == 1 or not self.interaction_only:
                for i in range(n_features):
                    self.interaction_indices_.append((i,))
                    self.feature_names_out_.append(self.feature_names_in_[i])
        for d in range(2, self.degree + 1):
            if self.interaction_only:
                if d <= n_features:
                    combos = combinations(range(n_features), d)
                    for combo in combos:
                        self.interaction_indices_.append(combo)
                        name = '*'.join([self.feature_names_in_[idx] for idx in combo])
                        self.feature_names_out_.append(name)
            else:
                combos = combinations_with_replacement(range(n_features), d)
                for combo in combos:
                    self.interaction_indices_.append(combo)
                    if len(combo) == 1:
                        name = f'{self.feature_names_in_[combo[0]]}^{d}'
                    else:
                        from collections import Counter
                        counts = Counter(combo)
                        terms = []
                        for (idx, count) in sorted(counts.items()):
                            if count == 1:
                                terms.append(self.feature_names_in_[idx])
                            else:
                                terms.append(f'{self.feature_names_in_[idx]}^{count}')
                        name = '*'.join(terms)
                    self.feature_names_out_.append(name)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the interaction feature transformation to input data.

        Creates new features by combining existing features according to the
        specified interaction rules (degree, interaction_only).

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Must have same number of features as fitted data.
        **kwargs : dict
            Additional parameters for transformation (not used in this implementation)

        Returns
        -------
        FeatureSet
            Transformed data with interaction features

        Raises
        ------
        ValueError
            If transformer has not been fitted or data dimensions don't match
        """
        if self.interaction_indices_ is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        else:
            X = data
            feature_names = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if n_features != len(self.feature_names_in_):
            raise ValueError(f'Number of features in input data ({n_features}) does not match number of features during fitting ({len(self.feature_names_in_)})')
        interaction_features = []
        for combo in self.interaction_indices_:
            if len(combo) == 0:
                interaction_features.append(np.ones(n_samples))
            else:
                feature_product = np.ones(n_samples)
                for idx in combo:
                    feature_product *= X[:, idx]
                interaction_features.append(feature_product)
        if interaction_features:
            transformed_features = np.column_stack(interaction_features)
        else:
            transformed_features = np.empty((n_samples, 0))
        return FeatureSet(features=transformed_features, feature_names=self.feature_names_out_, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation if possible.

        Since interaction features are generally not invertible to original space,
        this method will attempt to extract the original features from the
        transformed data if they're present.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to invert
        **kwargs : dict
            Additional parameters for inverse transformation

        Returns
        -------
        FeatureSet
            Data with original features extracted if possible

        Raises
        ------
        NotImplementedError
            If inverse transformation is not possible for the current configuration
        """
        if self.interaction_indices_ is None:
            raise NotImplementedError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X_transformed = data.features
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        else:
            X_transformed = data
            sample_ids = None
            metadata = {}
            quality_scores = {}
        if X_transformed.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        n_transformed_features = X_transformed.shape[1]
        if n_transformed_features != len(self.feature_names_out_):
            raise ValueError(f'Number of features in transformed data ({n_transformed_features}) does not match expected number of output features ({len(self.feature_names_out_)})')
        if self.degree > 1:
            raise NotImplementedError('Inverse transformation is not possible for degree > 1')
        if not self.interaction_only and self.degree > 1:
            raise NotImplementedError('Inverse transformation is not possible when interaction_only=False and degree > 1')
        original_feature_indices = []
        for (i, indices) in enumerate(self.interaction_indices_):
            if len(indices) == 1:
                original_feature_indices.append((indices[0], i))
        if len(original_feature_indices) != len(self.feature_names_in_):
            raise NotImplementedError('Inverse transformation is not possible because original features are not preserved in the transformed data')
        original_feature_indices.sort()
        original_features = []
        original_feature_names = []
        for (orig_idx, trans_idx) in original_feature_indices:
            original_features.append(X_transformed[:, trans_idx])
            original_feature_names.append(self.feature_names_out_[trans_idx])
        if original_features:
            X_original = np.column_stack(original_features)
        else:
            X_original = np.empty((X_transformed.shape[0], 0))
        return FeatureSet(features=X_original, feature_names=original_feature_names, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names for the interaction terms.

        Parameters
        ----------
        input_features : List[str], optional
            Names of input features. If not provided, uses names from fitting.

        Returns
        -------
        List[str]
            Names of output features after interaction transformation

        Raises
        ------
        ValueError
            If transformer has not been fitted and no input features provided
        """
        if self.feature_names_out_ is not None:
            return list(self.feature_names_out_)
        elif input_features is not None:
            n_features = len(input_features)
            feature_names_out = []
            interaction_indices = []
            if self.include_bias:
                interaction_indices.append(())
                feature_names_out.append('1')
            if self.degree >= 1:
                if not self.interaction_only or self.degree == 1:
                    for i in range(n_features):
                        interaction_indices.append((i,))
                        feature_names_out.append(input_features[i])
            for d in range(2, self.degree + 1):
                if self.interaction_only:
                    if d <= n_features:
                        combos = combinations(range(n_features), d)
                        for combo in combos:
                            interaction_indices.append(combo)
                            name = '*'.join([input_features[idx] for idx in combo])
                            feature_names_out.append(name)
                else:
                    combos = combinations_with_replacement(range(n_features), d)
                    for combo in combos:
                        interaction_indices.append(combo)
                        if len(combo) == 1:
                            name = f'{input_features[combo[0]]}^{d}'
                        else:
                            from collections import Counter
                            counts = Counter(combo)
                            terms = []
                            for (idx, count) in sorted(counts.items()):
                                if count == 1:
                                    terms.append(input_features[idx])
                                else:
                                    terms.append(f'{input_features[idx]}^{count}')
                            name = '*'.join(terms)
                        feature_names_out.append(name)
            return feature_names_out
        else:
            raise ValueError('Transformer has not been fitted and no input feature names provided. Either call fit() first or provide input_features.')