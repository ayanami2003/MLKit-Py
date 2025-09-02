from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from itertools import combinations_with_replacement, combinations
from copy import deepcopy

class PolynomialInteractionTransformer(BaseTransformer):
    """
    Transformer for generating polynomial and interaction features from input data.

    This transformer creates polynomial features of specified degree and interaction terms
    from the input features. It supports customization of interaction-only terms,
    inclusion of bias terms, and feature naming.

    The transformation follows sklearn's PolynomialFeatures approach but integrates
    with the system's FeatureSet structure.

    Attributes
    ----------
    degree : int
        The degree of polynomial features to generate (default=2)
    interaction_only : bool
        Whether to only produce interaction features (default=False)
    include_bias : bool
        Whether to include a bias column (default=True)
    name : str
        Name identifier for the transformer instance

    Methods
    -------
    fit() : Fits the transformer to input data
    transform() : Applies polynomial feature transformation
    inverse_transform() : Not supported for this transformer
    get_feature_names() : Gets names of generated features
    """

    def __init__(self, degree: int=2, interaction_only: bool=False, include_bias: bool=True, name: Optional[str]=None):
        """
        Initialize the PolynomialInteractionTransformer.

        Parameters
        ----------
        degree : int, default=2
            The degree of polynomial features.
        interaction_only : bool, default=False
            If True, only interaction features are produced.
        include_bias : bool, default=True
            If True, include a bias column (intercept).
        name : str, optional
            Name identifier for the transformer.
        """
        super().__init__(name=name)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self._feature_names: Optional[List[str]] = None
        self._n_input_features: Optional[int] = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'PolynomialInteractionTransformer':
        """
        Fit the transformer to the input data.

        This method determines the structure of the input data and prepares
        for feature transformation.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on. Can be a FeatureSet or numpy array.
        **kwargs : dict
            Additional parameters for fitting.

        Returns
        -------
        PolynomialInteractionTransformer
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        else:
            X = data
            self._feature_names = None
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self._n_input_features = X.shape[1]
        if self._feature_names is None:
            self._feature_names = [f'x{i}' for i in range(self._n_input_features)]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply polynomial feature transformation to input data.

        Generates polynomial and interaction features according to the specified
        degree and interaction settings.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Must have same number of features as fitted data.
        **kwargs : dict
            Additional parameters for transformation.

        Returns
        -------
        FeatureSet
            Transformed data with polynomial features.

        Raises
        ------
        ValueError
            If data has different number of features than fitted data.
        """
        if self._n_input_features is None:
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
        (n_samples, n_features) = X.shape
        if n_features == 0:
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
                    name = self._feature_names[indices[0]]
                else:
                    name = '*'.join([self._feature_names[i] for i in indices])
                feature_names_out.append(name)
            XP = np.hstack(features_list)
        return FeatureSet(features=XP, feature_names=feature_names_out, feature_types=['numeric'] * XP.shape[1], sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply inverse transformation (not supported for polynomial features).

        Polynomial feature transformation is not invertible, so this method
        raises a NotImplementedError.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to invert.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        FeatureSet
            Original data format.

        Raises
        ------
        NotImplementedError
            Always raised as polynomial transformation is not invertible.
        """
        raise NotImplementedError('Polynomial feature transformation is not invertible.')

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get names of the generated polynomial features.

        Constructs feature names based on input feature names and polynomial
        transformation parameters.

        Parameters
        ----------
        input_features : List[str], optional
            Names of input features. If None, uses stored names or generates defaults.

        Returns
        -------
        List[str]
            Names of generated polynomial features.
        """
        if input_features is not None:
            if len(input_features) != self._n_input_features:
                raise ValueError(f'input_features has {len(input_features)} elements, expected {self._n_input_features}')
            feature_names = input_features
        elif self._feature_names is not None:
            feature_names = self._feature_names
        else:
            feature_names = [f'x{i}' for i in range(self._n_input_features)]
        if self._n_input_features == 0:
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
                name = feature_names[indices[0]]
            else:
                name = '*'.join([feature_names[i] for i in indices])
            feature_names_out.append(name)
        return feature_names_out