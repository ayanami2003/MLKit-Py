import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union

class RatioFeatureTransformer(BaseTransformer):

    def __init__(self, feature_pairs=None, fill_value=np.nan, include_self_ratios=False, name=None):
        """
        Initialize the RatioFeatureTransformer.
        
        Parameters
        ----------
        feature_pairs : Optional[List[tuple]], optional
            List of tuples specifying which feature indices or names to use for ratio computations.
            Each tuple should contain two elements: (numerator_feature, denominator_feature).
            If None, ratios will be computed for all possible pairs. Default is None.
        fill_value : float, optional
            Value to use when denominator is zero. Default is NaN.
        include_self_ratios : bool, optional
            Whether to include ratios where numerator and denominator are the same feature.
            Default is False.
        name : Optional[str], optional
            Name of the transformer instance. If None, uses class name.
        """
        super().__init__(name=name)
        self.feature_pairs = feature_pairs
        self.fill_value = fill_value
        self.include_self_ratios = include_self_ratios
        self._feature_names = None
        self._n_features = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'RatioFeatureTransformer':
        """
        Fit the transformer to the input data.
        
        This method validates the input data and prepares the transformer for ratio computation.
        For ratio features, fitting primarily involves validating feature pairs if specified.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing features to compute ratios from. If FeatureSet, uses its features attribute.
        **kwargs : dict
            Additional parameters (not used in this implementation).
            
        Returns
        -------
        RatioFeatureTransformer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If specified feature pairs are invalid or out of bounds.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names if data.feature_names is not None else None
        elif isinstance(data, np.ndarray):
            X = data
            self._feature_names = None
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self._n_features = X.shape[1]
        self.original_feature_pairs_ = self.feature_pairs
        if self.feature_pairs is not None:
            validated_pairs = []
            for pair in self.feature_pairs:
                if len(pair) != 2:
                    raise ValueError('Each feature pair must contain exactly two elements')
                processed_pair = []
                for feature in pair:
                    if isinstance(feature, str):
                        if self._feature_names is None:
                            raise ValueError('Cannot use feature names when input is numpy array')
                        try:
                            idx = self._feature_names.index(feature)
                            processed_pair.append(idx)
                        except ValueError:
                            raise ValueError(f"Feature '{feature}' not found in feature names")
                    elif isinstance(feature, int):
                        if feature < 0 or feature >= self._n_features:
                            raise ValueError(f'Feature index {feature} is out of bounds')
                        processed_pair.append(feature)
                    else:
                        raise TypeError('Feature identifiers must be strings or integers')
                validated_pairs.append(tuple(processed_pair))
            self.feature_pairs_ = validated_pairs
        else:
            self.feature_pairs_ = []
            for i in range(self._n_features):
                for j in range(self._n_features):
                    if not self.include_self_ratios and i == j:
                        continue
                    self.feature_pairs_.append((i, j))
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the ratio feature transformation to the input data.
        
        Creates new features by computing ratios between selected feature pairs. Handles division
        by zero according to the fill_value parameter.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing features to compute ratios from. If FeatureSet, uses its features attribute.
        **kwargs : dict
            Additional parameters (not used in this implementation).
            
        Returns
        -------
        FeatureSet
            New FeatureSet containing original features plus the newly created ratio features.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet or if input data is incompatible.
        """
        if not self._is_fitted:
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self._n_features:
            raise ValueError('Number of features in input data does not match fitted data')
        ratio_features = []
        for (i, j) in self.feature_pairs_:
            numerator = X[:, i]
            denominator = X[:, j]
            ratio = np.where(denominator != 0, numerator / denominator, self.fill_value)
            ratio_features.append(ratio)
        if ratio_features:
            ratio_features = np.column_stack(ratio_features)
            combined_features = np.hstack([X, ratio_features])
        else:
            combined_features = X
        if isinstance(data, FeatureSet) and data.feature_names is not None:
            original_names = data.feature_names
        elif isinstance(data, np.ndarray):
            original_names = [str(i) for i in range(X.shape[1])]
        else:
            original_names = [str(i) for i in range(X.shape[1])]
        ratio_names = self.get_feature_names(original_names)
        combined_names = original_names + ratio_names
        return FeatureSet(features=combined_features, feature_names=combined_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for ratio features.
        
        Since ratio features are derived from original features, we cannot uniquely reconstruct
        the original features from the ratios alone.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            The input data unchanged.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for ratio features.')

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get names of the generated ratio features.
        
        Constructs descriptive names for the ratio features based on the input feature names
        and the specified feature pairs.
        
        Parameters
        ----------
        input_features : Optional[List[str]], optional
            List of input feature names. If None, generates generic names.
            
        Returns
        -------
        List[str]
            List of feature names for the generated ratio features.
        """
        if not self._is_fitted:
            raise ValueError('Transformer has not been fitted yet.')
        if input_features is None:
            if self._feature_names is not None:
                input_features = self._feature_names
            else:
                input_features = [str(i) for i in range(self._n_features)]
        ratio_names = []
        if self.original_feature_pairs_ is not None:
            for pair in self.original_feature_pairs_:
                if isinstance(pair, tuple) and len(pair) == 2:
                    if isinstance(pair[0], str) and isinstance(pair[1], str):
                        name = f'{pair[0]}_to_{pair[1]}'
                        ratio_names.append(name)
                    else:
                        (i, j) = pair
                        if isinstance(i, int) and isinstance(j, int) and (i < len(input_features)) and (j < len(input_features)):
                            name = f'{input_features[i]}_to_{input_features[j]}'
                        else:
                            name = f'{i}_to_{j}'
                        ratio_names.append(name)
                else:
                    ratio_names.append('unknown_ratio')
        else:
            for (i, j) in self.feature_pairs_:
                if isinstance(i, int) and isinstance(j, int) and (i < len(input_features)) and (j < len(input_features)):
                    name = f'{input_features[i]}_to_{input_features[j]}'
                else:
                    name = f'{i}_to_{j}'
                ratio_names.append(name)
        return ratio_names