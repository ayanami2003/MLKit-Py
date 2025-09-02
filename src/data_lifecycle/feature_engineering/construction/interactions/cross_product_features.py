from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from itertools import combinations

class CrossProductFeatureTransformer(BaseTransformer):
    """
    Transformer for generating cross-product interaction features from input features.
    
    This transformer creates interaction terms by computing the element-wise product
    between pairs of features. It supports selecting specific feature pairs or
    generating all pairwise combinations up to a specified degree.
    
    Attributes
    ----------
    feature_pairs : Optional[List[tuple]]
        Specific pairs of feature indices to create cross products for.
        If None, all pairwise combinations will be generated.
    degree : int
        Maximum degree of interactions to generate (default=2 for pairwise).
    include_bias : bool
        Whether to include a bias column of ones in the output (default=False).
    name : Optional[str]
        Name of the transformer instance.
    """

    def __init__(self, feature_pairs: Optional[List[tuple]]=None, degree: int=2, include_bias: bool=False, name: Optional[str]=None):
        super().__init__(name=name)
        self.feature_pairs = feature_pairs
        self.degree = degree
        self.include_bias = include_bias
        self._feature_names = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'CrossProductFeatureTransformer':
        """
        Fit the transformer to the input data.
        
        This method identifies the feature pairs to be used for cross-product
        generation if not explicitly provided during initialization.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing features to be transformed.
        **kwargs : dict
            Additional parameters for fitting (not used).
            
        Returns
        -------
        CrossProductFeatureTransformer
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._input_feature_names = data.feature_names
        else:
            X = data
            self._input_feature_names = None
        self.n_features_in_ = X.shape[1]
        if self.feature_pairs is None:
            feature_indices = list(range(self.n_features_in_))
            self.feature_pairs_ = []
            for d in range(2, min(self.degree + 1, self.n_features_in_ + 1)):
                self.feature_pairs_.extend(list(combinations(feature_indices, d)))
        else:
            self.feature_pairs_ = self.feature_pairs
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Generate cross-product features from the input data.
        
        Creates new features by computing element-wise products of selected
        feature pairs. Optionally includes a bias column.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters for transformation (not used).
            
        Returns
        -------
        FeatureSet
            Transformed data with cross-product features.
        """
        if not hasattr(self, 'feature_pairs_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        else:
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        interaction_features = []
        for pair in self.feature_pairs_:
            interaction_feature = np.ones(X.shape[0])
            for idx in pair:
                interaction_feature *= X[:, idx]
            interaction_features.append(interaction_feature)
        if interaction_features:
            interaction_matrix = np.column_stack(interaction_features)
            if interaction_matrix.ndim == 1:
                interaction_matrix = interaction_matrix.reshape(-1, 1)
            X_combined = np.hstack([X, interaction_matrix])
        else:
            X_combined = X
        if self.include_bias:
            bias_column = np.ones((X_combined.shape[0], 1))
            X_transformed = np.hstack([bias_column, X_combined])
        else:
            X_transformed = X_combined
        new_feature_names = None
        if feature_names is not None:
            new_feature_names = feature_names.copy()
        elif self._input_feature_names is not None:
            new_feature_names = self._input_feature_names.copy()
        else:
            new_feature_names = [f'x{i}' for i in range(self.n_features_in_)]
        for pair in self.feature_pairs_:
            if self._input_feature_names is not None:
                names = [self._input_feature_names[i] for i in pair]
            elif feature_names is not None:
                names = [feature_names[i] for i in pair]
            else:
                names = [f'x{i}' for i in pair]
            interaction_name = '_x_'.join(names)
            new_feature_names.append(interaction_name)
        if self.include_bias:
            new_feature_names = ['bias'] + new_feature_names
        new_feature_types = None
        if feature_types is not None:
            new_feature_types = feature_types.copy()
            new_feature_types.extend(['numeric'] * len(self.feature_pairs_))
            if self.include_bias:
                new_feature_types = ['numeric'] + new_feature_types
        return FeatureSet(features=X_transformed, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for this transformer.
        
        Cross-product features are not invertible to the original feature space.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Original data without cross-product features.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for cross-product features.')

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get names of the generated cross-product features.
        
        Parameters
        ----------
        input_features : Optional[List[str]]
            Names of the input features.
            
        Returns
        -------
        List[str]
            Names of the generated cross-product features.
        """
        if not hasattr(self, 'feature_pairs_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if input_features is None:
            if hasattr(self, '_input_feature_names') and self._input_feature_names is not None:
                input_features = self._input_feature_names
            else:
                input_features = [f'x{i}' for i in range(self.n_features_in_)]
        feature_names = input_features.copy()
        for pair in self.feature_pairs_:
            names = [input_features[i] for i in pair]
            interaction_name = '_x_'.join(names)
            feature_names.append(interaction_name)
        if self.include_bias:
            feature_names = ['bias'] + feature_names
        return feature_names