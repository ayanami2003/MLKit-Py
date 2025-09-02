from typing import Optional, List, Union, Dict, Tuple
from general.structures.feature_set import FeatureSet
import numpy as np
from general.base_classes.transformer_base import BaseTransformer


# ...(code omitted)...


class FisherScoreCalculator(BaseTransformer):
    """
    Transformer that computes Fisher scores for feature relevance assessment.
    
    Fisher score measures the discriminatory power of continuous features with respect to
    class labels. It evaluates the ratio of between-class variance to within-class variance
    for each feature, providing a univariate measure of feature importance for classification.
    
    Attributes:
        name (Optional[str]): Name of the transformer instance.
    """

    def __init__(self, name: Optional[str]=None):
        """
        Initialize the FisherScoreCalculator.
        
        Args:
            name: Optional name for the transformer.
        """
        super().__init__(name)
        self._fisher_scores = None
        self._fitted = False

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'FisherScoreCalculator':
        """
        Fit the transformer to compute Fisher scores.
        
        Args:
            data: Input features as FeatureSet or numpy array of shape (n_samples, n_features).
            y: Target labels as numpy array of shape (n_samples,).
            **kwargs: Additional parameters (unused).
            
        Returns:
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = np.asarray(data)
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError('Target labels must be a 1D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of samples in features and labels must match')
        (n_samples, n_features) = X.shape
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError('At least two classes are required for Fisher score calculation')
        fisher_scores = np.zeros(n_features)
        global_mean = np.mean(X, axis=0)
        between_class_var = np.zeros(n_features)
        within_class_var = np.zeros(n_features)
        for cls in unique_classes:
            class_mask = y == cls
            class_samples = X[class_mask]
            n_class_samples = class_samples.shape[0]
            class_mean = np.mean(class_samples, axis=0)
            between_class_var += n_class_samples * (class_mean - global_mean) ** 2
            if n_class_samples > 1:
                class_var = np.var(class_samples, axis=0, ddof=1)
                within_class_var += (n_class_samples - 1) * class_var
        fisher_scores = np.divide(between_class_var, within_class_var, out=np.zeros_like(between_class_var), where=within_class_var != 0)
        self._fisher_scores = fisher_scores
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the input data (returns input unchanged as feature selection is not automatic).
        
        Args:
            data: Input data as FeatureSet or numpy array.
            **kwargs: Additional parameters (unused).
            
        Returns:
            FeatureSet with the same features (transformation is identity).
        """
        if not self._fitted:
            raise RuntimeError("Transformer has not been fitted. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            return data
        else:
            return FeatureSet(features=np.asarray(data))

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation returns input unchanged.
        
        Args:
            data: Transformed data.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Original data unchanged.
        """
        return data

    def get_fisher_scores(self) -> np.ndarray:
        """
        Retrieve the computed Fisher scores.
        
        Returns:
            Array of Fisher scores for each feature.
            
        Raises:
            RuntimeError: If transformer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Transformer has not been fitted. Call 'fit' first.")
        return self._fisher_scores.copy()

class CrossTabulationGenerator(BaseTransformer):
    """
    Transformer that generates cross-tabulation features from categorical variables.
    
    This transformer creates cross-tabulation tables between pairs of categorical features,
    computing statistics like counts, normalized frequencies, and association measures.
    These features help capture relationships between categorical variables.
    
    Attributes:
        feature_pairs (List[Tuple[int, int]]): Pairs of feature indices to cross-tabulate.
        normalize (bool): Whether to normalize counts to frequencies.
        name (Optional[str]): Name of the transformer instance.
    """

    def __init__(self, feature_pairs: List[Tuple[int, int]], normalize: bool=True, name: Optional[str]=None):
        """
        Initialize the CrossTabulationGenerator.
        
        Args:
            feature_pairs: Pairs of feature indices to cross-tabulate.
            normalize: Whether to normalize counts to frequencies.
            name: Optional name for the transformer.
        """
        super().__init__(name)
        self.feature_pairs = feature_pairs
        self.normalize = normalize

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'CrossTabulationGenerator':
        """
        Fit the generator to prepare for cross-tabulation.
        
        Validates feature pairs and prepares for cross-tabulation computation.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            num_features = data.X.shape[1]
        elif isinstance(data, np.ndarray):
            num_features = data.shape[1]
        else:
            raise TypeError('Input data must be either a FeatureSet or a numpy array.')
        for pair in self.feature_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError('Each feature pair must be a tuple or list of two integers.')
            if not all((isinstance(idx, int) and 0 <= idx < num_features for idx in pair)):
                raise IndexError(f'Feature indices in pair {pair} are out of bounds for data with {num_features} features.')
        self._num_features = num_features
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Generate cross-tabulation features from the input data.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            **kwargs: Additional parameters (unused).
            
        Returns:
            FeatureSet with cross-tabulation features.
        """
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError('Input numpy array must be 2-dimensional.')
            X = data
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            metadata = {}
        elif isinstance(data, FeatureSet):
            X = data.X
            feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
            metadata = data.metadata.copy() if data.metadata is not None else {}
        else:
            raise TypeError('Input data must be either a FeatureSet or a numpy array.')
        if X.shape[1] != self._num_features:
            raise ValueError(f'Number of features in data ({X.shape[1]}) does not match the number seen during fit ({self._num_features}).')
        new_features = []
        new_feature_names = []
        for (i, j) in self.feature_pairs:
            col_i = X[:, i]
            col_j = X[:, j]
            combined = np.array([col_i, col_j]).T
            (unique_combinations, _, inverse_indices) = np.unique(combined, axis=0, return_index=True, return_inverse=True)
            counts = np.bincount(inverse_indices)
            base_name = f'{feature_names[i]}_x_{feature_names[j]}'
            for (k, (val_i, val_j)) in enumerate(unique_combinations):
                feature_name = f'{base_name}_{val_i}_to_{val_j}'
                new_feature_values = (combined == [val_i, val_j]).all(axis=1).astype(float)
                if self.normalize:
                    total_count = len(col_i)
                    new_feature_values = new_feature_values / total_count
                new_features.append(new_feature_values)
                new_feature_names.append(feature_name)
        if new_features:
            cross_tab_features = np.column_stack(new_features)
        else:
            cross_tab_features = np.empty((X.shape[0], 0))
        combined_X = np.hstack([X, cross_tab_features])
        combined_feature_names = feature_names + new_feature_names
        return FeatureSet(X=combined_X, feature_names=combined_feature_names, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation is not supported for cross-tabulation features.
        
        Args:
            data: Transformed data (unused).
            **kwargs: Additional parameters (unused).
            
        Returns:
            Original data unchanged.
        """
        raise NotImplementedError('Inverse transformation is not supported for cross-tabulation features.')