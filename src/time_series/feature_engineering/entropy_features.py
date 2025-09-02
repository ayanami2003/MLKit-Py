from typing import Union, Optional, List, Tuple
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np


# ...(code omitted)...


class EntropyFeatureExtractor(BaseTransformer):

    def __init__(self, method: str='shannon', embedding_dimension: int=3, tolerance: float=0.1, feature_columns: Optional[List[str]]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.method = method
        self.embedding_dimension = embedding_dimension
        self.tolerance = tolerance
        self.feature_columns = feature_columns
        valid_methods = ['shannon', 'sample', 'approximate', 'permutation']
        if self.method not in valid_methods:
            raise ValueError(f'Method must be one of {valid_methods}, got {self.method}')

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'EntropyFeatureExtractor':
        """
        Fit the entropy feature extractor to the input data.
        
        For entropy calculations, fitting typically involves validating input data
        and storing any necessary parameters.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input time series data to fit on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        EntropyFeatureExtractor
            Self instance for method chaining.
        """
        if not isinstance(data, (FeatureSet, np.ndarray)):
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
            self._feature_types = data.feature_types if data.feature_types is not None else ['numeric'] * len(self._feature_names)
        else:
            X = data
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self._feature_types = ['numeric'] * len(self._feature_names)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        (self.n_samples_, self.n_features_) = X.shape
        if self.feature_columns is not None:
            invalid_columns = [col for col in self.feature_columns if col not in self._feature_names]
            if invalid_columns:
                raise ValueError(f'Specified feature columns not found in data: {invalid_columns}')
        if self.embedding_dimension <= 0:
            raise ValueError('embedding_dimension must be positive')
        if self.tolerance <= 0:
            raise ValueError('tolerance must be positive')
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Compute entropy-based features for the input time series data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input time series data to transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            FeatureSet containing computed entropy features.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names if data.feature_names is not None else self._feature_names
            feature_types = data.feature_types if data.feature_types is not None else self._feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        else:
            X = data
            feature_names = self._feature_names
            feature_types = self._feature_types
            sample_ids = None
            metadata = {}
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.feature_columns is not None:
            column_indices = [feature_names.index(col) for col in self.feature_columns]
            process_columns = self.feature_columns
        else:
            column_indices = list(range(len(feature_names)))
            process_columns = feature_names
        entropy_features = []
        entropy_feature_names = []
        entropy_feature_types = []
        for (col_idx, col_name) in zip(column_indices, process_columns):
            series = X[:, col_idx]
            if np.std(series) == 0:
                entropy_val = 0.0
            elif self.method == 'shannon':
                entropy_val = self._shannon_entropy(series)
            elif self.method == 'sample':
                entropy_val = self._sample_entropy(series)
            elif self.method == 'approximate':
                entropy_val = self._approximate_entropy(series)
            elif self.method == 'permutation':
                entropy_val = self._permutation_entropy(series)
            else:
                raise ValueError(f'Unsupported method: {self.method}')
            entropy_val = max(0.0, entropy_val)
            entropy_features.append(entropy_val)
            entropy_feature_names.append(f'{col_name}_entropy')
            entropy_feature_types.append('numeric')
        n_samples = X.shape[0]
        entropy_features_array = np.tile(entropy_features, (n_samples, 1))
        new_features = np.hstack([X, entropy_features_array])
        new_feature_names = list(feature_names) + entropy_feature_names
        new_feature_types = list(feature_types) + entropy_feature_types
        result_metadata = metadata.copy()
        result_metadata.update({'entropy_method': self.method, 'embedding_dimension': self.embedding_dimension, 'tolerance': self.tolerance})
        return FeatureSet(features=new_features, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=sample_ids, metadata=result_metadata)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for entropy features.
        
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
            Always raised as inverse transformation is not meaningful for entropy features.
        """
        raise NotImplementedError('Inverse transformation is not supported for entropy features.')

    def _shannon_entropy(self, series: np.ndarray) -> float:
        """
        Compute Shannon entropy of a time series.
        
        Parameters
        ----------
        series : np.ndarray
            Input time series.
            
        Returns
        -------
        float
            Shannon entropy value.
        """
        (hist, _) = np.histogram(series, bins='auto', density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        return -np.sum(hist * np.log2(hist))

    def _sample_entropy(self, series: np.ndarray) -> float:
        """
        Compute Sample entropy of a time series.
        
        Parameters
        ----------
        series : np.ndarray
            Input time series.
            
        Returns
        -------
        float
            Sample entropy value.
        """
        import math
        N = len(series)
        if N <= self.embedding_dimension:
            return 0.0
        D = self.embedding_dimension
        r = self.tolerance * np.std(series)
        if r == 0:
            return 0.0

        def _get_patterns(dim):
            patterns = []
            for i in range(N - dim + 1):
                patterns.append(series[i:i + dim])
            return np.array(patterns)
        patterns_m = _get_patterns(D)
        patterns_m1 = _get_patterns(D + 1)
        if len(patterns_m) == 0 or len(patterns_m1) == 0:
            return 0.0

        def _count_matches(patterns, r):
            count = 0
            total = 0
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    total += 1
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                        count += 1
            return (count, total)
        (A, B) = _count_matches(patterns_m, r)
        (C, D_count) = _count_matches(patterns_m1, r)
        if A == 0 or C == 0 or B == 0 or (D_count == 0):
            return 0.0
        phi_m = A / B
        phi_m1 = C / D_count
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
        return -math.log(phi_m1 / phi_m)

    def _approximate_entropy(self, series: np.ndarray) -> float:
        """
        Compute Approximate entropy of a time series.
        
        Parameters
        ----------
        series : np.ndarray
            Input time series.
            
        Returns
        -------
        float
            Approximate entropy value.
        """
        import math
        N = len(series)
        if N <= self.embedding_dimension:
            return 0.0
        D = self.embedding_dimension
        r = self.tolerance * np.std(series)
        if r == 0:
            return 0.0

        def _get_patterns(dim):
            patterns = []
            for i in range(N - dim + 1):
                patterns.append(series[i:i + dim])
            return np.array(patterns)

        def _compute_phi(dim):
            patterns = _get_patterns(dim)
            if len(patterns) == 0:
                return 0.0
            C = np.zeros(len(patterns))
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j and np.max(np.abs(patterns[i] - patterns[j])) <= r:
                        C[i] += 1
            C = C / (len(patterns) - 1)
            C_nonzero = C[C > 0]
            if len(C_nonzero) == 0:
                return 0.0
            return np.mean(np.log(C_nonzero))
        phi_m = _compute_phi(D)
        phi_m1 = _compute_phi(D + 1)
        return phi_m - phi_m1

    def _permutation_entropy(self, series: np.ndarray) -> float:
        """
        Compute Permutation entropy of a time series.
        
        Parameters
        ----------
        series : np.ndarray
            Input time series.
            
        Returns
        -------
        float
            Permutation entropy value.
        """
        import math
        N = len(series)
        D = self.embedding_dimension
        if N < D:
            return 0.0
        permutations = []
        for i in range(N - D + 1):
            permutations.append(series[i:i + D])
        if len(permutations) == 0:
            return 0.0
        order_patterns = []
        for perm in permutations:
            sorted_indices = np.argsort(perm)
            pattern = tuple(sorted_indices)
            order_patterns.append(pattern)
        pattern_counts = {}
        for pattern in order_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        total_patterns = len(order_patterns)
        probabilities = [count / total_patterns for count in pattern_counts.values()]
        entropy = -sum((p * np.log(p) for p in probabilities if p > 0))
        max_entropy = np.log(math.factorial(D))
        if max_entropy > 0:
            entropy = entropy / max_entropy
        return entropy