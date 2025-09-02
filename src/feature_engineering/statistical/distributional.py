from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Union, Optional, List
from general.base_classes.transformer_base import BaseTransformer
from scipy.stats import entropy


# ...(code omitted)...


class ConditionalEntropyCalculator(BaseTransformer):
    """
    A transformer that computes conditional entropy H(Y|X) between feature pairs.
    
    This class calculates how much uncertainty remains in one variable (Y) given
    knowledge of another variable (X). Lower conditional entropy indicates stronger
    dependency between variables.
    
    Attributes:
        base (float): The logarithm base for entropy calculation (default: e for natural logarithm)
        normalize (bool): Whether to normalize entropy values
        handle_missing (str): How to handle missing values ('ignore', 'drop', or 'error')
    """

    def __init__(self, base: float=np.e, normalize: bool=False, handle_missing: str='ignore', name: Optional[str]=None):
        """
        Initialize the ConditionalEntropyCalculator.
        
        Args:
            base: Logarithm base for entropy calculation. Default is e (natural log).
            normalize: If True, normalizes entropy by maximum possible entropy.
            handle_missing: Strategy for handling missing values. Options are:
                          'ignore' (treat as separate category), 'drop' (exclude from calculation),
                          or 'error' (raise exception when encountered).
            name: Optional name for the transformer instance.
        """
        super().__init__(name=name)
        self.base = base
        self.normalize = normalize
        self.handle_missing = handle_missing

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ConditionalEntropyCalculator':
        """
        Fit the transformer to the input data (no actual fitting required for conditional entropy calculation).
        
        Args:
            data: Input data as FeatureSet or numpy array containing categorical features
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            self: Returns the fitted transformer instance
        """
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], condition_features: Optional[List[int]]=None, target_features: Optional[List[int]]=None, **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Calculate conditional entropy H(Y|X) for specified feature pairs.
        
        Args:
            data: Input data as FeatureSet or numpy array containing categorical features
            condition_features: Indices of features to use as conditioning variables (X).
                              If None, uses all features as conditioning variables.
            target_features: Indices of features to use as target variables (Y).
                           If None, uses all features as target variables.
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            Either a FeatureSet with conditional entropy values or numpy array of entropy values,
            depending on input type
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            return_feature_set = True
        else:
            X = data
            feature_names = None
            return_feature_set = False
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 2D array or FeatureSet with 2D features')
        (n_samples, n_features) = X.shape
        if self.handle_missing == 'error':
            if np.any([self._is_nan(val) for val in X.flatten()]):
                raise ValueError("Missing values found in data but handle_missing='error'")
        elif self.handle_missing == 'drop':
            valid_rows = np.array([not any((self._is_nan(val) for val in row)) for row in X])
            X = X[valid_rows]
            if X.shape[0] == 0:
                raise ValueError("All rows contain missing values and handle_missing='drop'")
        if condition_features is None:
            condition_features = list(range(n_features))
        if target_features is None:
            target_features = list(range(n_features))
        if not all((0 <= idx < n_features for idx in condition_features)):
            raise ValueError('Invalid condition feature index')
        if not all((0 <= idx < n_features for idx in target_features)):
            raise ValueError('Invalid target feature index')
        entropy_matrix = np.full((len(condition_features), len(target_features)), np.nan)
        for (i, cond_idx) in enumerate(condition_features):
            for (j, targ_idx) in enumerate(target_features):
                if cond_idx == targ_idx:
                    entropy_matrix[i, j] = 0.0
                else:
                    entropy_matrix[i, j] = self._compute_conditional_entropy(X[:, cond_idx], X[:, targ_idx])
        if self.normalize:
            for (i, cond_idx) in enumerate(condition_features):
                for (j, targ_idx) in enumerate(target_features):
                    if cond_idx != targ_idx:
                        h_y = self._compute_entropy(X[:, targ_idx])
                        if h_y > 0:
                            entropy_matrix[i, j] /= h_y
        if return_feature_set:
            cond_names = [feature_names[i] if feature_names else f'cond_{i}' for i in condition_features]
            targ_names = [feature_names[j] if feature_names else f'targ_{j}' for j in target_features]
            return FeatureSet(features=entropy_matrix, feature_names=targ_names, metadata={'condition_features': cond_names})
        else:
            return entropy_matrix

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation returns input unchanged.
        
        Args:
            data: Input data
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            The input data unchanged
        """
        return data

    def _is_nan(self, val) -> bool:
        """Check if a value is NaN, handling both float and non-float types."""
        if isinstance(val, float):
            return np.isnan(val)
        return False

    def _compute_entropy(self, values: np.ndarray) -> float:
        """Compute entropy of a discrete random variable."""
        if self.handle_missing == 'ignore':
            valid_values = [val for val in values if not self._is_nan(val)]
            if len(valid_values) == 0:
                return 0.0
            values = np.array(valid_values)
        elif self.handle_missing == 'drop':
            valid_values = [val for val in values if not self._is_nan(val)]
            if len(valid_values) == 0:
                return 0.0
            values = np.array(valid_values)
        elif self.handle_missing == 'error':
            if any((self._is_nan(val) for val in values)):
                raise ValueError("Missing values found but handle_missing='error'")
        if len(values) == 0:
            return 0.0
        (unique_vals, counts) = np.unique(values, return_counts=True)
        probabilities = counts / len(values)
        nonzero_probs = probabilities[probabilities > 0]
        if len(nonzero_probs) == 0:
            return 0.0
        if self.base == np.e:
            entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
        else:
            entropy = -np.sum(nonzero_probs * np.log(nonzero_probs) / np.log(self.base))
        return entropy

    def _compute_conditional_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute conditional entropy H(Y|X)."""
        if self.handle_missing == 'ignore':
            valid_pairs = [(x_val, y_val) for (x_val, y_val) in zip(x, y) if not (self._is_nan(x_val) or self._is_nan(y_val))]
            if len(valid_pairs) == 0:
                return 0.0
            (x_filtered, y_filtered) = zip(*valid_pairs)
            (x, y) = (np.array(x_filtered), np.array(y_filtered))
        elif self.handle_missing == 'drop':
            valid_pairs = [(x_val, y_val) for (x_val, y_val) in zip(x, y) if not (self._is_nan(x_val) or self._is_nan(y_val))]
            if len(valid_pairs) == 0:
                return 0.0
            (x_filtered, y_filtered) = zip(*valid_pairs)
            (x, y) = (np.array(x_filtered), np.array(y_filtered))
        elif self.handle_missing == 'error':
            if any((self._is_nan(x_val) or self._is_nan(y_val) for (x_val, y_val) in zip(x, y))):
                raise ValueError("Missing values found but handle_missing='error'")
        if len(x) == 0:
            return 0.0
        (x_unique, x_counts) = np.unique(x, return_counts=True)
        x_probs = x_counts / len(x)
        cond_entropy = 0.0
        for (x_val, x_prob) in zip(x_unique, x_probs):
            if x_prob > 0:
                indices = np.where(x == x_val)[0]
                y_given_x = y[indices]
                h_y_given_x = self._compute_entropy(y_given_x)
                cond_entropy += x_prob * h_y_given_x
        return cond_entropy

class ConditionalEntropyComputer(BaseTransformer):
    """
    A transformer that computes conditional entropy H(Y|X) between feature pairs.
    
    This class calculates how much uncertainty remains in target variables (Y) given
    knowledge of conditioning variables (X). Lower conditional entropy indicates stronger
    dependency between variable sets. The computation supports both discrete and continuous
    features with appropriate discretization for continuous variables.
    
    The transformer can compute conditional entropies for:
    1. All feature pairs (every feature as target conditioned on every other feature)
    2. Specific target-condition pairs as specified by indices
    3. All features as targets conditioned on a specific set of features
    
    Attributes:
        base (float): The logarithm base for entropy calculation (default: e for natural logarithm)
        discretize_continuous (bool): Whether to discretize continuous features before computation
        bins (int): Number of bins for discretizing continuous features
        handle_missing (str): How to handle missing values ('ignore', 'drop', or 'error')
    """

    def __init__(self, base: float=np.e, discretize_continuous: bool=True, bins: int=10, handle_missing: str='ignore', name: Optional[str]=None):
        """
        Initialize the ConditionalEntropyComputer.
        
        Args:
            base: Logarithm base for entropy calculation. Default is e (natural log).
            discretize_continuous: If True, continuous features are discretized before computation.
            bins: Number of bins to use when discretizing continuous features.
            handle_missing: Strategy for handling missing values. Options are:
                          'ignore' (treat as separate category), 'drop' (exclude from calculation),
                          or 'error' (raise exception when encountered).
            name: Optional name for the transformer instance.
        """
        super().__init__(name=name)
        self.base = base
        self.discretize_continuous = discretize_continuous
        self.bins = bins
        self.handle_missing = handle_missing

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ConditionalEntropyComputer':
        """
        Fit the transformer to the input data (prepares for conditional entropy computation).
        
        Args:
            data: Input data as FeatureSet or numpy array containing features
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            self: Returns the fitted transformer instance
        """
        if isinstance(data, FeatureSet):
            self._feature_names = data.feature_names
            self._data = data.features
        elif isinstance(data, np.ndarray):
            self._data = data
            self._feature_names = [f'feature_{i}' for i in range(data.shape[1])] if data.ndim > 1 else ['feature_0']
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        return self

    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """
        Handle missing values in the data according to the specified strategy.
        
        Args:
            data: Input data array
            
        Returns:
            Processed data with missing values handled
        """
        if self.handle_missing == 'error':
            if np.isnan(data).any() or (data.dtype.kind == 'O' and np.equal(data, None).any()):
                raise ValueError("Missing values found in data while handle_missing='error'")
        elif self.handle_missing == 'drop':
            if data.dtype.kind == 'O':
                mask = ~np.equal(data, None).any(axis=1)
            else:
                mask = ~np.isnan(data).any(axis=1)
            data = data[mask]
        return data

    def _discretize_feature(self, feature: np.ndarray) -> np.ndarray:
        """
        Discretize a continuous feature using equal-width binning.
        
        Args:
            feature: Continuous feature array
            
        Returns:
            Discretized feature array
        """
        finite_mask = np.isfinite(feature) if feature.dtype.kind in 'fc' else np.ones(len(feature), dtype=bool)
        if not finite_mask.all():
            feature = feature.copy()
            feature[~finite_mask] = np.nan
        finite_values = feature[finite_mask]
        if len(finite_values) == 0:
            return np.full_like(feature, 0, dtype=int)
        (min_val, max_val) = (finite_values.min(), finite_values.max())
        if min_val == max_val:
            return np.where(np.isfinite(feature), self.bins // 2, -1)
        discretized = np.floor((feature - min_val) / (max_val - min_val) * self.bins).astype(int)
        discretized = np.clip(discretized, 0, self.bins - 1)
        discretized[~np.isfinite(feature)] = -1
        return discretized

    def _compute_entropy(self, data: np.ndarray) -> float:
        """
        Compute entropy of a discrete variable.
        
        Args:
            data: Discrete data array
            
        Returns:
            Entropy value
        """
        if self.handle_missing in ['ignore', 'drop']:
            if data.dtype.kind == 'O':
                valid_mask = ~np.equal(data, None)
            else:
                valid_mask = ~np.isnan(data)
            data = data[valid_mask]
        if len(data) == 0:
            return 0.0
        (unique_vals, counts) = np.unique(data, return_counts=True)
        probs = counts / len(data)
        return entropy(probs, base=self.base)

    def _compute_joint_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute joint entropy H(X,Y).
        
        Args:
            x: First variable data
            y: Second variable data
            
        Returns:
            Joint entropy value
        """
        if self.handle_missing in ['ignore', 'drop']:
            if x.dtype.kind == 'O' or y.dtype.kind == 'O':
                valid_mask = ~(np.equal(x, None) | np.equal(y, None))
            else:
                valid_mask = ~(np.isnan(x) | np.isnan(y))
            (x, y) = (x[valid_mask], y[valid_mask])
        if len(x) == 0:
            return 0.0
        joint = np.vstack((x, y)).T
        (unique_pairs, counts) = np.unique(joint, axis=0, return_counts=True)
        probs = counts / len(joint)
        return entropy(probs, base=self.base)

    def _compute_conditional_entropy(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Compute conditional entropy H(Y|X) = H(X,Y) - H(X).
        
        Args:
            y: Target variable data
            x: Conditioning variable data
            
        Returns:
            Conditional entropy value
        """
        return self._compute_joint_entropy(x, y) - self._compute_entropy(x)

    def transform(self, data: Union[FeatureSet, np.ndarray], condition_indices: Optional[List[int]]=None, target_indices: Optional[List[int]]=None, **kwargs) -> FeatureSet:
        """
        Calculate conditional entropy H(Y|X) for specified feature pairs.
        
        Args:
            data: Input data as FeatureSet or numpy array containing features
            condition_indices: Indices of features to use as conditioning variables (X).
                             If None, uses all features as conditioning variables.
            target_indices: Indices of features to use as target variables (Y).
                          If None, uses all features as target variables.
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            FeatureSet: Contains conditional entropy values organized as a matrix where
                       row i, column j represents H(feature_j | feature_i) when both
                       indices are None, or appropriate subset when specified.
                       
        Raises:
            ValueError: If condition_indices or target_indices contain invalid indices
            TypeError: If data type is not supported
        """
        if isinstance(data, FeatureSet):
            raw_data = data.features
            feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            raw_data = data
            feature_names = [f'feature_{i}' for i in range(data.shape[1])] if data.ndim > 1 else ['feature_0']
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        processed_data = self._handle_missing_values(raw_data)
        if processed_data.ndim == 1:
            processed_data = processed_data.reshape(-1, 1)
        if self.discretize_continuous:
            discretized_data = np.empty_like(processed_data, dtype=int)
            for i in range(processed_data.shape[1]):
                discretized_data[:, i] = self._discretize_feature(processed_data[:, i])
            processed_data = discretized_data
        n_features = processed_data.shape[1]
        if condition_indices is not None:
            if not all((0 <= idx < n_features for idx in condition_indices)):
                raise ValueError('All condition_indices must be valid feature indices')
        if target_indices is not None:
            if not all((0 <= idx < n_features for idx in target_indices)):
                raise ValueError('All target_indices must be valid feature indices')
        if condition_indices is None:
            condition_indices = list(range(n_features))
        if target_indices is None:
            target_indices = list(range(n_features))
        result_matrix = np.zeros((len(condition_indices), len(target_indices)))
        for (i, cond_idx) in enumerate(condition_indices):
            for (j, target_idx) in enumerate(target_indices):
                result_matrix[i, j] = self._compute_conditional_entropy(processed_data[:, target_idx], processed_data[:, cond_idx])
        condition_names = [feature_names[idx] for idx in condition_indices]
        target_names = [feature_names[idx] for idx in target_indices]
        return FeatureSet(features=result_matrix, feature_names=target_names, metadata={'condition_features': condition_names})

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation returns input unchanged.
        
        Args:
            data: Input data
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            The input data unchanged
        """
        return data


# ...(code omitted)...


class SkewnessTransformer(BaseTransformer):
    """
    A transformer that applies skewness correction transformations to numerical features.
    
    This class implements various transformations to reduce or eliminate skewness in 
    numerical data distributions. Skewness measures the asymmetry of a distribution -
    positive skew indicates a longer right tail, while negative skew indicates a 
    longer left tail. Transforming skewed data is often necessary for algorithms 
    that assume normality or for improving model performance.
    
    Supported transformation methods:
    - Box-Cox transformation (for positive data only)
    - Yeo-Johnson transformation (works with negative values)
    - Log transformation (with shift for non-positive values)
    - Cube root transformation (preserves sign, works with all real numbers)
    - Square root transformation (for non-negative data)
    
    Attributes:
        method (str): Transformation method to use ('box-cox', 'yeo-johnson', 'log', 'cube-root', 'sqrt')
        lambda_param (Optional[float]): Fixed lambda parameter for Box-Cox/Yeo-Johnson; 
                                       if None, optimal lambda is estimated
        handle_negative (str): How to handle negative values for log/sqrt transforms 
                              ('shift', 'clip', 'error')
        preserve_mean (bool): If True, adjust transformed data to preserve original mean
    """

    def __init__(self, method: str='yeo-johnson', lambda_param: Optional[float]=None, handle_negative: str='shift', preserve_mean: bool=False, name: Optional[str]=None):
        """
        Initialize the SkewnessTransformer.
        
        Args:
            method: Transformation method to use. Options are:
                   'box-cox' (requires positive data), 
                   'yeo-johnson' (works with all real numbers),
                   'log' (natural log with handling for non-positive values),
                   'cube-root' (preserves sign, works with all real numbers),
                   'sqrt' (square root, requires non-negative data).
            lambda_param: Fixed lambda parameter for parametric transformations.
                         If None, optimal lambda is estimated from data.
            handle_negative: How to handle negative values for log/sqrt transforms.
                            Options are 'shift' (add constant to make positive),
                            'clip' (set negatives to small positive value),
                            'error' (raise exception for negative values).
            preserve_mean: If True, adjust transformed data to preserve original mean.
            name: Optional name for the transformer instance.
        """
        super().__init__(name=name)
        self.method = method
        self.lambda_param = lambda_param
        self.handle_negative = handle_negative
        self.preserve_mean = preserve_mean
        self._fitted_params = {}
        self.is_fitted_ = False

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'SkewnessTransformer':
        """
        Fit the transformer to estimate optimal transformation parameters.
        
        For parametric methods (Box-Cox, Yeo-Johnson), this estimates optimal lambda
        parameters for each feature. For other methods, it determines necessary 
        adjustments like shift constants for handling negative values.
        
        Args:
            data: Input data as FeatureSet or numpy array containing numerical features
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            self: Returns the fitted transformer instance with stored parameters
            
        Raises:
            ValueError: If data requirements for chosen method are not met
            TypeError: If data type is not supported
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            self._feature_names = None
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        (n_samples, n_features) = X.shape
        self._fitted_params = {'n_features_in_': n_features, 'original_means': np.mean(X, axis=0) if self.preserve_mean else None}
        valid_methods = ['box-cox', 'yeo-johnson', 'log', 'cube-root', 'sqrt']
        if self.method not in valid_methods:
            raise ValueError(f'Method must be one of {valid_methods}')
        if self.method in ['box-cox', 'yeo-johnson']:
            self._fitted_params['lambdas'] = np.zeros(n_features)
            for i in range(n_features):
                feature_data = X[:, i]
                if self.method == 'box-cox':
                    if np.any(feature_data <= 0):
                        raise ValueError('Box-Cox transformation requires all positive values')
                    if self.lambda_param is not None:
                        self._fitted_params['lambdas'][i] = self.lambda_param
                    else:
                        try:
                            (_, lambda_opt) = stats.boxcox(feature_data)
                            self._fitted_params['lambdas'][i] = lambda_opt
                        except Exception:
                            self._fitted_params['lambdas'][i] = 1.0
                elif self.method == 'yeo-johnson':
                    if self.lambda_param is not None:
                        self._fitted_params['lambdas'][i] = self.lambda_param
                    else:
                        try:
                            (_, lambda_opt) = stats.yeojohnson(feature_data)
                            self._fitted_params['lambdas'][i] = lambda_opt
                        except Exception:
                            self._fitted_params['lambdas'][i] = 1.0
        elif self.method == 'log':
            self._fitted_params['shift_amounts'] = np.zeros(n_features)
            for i in range(n_features):
                feature_data = X[:, i]
                min_val = np.min(feature_data)
                if min_val <= 0:
                    if self.handle_negative == 'error':
                        raise ValueError(f'Log transformation cannot handle non-positive values in feature {i}')
                    elif self.handle_negative == 'shift':
                        self._fitted_params['shift_amounts'][i] = 1 - min_val
                    elif self.handle_negative == 'clip':
                        self._fitted_params['shift_amounts'][i] = 0.0
                    else:
                        raise ValueError("handle_negative must be 'shift', 'clip', or 'error'")
        elif self.method == 'sqrt':
            for i in range(n_features):
                feature_data = X[:, i]
                if np.any(feature_data < 0):
                    if self.handle_negative == 'error':
                        raise ValueError(f'Square root transformation cannot handle negative values in feature {i}')
                    elif self.handle_negative == 'clip':
                        pass
                    elif self.handle_negative == 'shift':
                        pass
                    else:
                        raise ValueError("handle_negative must be 'shift', 'clip', or 'error'")
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply skewness transformation to the input data.
        
        Args:
            data: Input data as FeatureSet or numpy array containing numerical features
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            Transformed data with reduced skewness, in same format as input
            
        Raises:
            ValueError: If transformer was not fitted or data dimensions don't match
            RuntimeError: If transformation fails due to numerical issues
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            input_is_feature_set = True
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            input_is_feature_set = False
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if X.shape[1] != self._fitted_params['n_features_in_']:
            raise ValueError(f"Input data has {X.shape[1]} features, but transformer was fitted on {self._fitted_params['n_features_in_']} features")
        X_transformed = np.zeros_like(X, dtype=float)
        n_features = X.shape[1]
        for i in range(n_features):
            feature_data = X[:, i].copy()
            if self.method == 'box-cox':
                lambda_val = self._fitted_params['lambdas'][i]
                if lambda_val == 0:
                    X_transformed[:, i] = np.log(feature_data)
                else:
                    X_transformed[:, i] = (np.power(feature_data, lambda_val) - 1) / lambda_val
            elif self.method == 'yeo-johnson':
                lambda_val = self._fitted_params['lambdas'][i]
                X_transformed[:, i] = self._yeo_johnson_transform(feature_data, lambda_val)
            elif self.method == 'log':
                if self.handle_negative == 'shift':
                    shifted_data = feature_data + self._fitted_params['shift_amounts'][i]
                    shifted_data = np.maximum(shifted_data, 1e-10)
                    X_transformed[:, i] = np.log(shifted_data)
                elif self.handle_negative == 'clip':
                    clipped_data = np.clip(feature_data, 1e-10, None)
                    X_transformed[:, i] = np.log(clipped_data)
                else:
                    X_transformed[:, i] = np.log(np.maximum(feature_data, 1e-10))
            elif self.method == 'cube-root':
                X_transformed[:, i] = np.sign(feature_data) * np.power(np.abs(feature_data), 1 / 3)
            elif self.method == 'sqrt':
                if self.handle_negative == 'clip':
                    feature_data = np.clip(feature_data, 0, None)
                elif self.handle_negative == 'shift':
                    min_val = np.min(feature_data)
                    if min_val < 0:
                        feature_data = feature_data - min_val + 1e-10
                feature_data = np.maximum(feature_data, 0)
                X_transformed[:, i] = np.sqrt(feature_data)
        if self.preserve_mean and self._fitted_params['original_means'] is not None:
            current_means = np.mean(X_transformed, axis=0)
            X_transformed = X_transformed - current_means + self._fitted_params['original_means']
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            if data.ndim == 1:
                return X_transformed.flatten()
            return X_transformed

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """Apply Yeo-Johnson transformation."""
        result = np.zeros_like(x, dtype=float)
        pos = x >= 0
        if lmbda == 0:
            result[pos] = np.log(x[pos] + 1)
        else:
            result[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda
        neg = x < 0
        if lmbda == 2:
            result[neg] = -np.log(-x[neg] + 1)
        else:
            result[neg] = -(np.power(-x[neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        return result

    def _yeo_johnson_inverse_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """Apply inverse Yeo-Johnson transformation."""
        result = np.zeros_like(x, dtype=float)
        pos = x >= 0
        if lmbda == 0:
            result[pos] = np.exp(x[pos]) - 1
        else:
            result[pos] = np.power(lmbda * x[pos] + 1, 1 / lmbda) - 1
        neg = x < 0
        if lmbda == 2:
            result[neg] = 1 - np.exp(-x[neg])
        else:
            result[neg] = 1 - np.power(-(2 - lmbda) * x[neg] + 1, 1 / (2 - lmbda))
        return result

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply inverse skewness transformation to revert data to original scale.
        
        Args:
            data: Transformed data as FeatureSet or numpy array
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            Data in original scale, in same format as input
            
        Raises:
            ValueError: If transformer was not fitted or data dimensions don't match
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            input_is_feature_set = True
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            input_is_feature_set = False
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if X.shape[1] != self._fitted_params['n_features_in_']:
            raise ValueError(f"Input data has {X.shape[1]} features, but transformer was fitted on {self._fitted_params['n_features_in_']} features")
        X_inverse = np.zeros_like(X, dtype=float)
        n_features = X.shape[1]
        for i in range(n_features):
            feature_data = X[:, i].copy()
            if self.method == 'box-cox':
                lambda_val = self._fitted_params['lambdas'][i]
                if lambda_val == 0:
                    X_inverse[:, i] = np.exp(feature_data)
                else:
                    X_inverse[:, i] = np.power(lambda_val * feature_data + 1, 1 / lambda_val)
            elif self.method == 'yeo-johnson':
                lambda_val = self._fitted_params['lambdas'][i]
                X_inverse[:, i] = self._yeo_johnson_inverse_transform(feature_data, lambda_val)
            elif self.method == 'log':
                if self.handle_negative == 'shift':
                    X_inverse[:, i] = np.exp(feature_data) - self._fitted_params['shift_amounts'][i]
                elif self.handle_negative == 'clip':
                    X_inverse[:, i] = np.exp(feature_data)
                else:
                    X_inverse[:, i] = np.exp(feature_data)
            elif self.method == 'cube-root':
                X_inverse[:, i] = np.power(feature_data, 3)
            elif self.method == 'sqrt':
                X_inverse[:, i] = np.power(feature_data, 2)
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_inverse, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            if data.ndim == 1:
                return X_inverse.flatten()
            return X_inverse

class KurtosisCalculator(BaseTransformer):
    """
    Transformer to compute kurtosis for numerical features.
    
    Calculates kurtosis using scipy.stats.kurtosis with support for both 
    Fisher's (excess) and Pearson's definitions, optional bias correction,
    and configurable NaN handling policies.
    
    Attributes
    ----------
    fisher : bool, default=True
        If True, uses Fisher's definition (normal ==> 0.0).
        If False, uses Pearson's definition (normal ==> 3.0).
    bias : bool, default=False
        If False, bias correction is applied in calculations.
    nan_policy : {'propagate', 'raise', 'omit'}, default='propagate'
        Defines how to handle NaN values in input data:
        - 'propagate': returns NaN if any NaN values are present
        - 'raise': throws an error if NaN values are present
        - 'omit': ignores NaN values during calculation
        
    Examples
    --------
    >>> import numpy as np
    >>> from src.feature_engineering.statistical.distributional import KurtosisCalculator
    >>> data = np.random.normal(0, 1, (100, 2))
    >>> calculator = KurtosisCalculator()
    >>> calculator.fit(data)
    >>> kurtosis_values = calculator.transform(data)
    >>> print(kurtosis_values)
    """

    def __init__(self, fisher: bool=True, bias: bool=False, nan_policy: str='propagate'):
        super().__init__()
        self.fisher = fisher
        self.bias = bias
        self.nan_policy = nan_policy

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'KurtosisCalculator':
        """
        Fit the transformer to input data by validating numerical features.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing numerical features
        **kwargs : dict
            Additional parameters (unused)
            
        Returns
        -------
        KurtosisCalculator
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If input data contains non-numerical features
        TypeError
            If input data is neither FeatureSet nor numpy array
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('All features must be numerical')
        self._fitted_params = {'n_features_in_': X.shape[1], 'is_fitted_': True}
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Compute kurtosis for each feature in the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data for kurtosis calculation
        **kwargs : dict
            Additional parameters (unused)
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Kurtosis values for each feature in same format as input
            
        Raises
        ------
        ValueError
            If transformer was not fitted or data dimensions don't match
        """
        if not hasattr(self, '_fitted_params') or not self._fitted_params.get('is_fitted_', False):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            input_is_feature_set = True
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            input_is_feature_set = False
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if X.shape[1] != self._fitted_params['n_features_in_']:
            raise ValueError(f"Input data has {X.shape[1]} features, but transformer was fitted on {self._fitted_params['n_features_in_']} features")
        from scipy.stats import kurtosis
        kurtosis_values = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            kurtosis_values[i] = kurtosis(X[:, i], fisher=self.fisher, bias=self.bias, nan_policy=self.nan_policy)
        if input_is_feature_set:
            kurtosis_2d = kurtosis_values.reshape(1, -1)
            return FeatureSet(features=kurtosis_2d, feature_names=feature_names, metadata={'transformation': 'kurtosis'})
        else:
            if data.ndim == 1:
                return kurtosis_values.flatten()
            return kurtosis_values

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Return input data unchanged as kurtosis transformation is not invertible.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to pass through unchanged
        **kwargs : dict
            Additional parameters (unused)
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Input data unchanged
        """
        return data

def calculate_entropy(data: Union[np.ndarray, FeatureSet], base: float=np.e, normalize: bool=False, handle_missing: str='ignore') -> Union[float, np.ndarray]:
    """
    Calculate Shannon entropy for categorical features.
    
    This function computes the Shannon entropy of categorical variables, which measures
    the uncertainty or randomness in the distribution of categories. Higher entropy
    indicates more uniform distribution, while lower entropy suggests more predictable
    category assignments.
    
    Args:
        data: Input data as numpy array or FeatureSet containing categorical features.
             If numpy array, assumed to be 1D (single feature) or 2D (multiple features).
        base: Logarithm base for entropy calculation. Default is e (natural log).
        normalize: If True, normalizes entropy by dividing by log(base, n_categories).
        handle_missing: Strategy for handling missing values. Options are:
                      'ignore' (treat as separate category), 'drop' (exclude from calculation),
                      or 'error' (raise exception when encountered).
                      
    Returns:
        float or numpy array: Entropy value(s) for the input feature(s). Returns a single
        float for 1D input or array of entropy values for 2D input (one per column).
        
    Raises:
        ValueError: If handle_missing is 'error' and missing values are found,
                   or if data dimensions are incompatible.
        TypeError: If data type is not supported.
    """
    if isinstance(data, FeatureSet):
        data = data.features
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim == 1:
        return _compute_entropy_for_column(data, base, normalize, handle_missing)
    elif data.ndim == 2:
        entropies = []
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            entropy_val = _compute_entropy_for_column(col_data, base, normalize, handle_missing)
            entropies.append(entropy_val)
        return np.array(entropies)
    else:
        raise ValueError('Input data must be 1D or 2D')