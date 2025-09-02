from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, Union, Dict, Any, List
from general.base_classes.transformer_base import BaseTransformer


# ...(code omitted)...


class SimpleBayesianTargetEncoder(BaseTransformer):
    """
    Simplified Bayesian target encoder with fixed smoothing parameter.
    
    This encoder applies Bayesian smoothing to target statistics, shrinking estimates
    toward the global mean based on the number of samples per category. This helps
    reduce overfitting for categories with few observations.
    
    Attributes
    ----------
    smoothing : float
        Smoothing parameter controlling the influence of global mean
    handle_unknown : str
        Strategy for handling unknown categories during transform
    category_stats_ : dict
        Dictionary mapping categories to their Bayesian-smoothed target statistics
    global_mean_ : float
        Global mean of the target variable
        
    Methods
    -------
    fit() : Fit encoder with Bayesian smoothing
    transform() : Apply Bayesian smoothed target encoding
    inverse_transform() : Not supported
    """

    def __init__(self, smoothing: float=1.0, handle_unknown: str='return_nan', name: Optional[str]=None):
        super().__init__(name=name)
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self.category_stats_ = {}
        self.global_mean_ = None

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'SimpleBayesianTargetEncoder':
        """
        Fit Bayesian target encoder using feature and target data.
        
        Computes smoothed target statistics for each category using Bayesian
        shrinkage towards the global mean.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Categorical feature data to encode
        y : np.ndarray
            Target values used for encoding statistics
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        SimpleBayesianTargetEncoder
            Fitted encoder instance with Bayesian-smoothed statistics
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim > 1 and X.shape[1] == 1:
            X = X.flatten()
        elif X.ndim > 1:
            raise ValueError('SimpleBayesianTargetEncoder expects a single categorical feature')
        X_str = np.array([str(x) for x in X])
        self.global_mean_ = np.mean(y)
        unique_categories = np.unique(X_str)
        self.category_stats_ = {}
        for category in unique_categories:
            mask = X_str == category
            category_targets = y[mask]
            category_count = len(category_targets)
            category_mean = np.mean(category_targets)
            smoothed_value = (category_count * category_mean + self.smoothing * self.global_mean_) / (category_count + self.smoothing)
            self.category_stats_[category] = smoothed_value
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical features using Bayesian smoothed target statistics.
        
        Applies precomputed Bayesian estimates to convert categories to numerical values.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Categorical feature data to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Encoded features with Bayesian-smoothed target statistics
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        if X.ndim > 1 and X.shape[1] == 1:
            X = X.flatten()
        elif X.ndim > 1:
            raise ValueError('SimpleBayesianTargetEncoder expects a single categorical feature')
        X_str = np.array([str(x) for x in X])
        encoded_values = []
        for category in X_str:
            if category in self.category_stats_:
                encoded_values.append(self.category_stats_[category])
            elif self.handle_unknown == 'return_nan':
                encoded_values.append(np.nan)
            else:
                raise ValueError(f"Unknown category '{category}' encountered during transform")
        encoded_array = np.array(encoded_values).reshape(-1, 1)
        if isinstance(data, FeatureSet):
            return FeatureSet(features=encoded_array, feature_names=feature_names, feature_types=['numeric'] if feature_names else None, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        else:
            return FeatureSet(features=encoded_array)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for Bayesian target encoding.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Encoded data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Original data (raises NotImplementedError)
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not supported
        """
        raise NotImplementedError('Inverse transform not supported for Bayesian target encoding')


# ...(code omitted)...


class MedianTargetEncoder(BaseTransformer):
    """
    Robust categorical encoder using median target statistics.
    
    This transformer encodes categorical features by replacing each category
    with the median target value for that category, providing outlier-resistant
    encoding compared to mean-based approaches.
    
    Attributes
    ----------
    min_samples_leaf : int
        Minimum number of samples required to compute per-category statistics
    handle_unknown : str
        Strategy for handling unknown categories ('return_nan' or 'error')
    _target_stats : dict
        Dictionary mapping categories to their median target values
    _global_median : float
        Global median of the target variable for fallback encoding
        
    Methods
    -------
    fit() : Compute median target statistics for each category
    transform() : Apply median target encoding to categorical features
    inverse_transform() : Not supported (raises NotImplementedError)
    """

    def __init__(self, min_samples_leaf: int=1, handle_unknown: str='return_nan', name: Optional[str]=None):
        super().__init__(name=name)
        self.min_samples_leaf = min_samples_leaf
        self.handle_unknown = handle_unknown
        self._target_stats = {}
        self._global_median = None

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'MedianTargetEncoder':
        """
        Fit median target encoder using feature and target data.
        
        Computes per-category median target values, ensuring categories meet
        the minimum samples threshold. Categories with insufficient samples
        fallback to global target median.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Categorical feature data to encode
        y : np.ndarray
            Target values used for encoding statistics
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        MedianTargetEncoder
            Fitted encoder instance with median target statistics
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim > 1 and X.shape[1] == 1:
            X = X.flatten()
        elif X.ndim > 1:
            raise ValueError('MedianTargetEncoder expects a single categorical feature')
        X_str = np.array([str(x) for x in X])
        self._global_median = np.median(y)
        unique_categories = np.unique(X_str)
        self._target_stats = {}
        for category in unique_categories:
            mask = X_str == category
            category_targets = y[mask]
            category_count = len(category_targets)
            if category_count >= self.min_samples_leaf:
                category_median = np.median(category_targets)
            else:
                category_median = self._global_median
            self._target_stats[category] = category_median
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical features using median target statistics.
        
        Maps categories to their corresponding median target values, handling
        unknown categories according to the specified strategy.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Categorical feature data to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Encoded features with median target values
        """
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
        if X.ndim > 1 and X.shape[1] == 1:
            X = X.flatten()
        elif X.ndim > 1:
            raise ValueError('MedianTargetEncoder expects a single categorical feature')
        X_str = np.array([str(x) for x in X])
        transformed_values = np.zeros(len(X_str), dtype=float)
        for (i, category) in enumerate(X_str):
            if category in self._target_stats:
                transformed_values[i] = self._target_stats[category]
            elif self.handle_unknown == 'return_nan':
                transformed_values[i] = np.nan
            else:
                raise ValueError(f"Unknown category '{category}' encountered during transform")
        transformed_features = transformed_values.reshape(-1, 1)
        new_feature_names = None
        if feature_names is not None:
            new_feature_names = [f'{feature_names[0]}_median_encoded'] if len(feature_names) > 0 else ['median_encoded_feature']
        metadata['encoding_type'] = 'median_target_encoding'
        metadata['original_feature_names'] = feature_names
        return FeatureSet(features=transformed_features, feature_names=new_feature_names, feature_types=['numeric'], sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for median target encoding.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Encoded data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Original data (raises NotImplementedError)
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not supported
        """
        raise NotImplementedError('Inverse transform is not supported for MedianTargetEncoder')


# ...(code omitted)...


class CVTargetEncoder(BaseTransformer):
    """
    Encode categorical features using cross-validated target encoding.
    
    This encoder applies cross-validation during fitting to compute target statistics,
    preventing overfitting by ensuring encoding is based on out-of-fold data.
    
    Attributes
    ----------
    cv_folds : int
        Number of cross-validation folds to use
    smoothing : float
        Smoothing parameter for regularization
    handle_unknown : str
        Strategy for handling unknown categories during transform
        
    Methods
    -------
    fit() : Fit encoder using cross-validated target statistics
    transform() : Apply cross-validated target encoding
    inverse_transform() : Not supported
    """

    def __init__(self, cv_folds: int=5, smoothing: float=1.0, handle_unknown: str='return_nan', min_samples_leaf: int=1, name: Optional[str]=None):
        super().__init__(name=name)
        self.cv_folds = cv_folds
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self.min_samples_leaf = min_samples_leaf
        self._target_stats = {}
        self._global_mean = None

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'CVTargetEncoder':
        """
        Fit CV target encoder using feature and target data with cross-validation.
        
        Computes out-of-fold target statistics using cross-validation to prevent
        overfitting. For each fold, categories are encoded using statistics from
        the other folds.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Categorical feature data to encode
        y : np.ndarray
            Target values used for encoding statistics
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        CVTargetEncoder
            Fitted encoder instance with cross-validated statistics
        """
        if isinstance(data, FeatureSet):
            X = data.features
            if X.shape[1] != 1:
                raise ValueError('CVTargetEncoder expects exactly one categorical feature')
            X = X.flatten()
        elif data.ndim == 2 and data.shape[1] == 1:
            X = data.flatten()
        elif data.ndim == 1:
            X = data
        else:
            raise ValueError('CVTargetEncoder expects 1D array or 2D array with one column')
        if len(X) != len(y):
            raise ValueError('Feature and target arrays must have the same length')
        self._global_mean = np.mean(y)
        category_sums = {}
        category_counts = {}
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.cv_folds, n_samples // self.cv_folds)
        fold_sizes[:n_samples % self.cv_folds] += 1
        current = 0
        folds = []
        for fold_size in fold_sizes:
            (start, stop) = (current, current + fold_size)
            folds.append(indices[start:stop])
            current = stop
        out_of_fold_predictions = np.zeros(n_samples)
        for i in range(self.cv_folds):
            val_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(self.cv_folds) if j != i])
            X_train = X[train_indices]
            y_train = y[train_indices]
            train_category_sums = {}
            train_category_counts = {}
            for (cat, target_val) in zip(X_train, y_train):
                cat_key = str(cat)
                if cat_key not in train_category_sums:
                    train_category_sums[cat_key] = 0.0
                    train_category_counts[cat_key] = 0
                train_category_sums[cat_key] += target_val
                train_category_counts[cat_key] += 1
            train_category_stats = {}
            for cat_key in train_category_sums:
                count = train_category_counts[cat_key]
                mean_val = train_category_sums[cat_key] / count
                smoothed_mean = (count * mean_val + self.smoothing * self._global_mean) / (count + self.smoothing)
                train_category_stats[cat_key] = smoothed_mean
            for idx in val_indices:
                cat_key = str(X[idx])
                if cat_key in train_category_stats and train_category_counts[cat_key] >= self.min_samples_leaf:
                    out_of_fold_predictions[idx] = train_category_stats[cat_key]
                else:
                    out_of_fold_predictions[idx] = self._global_mean
        final_category_sums = {}
        final_category_counts = {}
        for (cat, pred_val) in zip(X, out_of_fold_predictions):
            cat_key = str(cat)
            if cat_key not in final_category_sums:
                final_category_sums[cat_key] = 0.0
                final_category_counts[cat_key] = 0
            final_category_sums[cat_key] += pred_val
            final_category_counts[cat_key] += 1
        for cat_key in final_category_sums:
            count = final_category_counts[cat_key]
            mean_val = final_category_sums[cat_key] / count
            smoothed_mean = (count * mean_val + self.smoothing * self._global_mean) / (count + self.smoothing)
            self._target_stats[cat_key] = smoothed_mean
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical features using fitted cross-validated target statistics.
        
        Maps categories to their corresponding smoothed target statistics computed
        during fitting. Unknown categories are handled according to the specified strategy.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Categorical feature data to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Encoded features with cross-validated target statistics
        """
        if isinstance(data, FeatureSet):
            X_original = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
            if X_original.shape[1] != 1:
                raise ValueError('CVTargetEncoder expects exactly one categorical feature')
            X = X_original.flatten()
        else:
            X_original = data
            if data.ndim == 2 and data.shape[1] == 1:
                X = data.flatten()
            elif data.ndim == 1:
                X = data
            else:
                raise ValueError('CVTargetEncoder expects 1D array or 2D array with one column')
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        encoded_values = []
        for cat in X:
            cat_key = str(cat)
            if cat_key in self._target_stats:
                encoded_values.append(self._target_stats[cat_key])
            elif self.handle_unknown == 'return_nan':
                encoded_values.append(np.nan)
            else:
                raise ValueError(f"Unknown category '{cat}' encountered during transform")
        encoded_array = np.array(encoded_values).reshape(-1, 1)
        if feature_names:
            new_feature_names = [f'{feature_names[0]}_cv_encoded']
        else:
            new_feature_names = ['cv_encoded_feature']
        if feature_types:
            new_feature_types = ['numeric']
        else:
            new_feature_types = ['numeric']
        metadata['encoder_type'] = 'CVTargetEncoder'
        metadata['original_feature_count'] = 1
        result = FeatureSet(features=encoded_array, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        return result

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for cross-validated target encoding.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Encoded data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Original data (raises NotImplementedError)
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not supported
        """
        raise NotImplementedError('Inverse transform not supported for cross-validated target encoding')


# ...(code omitted)...


class OptimizedTargetEncoder(BaseTransformer):
    """
    Encode categorical features using optimization-based target encoding.
    
    This encoder uses numerical optimization techniques to compute target encoding
    that maximizes a specified objective function, such as likelihood or information gain.
    
    Attributes
    ----------
    objective : str
        Objective function to optimize ('likelihood', 'information_gain', 'mse')
    max_iterations : int
        Maximum number of optimization iterations
    tolerance : float
        Convergence tolerance for optimization
    handle_unknown : str
        Strategy for handling unknown categories during transform
        
    Methods
    -------
    fit() : Fit encoder using numerical optimization
    transform() : Apply optimized target encoding
    inverse_transform() : Not supported
    """

    def __init__(self, objective: str='mse', smoothing: float=1.0, handle_unknown: str='return_nan', name: Optional[str]=None):
        super().__init__(name=name)
        self.objective = objective
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self._target_stats = {}
        self._global_mean = None
        self._is_fitted = False
        if self.objective not in ['mse', 'likelihood', 'information_gain']:
            raise ValueError("objective must be one of 'mse', 'likelihood', or 'information_gain'")

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'OptimizedTargetEncoder':
        """
        Fit optimized target encoder using feature and target data.
        
        Computes target statistics by optimizing an objective function for each
        category to prevent overfitting and improve encoding quality.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Categorical feature data to encode
        y : np.ndarray
            Target values used for encoding statistics
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        OptimizedTargetEncoder
            Fitted encoder instance with optimized statistics
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim > 1 and X.shape[1] == 1:
            X = X.flatten()
        elif X.ndim > 1:
            raise ValueError('OptimizedTargetEncoder expects a single categorical feature')
        X_str = np.array([str(x) for x in X])
        self._global_mean = np.mean(y)
        unique_categories = np.unique(X_str)
        for category in unique_categories:
            category_mask = X_str == category
            category_targets = y[category_mask]
            category_count = len(category_targets)
            category_mean = np.mean(category_targets) if len(category_targets) > 0 else self._global_mean
            smoothed_mean = (category_count * category_mean + self.smoothing * self._global_mean) / (category_count + self.smoothing)
            self._target_stats[category] = smoothed_mean
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical features using optimized target statistics.
        
        Converts categories to numerical values based on precomputed optimized targets.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Categorical feature data to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Encoded features with optimized target statistics
        """
        if not self._is_fitted:
            raise ValueError("This OptimizedTargetEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
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
        if X.ndim > 1 and X.shape[1] == 1:
            X = X.flatten()
        elif X.ndim > 1:
            raise ValueError('OptimizedTargetEncoder expects a single categorical feature')
        X_str = np.array([str(x) for x in X])
        transformed_values = np.zeros(len(X_str), dtype=float)
        for (i, category) in enumerate(X_str):
            if category in self._target_stats:
                transformed_values[i] = self._target_stats[category]
            elif self.handle_unknown == 'return_nan':
                transformed_values[i] = np.nan
            else:
                raise ValueError(f"Unknown category '{category}' encountered during transform")
        transformed_features = transformed_values.reshape(-1, 1)
        new_feature_names = None
        if feature_names is not None:
            new_feature_names = [f'{feature_names[0]}_optimized_encoded'] if len(feature_names) > 0 else ['optimized_encoded_feature']
        metadata['encoding_type'] = 'optimized_target_encoding'
        metadata['original_feature_names'] = feature_names
        metadata['objective'] = self.objective
        return FeatureSet(features=transformed_features, feature_names=new_feature_names, feature_types=['numeric'], sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for optimized target encoding.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Encoded data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Original data (raises NotImplementedError)
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not supported
        """
        raise NotImplementedError('Inverse transform is not supported for OptimizedTargetEncoder')