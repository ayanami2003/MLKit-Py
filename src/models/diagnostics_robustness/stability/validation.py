import numpy as np
from typing import Union, Dict, Any, Callable, Optional, List
from sklearn.model_selection import StratifiedKFold
import copy
from copy import deepcopy
import inspect
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.data_manipulation.data_splitting.split_operations.stratified_split import stratified_split
from typing import Union, Dict, Any, Optional, List
from src.data_lifecycle.computational_utilities.random_operations.sampling_methods import generate_random_samples
import time

def stratified_sampling_validation(model: BaseModel, X: Union[np.ndarray, DataBatch, FeatureSet], y: Union[np.ndarray, List], n_splits: int=5, test_size: float=0.2, random_state: Optional[int]=None) -> Dict[str, Any]:
    """
    Perform stratified sampling validation to assess model stability across different data splits while preserving class distributions.

    This function evaluates a model's performance using stratified sampling, which maintains the proportion of samples for each
    target class in both training and testing sets. It helps diagnose potential instability due to data distribution shifts.

    Args:
        model (BaseModel): The trained model to validate. Must implement fit and score methods.
        X (Union[np.ndarray, DataBatch, FeatureSet]): Input features for validation.
        y (Union[np.ndarray, List]): Target values corresponding to X.
        n_splits (int): Number of stratified splits to generate for validation. Defaults to 5.
        test_size (float): Proportion of data to reserve for testing in each split (between 0 and 1). Defaults to 0.2.
        random_state (Optional[int]): Random seed for reproducibility of splits. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'scores': List of performance scores for each split.
            - 'mean_score': Average performance across all splits.
            - 'std_score': Standard deviation of performance scores.
            - 'split_details': Detailed information about each split (optional).

    Raises:
        ValueError: If test_size is not between 0 and 1, or if n_splits is less than 2.
        TypeError: If model does not conform to BaseModel interface.
    """
    if not isinstance(model, BaseModel):
        raise TypeError('Model must be an instance of BaseModel')
    if not hasattr(model, 'fit') or not callable(getattr(model, 'fit')):
        raise TypeError("Model must have a callable 'fit' method")
    if not hasattr(model, 'score') or not callable(getattr(model, 'score')):
        raise TypeError("Model must have a callable 'score' method")
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1')
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2')
    if isinstance(X, DataBatch):
        X_array = np.array(X.data)
        y_array = np.array(y if y is not None else X.labels)
    elif isinstance(X, FeatureSet):
        X_array = X.features
        y_array = np.array(y)
    else:
        X_array = X
        y_array = np.array(y)
    if len(X_array) != len(y_array):
        raise ValueError('Length of X and y must match')
    scores = []
    split_details = []
    rng = np.random.default_rng(random_state)
    for i in range(n_splits):
        split_random_state = None if random_state is None else rng.integers(0, 2 ** 32 - 1)
        try:
            (X_train, X_test, y_train, y_test) = stratified_split(X_array, y_array, test_size=test_size, random_state=split_random_state)
        except Exception as e:
            raise ValueError(f'Error during stratified split: {str(e)}')
        try:
            try:
                model_params = {}
                if hasattr(model, '__dict__'):
                    for (key, value) in model.__dict__.items():
                        if not key.startswith('_') and (not callable(value)):
                            model_params[key] = value
                model_clone = model.__class__(**model_params)
            except Exception:
                try:
                    model_clone = model.__class__()
                except Exception:
                    model_clone = model
        except Exception as e:
            raise ValueError(f'Error cloning model for split {i}: {str(e)}')
        try:
            model_clone.fit(X_train, y_train)
        except Exception as e:
            raise ValueError(f'Error fitting model on split {i}: {str(e)}')
        try:
            score = model_clone.score(X_test, y_test)
            scores.append(score)
        except Exception as e:
            raise ValueError(f'Error scoring model on split {i}: {str(e)}')
        split_details.append({'split_index': i, 'train_size': len(y_train), 'test_size': len(y_test), 'score': score})
    if not scores:
        raise ValueError('No valid scores were computed')
    return {'scores': scores, 'mean_score': float(np.mean(scores)), 'std_score': float(np.std(scores)), 'split_details': split_details}

def bootstrap_validation(model: BaseModel, X: Union[np.ndarray, DataBatch, FeatureSet], y: Union[np.ndarray, List], n_iterations: int=100, sample_ratio: float=1.0, random_state: Optional[int]=None) -> Dict[str, Any]:
    """
    Perform bootstrap validation to assess model stability through repeated sampling with replacement.

    This function evaluates a model's performance using bootstrap sampling, where multiple datasets are generated
    by sampling with replacement from the original dataset. It provides insights into model variability and robustness.

    Args:
        model (BaseModel): The trained model to validate. Must implement fit and score methods.
        X (Union[np.ndarray, DataBatch, FeatureSet]): Input features for validation.
        y (Union[np.ndarray, List]): Target values corresponding to X.
        n_iterations (int): Number of bootstrap iterations to perform. Defaults to 100.
        sample_ratio (float): Ratio of samples to draw in each iteration (relative to original dataset size). Defaults to 1.0.
        random_state (Optional[int]): Random seed for reproducibility of sampling. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'scores': List of performance scores for each bootstrap iteration.
            - 'mean_score': Average performance across all iterations.
            - 'std_score': Standard deviation of performance scores.
            - 'confidence_interval': Confidence interval for the performance metric (optional).

    Raises:
        ValueError: If sample_ratio is not between 0 and 1, or if n_iterations is less than 1.
        TypeError: If model does not conform to BaseModel interface.
    """
    if not isinstance(model, BaseModel):
        raise TypeError('Model must conform to BaseModel interface')
    if not 0 < sample_ratio <= 1:
        raise ValueError('sample_ratio must be between 0 and 1')
    if n_iterations < 1:
        raise ValueError('n_iterations must be at least 1')
    if isinstance(X, DataBatch):
        X_data = X.data
        y_data = X.labels if X.labels is not None else y
    elif isinstance(X, FeatureSet):
        X_data = X.features
        y_data = y
    else:
        X_data = X
        y_data = y
    if len(X_data) != len(y_data):
        raise ValueError('X and y must have the same number of samples')
    n_samples = len(X_data)
    n_bootstrap_samples = int(sample_ratio * n_samples)
    scores = []
    rng = np.random.default_rng(random_state)
    for i in range(n_iterations):
        model_instance = model.__class__()
        bootstrap_indices = rng.choice(n_samples, size=n_bootstrap_samples, replace=True)
        if isinstance(X, DataBatch):
            if isinstance(X_data, np.ndarray):
                X_bootstrap = X_data[bootstrap_indices]
            else:
                X_bootstrap = [X_data[idx] for idx in bootstrap_indices]
            if isinstance(y_data, np.ndarray):
                y_bootstrap = y_data[bootstrap_indices]
            else:
                y_bootstrap = [y_data[idx] for idx in bootstrap_indices]
        elif isinstance(X, FeatureSet):
            X_bootstrap = X_data[bootstrap_indices]
            if isinstance(y_data, np.ndarray):
                y_bootstrap = y_data[bootstrap_indices]
            else:
                y_bootstrap = [y_data[idx] for idx in bootstrap_indices]
        else:
            if isinstance(X_data, np.ndarray):
                X_bootstrap = X_data[bootstrap_indices]
            else:
                X_bootstrap = [X_data[idx] for idx in bootstrap_indices]
            if isinstance(y_data, np.ndarray):
                y_bootstrap = y_data[bootstrap_indices]
            else:
                y_bootstrap = [y_data[idx] for idx in bootstrap_indices]
        model_instance.fit(X_bootstrap, y_bootstrap)
        score = model_instance.score(X_data, y_data)
        scores.append(score)
    scores_array = np.array(scores)
    mean_score = float(np.mean(scores_array))
    std_score = float(np.std(scores_array))
    confidence_interval = [float(np.percentile(scores_array, 2.5)), float(np.percentile(scores_array, 97.5))]
    return {'scores': scores, 'mean_score': mean_score, 'std_score': std_score, 'confidence_interval': confidence_interval}

def jackknife_resampling(model: BaseModel, X: Union[np.ndarray, DataBatch, FeatureSet], y: Union[np.ndarray, List], scoring_func: Optional[str]=None, random_state: Optional[int]=None) -> Dict[str, Any]:
    """
    Perform jackknife resampling to estimate model performance and its variability by systematically leaving out one observation at a time.

    This function evaluates a model's performance using the jackknife method, where the model is trained and tested
    repeatedly, each time leaving out one sample from the training set. It provides an estimate of model bias and variance.

    Args:
        model (BaseModel): The trained model to evaluate. Must implement fit and score methods.
        X (Union[np.ndarray, DataBatch, FeatureSet]): Input features for evaluation.
        y (Union[np.ndarray, List]): Target values corresponding to X.
        scoring_func (Optional[str]): Scoring method to use (e.g., 'accuracy', 'mse'). If None, uses model's default score. Defaults to None.
        random_state (Optional[int]): Random seed for reproducibility (used for data shuffling if needed). Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'scores': List of performance scores from each jackknife iteration.
            - 'mean_score': Mean of the jackknife scores.
            - 'bias_estimate': Estimated bias of the model performance.
            - 'variance_estimate': Estimated variance of the model performance.
            - 'standard_error': Standard error of the performance estimate.

    Raises:
        ValueError: If the dataset has fewer than 2 samples.
        TypeError: If model does not conform to BaseModel interface.
    """
    if not isinstance(model, BaseModel):
        raise TypeError('Model must be an instance of BaseModel.')
    required_methods = ['fit', 'score']
    for method in required_methods:
        if not callable(getattr(model, method, None)):
            raise TypeError(f"Model must implement '{method}' method.")
    if isinstance(X, DataBatch):
        X_array = X.data
        y_array = y if y is not None else X.labels
    elif isinstance(X, FeatureSet):
        X_array = X.features
        y_array = y
    else:
        X_array = X
        y_array = y
    if not isinstance(y_array, np.ndarray):
        y_array = np.array(y_array)
    n_samples = len(X_array)
    if n_samples < 2:
        raise ValueError('Dataset must contain at least 2 samples for jackknife resampling.')
    if len(y_array) != n_samples:
        raise ValueError('Length of target values must match number of samples in features.')
    if random_state is not None:
        np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    scores = []
    for i in range(n_samples):
        train_indices = np.concatenate([indices[:i], indices[i + 1:]])
        test_index = indices[i]
        X_train = X_array[train_indices]
        y_train = y_array[train_indices]
        X_test = X_array[test_index:test_index + 1]
        y_test = y_array[test_index:test_index + 1]
        model_clone = deepcopy(model)
        model_clone.fit(X_train, y_train)
        if scoring_func is not None:
            score = model_clone.score(X_test, y_test, scoring_func)
        else:
            score = model_clone.score(X_test, y_test)
        scores.append(score)
    scores_array = np.array(scores)
    mean_score = np.mean(scores_array)
    full_model = deepcopy(model)
    full_model.fit(X_array, y_array)
    if scoring_func is not None:
        full_score = full_model.score(X_array, y_array, scoring_func)
    else:
        full_score = full_model.score(X_array, y_array)
    bias_estimate = mean_score - full_score
    variance_estimate = np.var(scores_array, ddof=1)
    standard_error = np.sqrt(variance_estimate / n_samples)
    return {'scores': scores, 'mean_score': mean_score, 'bias_estimate': bias_estimate, 'variance_estimate': variance_estimate, 'standard_error': standard_error}

def permutation_test(model: BaseModel, X: Union[np.ndarray, DataBatch, FeatureSet], y: Union[np.ndarray, List], n_permutations: int=1000, scoring_func: Optional[str]=None, random_state: Optional[int]=None) -> Dict[str, Any]:
    """
    Perform permutation testing to assess statistical significance of model performance.

    This function evaluates whether the model's performance is significantly better than random chance by
    repeatedly shuffling the target variable and measuring performance on these permuted datasets.
    It helps determine if the model has learned meaningful patterns.

    Args:
        model (BaseModel): The trained model to evaluate. Must implement fit and score methods.
        X (Union[np.ndarray, DataBatch, FeatureSet]): Input features for evaluation.
        y (Union[np.ndarray, List]): Target values corresponding to X.
        n_permutations (int): Number of permutations to perform. Defaults to 1000.
        scoring_func (Optional[str]): Scoring method to use (e.g., 'accuracy', 'mse'). If None, uses model's default score. Defaults to None.
        random_state (Optional[int]): Random seed for reproducibility of permutations. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'observed_score': Performance score on the original (non-permuted) dataset.
            - 'permuted_scores': List of performance scores from permuted datasets.
            - 'p_value': Probability of obtaining the observed score by chance.
            - 'significant': Boolean indicating if the result is statistically significant (p < 0.05).

    Raises:
        ValueError: If n_permutations is less than 1.
        TypeError: If model does not conform to BaseModel interface.
    """
    if n_permutations < 1:
        raise ValueError('n_permutations must be at least 1.')
    if not hasattr(model, 'fit') or not callable(getattr(model, 'fit')):
        raise TypeError("Model must implement a callable 'fit' method.")
    if not hasattr(model, 'score') or not callable(getattr(model, 'score')):
        raise TypeError("Model must implement a callable 'score' method.")
    if isinstance(X, DataBatch):
        X_array = X.data if hasattr(X.data, 'values') else X.data
        if hasattr(X_array, 'values'):
            X_array = X_array.values
    elif isinstance(X, FeatureSet):
        X_array = X.features if hasattr(X.features, 'values') else X.features
        if hasattr(X_array, 'values'):
            X_array = X_array.values
    else:
        X_array = np.asarray(X)
    y_array = np.asarray(y)
    if scoring_func is not None and hasattr(model, scoring_func) and callable(getattr(model, scoring_func)):
        observed_score = getattr(model, scoring_func)(X_array, y_array)
    else:
        observed_score = model.score(X_array, y_array)
    permuted_scores = []
    rng = np.random.default_rng(random_state)
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_array)
        if scoring_func is not None and hasattr(model, scoring_func) and callable(getattr(model, scoring_func)):
            score = getattr(model, scoring_func)(X_array, y_shuffled)
        else:
            score = model.score(X_array, y_shuffled)
        permuted_scores.append(score)
    permuted_scores = np.array(permuted_scores)
    n_better_or_equal = np.sum(permuted_scores >= observed_score)
    p_value = (n_better_or_equal + 1) / (n_permutations + 1)
    significant = bool(p_value < 0.05)
    return {'observed_score': float(observed_score), 'permuted_scores': permuted_scores.tolist(), 'p_value': float(p_value), 'significant': significant}


# ...(code omitted)...


def stability_learning_curve_analysis(model: BaseModel, X: Union[np.ndarray, DataBatch, FeatureSet], y: Union[np.ndarray, List], train_sizes: Optional[List[float]]=None, cv_folds: int=5, scoring: str='accuracy', random_state: Optional[int]=None) -> Dict[str, Any]:
    """
    Generate learning curves to evaluate model performance and stability as a function of training set size.

    This function computes learning curves by training the model on progressively larger subsets of the data
    and evaluating performance on both training and validation sets. It helps diagnose bias, variance, and
    stability issues, showing how performance changes with more data.

    Args:
        model (BaseModel): The model to analyze. Must implement fit and score methods.
        X (Union[np.ndarray, DataBatch, FeatureSet]): Input features for analysis.
        y (Union[np.ndarray, List]): Target values corresponding to X.
        train_sizes (Optional[List[float]]): Fractions of training data to use for learning curve (e.g., [0.2, 0.4, 0.6, 0.8, 1.0]). 
                                             If None, defaults to 5 evenly spaced points between 0.1 and 1.0.
        cv_folds (int): Number of cross-validation folds to use for validation scores. Defaults to 5.
        scoring (str): Scoring metric to evaluate (e.g., 'accuracy', 'mse', 'f1'). Defaults to 'accuracy'.
        random_state (Optional[int]): Random seed for reproducibility of data splits. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'train_sizes': Actual number of training samples for each point.
            - 'train_scores': Scores on training sets for each size and fold.
            - 'validation_scores': Scores on validation sets for each size and fold.
            - 'mean_train_scores': Mean training scores for each size.
            - 'std_train_scores': Standard deviation of training scores for each size.
            - 'mean_validation_scores': Mean validation scores for each size.
            - 'std_validation_scores': Standard deviation of validation scores for each size.
            - 'fit_times': Time taken to fit the model for each size.

    Raises:
        ValueError: If train_sizes contains values outside [0, 1], or if cv_folds < 2.
        TypeError: If model does not conform to BaseModel interface.
    """
    if not isinstance(model, BaseModel):
        raise TypeError('Model must conform to BaseModel interface')
    if cv_folds < 2:
        raise ValueError('cv_folds must be at least 2')
    if train_sizes is None:
        train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    if not all((0 < size <= 1.0 for size in train_sizes)):
        raise ValueError('All train_sizes must be between 0 and 1')
    if isinstance(X, DataBatch):
        X_data = X.data
        y_data = X.labels if X.labels is not None else y
    elif isinstance(X, FeatureSet):
        X_data = X.features
        y_data = y
    else:
        X_data = X
        y_data = y
    if isinstance(y_data, list):
        y_data = np.array(y_data)
    if not isinstance(X_data, np.ndarray):
        X_data = np.array(X_data)
    n_samples = len(X_data)
    if n_samples < cv_folds:
        raise ValueError(f'Not enough samples ({n_samples}) for {cv_folds} cross-validation folds')
    train_sizes = sorted(train_sizes)
    n_train_sizes = len(train_sizes)
    train_scores = np.zeros((n_train_sizes, cv_folds))
    validation_scores = np.zeros((n_train_sizes, cv_folds))
    fit_times = np.zeros((n_train_sizes, cv_folds))
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for (i, train_size) in enumerate(train_sizes):
        n_train_samples = max(int(train_size * n_samples), 1)
        if n_train_samples >= n_samples:
            n_train_samples = n_samples
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.arange(n_samples)
        if random_state is not None:
            np.random.shuffle(indices)
        subset_indices = indices[:n_train_samples]
        X_subset = X_data[subset_indices]
        y_subset = y_data[subset_indices]
        for (fold_idx, (train_idx, val_idx)) in enumerate(cv_splitter.split(X_subset, y_subset)):
            X_train_fold = X_subset[train_idx]
            y_train_fold = y_subset[train_idx]
            X_val_fold = X_subset[val_idx]
            y_val_fold = y_subset[val_idx]
            model_instance = model.__class__()
            start_time = time.time()
            model_instance.fit(X_train_fold, y_train_fold)
            fit_time = time.time() - start_time
            fit_times[i, fold_idx] = fit_time
            try:
                if hasattr(model_instance, scoring) and callable(getattr(model_instance, scoring)):
                    train_score = getattr(model_instance, scoring)(X_train_fold, y_train_fold)
                else:
                    train_score = model_instance.score(X_train_fold, y_train_fold)
                train_scores[i, fold_idx] = train_score
            except Exception:
                train_scores[i, fold_idx] = np.nan
            try:
                if hasattr(model_instance, scoring) and callable(getattr(model_instance, scoring)):
                    val_score = getattr(model_instance, scoring)(X_val_fold, y_val_fold)
                else:
                    val_score = model_instance.score(X_val_fold, y_val_fold)
                validation_scores[i, fold_idx] = val_score
            except Exception:
                validation_scores[i, fold_idx] = np.nan
    train_scores_mean = np.nanmean(train_scores, axis=1)
    train_scores_std = np.nanstd(train_scores, axis=1)
    validation_scores_mean = np.nanmean(validation_scores, axis=1)
    validation_scores_std = np.nanstd(validation_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    train_sizes_abs = [int(size * n_samples) for size in train_sizes]
    return {'train_sizes': np.array(train_sizes_abs), 'train_scores': train_scores, 'validation_scores': validation_scores, 'train_scores_mean': train_scores_mean, 'train_scores_std': train_scores_std, 'validation_scores_mean': validation_scores_mean, 'validation_scores_std': validation_scores_std, 'fit_times': fit_times, 'fit_times_mean': fit_times_mean, 'fit_times_std': fit_times_std}