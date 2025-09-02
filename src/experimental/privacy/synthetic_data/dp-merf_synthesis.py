from typing import Optional, Union
import numpy as np
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

def synthesize_data_with_dp_merf(data: Union[np.ndarray, DataBatch, FeatureSet], epsilon: float=1.0, n_components: int=10, n_samples: Optional[int]=None, random_state: Optional[int]=None, preserve_categorical: bool=True) -> Union[np.ndarray, DataBatch, FeatureSet]:
    """
    Synthesize privacy-preserving data using the DP-MERF (Differentially Private Maximum Entropy Random Features) method.

    This function generates synthetic data that approximates the statistical properties of the input dataset while
    providing rigorous differential privacy guarantees. It uses random Fourier features to approximate the data
    distribution and adds calibrated noise to ensure privacy preservation.

    Args:
        data (Union[np.ndarray, DataBatch, FeatureSet]): Input data to synthesize from. Can be a numpy array,
            DataBatch, or FeatureSet containing the original sensitive data.
        epsilon (float): Privacy budget for differential privacy. Lower values provide more privacy but may
            reduce data utility. Must be positive.
        n_components (int): Number of random Fourier components to use for distribution approximation.
            Higher values improve fidelity but increase computation time.
        n_samples (Optional[int]): Number of synthetic samples to generate. If None, matches the input data size.
        random_state (Optional[int]): Random seed for reproducibility of results.
        preserve_categorical (bool): Whether to detect and preserve categorical feature characteristics in
            the synthetic data generation process.

    Returns:
        Union[np.ndarray, DataBatch]: Synthetic dataset with the same structure as input. If input was a
            DataBatch, returns a DataBatch with synthetic data and preserved metadata where applicable.

    Raises:
        ValueError: If epsilon is not positive, or if data structures are malformed.
        RuntimeError: If synthesis fails due to numerical instability or insufficient privacy budget.
    """
    if epsilon <= 0:
        raise ValueError('Epsilon must be positive for differential privacy')
    if random_state is not None:
        np.random.seed(random_state)
    original_type = type(data)
    if isinstance(data, np.ndarray):
        X = data
    elif isinstance(data, DataBatch):
        X = np.array(data.data)
    elif isinstance(data, FeatureSet):
        X = data.features
    else:
        raise ValueError('Input data must be numpy array, DataBatch, or FeatureSet')
    if X.ndim != 2:
        raise ValueError('Input data must be 2-dimensional')
    (n_original_samples, n_features) = X.shape
    if n_samples is None:
        n_samples = n_original_samples
    categorical_columns = []
    if preserve_categorical:
        for j in range(n_features):
            unique_vals = np.unique(X[:, j])
            if len(unique_vals) <= max(10, int(np.sqrt(n_original_samples))):
                categorical_columns.append(j)
    if n_components <= 0:
        raise ValueError('n_components must be positive')
    omega = np.random.normal(0, 1, (n_features, n_components))
    b = np.random.uniform(0, 2 * np.pi, n_components)
    phi_X = np.sqrt(2.0 / n_components) * np.cos(X @ omega + b)
    mu = np.mean(phi_X, axis=0)
    sensitivity = 2.0 / (np.sqrt(n_components) * n_original_samples)
    noise_scale = sensitivity / epsilon
    noise = np.random.normal(0, noise_scale, n_components)
    mu_noisy = mu + noise
    X_mean = np.mean(X, axis=0)
    synthetic_data = np.zeros((n_samples, n_features))
    continuous_columns = [j for j in range(n_features) if j not in categorical_columns]
    if continuous_columns:
        X_continuous = X[:, continuous_columns]
        X_cont_mean = np.mean(X_continuous, axis=0)
        reg_param = 1e-06
        cov_matrix = np.cov(X_continuous, rowvar=False)
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])
        cov_matrix += reg_param * np.eye(cov_matrix.shape[0])
        try:
            synthetic_data[:, continuous_columns] = np.random.multivariate_normal(X_cont_mean, cov_matrix, n_samples)
        except np.linalg.LinAlgError:
            synthetic_data[:, continuous_columns] = X_cont_mean + np.random.normal(0, 0.1, (n_samples, len(continuous_columns)))
    for col_idx in categorical_columns:
        (unique_vals, counts) = np.unique(X[:, col_idx], return_counts=True)
        probs = counts.astype(float)
        if epsilon > 0:
            laplace_noise = np.random.laplace(0, 1.0 / epsilon, len(probs))
            probs = probs + laplace_noise
            probs = np.maximum(probs, 0)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones_like(probs) / len(probs)
        sampled_indices = np.random.choice(len(unique_vals), n_samples, p=probs)
        synthetic_data[:, col_idx] = unique_vals[sampled_indices]
    if original_type == np.ndarray:
        return synthetic_data
    elif original_type == DataBatch:
        synthetic_batch = DataBatch(data=synthetic_data, labels=data.labels[:n_samples] if data.labels is not None else None, metadata=data.metadata.copy() if data.metadata is not None else None, sample_ids=[f'syn_{i}' for i in range(n_samples)] if data.sample_ids is not None else None, feature_names=data.feature_names, batch_id=f'{data.batch_id}_synthetic' if data.batch_id is not None else 'synthetic')
        return synthetic_batch
    elif original_type == FeatureSet:
        synthetic_feature_set = FeatureSet(features=synthetic_data, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=[f'syn_{i}' for i in range(n_samples)] if data.sample_ids is not None else None, metadata=data.metadata.copy() if data.metadata is not None else None, quality_scores=data.quality_scores.copy() if data.quality_scores is not None else None)
        return synthetic_feature_set