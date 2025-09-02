from typing import Union, Optional, Tuple
import numpy as np
from general.structures.data_batch import DataBatch

def time_series_split(data: Union[DataBatch, np.ndarray], labels: Optional[Union[np.ndarray, list]]=None, n_splits: int=5, gap: int=0, test_size: Optional[int]=None, stride: Optional[int]=None) -> Tuple[Union[DataBatch, np.ndarray], Union[DataBatch, np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Split time series data into consecutive training and testing segments while preserving temporal order.

    This function creates multiple train/test splits where each test set follows its corresponding
    training set in time. It ensures that no future data leaks into earlier training periods,
    which is crucial for realistic time series model evaluation.

    Args:
        data (Union[DataBatch, np.ndarray]): Input time series data to be split. If DataBatch,
                                             must contain time-ordered samples.
        labels (Optional[Union[np.ndarray, list]]): Optional target values corresponding to the data.
        n_splits (int): Number of train/test splits to generate. Defaults to 5.
        gap (int): Number of samples to skip between training and testing sets. Defaults to 0.
        test_size (Optional[int]): Number of samples in each test set. If None, inferred from data size
                                   and other parameters.
        stride (Optional[int]): Number of samples to advance the split window by in each iteration.
                                If None, uses test_size as stride.

    Returns:
        Tuple[Union[DataBatch, np.ndarray], Union[DataBatch, np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            A tuple containing:
            - train_data: Time series training data segment
            - test_data: Time series testing data segment that follows temporally after train_data
            - train_labels: Labels for training data (if labels provided)
            - test_labels: Labels for testing data (if labels provided)

    Raises:
        ValueError: If data has insufficient samples for requested splits or if parameters are inconsistent.
    """
    if not isinstance(n_splits, int) or n_splits <= 0:
        raise ValueError('n_splits must be a positive integer')
    if not isinstance(gap, int) or gap < 0:
        raise ValueError('gap must be a non-negative integer')
    if isinstance(data, DataBatch):
        features = data.data
        if labels is None and data.is_labeled():
            labels = data.labels
    else:
        features = data
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    elif features.ndim != 2:
        raise ValueError('features must be 1D or 2D array')
    n_samples = features.shape[0]
    if labels is not None:
        if not isinstance(labels, (np.ndarray, list)):
            labels = np.array(labels)
        if len(labels) != n_samples:
            raise ValueError('Number of labels must match number of samples in data')
    if test_size is None:
        test_size = max(1, n_samples // (n_splits + 4))
    if not isinstance(test_size, int) or test_size <= 0:
        raise ValueError('test_size must be a positive integer')
    if stride is None:
        stride = test_size
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError('stride must be a positive integer')
    min_required_samples = (n_splits - 1) * stride + test_size + gap
    if n_samples < min_required_samples:
        raise ValueError(f'Insufficient samples ({n_samples}) for {n_splits} splits with test_size={test_size}, gap={gap}, and stride={stride}. At least {min_required_samples} samples required.')
    for i in range(n_splits):
        train_end = (i + 1) * stride
        train_start = 0
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n_samples:
            raise ValueError(f'Split {i + 1}/{n_splits} exceeds data boundaries: test_end ({test_end}) > n_samples ({n_samples})')
        train_indices = np.arange(train_start, train_end)
        test_indices = np.arange(test_start, test_end)
        if isinstance(data, DataBatch):
            train_features = features[train_indices]
            test_features = features[test_indices]
            train_metadata = data.metadata.copy() if data.metadata else {}
            test_metadata = data.metadata.copy() if data.metadata else {}
            train_sample_ids = None
            test_sample_ids = None
            if data.sample_ids:
                train_sample_ids = [data.sample_ids[idx] for idx in train_indices]
                test_sample_ids = [data.sample_ids[idx] for idx in test_indices]
            train_labels_split = None
            test_labels_split = None
            if labels is not None:
                if isinstance(labels, np.ndarray):
                    train_labels_split = labels[train_indices]
                    test_labels_split = labels[test_indices]
                else:
                    train_labels_split = [labels[idx] for idx in train_indices]
                    test_labels_split = [labels[idx] for idx in test_indices]
            train_data = DataBatch(data=train_features, labels=train_labels_split, metadata=train_metadata, sample_ids=train_sample_ids, feature_names=data.feature_names, batch_id=f'{data.batch_id}_train_{i}' if data.batch_id else None)
            test_data = DataBatch(data=test_features, labels=test_labels_split, metadata=test_metadata, sample_ids=test_sample_ids, feature_names=data.feature_names, batch_id=f'{data.batch_id}_test_{i}' if data.batch_id else None)
        else:
            train_data = features[train_indices]
            test_data = features[test_indices]
            train_labels_split = None
            test_labels_split = None
            if labels is not None:
                if isinstance(labels, np.ndarray):
                    train_labels_split = labels[train_indices]
                    test_labels_split = labels[test_indices]
                else:
                    train_labels_split = [labels[idx] for idx in train_indices]
                    test_labels_split = [labels[idx] for idx in test_indices]
        yield (train_data, test_data, train_labels_split, test_labels_split)