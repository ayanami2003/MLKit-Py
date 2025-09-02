from typing import Union, Optional, Tuple
import numpy as np
from general.structures.data_batch import DataBatch

def random_train_test_split(data: Union[np.ndarray, DataBatch], test_size: Union[float, int]=0.2, random_state: Optional[int]=None) -> Tuple[Union[np.ndarray, DataBatch], Union[np.ndarray, DataBatch]]:
    """
    Split data into random train and test subsets.

    This function divides the input data into two disjoint sets: a training set and a testing set,
    using random sampling. The split can be specified as a fraction of the total data or as an absolute number of samples.
    When working with labeled data, both features and labels are split accordingly while maintaining alignment.

    Args:
        data (Union[np.ndarray, DataBatch]): The input dataset to split. If DataBatch, it must contain aligned data and labels.
        test_size (Union[float, int]): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
                                       If int, represents the absolute number of test samples.
        random_state (Optional[int]): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

    Returns:
        Tuple[Union[np.ndarray, DataBatch], Union[np.ndarray, DataBatch]]: A tuple containing (train_data, test_data).
            If input was DataBatch, returns tuple of DataBatch objects with appropriately split data and labels.
            If input was np.ndarray, returns tuple of np.ndarrays.

    Raises:
        ValueError: If test_size is not in (0, 1) when float or not positive when int.
        ValueError: If test_size is larger than the number of samples.
    """
    if not isinstance(data, (np.ndarray, DataBatch)):
        raise TypeError('Input data must be either a numpy array or a DataBatch instance.')
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('Input numpy array must be 2-dimensional.')
        n_samples = data.shape[0]
    else:
        n_samples = len(data.data)
        if n_samples == 0:
            return (DataBatch(data=[], labels=[] if data.labels is not None else None, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=[], feature_names=data.feature_names, batch_id=data.batch_id), DataBatch(data=[], labels=[] if data.labels is not None else None, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=[], feature_names=data.feature_names, batch_id=data.batch_id))
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError('test_size must be between 0.0 and 1.0 when provided as a float.')
        n_test = int(np.floor(test_size * n_samples))
    elif isinstance(test_size, int):
        if test_size <= 0:
            raise ValueError('test_size must be positive when provided as an integer.')
        n_test = test_size
    else:
        raise TypeError('test_size must be either float or int.')
    if n_test >= n_samples:
        raise ValueError('test_size must be smaller than the number of samples.')
    n_train = n_samples - n_test
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    if isinstance(data, np.ndarray):
        train_data = data[train_indices]
        test_data = data[test_indices]
        return (train_data, test_data)
    else:
        if isinstance(data.data, np.ndarray):
            train_data_arr = data.data[train_indices]
            test_data_arr = data.data[test_indices]
        else:
            train_data_arr = [data.data[i] for i in train_indices]
            test_data_arr = [data.data[i] for i in test_indices]
        train_labels = None
        test_labels = None
        if data.labels is not None:
            if isinstance(data.labels, np.ndarray):
                train_labels = data.labels[train_indices]
                test_labels = data.labels[test_indices]
            else:
                train_labels = [data.labels[i] for i in train_indices]
                test_labels = [data.labels[i] for i in test_indices]
        train_sample_ids = None
        test_sample_ids = None
        if data.sample_ids is not None:
            train_sample_ids = [data.sample_ids[i] for i in train_indices]
            test_sample_ids = [data.sample_ids[i] for i in test_indices]
        train_batch = DataBatch(data=train_data_arr, labels=train_labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=train_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        test_batch = DataBatch(data=test_data_arr, labels=test_labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=test_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        return (train_batch, test_batch)

def random_percentage_split(data: Union[np.ndarray, DataBatch], percentages: Tuple[float, ...], random_state: Optional[int]=None) -> Tuple[Union[np.ndarray, DataBatch], ...]:
    """
    Split data into multiple random subsets based on specified percentages.

    This function divides the input data into multiple disjoint subsets according to the provided percentages.
    The sum of percentages should be less than or equal to 1.0. If less than 1.0, the remaining data forms an additional subset.
    When working with labeled data, both features and labels are split accordingly while maintaining alignment.

    Args:
        data (Union[np.ndarray, DataBatch]): The input dataset to split. If DataBatch, it must contain aligned data and labels.
        percentages (Tuple[float, ...]): Tuple of percentages for each split. Each value should be between 0.0 and 1.0.
                                         The sum of percentages should be <= 1.0.
        random_state (Optional[int]): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

    Returns:
        Tuple[Union[np.ndarray, DataBatch], ...]: A tuple containing the split datasets in the order specified by percentages,
            followed by any remaining data if the sum of percentages is less than 1.0.
            If input was DataBatch, returns tuple of DataBatch objects with appropriately split data and labels.
            If input was np.ndarray, returns tuple of np.ndarrays.

    Raises:
        ValueError: If any percentage is not in [0.0, 1.0].
        ValueError: If the sum of percentages exceeds 1.0.
        ValueError: If any calculated split size is zero.
        ValueError: If data is DataBatch and lengths of data and labels do not match.
    """
    if not isinstance(data, (np.ndarray, DataBatch)):
        raise TypeError('Input data must be either a numpy array or a DataBatch instance.')
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('Input numpy array must be 2-dimensional.')
        n_samples = data.shape[0]
    else:
        n_samples = len(data.data)
        if data.labels is not None and len(data.data) != len(data.labels):
            raise ValueError('Length of data and labels must match in DataBatch.')
        if n_samples == 0:
            result = []
            for _ in percentages:
                result.append(DataBatch(data=[], labels=[] if data.labels is not None else None, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=[], feature_names=data.feature_names, batch_id=data.batch_id))
            if sum(percentages) < 1.0:
                result.append(DataBatch(data=[], labels=[] if data.labels is not None else None, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=[], feature_names=data.feature_names, batch_id=data.batch_id))
            return tuple(result)
    if not all((0.0 <= p <= 1.0 for p in percentages)):
        raise ValueError('All percentages must be between 0.0 and 1.0.')
    total_percentage = sum(percentages)
    if total_percentage > 1.0:
        raise ValueError('Sum of percentages must not exceed 1.0.')
    split_sizes = [round(p * n_samples) for p in percentages]
    if total_percentage == 1.0:
        current_total = sum(split_sizes)
        difference = n_samples - current_total
        if difference != 0:
            fractional_parts = [(p * n_samples - int(p * n_samples), i) for (i, p) in enumerate(percentages)]
            fractional_parts.sort(reverse=True)
            if difference > 0:
                for i in range(min(difference, len(split_sizes))):
                    idx = fractional_parts[i][1]
                    split_sizes[idx] += 1
            else:
                for i in range(min(-difference, len(split_sizes))):
                    idx = fractional_parts[len(fractional_parts) - 1 - i][1]
                    if split_sizes[idx] > 0:
                        split_sizes[idx] -= 1
    for (i, (size, pct)) in enumerate(zip(split_sizes, percentages)):
        if size == 0 and pct > 0.0:
            raise ValueError(f'Split size for percentage {pct} is zero. Consider adjusting percentages or data size.')
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    result_indices = []
    start_idx = 0
    for size in split_sizes:
        result_indices.append(indices[start_idx:start_idx + size])
        start_idx += size
    if total_percentage < 1.0:
        result_indices.append(indices[start_idx:])
    results = []
    for idx_group in result_indices:
        if isinstance(data, np.ndarray):
            results.append(data[idx_group])
        else:
            if isinstance(data.data, np.ndarray):
                split_data = data.data[idx_group]
            else:
                split_data = [data.data[i] for i in idx_group]
            split_labels = None
            if data.labels is not None:
                if isinstance(data.labels, np.ndarray):
                    split_labels = data.labels[idx_group]
                else:
                    split_labels = [data.labels[i] for i in idx_group]
            split_sample_ids = None
            if data.sample_ids is not None:
                split_sample_ids = [data.sample_ids[i] for i in idx_group]
            results.append(DataBatch(data=split_data, labels=split_labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=split_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id))
    return tuple(results)