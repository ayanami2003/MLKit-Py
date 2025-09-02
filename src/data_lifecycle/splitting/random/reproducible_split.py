from general.structures.data_batch import DataBatch
from typing import Union, Tuple, Optional
import numpy as np
import copy

def reproducible_random_split(data: Union[np.ndarray, DataBatch], test_size: Union[float, int]=0.2, random_state: Optional[int]=None) -> Tuple[Union[np.ndarray, DataBatch], Union[np.ndarray, DataBatch]]:
    """
    Split data into train and test sets with reproducible randomness based on a fixed seed.

    This function divides the input data into two parts: a training set and a testing set,
    ensuring that the split is reproducible when the same random_state is provided.
    It supports both raw numpy arrays and DataBatch objects, maintaining the input type in the output.

    Args:
        data (Union[np.ndarray, DataBatch]): The input data to be split. Can be either a
            numpy array or a DataBatch object containing the data.
        test_size (Union[float, int], optional): If float, should be between 0.0 and 1.0 and 
            represent the proportion of the dataset to include in the test split. If int, 
            represents the absolute number of test samples. Defaults to 0.2.
        random_state (Optional[int], optional): Controls the shuffling applied to the data
            before applying the split. Pass an int for reproducible output across multiple 
            function calls. Defaults to None.

    Returns:
        Tuple[Union[np.ndarray, DataBatch], Union[np.ndarray, DataBatch]]: A tuple containing
            (train_data, test_data) where both elements are of the same type as the input data.
            
    Raises:
        ValueError: If test_size is not a valid float (between 0 and 1) or int (positive),
            or if the input data is empty.
    """
    if isinstance(data, np.ndarray):
        if data.size == 0:
            raise ValueError('Input data cannot be empty.')
        n_samples = data.shape[0]
    elif isinstance(data, DataBatch):
        if len(data.data) == 0:
            raise ValueError('Input data cannot be empty.')
        n_samples = len(data.data)
    else:
        raise TypeError('Input data must be either a numpy array or a DataBatch instance.')
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError('test_size must be between 0 and 1 when provided as a float.')
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        if test_size <= 0:
            raise ValueError('test_size must be a positive integer when provided as an int.')
        if test_size >= n_samples:
            raise ValueError('test_size must be less than the total number of samples.')
        n_test = test_size
    else:
        raise TypeError('test_size must be either a float or an integer.')
    n_train = n_samples - n_test
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    if isinstance(data, np.ndarray):
        train_data = data[train_indices]
        test_data = data[test_indices]
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
        train_metadata = copy.deepcopy(data.metadata) if data.metadata is not None else {}
        test_metadata = copy.deepcopy(data.metadata) if data.metadata is not None else {}
        train_batch_id = f'{data.batch_id}_train' if data.batch_id else 'train'
        test_batch_id = f'{data.batch_id}_test' if data.batch_id else 'test'
        train_data = DataBatch(data=train_data_arr, labels=train_labels, metadata=train_metadata, sample_ids=train_sample_ids, feature_names=data.feature_names, batch_id=train_batch_id)
        test_data = DataBatch(data=test_data_arr, labels=test_labels, metadata=test_metadata, sample_ids=test_sample_ids, feature_names=data.feature_names, batch_id=test_batch_id)
    return (train_data, test_data)