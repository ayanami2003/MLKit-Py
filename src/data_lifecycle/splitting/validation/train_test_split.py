from general.structures.data_batch import DataBatch
import numpy as np
from typing import Tuple, Optional, Union
from src.data_lifecycle.utilities.shuffling.data_shuffling import shuffle_rows
import copy

def train_test_split(data: DataBatch, test_size: Union[float, int]=0.2, shuffle: bool=True, random_state: Optional[int]=None) -> Tuple[DataBatch, DataBatch]:
    """
    Split a dataset into training and testing sets.

    This function divides the input data into two separate batches: one for training and one for testing.
    The split can be configured to use a specific proportion or absolute number of samples for the test set.
    Optionally, the data can be shuffled before splitting to ensure randomness.

    Args:
        data (DataBatch): The input dataset to be split. Must contain features and optionally labels.
        test_size (Union[float, int], optional): If float, should be between 0.0 and 1.0 and represent the 
            proportion of the dataset to include in the test split. If int, represents the absolute number 
            of test samples. Defaults to 0.2.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
        random_state (Optional[int], optional): Random seed for reproducibility when shuffling. 
            If None, no seed is set. Defaults to None.

    Returns:
        Tuple[DataBatch, DataBatch]: A tuple containing two DataBatch objects:
            - The first element is the training set
            - The second element is the testing set

    Raises:
        ValueError: If test_size is not a valid float (between 0 and 1) or int (positive), or if 
            the resulting train or test sets would be empty.
    """
    if not isinstance(data, DataBatch):
        raise TypeError('data must be a DataBatch instance')
    n_samples = len(data.data) if hasattr(data.data, '__len__') else 0
    if n_samples == 0:
        raise ValueError('Cannot split an empty dataset')
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError('test_size float must be between 0.0 and 1.0')
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        if test_size <= 0:
            raise ValueError('test_size int must be positive')
        if test_size >= n_samples:
            raise ValueError('test_size must be less than the number of samples')
        n_test = test_size
    else:
        raise ValueError('test_size must be either float or int')
    n_train = n_samples - n_test
    if n_train == 0:
        raise ValueError('train set would be empty - adjust test_size')
    data_copy = DataBatch(data=data.data, labels=data.labels, metadata=copy.deepcopy(data.metadata) if data.metadata else None, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
    if shuffle:
        data_copy = shuffle_rows(data_copy, random_state=random_state)
    if isinstance(data_copy.data, np.ndarray):
        train_data = data_copy.data[:n_train]
        test_data = data_copy.data[n_train:]
    else:
        train_data = data_copy.data[:n_train]
        test_data = data_copy.data[n_train:]
    (train_labels, test_labels) = (None, None)
    if data_copy.labels is not None:
        if isinstance(data_copy.labels, np.ndarray):
            train_labels = data_copy.labels[:n_train]
            test_labels = data_copy.labels[n_train:]
        else:
            train_labels = data_copy.labels[:n_train]
            test_labels = data_copy.labels[n_train:]
    (train_sample_ids, test_sample_ids) = (None, None)
    if data_copy.sample_ids is not None:
        train_sample_ids = data_copy.sample_ids[:n_train]
        test_sample_ids = data_copy.sample_ids[n_train:]
    train_metadata = copy.deepcopy(data.metadata) if data.metadata else None
    test_metadata = copy.deepcopy(data.metadata) if data.metadata else None
    train_batch = DataBatch(data=train_data, labels=train_labels, metadata=train_metadata, sample_ids=train_sample_ids, feature_names=data.feature_names, batch_id=f'{data.batch_id}_train' if data.batch_id else 'train')
    test_batch = DataBatch(data=test_data, labels=test_labels, metadata=test_metadata, sample_ids=test_sample_ids, feature_names=data.feature_names, batch_id=f'{data.batch_id}_test' if data.batch_id else 'test')
    return (train_batch, test_batch)