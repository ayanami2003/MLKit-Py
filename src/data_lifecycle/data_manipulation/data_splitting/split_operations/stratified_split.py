from general.structures.data_batch import DataBatch
from typing import Union, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_split(data: Union[DataBatch, np.ndarray], labels: Union[np.ndarray, list], test_size: Union[float, int]=0.2, random_state: Optional[int]=None) -> Tuple[Union[DataBatch, np.ndarray], Union[DataBatch, np.ndarray], np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets while preserving the distribution of target labels.

    This function performs stratified sampling to ensure that the proportion of samples from each 
    target class is maintained in both training and testing subsets. It supports both raw NumPy arrays
    and DataBatch objects as input.

    Args:
        data (Union[DataBatch, np.ndarray]): Input data to be split. Can be a DataBatch object 
            containing features or a NumPy array representing feature matrix.
        labels (Union[np.ndarray, list]): Target labels corresponding to the data samples. 
            Must be the same length as the number of samples in data.
        test_size (Union[float, int], optional): Size of the test subset. If float, should be 
            between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
            If int, represents the absolute number of test samples. Defaults to 0.2.
        random_state (Optional[int], optional): Controls the randomness of the sampling. Pass an int 
            for reproducible output across multiple function calls. Defaults to None.

    Returns:
        Tuple[Union[DataBatch, np.ndarray], Union[DataBatch, np.ndarray], np.ndarray, np.ndarray]: 
            A tuple containing:
            - train_data: Training subset of input data
            - test_data: Testing subset of input data
            - train_labels: Labels corresponding to training data
            - test_labels: Labels corresponding to testing data

    Raises:
        ValueError: If the length of labels does not match the number of samples in data.
        ValueError: If test_size is not a valid float or int value.
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(data, DataBatch):
        data_array = np.array(data.data)
        n_samples = data_array.shape[0]
    elif isinstance(data, np.ndarray):
        data_array = data
        n_samples = data.shape[0]
    else:
        raise TypeError('Data must be either a numpy array or a DataBatch object')
    if len(labels) != n_samples:
        raise ValueError(f'Length of labels ({len(labels)}) does not match number of samples in data ({n_samples})')
    if isinstance(test_size, float):
        if test_size <= 0.0 or test_size >= 1.0:
            raise ValueError('test_size float must be between 0.0 and 1.0')
    elif isinstance(test_size, int):
        if test_size <= 0 or test_size >= n_samples:
            raise ValueError(f'test_size int must be between 0 and {n_samples}')
    else:
        raise ValueError('test_size must be either float or int')
    from sklearn.model_selection import train_test_split
    if isinstance(data, np.ndarray):
        (train_data, test_data, train_labels, test_labels) = train_test_split(data, labels, test_size=test_size, random_state=random_state, stratify=labels)
    else:
        indices = np.arange(n_samples)
        (train_indices, test_indices, train_labels, test_labels) = train_test_split(indices, labels, test_size=test_size, random_state=random_state, stratify=labels)
        train_data_array = data_array[train_indices]
        test_data_array = data_array[test_indices]
        train_sample_ids = [data.sample_ids[i] for i in train_indices] if data.sample_ids else None
        test_sample_ids = [data.sample_ids[i] for i in test_indices] if data.sample_ids else None
        train_data = DataBatch(data=train_data_array, labels=train_labels, metadata=data.metadata.copy() if data.metadata else None, sample_ids=train_sample_ids, feature_names=data.feature_names, batch_id=f'{data.batch_id}_train' if data.batch_id else None)
        test_data = DataBatch(data=test_data_array, labels=test_labels, metadata=data.metadata.copy() if data.metadata else None, sample_ids=test_sample_ids, feature_names=data.feature_names, batch_id=f'{data.batch_id}_test' if data.batch_id else None)
    return (train_data, test_data, train_labels, test_labels)