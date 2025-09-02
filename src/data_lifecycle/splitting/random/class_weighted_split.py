from typing import Union, Tuple, Optional
import numpy as np
from general.structures.data_batch import DataBatch


# ...(code omitted)...


def weighted_by_class_split(data: Union[np.ndarray, DataBatch], labels: Union[np.ndarray, list], test_size: float=0.2, random_state: Optional[int]=None) -> Tuple[Union[np.ndarray, DataBatch], Union[np.ndarray, DataBatch], np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets while preserving class distribution.

    This function performs a stratified split based on class labels, ensuring that 
    the relative frequency of each class is maintained in both training and testing sets. 
    It is particularly useful for classification tasks where maintaining class balance 
    is important.

    Args:
        data (Union[np.ndarray, DataBatch]): The input dataset to be split. Can be 
            a NumPy array or a DataBatch object containing the data.
        labels (Union[np.ndarray, list]): The corresponding class labels for the data. 
            Must be the same length as the number of samples in the data.
        test_size (float): Proportion of the dataset to include in the test split, 
            between 0.0 and 1.0. Defaults to 0.2 (20%).
        random_state (Optional[int]): Seed for the random number generator to ensure 
            reproducible splits. If None, the split will be non-reproducible.

    Returns:
        Tuple[Union[np.ndarray, DataBatch], Union[np.ndarray, DataBatch], np.ndarray, np.ndarray]: 
        A tuple containing four elements:
        - Training set data
        - Testing set data
        - Training set labels
        - Testing set labels
        Data portions maintain the same type as the input data.

    Raises:
        ValueError: If the length of data and labels do not match, if test_size is 
            not between 0.0 and 1.0, or if there are insufficient samples per class 
            to perform the requested split.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError('test_size must be between 0.0 and 1.0')
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('Input numpy array must be 2-dimensional')
        n_samples = data.shape[0]
        data_array = data
    elif isinstance(data, DataBatch):
        n_samples = len(data.data) if hasattr(data.data, '__len__') else 0
        data_array = np.asarray(data.data) if not isinstance(data.data, np.ndarray) else data.data
    else:
        raise TypeError('data must be either a numpy array or a DataBatch instance')
    labels_array = np.asarray(labels)
    if n_samples != len(labels_array):
        raise ValueError('Length of data and labels must match')
    rng = np.random.default_rng(random_state)
    (unique_classes, class_counts) = np.unique(labels_array, return_counts=True)
    min_samples_per_class = np.ceil(class_counts * test_size).astype(int)
    if np.any(min_samples_per_class == 0):
        raise ValueError('Insufficient samples in at least one class to perform the requested split')
    train_indices = []
    test_indices = []
    for cls in unique_classes:
        class_indices = np.where(labels_array == cls)[0]
        n_test = max(1, int(len(class_indices) * test_size))
        shuffled_indices = rng.permutation(class_indices)
        test_indices.extend(shuffled_indices[:n_test])
        train_indices.extend(shuffled_indices[n_test:])
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    train_labels = labels_array[train_indices]
    test_labels = labels_array[test_indices]
    if isinstance(data, np.ndarray):
        train_data = data[train_indices]
        test_data = data[test_indices]
        return (train_data, test_data, train_labels, test_labels)
    else:
        if isinstance(data.data, np.ndarray):
            train_data_content = data.data[train_indices]
            test_data_content = data.data[test_indices]
        else:
            train_data_content = [data.data[i] for i in train_indices]
            test_data_content = [data.data[i] for i in test_indices]
        train_labels_content = None
        test_labels_content = None
        if data.labels is not None:
            if isinstance(data.labels, np.ndarray):
                train_labels_content = data.labels[train_indices]
                test_labels_content = data.labels[test_indices]
            else:
                train_labels_content = [data.labels[i] for i in train_indices]
                test_labels_content = [data.labels[i] for i in test_indices]
        train_sample_ids = None
        test_sample_ids = None
        if data.sample_ids is not None:
            train_sample_ids = [data.sample_ids[i] for i in train_indices]
            test_sample_ids = [data.sample_ids[i] for i in test_indices]
        train_data_batch = DataBatch(data=train_data_content, labels=train_labels_content, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=train_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        test_data_batch = DataBatch(data=test_data_content, labels=test_labels_content, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=test_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        return (train_data_batch, test_data_batch, train_labels, test_labels)