from general.structures.data_batch import DataBatch
import numpy as np
from typing import Union, Tuple, Optional

def train_test_split(data: Union[DataBatch, np.ndarray], labels: Optional[Union[np.ndarray, list]]=None, test_size: Union[float, int]=0.2, stratify: Optional[Union[np.ndarray, list]]=None, random_state: Optional[int]=None, shuffle: bool=True) -> Tuple[Union[DataBatch, np.ndarray], Union[DataBatch, np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Split data into random train and test subsets.

    This function splits input data into training and testing sets according to the specified test size.
    It supports both labeled and unlabeled data, with optional stratification to maintain class distribution
    in both sets. The function works with both DataBatch objects and raw numpy arrays.

    Args:
        data (Union[DataBatch, np.ndarray]): Input data to split. Can be a DataBatch object containing 
            features or a numpy array representing feature matrix.
        labels (Optional[Union[np.ndarray, list]]): Target values or labels corresponding to the data. 
            Required for supervised learning scenarios. Defaults to None.
        test_size (Union[float, int]): Size of the test set. If float, should be between 0.0 and 1.0 and 
            represent the proportion of the dataset to include in the test split. If int, represents the 
            absolute number of test samples. Defaults to 0.2.
        stratify (Optional[Union[np.ndarray, list]]): Array-like object used to stratify the split. 
            If not None, data is split in a stratified fashion, using this as the class labels. 
            Defaults to None.
        random_state (Optional[int]): Controls the shuffling applied to the data before applying the split. 
            Pass an int for reproducible output across multiple function calls. Defaults to None.
        shuffle (bool): Whether or not to shuffle the data before splitting. If shuffle=False then 
            stratify must be None. Defaults to True.

    Returns:
        Tuple[Union[DataBatch, np.ndarray], Union[DataBatch, np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]: 
        A tuple containing:
        - train_data: Training data subset
        - test_data: Testing data subset
        - train_labels: Training labels subset (if labels provided)
        - test_labels: Testing labels subset (if labels provided)

    Raises:
        ValueError: If test_size is invalid, stratify is used when shuffle=False, or data and labels shapes don't match.
        TypeError: If data type is not supported.
    """
    if not isinstance(data, (DataBatch, np.ndarray)):
        raise TypeError('data must be either a DataBatch or numpy array')
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
    if n_samples == 0:
        raise ValueError('Cannot split empty dataset')
    if labels is not None:
        if not isinstance(labels, (np.ndarray, list)):
            labels = np.array(labels)
        if len(labels) != n_samples:
            raise ValueError('Number of labels must match number of samples in data')
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError('test_size must be between 0.0 and 1.0 when float')
        n_test = int(np.ceil(test_size * n_samples))
        if n_test == 0 and n_samples > 0:
            n_test = 1
    elif isinstance(test_size, int):
        if test_size < 0:
            raise ValueError('test_size must be non-negative when int')
        n_test = test_size
    else:
        raise TypeError('test_size must be either float or int')
    if n_test >= n_samples and n_samples > 0:
        if n_samples == 1:
            n_test = 1
        else:
            raise ValueError('test_size too large: no samples would remain for training set')
    if n_samples == 1:
        n_test = 1
    if n_test <= 0 and n_samples > 0:
        raise ValueError('test_size too small: resulting test set would be empty')
    if not shuffle and stratify is not None:
        raise ValueError('stratify cannot be used when shuffle=False')
    if stratify is not None:
        if labels is None:
            raise ValueError('labels must be provided when stratify is used')
        if not isinstance(stratify, (np.ndarray, list)):
            stratify = np.array(stratify)
        if len(stratify) != n_samples:
            raise ValueError('stratify array must have same length as data')
        (unique_classes, class_counts) = np.unique(stratify, return_counts=True)
        if np.any(class_counts < 2):
            raise ValueError('Each class must have at least 2 samples for stratified splitting')
        test_counts = {}
        remaining = n_test
        for cls in unique_classes:
            count = class_counts[unique_classes == cls][0]
            cls_test_count = max(1, int(np.round(count * n_test / n_samples)))
            cls_test_count = min(cls_test_count, remaining, count - 1)
            test_counts[cls] = cls_test_count
            remaining -= cls_test_count
        while remaining > 0:
            for cls in unique_classes:
                if remaining <= 0:
                    break
                if test_counts[cls] < class_counts[unique_classes == cls][0] - 1:
                    test_counts[cls] += 1
                    remaining -= 1
        train_indices = []
        test_indices = []
        if shuffle:
            rng = np.random.default_rng(random_state)
        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            if shuffle:
                rng.shuffle(cls_indices)
            cls_test_count = test_counts[cls]
            test_indices.extend(cls_indices[:cls_test_count])
            train_indices.extend(cls_indices[cls_test_count:])
        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)
    else:
        indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(indices)
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
        else:
            test_indices = indices[-n_test:]
            train_indices = indices[:-n_test]
    if n_samples == 1:
        pass
    if isinstance(data, DataBatch):
        train_features = features[train_indices]
        test_features = features[test_indices]
        train_metadata = data.metadata.copy() if data.metadata else {}
        test_metadata = data.metadata.copy() if data.metadata else {}
        train_sample_ids = None
        test_sample_ids = None
        if data.sample_ids:
            train_sample_ids = [data.sample_ids[i] for i in train_indices]
            test_sample_ids = [data.sample_ids[i] for i in test_indices]
        train_labels_split = None
        test_labels_split = None
        if labels is not None:
            if isinstance(labels, np.ndarray):
                train_labels_split = labels[train_indices]
                test_labels_split = labels[test_indices]
            else:
                train_labels_split = [labels[i] for i in train_indices]
                test_labels_split = [labels[i] for i in test_indices]
        train_data = DataBatch(data=train_features, labels=train_labels_split, metadata=train_metadata, sample_ids=train_sample_ids, feature_names=data.feature_names, batch_id=f'{data.batch_id}_train' if data.batch_id else None)
        test_data = DataBatch(data=test_features, labels=test_labels_split, metadata=test_metadata, sample_ids=test_sample_ids, feature_names=data.feature_names, batch_id=f'{data.batch_id}_test' if data.batch_id else None)
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
                train_labels_split = [labels[i] for i in train_indices]
                test_labels_split = [labels[i] for i in test_indices]
    return (train_data, test_data, train_labels_split, test_labels_split)