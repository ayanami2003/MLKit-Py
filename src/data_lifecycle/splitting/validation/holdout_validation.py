from general.structures.data_batch import DataBatch
from typing import Tuple, Optional, Union
import numpy as np

def holdout_validation_split(data: DataBatch, validation_size: Union[float, int]=0.2, shuffle: bool=True, random_state: Optional[int]=None) -> Tuple[DataBatch, DataBatch]:
    """
    Split a dataset into training and validation sets using a holdout strategy.

    This function implements a holdout validation approach where a specified portion of the data
    is reserved for validation while the remainder is used for training. It maintains the
    integrity of the DataBatch structure and supports both proportional and absolute sizing
    for the validation set.

    Args:
        data (DataBatch): The input dataset to be split, containing features and optionally labels.
        validation_size (Union[float, int], optional): Size of the validation set. If float, should be
            between 0.0 and 1.0 and represents the proportion of the dataset to include in the
            validation split. If int, represents the absolute number of validation samples.
            Defaults to 0.2 (20%).
        shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
        random_state (Optional[int], optional): Random seed for reproducibility when shuffling.
            If None, no seed is used. Defaults to None.

    Returns:
        Tuple[DataBatch, DataBatch]: A tuple containing:
            - training_batch (DataBatch): The training subset.
            - validation_batch (DataBatch): The validation subset.

    Raises:
        ValueError: If validation_size is not in the proper range (0.0 to 1.0 for float, or
            0 to len(data) for int).
        TypeError: If data is not a DataBatch instance.
    """
    if not isinstance(data, DataBatch):
        raise TypeError('data must be a DataBatch instance.')
    n_samples = len(data.data) if hasattr(data.data, '__len__') else 1 if data.data is not None else 0
    if isinstance(validation_size, float):
        if not 0.0 <= validation_size <= 1.0:
            raise ValueError('validation_size must be between 0.0 and 1.0 when provided as a float.')
        n_val = int(np.round(validation_size * n_samples))
    elif isinstance(validation_size, int):
        if validation_size < 0:
            raise ValueError('validation_size must be non-negative when provided as an integer.')
        if n_samples == 0 and validation_size > 0:
            raise ValueError(f'validation_size ({validation_size}) must be 0 for empty datasets when provided as an integer.')
        if validation_size > n_samples:
            raise ValueError(f'validation_size ({validation_size}) must be less than or equal to the number of samples ({n_samples}) when provided as an integer.')
        n_val = validation_size
    else:
        raise TypeError('validation_size must be either float or int.')
    if n_samples == 0:
        empty_metadata = data.metadata.copy() if data.metadata else {}
        empty_batch = DataBatch(data=[], labels=[] if data.labels is not None else None, metadata=empty_metadata, sample_ids=[], feature_names=data.feature_names, batch_id=data.batch_id)
        return (empty_batch, empty_batch)
    if n_val < 0:
        raise ValueError('validation_size resulted in negative number of samples.')
    if n_val > n_samples:
        raise ValueError(f'validation_size ({n_val}) cannot be greater than the number of samples ({n_samples}).')
    if n_val == 0:
        train_metadata = data.metadata.copy() if data.metadata else {}
        val_metadata = data.metadata.copy() if data.metadata else {}
        empty_data = [] if not isinstance(data.data, np.ndarray) else np.array([]).reshape(0, data.data.shape[1] if hasattr(data.data, 'shape') and len(data.data.shape) > 1 else 0)
        empty_labels = [] if data.labels is not None and (not isinstance(data.labels, np.ndarray)) else np.array([]) if data.labels is not None else None
        empty_sample_ids = [] if data.sample_ids is not None else None
        training_batch = DataBatch(data=data.data, labels=data.labels, metadata=train_metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        validation_batch = DataBatch(data=empty_data, labels=empty_labels, metadata=val_metadata, sample_ids=empty_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        return (training_batch, validation_batch)
    elif n_val == n_samples:
        train_metadata = data.metadata.copy() if data.metadata else {}
        val_metadata = data.metadata.copy() if data.metadata else {}
        empty_data = [] if not isinstance(data.data, np.ndarray) else np.array([]).reshape(0, data.data.shape[1] if hasattr(data.data, 'shape') and len(data.data.shape) > 1 else 0)
        empty_labels = [] if data.labels is not None and (not isinstance(data.labels, np.ndarray)) else np.array([]) if data.labels is not None else None
        empty_sample_ids = [] if data.sample_ids is not None else None
        training_batch = DataBatch(data=empty_data, labels=empty_labels, metadata=train_metadata, sample_ids=empty_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        validation_batch = DataBatch(data=data.data, labels=data.labels, metadata=val_metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        return (training_batch, validation_batch)
    n_train = n_samples - n_val
    if shuffle:
        from src.data_lifecycle.utilities.shuffling.data_shuffling import shuffle_rows
        shuffled_data = shuffle_rows(data, random_state=random_state)
    else:
        shuffled_data = data
    if isinstance(shuffled_data.data, np.ndarray):
        train_data = shuffled_data.data[:n_train]
        val_data = shuffled_data.data[n_train:]
    else:
        train_data = shuffled_data.data[:n_train]
        val_data = shuffled_data.data[n_train:]
    train_labels = None
    val_labels = None
    if shuffled_data.labels is not None:
        if isinstance(shuffled_data.labels, np.ndarray):
            train_labels = shuffled_data.labels[:n_train]
            val_labels = shuffled_data.labels[n_train:]
        else:
            train_labels = shuffled_data.labels[:n_train]
            val_labels = shuffled_data.labels[n_train:]
    train_sample_ids = None
    val_sample_ids = None
    if shuffled_data.sample_ids is not None:
        train_sample_ids = shuffled_data.sample_ids[:n_train]
        val_sample_ids = shuffled_data.sample_ids[n_train:]
    train_metadata = shuffled_data.metadata.copy() if shuffled_data.metadata else {}
    val_metadata = shuffled_data.metadata.copy() if shuffled_data.metadata else {}
    training_batch = DataBatch(data=train_data, labels=train_labels, metadata=train_metadata, sample_ids=train_sample_ids, feature_names=shuffled_data.feature_names, batch_id=shuffled_data.batch_id)
    validation_batch = DataBatch(data=val_data, labels=val_labels, metadata=val_metadata, sample_ids=val_sample_ids, feature_names=shuffled_data.feature_names, batch_id=shuffled_data.batch_id)
    return (training_batch, validation_batch)