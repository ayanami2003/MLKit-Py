import numpy as np
from typing import Optional, Union
from general.structures.data_batch import DataBatch

def shuffle_rows(data: Union[np.ndarray, DataBatch], random_state: Optional[int]=None) -> Union[np.ndarray, DataBatch]:
    """
    Shuffle the rows of a dataset or DataBatch while maintaining row correspondence.

    This function randomly reorders the rows of the input data. If a DataBatch is provided,
    the shuffling is applied to the primary data while ensuring that any associated labels,
    sample IDs, and metadata maintain alignment with their respective rows.

    Args:
        data (Union[np.ndarray, DataBatch]): The input data to shuffle. If a numpy array,
            it should be 2D where each row represents a sample. If a DataBatch, the primary
            data attribute will be shuffled while preserving alignment of labels and metadata.
        random_state (Optional[int]): Random seed for reproducibility. If None, uses
            the system's default random number generator state.

    Returns:
        Union[np.ndarray, DataBatch]: The shuffled data in the same format as the input.
            If input was a numpy array, returns a shuffled numpy array. If input was a
            DataBatch, returns a new DataBatch with shuffled data and aligned components.

    Raises:
        ValueError: If the input data format is unsupported or invalid.
        TypeError: If data is neither a numpy array nor a DataBatch instance.
    """
    if not isinstance(data, (np.ndarray, DataBatch)):
        raise TypeError('Input data must be either a numpy array or a DataBatch instance.')
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('Input numpy array must be 2-dimensional.')
        shuffled_data = data.copy()
        rng = np.random.default_rng(random_state)
        n_rows = shuffled_data.shape[0]
        indices = rng.permutation(n_rows)
        shuffled_data = shuffled_data[indices]
        return shuffled_data
    else:
        if isinstance(data.data, np.ndarray) and data.data.ndim != 2:
            raise ValueError("DataBatch data must be 2-dimensional when it's a numpy array.")
        elif not isinstance(data.data, np.ndarray) and (not (hasattr(data.data, '__len__') and len(data.data) > 0 and hasattr(data.data[0], '__len__'))):
            raise ValueError('DataBatch data must be 2-dimensional.')
        rng = np.random.default_rng(random_state)
        n_rows = len(data.data)
        if n_rows == 0:
            return DataBatch(data=data.data, labels=data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        indices = rng.permutation(n_rows)
        if isinstance(data.data, np.ndarray):
            shuffled_data = data.data[indices]
        else:
            shuffled_data = [data.data[i] for i in indices]
        shuffled_labels = None
        if data.labels is not None:
            if isinstance(data.labels, np.ndarray):
                shuffled_labels = data.labels[indices]
            else:
                shuffled_labels = [data.labels[i] for i in indices]
        shuffled_sample_ids = None
        if data.sample_ids is not None:
            shuffled_sample_ids = [data.sample_ids[i] for i in indices]
        return DataBatch(data=shuffled_data, labels=shuffled_labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=shuffled_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)