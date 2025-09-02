import numpy as np
from typing import Union, Optional, List
from general.structures.data_batch import DataBatch

def generate_random_samples(data: Union[np.ndarray, DataBatch, List], n_samples: int, random_state: Optional[int]=None, replace: bool=False) -> Union[np.ndarray, DataBatch, List]:
    """
    Generate random samples from the input data.

    This function randomly selects a specified number of samples from the provided data
    using either sampling with or without replacement. It maintains the original data
    structure type in the returned result.

    Args:
        data (Union[np.ndarray, DataBatch, List]): The input data to sample from.
            Can be a numpy array, DataBatch object, or Python list.
        n_samples (int): The number of samples to generate. Must be positive.
        random_state (Optional[int]): Random seed for reproducibility. If None,
            uses system time as seed.
        replace (bool): Whether to sample with replacement. Defaults to False.

    Returns:
        Union[np.ndarray, DataBatch, List]: Randomly sampled data in the same
            format as the input. For DataBatch objects, maintains metadata and
            structure while updating sample_ids if present.

    Raises:
        ValueError: If n_samples is negative or exceeds population size when
            replace=False.
        TypeError: If data is not a supported type.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError('n_samples must be a positive integer.')
    rng = np.random.default_rng(random_state)
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            raise ValueError('Cannot sample from 0-dimensional array.')
        population_size = data.shape[0]
        if not replace and n_samples > population_size:
            raise ValueError(f'Cannot sample {n_samples} samples without replacement from a population of size {population_size}.')
        indices = rng.choice(population_size, size=n_samples, replace=replace)
        return data[indices]
    elif isinstance(data, DataBatch):
        if len(data.data) == 0:
            raise ValueError('Cannot sample from empty DataBatch.')
        population_size = len(data.data)
        if not replace and n_samples > population_size:
            raise ValueError(f'Cannot sample {n_samples} samples without replacement from a population of size {population_size}.')
        indices = rng.choice(population_size, size=n_samples, replace=replace)
        if isinstance(data.data, np.ndarray):
            sampled_data = data.data[indices]
        else:
            sampled_data = [data.data[i] for i in indices]
        sampled_labels = None
        if data.labels is not None:
            if isinstance(data.labels, np.ndarray):
                sampled_labels = data.labels[indices]
            else:
                sampled_labels = [data.labels[i] for i in indices]
        sampled_sample_ids = None
        if data.sample_ids is not None:
            sampled_sample_ids = [data.sample_ids[i] for i in indices]
        return DataBatch(data=sampled_data, labels=sampled_labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=sampled_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValueError('Cannot sample from empty list.')
        population_size = len(data)
        if not replace and n_samples > population_size:
            raise ValueError(f'Cannot sample {n_samples} samples without replacement from a population of size {population_size}.')
        indices = rng.choice(population_size, size=n_samples, replace=replace)
        return [data[i] for i in indices]
    else:
        raise TypeError(f'Unsupported data type: {type(data)}. Supported types are np.ndarray, DataBatch, and list.')