from typing import Any, Iterator, List, Optional, Tuple
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch

def k_fold_cross_validation(model: BaseModel, data: DataBatch, n_splits: int=5, shuffle: bool=False, random_state: Optional[int]=None) -> Tuple[List[float], List[float]]:
    """
    Perform K-Fold Cross-Validation on a given model and dataset.

    This function splits the provided dataset into K consecutive folds, trains the model 
    on K-1 folds, and validates it on the remaining fold. This process is repeated K times 
    (once for each fold), returning the training and validation scores for each iteration.

    Args:
        model (BaseModel): The machine learning model to validate. Must implement fit, predict, and score methods.
        data (DataBatch): The dataset to perform cross-validation on. Must include features and labels.
        n_splits (int): Number of folds to generate. Must be at least 2. Defaults to 5.
        shuffle (bool): Whether to shuffle the data before splitting. Defaults to False.
        random_state (Optional[int]): Random seed for shuffling. Used only if shuffle is True. Defaults to None.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists:
            - Training scores for each fold
            - Validation scores for each fold

    Raises:
        ValueError: If n_splits is less than 2 or greater than the number of samples.
        TypeError: If model does not conform to BaseModel or data is not a DataBatch.
    """
    if not isinstance(model, BaseModel):
        raise TypeError('Model must be an instance of BaseModel.')
    if not isinstance(data, DataBatch):
        raise TypeError('Data must be an instance of DataBatch.')
    if not data.is_labeled():
        raise ValueError('Data must contain labels for cross-validation.')
    n_samples = len(data.data) if hasattr(data.data, '__len__') else 1
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2.')
    if n_splits > n_samples:
        raise ValueError('n_splits cannot be greater than the number of samples.')
    if shuffle:
        from src.data_lifecycle.utilities.shuffling.data_shuffling import shuffle_rows
        shuffled_data = shuffle_rows(data, random_state=random_state)
    else:
        shuffled_data = data
    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits)
    fold_sizes[:n_samples % n_splits] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        (start, stop) = (current, current + fold_size)
        folds.append(indices[start:stop])
        current = stop
    train_scores = []
    val_scores = []
    for i in range(n_splits):
        val_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        if isinstance(shuffled_data.data, np.ndarray):
            X_train = shuffled_data.data[train_indices]
            X_val = shuffled_data.data[val_indices]
        else:
            X_train = [shuffled_data.data[idx] for idx in train_indices]
            X_val = [shuffled_data.data[idx] for idx in val_indices]
        if isinstance(shuffled_data.labels, np.ndarray):
            y_train = shuffled_data.labels[train_indices]
            y_val = shuffled_data.labels[val_indices]
        else:
            y_train = [shuffled_data.labels[idx] for idx in train_indices]
            y_val = [shuffled_data.labels[idx] for idx in val_indices]
        train_batch = DataBatch(data=X_train, labels=y_train)
        val_batch = DataBatch(data=X_val, labels=y_val)
        from copy import deepcopy
        model_clone = deepcopy(model)
        model_clone.fit(train_batch.data, train_batch.labels)
        train_score = model_clone.score(train_batch.data, train_batch.labels)
        val_score = model_clone.score(val_batch.data, val_batch.labels)
        train_scores.append(train_score)
        val_scores.append(val_score)
    return (train_scores, val_scores)