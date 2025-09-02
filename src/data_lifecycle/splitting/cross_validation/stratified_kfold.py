import numpy as np
from typing import Iterator, Optional, Union, List, Tuple
from general.structures.data_batch import DataBatch
from general.base_classes.pipeline_base import PipelineStep

class StratifiedKFold:
    """
    A class for generating stratified K-Fold cross-validation splits.
    
    This splitter ensures that each fold maintains the same proportion of samples 
    for each target class as in the original dataset, which is especially important 
    for classification tasks with imbalanced classes.
    
    Attributes:
        n_splits (int): Number of folds. Must be at least 2.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (Optional[int]): Random state for reproducibility when shuffling.
    """

    def __init__(self, n_splits: int=5, shuffle: bool=False, random_state: Optional[int]=None):
        """
        Initialize the StratifiedKFold splitter.
        
        Args:
            n_splits (int): Number of folds. Must be at least 2. Defaults to 5.
            shuffle (bool): Whether to shuffle the data before splitting. Defaults to False.
            random_state (Optional[int]): Random state for reproducibility when shuffling. Defaults to None.
        """
        if n_splits < 2:
            raise ValueError('n_splits must be at least 2')
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data: DataBatch) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Generate stratified K-Fold splits of the input data.
        
        Each iteration yields a tuple of (train_indices, test_indices) which are lists of integers
        representing the indices of samples in the training and test sets respectively.
        The splits maintain the same proportion of samples for each target class as in the original dataset.
        
        Args:
            data (DataBatch): Input data batch containing features and labels.
                              Must have non-null labels for stratification.
            
        Returns:
            Iterator[tuple[list[int], list[int]]]: An iterator over train-test index pairs.
            
        Raises:
            ValueError: If data does not contain labels, if n_splits is less than 2,
                        or if there are insufficient samples per class for stratification.
        """
        if not data.is_labeled():
            raise ValueError('Data must contain labels for stratification.')
        labels = data.labels
        n_samples = len(labels)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        (unique_labels, label_counts) = np.unique(labels, return_counts=True)
        min_label_count = np.min(label_counts)
        if min_label_count < self.n_splits:
            raise ValueError(f'StratifiedKFold requires at least {self.n_splits} samples per class, but smallest class has only {min_label_count} samples.')
        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
            shuffled_labels = labels[indices]
        else:
            shuffled_labels = labels
        class_indices = {}
        for label in unique_labels:
            class_indices[label] = np.where(shuffled_labels == label)[0]
        folds = [[] for _ in range(self.n_splits)]
        for label in unique_labels:
            label_indices = class_indices[label]
            for (i, idx) in enumerate(label_indices):
                fold_idx = i % self.n_splits
                folds[fold_idx].append(idx)
        for i in range(self.n_splits):
            folds[i] = np.array(folds[i])
        for i in range(self.n_splits):
            test_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            if self.shuffle:
                test_indices = indices[test_indices]
                train_indices = indices[train_indices]
            yield (train_indices.tolist(), test_indices.tolist())

class StratifiedKFoldSplitter(PipelineStep):
    """
    A pipeline component for performing stratified K-Fold cross-validation splits.
    
    This splitter ensures that each fold maintains the same proportion of samples 
    for each target class as in the original dataset, which is especially important 
    for classification tasks with imbalanced classes.
    
    Attributes:
        n_splits (int): Number of folds. Must be at least 2.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (Optional[int]): Random state for reproducibility when shuffling.
    """

    def __init__(self, n_splits: int=5, shuffle: bool=False, random_state: Optional[int]=None):
        """
        Initialize the StratifiedKFoldSplitter.
        
        Args:
            n_splits (int): Number of folds. Must be at least 2. Defaults to 5.
            shuffle (bool): Whether to shuffle the data before splitting. Defaults to False.
            random_state (Optional[int]): Random state for reproducibility when shuffling. Defaults to None.
        """
        super().__init__(name='StratifiedKFoldSplitter')
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def execute(self, data: DataBatch, **kwargs) -> Iterator[tuple[DataBatch, DataBatch]]:
        """
        Generate stratified K-Fold splits of the input data.
        
        Each iteration yields a tuple of (train_batch, test_batch) where both are DataBatch objects.
        The splits maintain the same proportion of samples for each target class as in the original dataset.
        
        Args:
            data (DataBatch): Input data batch containing features and labels.
                              Must have non-null labels for stratification.
            **kwargs: Additional parameters (not used in this implementation).
            
        Returns:
            Iterator[tuple[DataBatch, DataBatch]]: An iterator over train-test pairs.
            
        Raises:
            ValueError: If data does not contain labels or if n_splits is less than 2.
        """
        if not data.is_labeled():
            raise ValueError('Data must contain labels for stratification.')
        if self.n_splits < 2:
            raise ValueError('n_splits must be at least 2')
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for (train_indices, test_indices) in skf.split(data):
            if hasattr(data.data, 'iloc'):
                train_data = data.data.iloc[train_indices]
            else:
                train_data = [data.data[i] for i in train_indices]
            if data.labels is not None:
                if hasattr(data.labels, 'iloc'):
                    train_labels = data.labels.iloc[train_indices]
                else:
                    train_labels = [data.labels[i] for i in train_indices]
            else:
                train_labels = None
            train_sample_ids = [data.sample_ids[i] for i in train_indices] if data.sample_ids is not None else None
            train_metadata = data.metadata.copy() if data.metadata else None
            train_batch_id = f'{data.batch_id}_train_fold' if data.batch_id else None
            train_batch = DataBatch(data=train_data, labels=train_labels, metadata=train_metadata, sample_ids=train_sample_ids, feature_names=data.feature_names, batch_id=train_batch_id)
            if hasattr(data.data, 'iloc'):
                test_data = data.data.iloc[test_indices]
            else:
                test_data = [data.data[i] for i in test_indices]
            if data.labels is not None:
                if hasattr(data.labels, 'iloc'):
                    test_labels = data.labels.iloc[test_indices]
                else:
                    test_labels = [data.labels[i] for i in test_indices]
            else:
                test_labels = None
            test_sample_ids = [data.sample_ids[i] for i in test_indices] if data.sample_ids is not None else None
            test_metadata = data.metadata.copy() if data.metadata else None
            test_batch_id = f'{data.batch_id}_test_fold' if data.batch_id else None
            test_batch = DataBatch(data=test_data, labels=test_labels, metadata=test_metadata, sample_ids=test_sample_ids, feature_names=data.feature_names, batch_id=test_batch_id)
            yield (train_batch, test_batch)

class StratifiedKFoldStrategy:
    """
    A class for generating stratified K-Fold cross-validation splits with various strategies.
    
    This class implements different strategies for stratified K-Fold splitting, allowing users
    to choose the most appropriate method for their specific use case while maintaining
    class distribution across folds.
    
    Supported strategies include:
    - Standard stratified K-Fold
    - Shuffle-enabled stratified K-Fold
    - Custom stratification based on specific columns
    
    Attributes:
        n_splits (int): Number of folds. Must be at least 2.
        strategy (str): Strategy to use for splitting ('standard', 'shuffle', 'custom').
        shuffle (bool): Whether to shuffle the data before splitting (for 'shuffle' strategy).
        random_state (Optional[int]): Random state for reproducibility when shuffling.
        stratify_column (Optional[str]): Column name to use for custom stratification.
    """

    def __init__(self, n_splits: int=5, strategy: str='standard', shuffle: bool=False, random_state: Optional[int]=None, stratify_column: Optional[str]=None):
        """
        Initialize the StratifiedKFoldStrategy.
        
        Args:
            n_splits (int): Number of folds. Must be at least 2. Defaults to 5.
            strategy (str): Strategy to use for splitting ('standard', 'shuffle', 'custom'). Defaults to 'standard'.
            shuffle (bool): Whether to shuffle the data before splitting (for 'shuffle' strategy). Defaults to False.
            random_state (Optional[int]): Random state for reproducibility when shuffling. Defaults to None.
            stratify_column (Optional[str]): Column name to use for custom stratification. Defaults to None.
            
        Raises:
            ValueError: If n_splits is less than 2 or if strategy is not supported.
        """
        if n_splits < 2:
            raise ValueError('n_splits must be at least 2')
        supported_strategies = ['standard', 'shuffle', 'custom']
        if strategy not in supported_strategies:
            raise ValueError(f'strategy must be one of {supported_strategies}')
        self.n_splits = n_splits
        self.strategy = strategy
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify_column = stratify_column

    def split(self, data: DataBatch) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Generate stratified K-Fold splits of the input data according to the selected strategy.
        
        Each iteration yields a tuple of (train_indices, test_indices) which are lists of integers
        representing the indices of samples in the training and test sets respectively.
        The splits maintain the same proportion of samples for each target class as in the original dataset.
        
        Args:
            data (DataBatch): Input data batch containing features and labels.
                              Must have non-null labels for stratification.
            
        Returns:
            Iterator[tuple[list[int], list[int]]]: An iterator over train-test index pairs.
            
        Raises:
            ValueError: If data does not contain labels, if n_splits is less than 2,
                        if there are insufficient samples per class for stratification,
                        or if stratify_column is specified but not found in data.
        """
        if not data.is_labeled():
            raise ValueError('Data must contain labels for stratification')
        if self.strategy == 'custom':
            if self.stratify_column is None:
                raise ValueError("stratify_column must be specified for 'custom' strategy")
            if data.metadata is None or self.stratify_column not in data.metadata:
                raise ValueError(f"stratify_column '{self.stratify_column}' not found in data metadata")
            y = data.metadata[self.stratify_column]
        else:
            y = data.labels
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        n_samples = len(y)
        if n_samples < self.n_splits:
            raise ValueError(f'n_samples={n_samples} should be >= n_splits={self.n_splits}')
        (unique_classes, class_counts) = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        if min_class_count < self.n_splits:
            raise ValueError(f'Got only {min_class_count} samples for class {unique_classes[np.argmin(class_counts)]},but n_splits={self.n_splits}. Some classes have too few samples for stratification.')
        indices = np.arange(n_samples)
        if self.strategy == 'shuffle' or (self.strategy == 'standard' and self.shuffle):
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(indices)
            y_shuffled = y[indices]
        elif self.strategy == 'custom' and self.shuffle:
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(indices)
            y_shuffled = y[indices]
        else:
            y_shuffled = y
        class_indices = {}
        for cls in unique_classes:
            class_indices[cls] = np.where(y_shuffled == cls)[0]
        folds = [[] for _ in range(self.n_splits)]
        for cls in unique_classes:
            cls_indices = class_indices[cls]
            n_cls_samples = len(cls_indices)
            samples_per_fold = np.full(self.n_splits, n_cls_samples // self.n_splits)
            remaining = n_cls_samples % self.n_splits
            samples_per_fold[:remaining] += 1
            start_idx = 0
            for fold_idx in range(self.n_splits):
                end_idx = start_idx + samples_per_fold[fold_idx]
                folds[fold_idx].extend(cls_indices[start_idx:end_idx])
                start_idx = end_idx
        for test_fold_idx in range(self.n_splits):
            test_indices = [int(indices[idx]) for idx in folds[test_fold_idx]]
            train_indices = []
            for fold_idx in range(self.n_splits):
                if fold_idx != test_fold_idx:
                    train_indices.extend([int(indices[idx]) for idx in folds[fold_idx]])
            yield (train_indices, test_indices)


# ...(code omitted)...


class RepeatedKFold:
    """
    A class for generating repeated K-Fold cross-validation splits.
    
    This splitter repeatedly applies K-Fold cross-validation with different
    randomizations to provide more robust estimates of model performance.
    Each repetition uses a different random split of the data into K folds.
    
    Attributes:
        n_splits (int): Number of folds. Must be at least 2.
        n_repeats (int): Number of times to repeat the K-Fold split.
        random_state (Optional[int]): Random state for reproducibility.
    """

    def __init__(self, n_splits: int=5, n_repeats: int=10, random_state: Optional[int]=None):
        """
        Initialize the RepeatedKFold splitter.
        
        Args:
            n_splits (int): Number of folds. Must be at least 2. Defaults to 5.
            n_repeats (int): Number of times to repeat the K-Fold split. Defaults to 10.
            random_state (Optional[int]): Random state for reproducibility. Defaults to None.
            
        Raises:
            ValueError: If n_splits is less than 2 or n_repeats is less than 1.
        """
        if n_splits < 2:
            raise ValueError('n_splits must be at least 2')
        if n_repeats < 1:
            raise ValueError('n_repeats must be at least 1')
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, data: DataBatch) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Generate repeated K-Fold splits of the input data.
        
        Each repetition generates a complete set of K folds, resulting in 
        n_splits * n_repeats total train-test pairs. Each repetition uses
        a different randomization of the data.
        
        Args:
            data (DataBatch): Input data batch to split.
            
        Returns:
            Iterator[Tuple[List[int], List[int]]]: An iterator over train-test index pairs.
            
        Raises:
            ValueError: If data cannot be split due to insufficient samples.
        """
        n_samples = len(data.data) if hasattr(data.data, '__len__') else 0
        if n_samples < self.n_splits:
            raise ValueError(f'Cannot perform {self.n_splits}-fold cross-validation on {n_samples} samples. Need at least {self.n_splits} samples.')
        indices = np.arange(n_samples)
        for repeat in range(self.n_repeats):
            if self.random_state is not None:
                repeat_random_state = self.random_state + repeat
            else:
                repeat_random_state = None
            if repeat_random_state is not None:
                rng = np.random.default_rng(repeat_random_state)
                shuffled_indices = indices.copy()
                rng.shuffle(shuffled_indices)
            else:
                rng = np.random.default_rng()
                shuffled_indices = indices.copy()
                rng.shuffle(shuffled_indices)
            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
            fold_sizes[:n_samples % self.n_splits] += 1
            current = 0
            for fold_size in fold_sizes:
                (start, stop) = (current, current + fold_size)
                test_indices = shuffled_indices[start:stop]
                train_indices = np.concatenate([shuffled_indices[:start], shuffled_indices[stop:]])
                yield (train_indices.tolist(), test_indices.tolist())
                current = stop