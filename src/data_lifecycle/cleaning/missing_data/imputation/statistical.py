import pandas as pd
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer


# ...(code omitted)...


class ForwardBackwardFillTransformer(BaseTransformer):
    """
    Transformer that performs combined forward and backward filling on missing values.
    
    This transformer first applies forward fill to carry last observed values forward,
    then applies backward fill to propagate next observed values backward. This
    approach handles missing values at both beginning and end of sequences.
    
    Attributes
    ----------
    limit : Optional[int]
        Maximum number of consecutive missing values to fill in each direction.
        If None, no limit.
    columns : Optional[list]
        Specific columns to apply fill operations to. If None, applies to all.
    """

    def __init__(self, limit: Optional[int]=None, columns: Optional[list]=None, name: Optional[str]=None):
        """
        Initialize the ForwardBackwardFillTransformer.
        
        Parameters
        ----------
        limit : Optional[int], default=None
            Maximum number of consecutive NaN values to fill in each direction.
            Must be greater than 0 if specified.
        columns : Optional[list], default=None
            List of column names to apply fill operations. If None, applies to all columns.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError('limit must be a positive integer or None')
        self.limit = limit
        self.columns = columns

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'ForwardBackwardFillTransformer':
        """
        Fit the transformer to the input data.
        
        For combined fill operations, fitting is a no-op as no model parameters need to be learned.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data to fit the transformer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        ForwardBackwardFillTransformer
            Self instance for method chaining.
        """
        return self

    def _get_column_indices(self, data_container) -> List[int]:
        """Get column indices to apply transformation to."""
        if self.columns is None:
            if isinstance(data_container, np.ndarray):
                return list(range(data_container.shape[1]))
            elif isinstance(data_container, FeatureSet) and data_container.feature_names:
                return list(range(len(data_container.feature_names)))
            elif isinstance(data_container, DataBatch) and data_container.feature_names:
                return list(range(len(data_container.feature_names)))
            elif isinstance(data_container.data, np.ndarray):
                return list(range(data_container.data.shape[1]))
            else:
                return list(range(len(data_container.data[0]))) if data_container.data else []
        elif isinstance(data_container, FeatureSet) and data_container.feature_names:
            try:
                return [data_container.feature_names.index(col) for col in self.columns]
            except ValueError as e:
                raise ValueError(f'Column {str(e).split()[0]} not found in FeatureSet') from e
        elif isinstance(data_container, DataBatch) and data_container.feature_names:
            try:
                return [data_container.feature_names.index(col) for col in self.columns]
            except ValueError as e:
                raise ValueError(f'Column {str(e).split()[0]} not found in DataBatch') from e
        else:
            if not all((isinstance(col, int) for col in self.columns)):
                raise ValueError('For ndarray/DataBatch.data without feature_names, columns must be integer indices')
            return self.columns

    def _apply_fill_operations(self, arr: np.ndarray, column_indices: List[int]) -> np.ndarray:
        """Apply forward and backward fill operations to specified columns."""
        result = arr.copy()
        if result.size == 0:
            return result
        for col_idx in column_indices:
            col_data = result[:, col_idx]
            if np.all(np.isnan(col_data)):
                continue
            filled_data = pd.Series(col_data).fillna(method='ffill', limit=self.limit)
            filled_data = filled_data.fillna(method='bfill', limit=self.limit)
            result[:, col_idx] = filled_data.values
        return result

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Apply combined forward and backward fill to the input data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Data with combined fill operations applied.
        """
        import pandas as pd
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                column_indices = self._get_column_indices(data)
                result = self._apply_fill_operations(data, column_indices)
                return result.flatten()
            elif data.ndim == 2:
                column_indices = self._get_column_indices(data)
                return self._apply_fill_operations(data, column_indices)
            else:
                raise ValueError('Only 1D and 2D numpy arrays are supported')
        elif isinstance(data, FeatureSet):
            column_indices = self._get_column_indices(data)
            new_features = self._apply_fill_operations(data.features, column_indices)
            return FeatureSet(features=new_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        elif isinstance(data, DataBatch):
            if isinstance(data.data, np.ndarray):
                if data.data.ndim == 1:
                    arr_2d = data.data.reshape(-1, 1)
                    column_indices = self._get_column_indices(data)
                    result = self._apply_fill_operations(arr_2d, column_indices)
                    new_data = result.flatten()
                elif data.data.ndim == 2:
                    column_indices = self._get_column_indices(data)
                    new_data = self._apply_fill_operations(data.data, column_indices)
                else:
                    raise ValueError('Only 1D and 2D numpy arrays are supported in DataBatch')
            else:
                try:
                    arr = np.array(data.data, dtype=float)
                    if arr.ndim == 1:
                        arr_2d = arr.reshape(-1, 1)
                        column_indices = self._get_column_indices(data)
                        result = self._apply_fill_operations(arr_2d, column_indices)
                        new_data = result.flatten().tolist()
                    elif arr.ndim == 2:
                        column_indices = self._get_column_indices(data)
                        result = self._apply_fill_operations(arr, column_indices)
                        new_data = result.tolist()
                    else:
                        raise ValueError('Only 1D and 2D data arrays are supported in DataBatch')
                except (ValueError, TypeError) as e:
                    raise ValueError('Cannot convert DataBatch data to numeric array for fill operations') from e
            return DataBatch(data=new_data, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}. Supported types are DataBatch, FeatureSet, and np.ndarray')

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Apply inverse transformation (not implemented for this transformer).
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Transformed data to inverse transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Original data format (identity operation).
        """
        return data

class MissingDataDeletionTransformer(BaseTransformer):
    """
    Transformer that removes rows or columns with missing values from the dataset.
    
    This transformer provides flexible deletion strategies for handling missing data:
    - Row-wise deletion: Removes any rows containing missing values
    - Column-wise deletion: Removes any columns containing missing values
    - Threshold-based deletion: Removes rows/columns based on missing value proportion
    
    Attributes
    ----------
    strategy : str
        Deletion strategy ('row', 'column', or 'threshold')
    threshold : float
        Proportion threshold for threshold-based deletion (0.0 to 1.0)
    columns : Optional[list]
        Specific columns to consider for deletion. If None, considers all columns.
    """

    def __init__(self, strategy: str='row', threshold: float=0.5, columns: Optional[List[Union[str, int]]]=None, name: Optional[str]=None):
        """
        Initialize the MissingDataDeletionTransformer.
        
        Parameters
        ----------
        strategy : str, default='row'
            Deletion strategy. Options: 'row', 'column', 'threshold'.
        threshold : float, default=0.5
            Proportion threshold for threshold-based deletion (0.0 to 1.0).
        columns : Optional[List[Union[str, int]]], default=None
            Specific columns to consider for deletion. If None, uses all columns.
        name : Optional[str], default=None
            Name of the transformer instance.
            
        Raises
        ------
        ValueError
            If strategy is not one of 'row', 'column', 'threshold' or
            if threshold is not between 0.0 and 1.0.
        """
        super().__init__(name=name)
        if strategy not in ['row', 'column', 'threshold']:
            raise ValueError("strategy must be one of 'row', 'column', 'threshold'")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('threshold must be between 0.0 and 1.0')
        self.strategy = strategy
        self.threshold = threshold
        self.columns = columns
        self._drop_indices = []
        self._fitted = False

    def fit(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> 'MissingDataDeletionTransformer':
        """
        Determine which rows/columns to delete based on the strategy and threshold.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data to fit the transformer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        MissingDataDeletionTransformer
            Self instance for method chaining.
        """
        self._drop_indices = []
        if isinstance(data, np.ndarray):
            if data.size == 0:
                self._drop_indices = []
                self._fitted = True
                return self
            array_data = data
            feature_names = None
        elif isinstance(data, FeatureSet):
            if data.features.size == 0:
                self._drop_indices = []
                self._fitted = True
                return self
            array_data = data.features
            feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            if hasattr(data.data, 'size') and data.data.size == 0:
                self._drop_indices = []
                self._fitted = True
                return self
            array_data = np.array(data.data) if not isinstance(data.data, np.ndarray) else data.data
            feature_names = data.feature_names
        else:
            raise TypeError('Unsupported data type. Expected np.ndarray, FeatureSet, or DataBatch.')
        if array_data.ndim == 1:
            array_data = array_data.reshape(-1, 1)
        if self.columns is not None and feature_names is not None:
            try:
                column_indices = [feature_names.index(col) if isinstance(col, str) else col for col in self.columns]
            except (ValueError, IndexError) as e:
                raise ValueError(f'Invalid column specification: {self.columns}') from e
        elif self.columns is not None:
            column_indices = self.columns
        else:
            column_indices = list(range(array_data.shape[1]))
        if any((idx < 0 or idx >= array_data.shape[1] for idx in column_indices)):
            raise ValueError('Column indices out of bounds')
        if self.strategy == 'row':
            mask = self._create_missing_mask(array_data[:, column_indices])
            self._drop_indices = np.where(mask)[0].tolist()
        elif self.strategy == 'column':
            mask = self._create_missing_mask(array_data[:, column_indices])
            self._drop_indices = [column_indices[i] for i in range(len(column_indices)) if np.any(mask[:, i])]
        elif self.strategy == 'threshold':
            if array_data.shape[0] == 0:
                self._drop_indices = []
            elif self.strategy == 'threshold':
                mask = self._create_missing_mask(array_data[:, column_indices])
                missing_proportions = np.sum(mask, axis=1) / len(column_indices)
                self._drop_indices = np.where(missing_proportions > self.threshold)[0].tolist()
        self._fitted = True
        return self

    def _create_missing_mask(self, data: np.ndarray) -> np.ndarray:
        """Create a boolean mask indicating missing values."""
        if data.size == 0:
            return np.zeros(data.shape, dtype=bool)
        try:
            nan_mask = np.isnan(data)
        except (TypeError, ValueError):
            nan_mask = np.zeros(data.shape, dtype=bool)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if isinstance(data[i, j], (float, np.floating)) and np.isnan(data[i, j]):
                        nan_mask[i, j] = True
        none_mask = np.array([[element is None for element in row] for row in data]) if data.ndim > 0 else np.array([])
        return nan_mask | none_mask

    def transform(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> Union[np.ndarray, DataBatch, FeatureSet]:
        """
        Apply the deletion to remove missing values from the dataset.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data to transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[np.ndarray, DataBatch, FeatureSet]
            Data with missing values removed according to the deletion strategy.
        """
        if not self._fitted:
            raise RuntimeError('Transformer must be fitted before transform can be called.')
        if isinstance(data, np.ndarray):
            return self._transform_array(data)
        elif isinstance(data, FeatureSet):
            return self._transform_featureset(data)
        elif isinstance(data, DataBatch):
            return self._transform_databatch(data)
        else:
            raise TypeError('Unsupported data type. Expected np.ndarray, FeatureSet, or DataBatch.')

    def _transform_array(self, data: np.ndarray) -> np.ndarray:
        """Transform numpy array data."""
        if data.size == 0:
            return data.copy()
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            was_1d = True
        else:
            was_1d = False
        if self.strategy in ['row', 'threshold']:
            if len(self._drop_indices) == 0:
                result = data.copy()
            else:
                mask = np.ones(data.shape[0], dtype=bool)
                mask[self._drop_indices] = False
                result = data[mask]
        elif self.strategy == 'column':
            if len(self._drop_indices) == 0:
                result = data.copy()
            else:
                mask = np.ones(data.shape[1], dtype=bool)
                for idx in self._drop_indices:
                    mask[idx] = False
                result = data[:, mask]
        if was_1d and result.shape[1] == 1:
            result = result.flatten()
        elif was_1d and result.shape[1] > 1:
            result = result.flatten()
        return result

    def _transform_featureset(self, data: FeatureSet) -> FeatureSet:
        """Transform FeatureSet data."""
        if data.features.size == 0:
            return FeatureSet(features=data.features.copy(), feature_names=data.feature_names.copy() if data.feature_names else None, feature_types=data.feature_types.copy() if data.feature_types else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        if self.strategy in ['row', 'threshold']:
            if len(self._drop_indices) == 0:
                filtered_features = data.features.copy()
                filtered_sample_ids = data.sample_ids.copy() if data.sample_ids else None
            else:
                mask = np.ones(data.features.shape[0], dtype=bool)
                mask[self._drop_indices] = False
                filtered_features = data.features[mask]
                if data.sample_ids:
                    filtered_sample_ids = [data.sample_ids[i] for i in range(len(data.sample_ids)) if i not in self._drop_indices]
                else:
                    filtered_sample_ids = None
        elif self.strategy == 'column':
            if len(self._drop_indices) == 0:
                filtered_features = data.features.copy()
                filtered_feature_names = data.feature_names.copy() if data.feature_names else None
                filtered_feature_types = data.feature_types.copy() if data.feature_types else None
            else:
                mask = np.ones(data.features.shape[1], dtype=bool)
                for idx in self._drop_indices:
                    mask[idx] = False
                filtered_features = data.features[:, mask]
                if data.feature_names:
                    filtered_feature_names = [data.feature_names[i] for i in range(len(data.feature_names)) if i not in self._drop_indices]
                else:
                    filtered_feature_names = None
                if data.feature_types:
                    filtered_feature_types = [data.feature_types[i] for i in range(len(data.feature_types)) if i not in self._drop_indices]
                else:
                    filtered_feature_types = None
            return FeatureSet(features=filtered_features, feature_names=filtered_feature_names, feature_types=filtered_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        return FeatureSet(features=filtered_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=filtered_sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)

    def _transform_databatch(self, data: DataBatch) -> DataBatch:
        """Transform DataBatch data."""
        batch_data = data.data
        if hasattr(batch_data, 'size') and batch_data.size == 0:
            return DataBatch(data=batch_data.copy() if hasattr(batch_data, 'copy') else [], labels=data.labels.copy() if data.labels is not None and hasattr(data.labels, 'copy') else data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids.copy() if data.sample_ids is not None and hasattr(data.sample_ids, 'copy') else data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        if not isinstance(batch_data, np.ndarray):
            array_data = np.array(batch_data)
        else:
            array_data = batch_data
        if array_data.ndim == 1:
            array_data = array_data.reshape(-1, 1)
            was_1d = True
        else:
            was_1d = False
        if self.strategy in ['row', 'threshold']:
            if len(self._drop_indices) == 0:
                filtered_data = array_data.copy()
                filtered_labels = data.labels.copy() if data.labels is not None and hasattr(data.labels, 'copy') else data.labels
                filtered_sample_ids = data.sample_ids.copy() if data.sample_ids is not None and hasattr(data.sample_ids, 'copy') else data.sample_ids
            else:
                mask = np.ones(array_data.shape[0], dtype=bool)
                mask[self._drop_indices] = False
                filtered_data = array_data[mask]
                if data.labels is not None:
                    if isinstance(data.labels, np.ndarray):
                        filtered_labels = data.labels[mask]
                    else:
                        filtered_labels = [data.labels[i] for i in range(len(data.labels)) if i not in self._drop_indices]
                else:
                    filtered_labels = None
                if data.sample_ids is not None:
                    if isinstance(data.sample_ids, np.ndarray):
                        filtered_sample_ids = data.sample_ids[mask]
                    else:
                        filtered_sample_ids = [data.sample_ids[i] for i in range(len(data.sample_ids)) if i not in self._drop_indices]
                else:
                    filtered_sample_ids = None
            if was_1d and filtered_data.shape[1] == 1:
                filtered_data = filtered_data.flatten()
            elif was_1d:
                filtered_data = filtered_data.flatten()
            if not isinstance(batch_data, np.ndarray):
                filtered_data = filtered_data.tolist()
            return DataBatch(data=filtered_data, labels=filtered_labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=filtered_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        elif self.strategy == 'column':
            if len(self._drop_indices) == 0:
                filtered_data = array_data.copy()
                filtered_feature_names = data.feature_names
            else:
                mask = np.ones(array_data.shape[1], dtype=bool)
                for idx in self._drop_indices:
                    mask[idx] = False
                filtered_data = array_data[:, mask]
                if data.feature_names:
                    filtered_feature_names = [data.feature_names[i] for i in range(len(data.feature_names)) if i not in self._drop_indices]
                else:
                    filtered_feature_names = None
            if was_1d and filtered_data.shape[1] == 1:
                filtered_data = filtered_data.flatten()
            elif was_1d:
                filtered_data = filtered_data.flatten()
            if not isinstance(batch_data, np.ndarray):
                filtered_data = filtered_data.tolist()
            return DataBatch(data=filtered_data, labels=data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids, feature_names=filtered_feature_names, batch_id=data.batch_id)

    def inverse_transform(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> Union[np.ndarray, DataBatch, FeatureSet]:
        """
        Return the data unchanged since deletion is irreversible.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Transformed data to inverse transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[np.ndarray, DataBatch, FeatureSet]
            Original data format (identity operation).
        """
        return data

class MeanImputationTransformer(BaseTransformer):
    """
    Transformer that performs mean imputation on missing values in numerical data.
    
    This transformer replaces missing values with the mean of the respective column.
    It is suitable for numerical data where the mean is a reasonable estimate for
    missing values under the assumption of data being missing completely at random.
    
    Attributes
    ----------
    columns : Optional[list]
        Specific columns to apply mean imputation to. If None, applies to all numerical columns.
    """

    def __init__(self, columns: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the MeanImputationTransformer.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            List of column names to apply mean imputation. If None, applies to all numerical columns.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.columns = columns
        self._means = {}

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'MeanImputationTransformer':
        """
        Fit the transformer to the input data by calculating column means.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data to fit the transformer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        MeanImputationTransformer
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            X = data.data
            feature_names = data.feature_names
        elif isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
        else:
            raise TypeError('Input data must be DataBatch, FeatureSet, or numpy array')
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            if feature_names is None:
                feature_names = ['feature_0']
        if self.columns is None:
            if feature_names is not None:
                cols_to_process = feature_names
            else:
                cols_to_process = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            cols_to_process = self.columns
            if feature_names is not None:
                missing_cols = set(cols_to_process) - set(feature_names)
                if missing_cols:
                    raise ValueError(f'Specified columns not found in data: {missing_cols}')
        self._means = {}
        for (i, col_name) in enumerate(cols_to_process if feature_names is None else feature_names):
            if col_name in cols_to_process:
                col_idx = i if feature_names is None else feature_names.index(col_name)
                col_data = X[:, col_idx]
                if np.issubdtype(col_data.dtype, np.number):
                    self._means[col_name] = np.nanmean(col_data)
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Apply mean imputation to the input data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Data with mean imputation applied.
        """
        if isinstance(data, DataBatch):
            X = data.data.copy()
            feature_names = data.feature_names
            is_data_batch = True
        elif isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            is_feature_set = True
        elif isinstance(data, np.ndarray):
            X = data.copy()
            feature_names = None
            is_array = True
        else:
            raise TypeError('Input data must be DataBatch, FeatureSet, or numpy array')
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        original_ndim = X.ndim
        if original_ndim == 1:
            X = X.reshape(-1, 1)
            if feature_names is None:
                feature_names = ['feature_0']
        if self.columns is None:
            if feature_names is not None:
                cols_to_process = feature_names
            else:
                cols_to_process = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            cols_to_process = self.columns
        for col_name in cols_to_process:
            if col_name in self._means:
                if feature_names is not None:
                    if col_name in feature_names:
                        col_idx = feature_names.index(col_name)
                    else:
                        continue
                else:
                    col_idx = cols_to_process.index(col_name)
                    if col_idx >= X.shape[1]:
                        continue
                col_data = X[:, col_idx]
                if np.issubdtype(col_data.dtype, np.number):
                    nan_mask = np.isnan(col_data)
                    X[nan_mask, col_idx] = self._means[col_name]
        if original_ndim == 1:
            X = X.flatten()
        if isinstance(data, DataBatch):
            return DataBatch(data=X, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        elif isinstance(data, FeatureSet):
            return FeatureSet(features=X, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        else:
            return X

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Apply inverse transformation (not implemented for this transformer).
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Transformed data to inverse transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Original data format (identity operation).
        """
        return data


# ...(code omitted)...


class ModeFillTransformer(BaseTransformer):
    """
    Transformer that fills missing values with the mode (most frequent value) of each column.
    
    This transformer is similar to ModeImputationTransformer but with a focus on the
    filling operation rather than imputation semantics. It replaces missing values 
    with the mode of the respective column, suitable for categorical data.
    
    Attributes
    ----------
    columns : Optional[list]
        Specific columns to apply mode filling to. If None, applies to all categorical columns.
    """

    def __init__(self, columns: Optional[List[Union[str, int]]]=None, name: Optional[str]=None):
        """
        Initialize the ModeFillTransformer.
        
        Parameters
        ----------
        columns : Optional[List[Union[str, int]]], default=None
            List of column names or indices to apply mode filling. If None, applies to all categorical columns.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.columns = columns
        self._modes = {}
        self._feature_names = None

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'ModeFillTransformer':
        """
        Fit the transformer to the input data by calculating column modes.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data to fit the transformer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        ModeFillTransformer
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            X = data.data
            feature_names = data.feature_names
        elif isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
        else:
            raise TypeError('Input data must be DataBatch, FeatureSet, or numpy array')
        if hasattr(X, 'size') and X.size == 0:
            self._modes = {}
            return self
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            if feature_names is None:
                feature_names = ['feature_0']
        self._feature_names = feature_names
        if self.columns is None:
            if feature_names is not None:
                cols_to_process = list(range(len(feature_names)))
            else:
                cols_to_process = list(range(X.shape[1]))
        elif feature_names is not None and len(self.columns) > 0 and isinstance(self.columns[0], str):
            col_name_to_idx = {name: i for (i, name) in enumerate(feature_names) if name is not None}
            cols_to_process = [col_name_to_idx[col] for col in self.columns if col in col_name_to_idx]
        else:
            cols_to_process = self.columns
        self._modes = {}
        for col_idx in cols_to_process:
            if col_idx >= X.shape[1]:
                continue
            col_data = X[:, col_idx]
            if col_data.dtype.kind in ['U', 'S', 'O'] or (col_data.dtype == object and (not np.issubdtype(col_data.dtype, np.number))):
                valid_mask = ~pd.isna(col_data) & (col_data != 'nan')
            else:
                valid_mask = ~pd.isna(col_data)
            valid_data = col_data[valid_mask]
            if len(valid_data) > 0:
                (unique_vals, counts) = np.unique(valid_data, return_counts=True)
                max_count = np.max(counts)
                mode_candidates = unique_vals[counts == max_count]
                if mode_candidates.dtype.kind in ['U', 'S', 'O'] or (mode_candidates.dtype == object and (not np.issubdtype(mode_candidates.dtype, np.number))):
                    mode_candidates = sorted(mode_candidates)
                else:
                    mode_candidates = np.sort(mode_candidates)
                self._modes[col_idx] = mode_candidates[0]
            else:
                self._modes[col_idx] = None
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Transform the input data by filling missing values with computed modes.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Transformed data with missing values filled.
        """
        if not hasattr(self, '_modes') or self._modes is None:
            raise RuntimeError('Transformer must be fitted before transform can be called.')
        if isinstance(data, DataBatch):
            return self._transform_databatch(data)
        elif isinstance(data, FeatureSet):
            return self._transform_featureset(data)
        elif isinstance(data, np.ndarray):
            return self._transform_array(data)
        else:
            raise TypeError('Input data must be DataBatch, FeatureSet, or numpy array')

    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        """Transform numpy array data."""
        if hasattr(X, 'size') and X.size == 0:
            return X.copy() if hasattr(X, 'copy') else np.array([])
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            was_1d = True
        else:
            was_1d = False
        result = X.copy()
        for (col_idx, mode_val) in self._modes.items():
            if mode_val is not None and col_idx < result.shape[1]:
                col_data = result[:, col_idx]
                if col_data.dtype.kind in ['U', 'S', 'O'] or (col_data.dtype == object and (not np.issubdtype(col_data.dtype, np.number))):
                    nan_mask = pd.isna(col_data) | (col_data == 'nan')
                else:
                    nan_mask = pd.isna(col_data)
                result[nan_mask, col_idx] = mode_val
        if was_1d and result.shape[1] == 1:
            result = result.flatten()
        return result

    def _transform_featureset(self, data: FeatureSet) -> FeatureSet:
        """Transform FeatureSet data."""
        if hasattr(data.features, 'size') and data.features.size == 0:
            return FeatureSet(features=data.features.copy() if hasattr(data.features, 'copy') else [], feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        if not isinstance(data.features, np.ndarray):
            X = np.array(data.features, dtype=object)
        else:
            X = data.features.copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        result_features = self._transform_array(X)
        if X.shape[1] == 1 and result_features.ndim > 1 and (result_features.shape[1] == 1):
            result_features = result_features.flatten()
        return FeatureSet(features=result_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)

    def _transform_databatch(self, data: DataBatch) -> DataBatch:
        """Transform DataBatch data."""
        batch_data = data.data
        if hasattr(batch_data, 'size') and batch_data.size == 0:
            return DataBatch(data=batch_data.copy() if hasattr(batch_data, 'copy') else [], labels=data.labels.copy() if data.labels is not None and hasattr(data.labels, 'copy') else data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids.copy() if data.sample_ids is not None and hasattr(data.sample_ids, 'copy') else data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        if not isinstance(batch_data, np.ndarray):
            X = np.array(batch_data, dtype=object)
        else:
            X = batch_data.copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            was_1d = True
        else:
            was_1d = False
        result_data = self._transform_array(X)
        if was_1d and result_data.shape[1] == 1:
            result_data = result_data.flatten()
        elif was_1d:
            result_data = result_data.flatten()
        if not isinstance(batch_data, np.ndarray):
            result_data = result_data.tolist()
        return DataBatch(data=result_data, labels=data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)

    def inverse_transform(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> Union[np.ndarray, DataBatch, FeatureSet]:
        """
        Return the data unchanged since mode filling is not invertible.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Transformed data to inverse transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[np.ndarray, DataBatch, FeatureSet]
            Original data format (identity operation).
        """
        return data