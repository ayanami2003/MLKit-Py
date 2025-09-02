from typing import Any, Callable, Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class FunctionMapper(BaseTransformer):

    def __init__(self, map_func: Callable[[Any], Any], inverse_func: Optional[Callable[[Any], Any]]=None, columns: Optional[Union[str, int, list]]=None, validate_output: bool=True, name: Optional[str]=None):
        """
        Initialize the FunctionMapper transformer.

        Parameters
        ----------
        map_func : Callable[[Any], Any]
            The function to apply to transform values. Should accept a single value
            or appropriately shaped input and return the transformed value(s).
        inverse_func : Optional[Callable[[Any], Any]], optional
            Optional inverse function that can reverse the mapping. Used by
            inverse_transform method. Default is None.
        columns : Optional[Union[str, int, list]], optional
            Column names (str), indices (int), or list of either to apply the
            mapping to. If None, the function is applied to the entire data.
            Only relevant for structured data formats. Default is None.
        validate_output : bool, optional
            Whether to perform basic validation on the output to ensure it's
            compatible with the expected data structure. Default is True.
        name : Optional[str], optional
            Name for the transformer instance. If None, uses class name.
        """
        super().__init__(name)
        self.map_func = map_func
        self.inverse_func = inverse_func
        self.columns = columns
        self.validate_output = validate_output

    def fit(self, data: Union[DataBatch, FeatureSet, Any], **kwargs) -> 'FunctionMapper':
        """
        Fit the transformer to the data.

        This is a stateless transformer, so fitting is a no-op. This method
        exists to maintain compatibility with the BaseTransformer interface.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, Any]
            Input data to fit on. Not used in this transformer.
        **kwargs : dict
            Additional fitting parameters (ignored).

        Returns
        -------
        FunctionMapper
            Returns self for method chaining.
        """
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, Any], **kwargs) -> Union[DataBatch, FeatureSet, Any]:
        """
        Apply the mapping function to transform the data.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, Any]
            Input data to transform. Can be any supported data structure.
        **kwargs : dict
            Additional transformation parameters.

        Returns
        -------
        Union[DataBatch, FeatureSet, Any]
            Transformed data in the same format as input.

        Raises
        ------
        ValueError
            If columns are specified but data format doesn't support column selection.
        TypeError
            If the map_func cannot be applied to the data.
        """
        if isinstance(data, DataBatch):
            if self.columns is not None:
                transformed_data = self._transform_structured_data(data.data, self.columns)
                return DataBatch(data=transformed_data, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
            else:
                try:
                    transformed_data = self.map_func(data.data, **kwargs)
                    return DataBatch(data=transformed_data, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
                except Exception as e:
                    raise TypeError(f'Failed to apply map_func to DataBatch data: {e}')
        elif isinstance(data, FeatureSet):
            if self.columns is not None:
                transformed_features = self._transform_feature_set_columns(data, self.columns)
                return FeatureSet(features=transformed_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
            else:
                try:
                    transformed_features = self.map_func(data.features, **kwargs)
                    if self.validate_output and (not isinstance(transformed_features, np.ndarray)):
                        raise ValueError('Transformed features must be a numpy array when validate_output is True')
                    return FeatureSet(features=transformed_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
                except Exception as e:
                    raise TypeError(f'Failed to apply map_func to FeatureSet features: {e}')
        elif self.columns is not None:
            raise ValueError('Column specification is only supported for structured data (DataBatch, FeatureSet)')
        else:
            try:
                return self.map_func(data, **kwargs)
            except Exception as e:
                raise TypeError(f'Failed to apply map_func to input data: {e}')

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, Any], **kwargs) -> Union[DataBatch, FeatureSet, Any]:
        """
        Apply the inverse mapping function to revert the transformation.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, Any]
            Transformed data to invert.
        **kwargs : dict
            Additional inversion parameters.

        Returns
        -------
        Union[DataBatch, FeatureSet, Any]
            Data transformed back to original space using inverse_func.

        Raises
        ------
        NotImplementedError
            If no inverse_func was provided during initialization.
        """
        if self.inverse_func is None:
            raise ValueError('No inverse function provided during initialization')
        if isinstance(data, DataBatch):
            if self.columns is not None:
                inverted_data = self._transform_structured_data(data.data, self.columns, use_inverse=True)
                return DataBatch(data=inverted_data, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
            else:
                try:
                    inverted_data = self.inverse_func(data.data, **kwargs)
                    return DataBatch(data=inverted_data, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
                except Exception as e:
                    raise TypeError(f'Failed to apply inverse_func to DataBatch data: {e}')
        elif isinstance(data, FeatureSet):
            if self.columns is not None:
                inverted_features = self._transform_feature_set_columns(data, self.columns, use_inverse=True)
                return FeatureSet(features=inverted_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
            else:
                try:
                    inverted_features = self.inverse_func(data.features, **kwargs)
                    if self.validate_output and (not isinstance(inverted_features, np.ndarray)):
                        raise ValueError('Inverted features must be a numpy array when validate_output is True')
                    return FeatureSet(features=inverted_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
                except Exception as e:
                    raise TypeError(f'Failed to apply inverse_func to FeatureSet features: {e}')
        elif self.columns is not None:
            raise ValueError('Column specification is only supported for structured data (DataBatch, FeatureSet)')
        elif isinstance(data, list) and len(data) > 0 and (not isinstance(data[0], (list, np.ndarray))):
            try:
                return [self.inverse_func(item) for item in data]
            except Exception as e:
                raise TypeError(f'Failed to apply inverse_func element-wise to list: {e}')
        else:
            try:
                return self.inverse_func(data, **kwargs)
            except Exception as e:
                raise TypeError(f'Failed to apply inverse_func to input data: {e}')

    def _transform_structured_data(self, data: Union[np.ndarray, List], columns: Union[str, int, List]):
        """Helper method to transform structured data with column specification."""
        if not isinstance(columns, list):
            columns = [columns]
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                if len(columns) != 1:
                    raise ValueError('1D array can only be transformed with a single column specification')
                return self.map_func(data)
            elif data.ndim == 2:
                transformed_data = data.copy()
                for col in columns:
                    if isinstance(col, str):
                        raise ValueError('String column names are not supported for raw numpy arrays')
                    elif isinstance(col, int):
                        if col < 0 or col >= data.shape[1]:
                            raise IndexError(f'Column index {col} is out of bounds for array with {data.shape[1]} columns')
                        transformed_data[:, col] = self.map_func(data[:, col])
                    else:
                        raise TypeError('Column specification must be string or integer')
                return transformed_data
            else:
                return self.map_func(data)
        elif isinstance(data, list):
            if len(data) == 0:
                return data
            if isinstance(data[0], (list, np.ndarray)) and len(data[0]) > 0:
                transformed_data = [list(row) for row in data]
                for col in columns:
                    if isinstance(col, str):
                        raise ValueError('String column names are not supported for raw lists')
                    elif isinstance(col, int):
                        if col < 0 or col >= len(data[0]):
                            raise IndexError(f'Column index {col} is out of bounds for list with {len(data[0])} columns')
                        for i in range(len(data)):
                            transformed_data[i][col] = self.map_func(data[i][col])
                    else:
                        raise TypeError('Column specification must be string or integer')
                return transformed_data
            else:
                if len(columns) != 1:
                    raise ValueError('1D list can only be transformed with a single column specification')
                return [self.map_func(item) for item in data]
        else:
            return self.map_func(data)

    def _transform_feature_set_columns(self, feature_set: FeatureSet, columns: Union[str, int, List]):
        """Helper method to transform specific columns of a FeatureSet."""
        if not isinstance(columns, list):
            columns = [columns]
        transformed_features = feature_set.features.copy()
        for col in columns:
            if isinstance(col, str):
                if feature_set.feature_names is None:
                    raise ValueError('Feature names not available in FeatureSet')
                try:
                    col_idx = feature_set.feature_names.index(col)
                except ValueError:
                    raise KeyError(f"Feature '{col}' not found in feature names")
            elif isinstance(col, int):
                if col < 0 or col >= feature_set.features.shape[1]:
                    raise IndexError(f'Column index {col} is out of bounds for FeatureSet with {feature_set.features.shape[1]} features')
                col_idx = col
            else:
                raise TypeError('Column specification must be string or integer')
            transformed_features[:, col_idx] = self.map_func(feature_set.features[:, col_idx])
        return transformed_features