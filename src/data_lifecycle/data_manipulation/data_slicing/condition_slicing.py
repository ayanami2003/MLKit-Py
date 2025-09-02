from typing import Union, Optional, Callable, Any
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import pandas as pd
import numpy as np

class ConditionSlicer(BaseTransformer):
    """
    A transformer that slices data based on a specified condition function or expression.
    
    This component allows for flexible data filtering by applying a user-defined condition
    to select rows or samples from a dataset. It supports both FeatureSet and DataBatch inputs,
    enabling integration into various stages of the data processing pipeline.
    
    The condition can be specified as a callable function that takes a data row/sample and
    returns a boolean indicating whether to include that sample in the output.
    
    Attributes
    ----------
    condition : Union[Callable[[Any], bool], str]
        A function or string expression that defines the slicing condition.
        If a string, it should be a valid expression that can be evaluated
        in the context of a data row.
    name : Optional[str]
        Name identifier for the transformer instance.
        
    Examples
    --------
    >>> import numpy as np
    >>> from general.structures.feature_set import FeatureSet
    >>> 
    >>> # Create sample data
    >>> features = np.array([[1, 2], [3, 4], [5, 6]])
    >>> feature_set = FeatureSet(features=features, feature_names=['a', 'b'])
    >>> 
    >>> # Create condition slicer
    >>> slicer = ConditionSlicer(condition=lambda x: x[0] > 2)  # Select rows where first feature > 2
    >>> 
    >>> # Apply slicing
    >>> sliced_data = slicer.fit_transform(feature_set)
    >>> print(sliced_data.features)
    [[3 4]
     [5 6]]
    """

    def __init__(self, condition: Union[Callable[[Any], bool], str], name: Optional[str]=None):
        """
        Initialize the ConditionSlicer.
        
        Parameters
        ----------
        condition : Union[Callable[[Any], bool], str]
            A function that takes a data row and returns True for rows to keep,
            or a string expression that evaluates to a boolean in the context
            of a data row.
        name : Optional[str]
            Name identifier for the transformer instance.
        """
        super().__init__(name)
        self.condition = condition

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'ConditionSlicer':
        """
        Fit the transformer to the input data.
        
        For ConditionSlicer, fitting is a no-op since no model parameters
        need to be learned from the data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to fit the transformer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        ConditionSlicer
            Self instance for method chaining.
        """
        return self

    def _evaluate_condition(self, row: Union[np.ndarray, pd.Series], is_feature_set: bool) -> bool:
        """
        Evaluate the condition for a single row.
        
        Parameters
        ----------
        row : Union[np.ndarray, pd.Series]
            A single data row to evaluate the condition on.
        is_feature_set : bool
            Whether the data is a FeatureSet (affects how string conditions are evaluated).
            
        Returns
        -------
        bool
            Result of evaluating the condition on the row.
            
        Raises
        ------
        ValueError
            If the condition cannot be evaluated.
        """
        try:
            if callable(self.condition):
                return bool(self.condition(row))
            elif isinstance(self.condition, str):
                if is_feature_set and hasattr(row, '__len__') and (len(row) > 0):
                    if isinstance(row, np.ndarray):
                        env = {f'f{i}': val for (i, val) in enumerate(row)}
                    else:
                        env = {f'f{i}': val for (i, val) in enumerate(row)}
                    result = eval(self.condition, {'__builtins__': {}}, env)
                    return bool(result)
                else:
                    env = {'x': row}
                    result = eval(self.condition, {'__builtins__': {}}, env)
                    return bool(result)
            else:
                raise ValueError(f'Condition must be callable or string, got {type(self.condition)}')
        except Exception as e:
            raise ValueError(f"Failed to evaluate condition '{self.condition}' on data row: {str(e)}") from e

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Apply the condition-based slicing to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to slice based on the condition.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Sliced data containing only rows that satisfy the condition.
            
        Raises
        ------
        ValueError
            If the condition cannot be applied to the data.
        """
        mask = []
        is_feature_set = isinstance(data, FeatureSet)
        if is_feature_set:
            if data.features is None:
                raise ValueError('FeatureSet has no features to slice')
            iterable_data = data.features
        else:
            if data.samples is None:
                raise ValueError('DataBatch has no samples to slice')
            iterable_data = data.samples
        for row in iterable_data:
            mask.append(self._evaluate_condition(row, is_feature_set))
        mask = np.array(mask, dtype=bool)
        if is_feature_set:
            sliced_features = data.features[mask] if data.features is not None else None
            sliced_targets = data.targets[mask] if data.targets is not None else None
            return FeatureSet(features=sliced_features, targets=sliced_targets, feature_names=data.feature_names, target_names=data.target_names, metadata=data.metadata)
        else:
            sliced_samples = [sample for (i, sample) in enumerate(data.samples) if mask[i]] if data.samples is not None else None
            sliced_labels = data.labels[mask] if data.labels is not None else None
            return DataBatch(samples=sliced_samples, labels=sliced_labels, metadata=data.metadata)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Apply the inverse transformation (not applicable for slicing).
        
        Since slicing is a lossy operation (data is removed), the inverse
        transformation simply returns the input data unchanged.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data (ignored in this implementation).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            The same data passed as input.
        """
        return data