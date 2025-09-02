from typing import Any, Callable, Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class GenericDataTransformer(BaseTransformer):
    """
    A flexible transformer for applying generic transformations to data batches or feature sets.
    
    This transformer allows users to define custom transformation functions that can be applied
    to various data structures in the pipeline. It supports both stateful and stateless transformations
    and maintains compatibility with the standard transformer interface.
    
    Attributes
    ----------
    transform_func : Callable[[Any], Any]
        The function to apply to transform the data
    inverse_func : Optional[Callable[[Any], Any]]
        Optional function to apply for inverse transformation
    validate_output : bool
        Whether to perform basic validation on transformed outputs
    
    Examples
    --------
    >>> import numpy as np
    >>> from general.structures.feature_set import FeatureSet
    >>> 
    >>> # Define a simple transformation function
    >>> def log_transform(data):
    ...     if isinstance(data, FeatureSet):
    ...         return FeatureSet(
    ...             features=np.log1p(data.features),
    ...             feature_names=data.feature_names,
    ...             feature_types=data.feature_types
    ...         )
    ...     return np.log1p(data)
    >>> 
    >>> # Create transformer
    >>> transformer = GenericDataTransformer(log_transform)
    >>> 
    >>> # Apply to data
    >>> features = np.array([[1, 2], [3, 4]])
    >>> transformed = transformer.transform(features)
    """

    def __init__(self, transform_func: Callable[[Any], Any], inverse_func: Optional[Callable[[Any], Any]]=None, validate_output: bool=True, name: Optional[str]=None):
        """
        Initialize the GenericDataTransformer.
        
        Parameters
        ----------
        transform_func : Callable[[Any], Any]
            Function that takes data as input and returns transformed data
        inverse_func : Optional[Callable[[Any], Any]], optional
            Function that performs the inverse transformation, by default None
        validate_output : bool, optional
            Whether to validate the output structure, by default True
        name : Optional[str], optional
            Name for the transformer, by default None
        """
        super().__init__(name=name)
        self.transform_func = transform_func
        self.inverse_func = inverse_func
        self.validate_output = validate_output

    def fit(self, data: Union[DataBatch, FeatureSet, Any], **kwargs) -> 'GenericDataTransformer':
        """
        Fit the transformer to the data (no-op for generic transformer).
        
        This method exists to maintain compatibility with the BaseTransformer interface.
        For generic transformations, fitting is typically not required.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, Any]
            Input data to fit on
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        GenericDataTransformer
            Self instance for method chaining
        """
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, Any], **kwargs) -> Union[DataBatch, FeatureSet, Any]:
        """
        Apply the transformation function to the input data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, Any]
            Input data to transform
        **kwargs : dict
            Additional parameters to pass to the transform function
            
        Returns
        -------
        Union[DataBatch, FeatureSet, Any]
            Transformed data in the same format as input when possible
            
        Raises
        ------
        ValueError
            If validate_output is True and the output structure is invalid
        """
        transformed_data = self.transform_func(data, **kwargs)
        if self.validate_output:
            if isinstance(data, (DataBatch, FeatureSet)) and (not isinstance(transformed_data, (DataBatch, FeatureSet, type(data)))):
                raise ValueError(f'Transformed data type {type(transformed_data)} does not match expected types (DataBatch, FeatureSet) or input type {type(data)}')
        return transformed_data

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, Any], **kwargs) -> Union[DataBatch, FeatureSet, Any]:
        """
        Apply the inverse transformation if an inverse function was provided.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, Any]
            Transformed data to invert
        **kwargs : dict
            Additional parameters to pass to the inverse function
            
        Returns
        -------
        Union[DataBatch, FeatureSet, Any]
            Inverted data in the same format as input when possible
            
        Raises
        ------
        NotImplementedError
            If no inverse function was provided
        """
        if self.inverse_func is None:
            raise NotImplementedError('No inverse function was provided during initialization')
        inverted_data = self.inverse_func(data, **kwargs)
        if self.validate_output:
            if isinstance(data, (DataBatch, FeatureSet)) and (not isinstance(inverted_data, (DataBatch, FeatureSet, type(data)))):
                raise ValueError(f'Inverted data type {type(inverted_data)} does not match expected types (DataBatch, FeatureSet) or input type {type(data)}')
        return inverted_data