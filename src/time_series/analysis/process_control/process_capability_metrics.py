from typing import Union, Optional, Dict, Any, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
import numpy as np

class ProcessCapabilityIndicesCalculator(BaseTransformer):
    """
    A transformer that computes process capability indices (Cp and Cpk) for time series data.
    
    This class evaluates how well a process meets its specification limits by calculating
    the Cp (process potential) and Cpk (process capability) indices. These indices help
    determine if a process is capable of producing outputs within specified tolerance limits.
    
    Cp = (USL - LSL) / (6 * sigma)
    Cpk = min[(USL - mean) / (3 * sigma), (mean - LSL) / (3 * sigma)]
    
    Where:
    - USL: Upper Specification Limit
    - LSL: Lower Specification Limit
    - sigma: Process standard deviation
    - mean: Process mean
    
    Attributes
    ----------
    feature_column : str
        Name of the column containing the process measurements
    usl : float
        Upper Specification Limit
    lsl : float
        Lower Specification Limit
    mean : Optional[float]
        Process mean (computed during fit)
    std_dev : Optional[float]
        Process standard deviation (computed during fit)
    cp : Optional[float]
        Process capability index Cp (computed during fit)
    cpk : Optional[float]
        Process capability index Cpk (computed during fit)
    """

    def __init__(self, feature_column: str, usl: float, lsl: float, name: Optional[str]=None):
        """
        Initialize the ProcessCapabilityIndicesCalculator.
        
        Parameters
        ----------
        feature_column : str
            Name of the column containing the process measurements
        usl : float
            Upper Specification Limit
        lsl : float
            Lower Specification Limit
        name : Optional[str]
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.feature_column = feature_column
        self.usl = usl
        self.lsl = lsl
        self.mean = None
        self.std_dev = None
        self.cp = None
        self.cpk = None

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'ProcessCapabilityIndicesCalculator':
        """
        Compute process mean, standard deviation, Cp, and Cpk from the input data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Input data containing the process measurements
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        ProcessCapabilityIndicesCalculator
            Self instance for method chaining
        """
        if isinstance(data, DataBatch):
            if data.feature_names is not None and self.feature_column in data.feature_names:
                col_idx = data.feature_names.index(self.feature_column)
                if isinstance(data.data, np.ndarray):
                    values = data.data[:, col_idx]
                else:
                    values = np.array([row[col_idx] for row in data.data])
            else:
                raise ValueError(f"Feature column '{self.feature_column}' not found in DataBatch")
        elif isinstance(data, FeatureSet):
            if data.feature_names is not None and self.feature_column in data.feature_names:
                col_idx = data.feature_names.index(self.feature_column)
                values = data.features[:, col_idx]
            else:
                raise ValueError(f"Feature column '{self.feature_column}' not found in FeatureSet")
        else:
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        values = values[~np.isnan(values)]
        self.mean = float(np.mean(values))
        self.std_dev = float(np.std(values, ddof=1))
        if self.std_dev > 0:
            self.cp = (self.usl - self.lsl) / (6 * self.std_dev)
            cpk_upper = (self.usl - self.mean) / (3 * self.std_dev)
            cpk_lower = (self.mean - self.lsl) / (3 * self.std_dev)
            self.cpk = min(cpk_upper, cpk_lower)
        else:
            self.cp = np.inf if self.usl - self.lsl > 0 else 0.0
            self.cpk = np.inf if self.usl - self.lsl > 0 else 0.0
        return self

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Add Cp and Cpk indices as metadata to the input data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Input data to transform
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        FeatureSet
            Transformed data with capability indices in metadata
        """
        if isinstance(data, DataBatch):
            features = data.data if isinstance(data.data, np.ndarray) else np.array(data.data)
            feature_set = FeatureSet(features=features, feature_names=data.feature_names, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else {})
        else:
            feature_set = FeatureSet(features=data.features.copy(), feature_names=data.feature_names.copy() if data.feature_names else None, feature_types=data.feature_types.copy() if data.feature_types else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else {}, quality_scores=data.quality_scores.copy() if data.quality_scores else {})
        if feature_set.metadata is None:
            feature_set.metadata = {}
        capability_data = {'mean': self.mean, 'std_dev': self.std_dev, 'cp': self.cp, 'cpk': self.cpk, 'usl': self.usl, 'lsl': self.lsl, 'feature_column': self.feature_column}
        feature_set.metadata['capability_indices'] = capability_data
        return feature_set

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Return the input data unchanged (no inverse transformation needed).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to transform
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Unchanged input data
        """
        return data

    def get_capability_indices(self) -> Dict[str, Optional[float]]:
        """
        Retrieve the computed process capability indices.
        
        Returns
        -------
        Dict[str, Optional[float]]
            Dictionary containing Cp and Cpk values
        """
        return {'cp': self.cp, 'cpk': self.cpk}