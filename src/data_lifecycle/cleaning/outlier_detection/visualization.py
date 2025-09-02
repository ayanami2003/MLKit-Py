from general.structures.data_batch import DataBatch
from typing import Union, Optional, Dict, Any, Tuple, NamedTuple
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.base_classes.validator_base import BaseValidator


# ...(code omitted)...


class ZScoreOutlierRemover(BaseTransformer):
    """
    Transformer for removing outliers based on Z-score thresholds.
    
    Implements robust outlier filtering by calculating Z-scores and removing data points
    that exceed a specified threshold. Uses robust statistics (median and MAD) for
    improved resistance to outliers in the calculation process.
    """

    def __init__(self, threshold: float=3.0, use_modified_zscore: bool=True, axis: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the Z-score outlier remover.
        
        Args:
            threshold (float): Z-score threshold for outlier identification. Default is 3.0.
            use_modified_zscore (bool): Whether to use modified Z-score (based on median and MAD) 
                                      instead of standard Z-score (based on mean and std).
            axis (Optional[int]): Axis along which to compute statistics. If None, computes over all axes.
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name)
        self.threshold = threshold
        self.use_modified_zscore = use_modified_zscore
        self.axis = axis
        self.center = None
        self.scale = None

    def fit(self, data: Union[np.ndarray, DataBatch], **kwargs) -> 'ZScoreOutlierRemover':
        """
        Compute center and scale statistics from the input data for Z-score calculation.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to fit the transformer on.
            **kwargs: Additional parameters for fitting.
            
        Returns:
            ZScoreOutlierRemover: Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            data_array = data.data
        else:
            data_array = data
        if not isinstance(data_array, np.ndarray):
            data_array = np.asarray(data_array)
        if self.use_modified_zscore:
            self.center = np.median(data_array, axis=self.axis, keepdims=True)
            mad = np.median(np.abs(data_array - self.center), axis=self.axis, keepdims=True)
            self.scale = np.where(mad == 0, 1e-10, mad)
        else:
            self.center = np.mean(data_array, axis=self.axis, keepdims=True)
            self.scale = np.std(data_array, axis=self.axis, keepdims=True)
            self.scale = np.where(self.scale == 0, 1e-10, self.scale)
        return self

    def transform(self, data: Union[np.ndarray, DataBatch], **kwargs) -> Union[np.ndarray, DataBatch]:
        """
        Remove outliers from the input data based on Z-score thresholds.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to transform.
            **kwargs: Additional parameters for transformation.
            
        Returns:
            Union[np.ndarray, DataBatch]: Data with outliers removed.
        """
        if self.center is None or self.scale is None:
            raise RuntimeError("Transformer has not been fitted. Call 'fit' first.")
        if isinstance(data, DataBatch):
            data_array = data.data
            is_databatch = True
            original_sample_ids = data.sample_ids
            original_feature_names = data.feature_names
            original_metadata = data.metadata
            original_labels = data.labels
        else:
            data_array = data
            is_databatch = False
            original_sample_ids = None
            original_feature_names = None
            original_metadata = None
            original_labels = None
        if not isinstance(data_array, np.ndarray):
            data_array = np.asarray(data_array)
        z_scores = self.get_z_scores(data_array)
        mask = np.abs(z_scores) <= self.threshold
        original_shape = data_array.shape
        flat_data = data_array.reshape(-1, original_shape[-1]) if data_array.ndim > 1 else data_array.flatten()
        flat_mask = mask.flatten()
        cleaned_data = flat_data[flat_mask]
        if data_array.ndim > 1 and len(cleaned_data) > 0:
            try:
                if data_array.shape[-1] > 0:
                    cleaned_data = cleaned_data.reshape(-1, data_array.shape[-1])
            except ValueError:
                pass
        if is_databatch:
            new_sample_ids = None
            if original_sample_ids is not None:
                try:
                    flat_sample_ids = np.array(original_sample_ids).flatten()
                    if len(flat_sample_ids) == len(flat_mask):
                        new_sample_ids = flat_sample_ids[flat_mask].tolist()
                        if len(new_sample_ids) > len(cleaned_data):
                            new_sample_ids = new_sample_ids[:len(cleaned_data)]
                except:
                    pass
            new_labels = None
            if original_labels is not None:
                try:
                    flat_labels = np.array(original_labels).flatten()
                    if len(flat_labels) == len(flat_mask):
                        new_labels = flat_labels[flat_mask]
                        if original_labels.ndim > 1:
                            try:
                                new_labels = new_labels.reshape(-1, original_labels.shape[-1])
                            except:
                                pass
                except:
                    pass
            return DataBatch(data=cleaned_data, labels=new_labels, sample_ids=new_sample_ids, feature_names=original_feature_names, metadata=original_metadata)
        else:
            return cleaned_data

    def inverse_transform(self, data: Union[np.ndarray, DataBatch], **kwargs) -> Union[np.ndarray, DataBatch]:
        """
        Inverse transformation (identity operation - cannot reconstruct removed outliers).
        
        Args:
            data (Union[np.ndarray, DataBatch]): Transformed data.
            **kwargs: Additional parameters.
            
        Returns:
            Union[np.ndarray, DataBatch]: Unchanged data (inverse not supported).
        """
        return data

    def get_z_scores(self, data: Union[np.ndarray, DataBatch]) -> Union[np.ndarray, DataBatch]:
        """
        Calculate Z-scores for the input data based on fitted statistics.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data for Z-score calculation.
            
        Returns:
            Union[np.ndarray, DataBatch]: Calculated Z-scores.
        """
        if self.center is None or self.scale is None:
            raise RuntimeError("Transformer has not been fitted. Call 'fit' first.")
        if isinstance(data, DataBatch):
            data_array = data.data
            is_databatch = True
            original_sample_ids = data.sample_ids
            original_feature_names = data.feature_names
            original_metadata = data.metadata
            original_labels = data.labels
        else:
            data_array = data
            is_databatch = False
        if not isinstance(data_array, np.ndarray):
            data_array = np.asarray(data_array)
        if self.use_modified_zscore:
            z_scores = 0.6745 * (data_array - self.center) / self.scale
        else:
            z_scores = (data_array - self.center) / self.scale
        if is_databatch:
            return DataBatch(data=z_scores, labels=original_labels, sample_ids=original_sample_ids, feature_names=original_feature_names, metadata=original_metadata)
        else:
            return z_scores