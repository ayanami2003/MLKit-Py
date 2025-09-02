from typing import List, Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch

class ChangePointDetector(BaseTransformer):

    def __init__(self, method: str='pelt', cost_function: str='l2', min_size: int=5, penalty: Optional[float]=None, jump: int=5, name: Optional[str]=None):
        """
        Initialize the ChangePointDetector.
        
        Parameters
        ----------
        method : str, default='pelt'
            Algorithm to use for change point detection. Options include:
            - 'pelt': Pruned Exact Linear Time
            - 'binseg': Binary Segmentation
            - 'window': Sliding Window
        cost_function : str, default='l2'
            Cost function for evaluating changes. Options include:
            - 'l2': Mean squared error
            - 'l1': Mean absolute error
            - 'rbf': Radial basis function
            - 'mahalanobis': Mahalanobis distance
        min_size : int, default=5
            Minimum size of a segment between change points
        penalty : float, optional
            Penalty value to control number of change points. If None,
            uses a default value based on the data size
        jump : int, default=5
            Only consider every jump-th point as a potential change point
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.method = method
        self.cost_function = cost_function
        self.min_size = min_size
        self.penalty = penalty
        self.jump = jump

    def fit(self, data: DataBatch, **kwargs) -> 'ChangePointDetector':
        """
        Fit the change point detector to the time series data.
        
        This method analyzes the data to determine parameters needed for
        detecting change points but does not actually perform the detection.
        
        Parameters
        ----------
        data : DataBatch
            Time series data to analyze. The data should be univariate
            with the time dimension in the first axis
        **kwargs : dict
            Additional fitting parameters (ignored)
            
        Returns
        -------
        ChangePointDetector
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the data is not one-dimensional or empty
        """
        if not isinstance(data, DataBatch):
            raise TypeError('Input data must be a DataBatch instance')
        ts_data = data.data
        if not isinstance(ts_data, np.ndarray):
            ts_data = np.array(ts_data)
        if ts_data.ndim != 1:
            raise ValueError(f'Time series data must be 1-dimensional, got {ts_data.ndim}D array')
        if ts_data.size == 0:
            raise ValueError('Time series data cannot be empty')
        self._data_length = len(ts_data)
        if self.penalty is None:
            self._effective_penalty = 2 * np.log(self._data_length)
        else:
            self._effective_penalty = self.penalty
        self._is_fitted = True
        return self

    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Detect change points in the time series data.
        
        Applies the configured change point detection algorithm to identify
        significant changes in the statistical properties of the time series.
        The detected change points are added to the metadata of the returned
        DataBatch.
        
        Parameters
        ----------
        data : DataBatch
            Time series data to analyze for change points
        **kwargs : dict
            Additional transformation parameters (ignored)
            
        Returns
        -------
        DataBatch
            Original data with added metadata containing:
            - 'change_points': List of indices where changes occur
            - 'n_changes': Number of detected change points
            - 'segments': List of (start, end) tuples for each segment
            
        Raises
        ------
        ValueError
            If the detector has not been fitted or data is incompatible
        """
        pass

    def inverse_transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Inverse transform is not applicable for change point detection.
        
        Change point detection is a one-way analysis that identifies structural
        breaks in time series data. There is no meaningful inverse operation.
        
        Parameters
        ----------
        data : DataBatch
            Data to "invert" (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        DataBatch
            The same data passed as input
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not supported
        """
        pass

    def get_change_points(self) -> List[int]:
        """
        Get the indices of detected change points.
        
        Returns
        -------
        List[int]
            Sorted list of indices where change points were detected.
            Empty list if no change points were detected or transform
            has not yet been called.
        """
        pass

    def get_segments(self) -> List[tuple]:
        """
        Get the time series segments defined by change points.
        
        Returns
        -------
        List[tuple]
            List of (start_index, end_index) tuples defining the segments
            between change points. Includes the initial and final segments.
        """
        pass