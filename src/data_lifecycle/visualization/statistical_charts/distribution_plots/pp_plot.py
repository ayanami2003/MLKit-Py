import numpy as np
from typing import Optional, Union, List
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.data_batch import DataBatch
from scipy import stats

class PPPlotGenerator(BasePipelineComponent):

    def __init__(self, name: Optional[str]=None, distribution: str='norm', distribution_params: Optional[dict]=None, reference_line: bool=True):
        """
        Initialize the P-P Plot Generator.
        
        Parameters
        ----------
        name : Optional[str]
            Name identifier for the component
        distribution : str, default='norm'
            Name of the theoretical distribution to compare against
        distribution_params : Optional[dict], default=None
            Parameters for the theoretical distribution
        reference_line : bool, default=True
            Whether to draw the reference diagonal line (y=x)
        """
        super().__init__(name)
        self.distribution = distribution
        self.distribution_params = distribution_params or {}
        self.reference_line = reference_line

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate a P-P plot from input data.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data to generate P-P plot for. If DataBatch, uses the 'data' attribute.
        **kwargs : dict
            Additional processing parameters (unused in this implementation)
            
        Returns
        -------
        dict
            Dictionary containing plot data and metadata:
            - 'theoretical_probs': Theoretical cumulative probabilities
            - 'empirical_probs': Empirical cumulative probabilities
            - 'distribution': Name of theoretical distribution used
            - 'distribution_params': Parameters of theoretical distribution
            - 'reference_line': Boolean indicating if reference line is included
            
        Raises
        ------
        ValueError
            If data is empty or invalid distribution parameters are provided
        """
        if isinstance(data, DataBatch):
            raw_data = np.array(data.data)
        else:
            raw_data = np.array(data)
        if raw_data.size == 0:
            raise ValueError('Input data is empty')
        if not np.all(np.isfinite(raw_data)):
            raise ValueError('Input data contains NaN or infinite values')
        if raw_data.ndim > 1:
            raw_data = raw_data.flatten()
        sorted_data = np.sort(raw_data)
        n = len(sorted_data)
        empirical_probs = np.arange(1, n + 1) / n
        try:
            dist = getattr(stats, self.distribution)
        except AttributeError:
            raise ValueError(f'Invalid distribution name: {self.distribution}')
        try:
            theoretical_probs = dist.cdf(sorted_data, **self.distribution_params)
        except Exception as e:
            raise ValueError(f'Error calculating theoretical probabilities: {str(e)}')
        return {'theoretical_probs': theoretical_probs, 'empirical_probs': empirical_probs, 'distribution': self.distribution, 'distribution_params': self.distribution_params, 'reference_line': self.reference_line}