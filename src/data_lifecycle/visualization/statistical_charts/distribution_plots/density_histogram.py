import numpy as np
from typing import Optional, Union, List
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.data_batch import DataBatch

class DensityHistogramPlotter(BasePipelineComponent):
    """
    A pipeline component for generating density histograms to visualize the distribution of data.
    
    This class creates histogram plots where the y-axis represents density instead of raw counts,
    enabling comparisons across datasets with different sample sizes. It supports customization
    of bins, normalization methods, and visual styling.
    
    Attributes:
        name (Optional[str]): Name identifier for the component.
        bins (Union[int, str, List[float]]): Number of bins, binning strategy ('auto', 'sturges', etc.), or bin edges.
        density (bool): If True, normalizes the histogram to form a probability density.
        color (Optional[str]): Color for the histogram bars.
        alpha (float): Transparency level for the histogram (0.0 to 1.0).
        edgecolor (Optional[str]): Color for the edges of the bars.
        linewidth (float): Width of the bar edges.
        
    Methods:
        process: Generates the density histogram from input data.
    """

    def __init__(self, name: Optional[str]=None, bins: Union[int, str, List[float]]='auto', density: bool=True, color: Optional[str]=None, alpha: float=0.7, edgecolor: Optional[str]=None, linewidth: float=0.5):
        """
        Initialize the DensityHistogramPlotter.
        
        Args:
            name (Optional[str]): Name identifier for the component.
            bins (Union[int, str, List[float]]): Number of bins, binning strategy ('auto', 'sturges', etc.), or bin edges.
            density (bool): If True, normalizes the histogram to form a probability density.
            color (Optional[str]): Color for the histogram bars.
            alpha (float): Transparency level for the histogram (0.0 to 1.0).
            edgecolor (Optional[str]): Color for the edges of the bars.
            linewidth (float): Width of the bar edges.
        """
        super().__init__(name=name)
        self.bins = bins
        self.density = density
        self.color = color
        self.alpha = alpha
        self.edgecolor = edgecolor
        self.linewidth = linewidth

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate a density histogram from input data.
        
        This method takes either a DataBatch or numpy array and produces a density histogram
        visualization. For DataBatch inputs, it uses the primary data attribute.
        
        Args:
            data (Union[DataBatch, np.ndarray]): Input data to visualize. If DataBatch, uses data.data.
            **kwargs: Additional keyword arguments for histogram generation.
            
        Returns:
            dict: A dictionary containing:
                - 'figure': The matplotlib figure object
                - 'axes': The matplotlib axes object
                - 'bins': The computed bin edges
                - 'counts': The normalized counts for each bin
                
        Raises:
            ValueError: If data is empty or not in a supported format.
            TypeError: If data is not a DataBatch or numpy array.
        """
        import matplotlib.pyplot as plt
        if isinstance(data, DataBatch):
            values = data.data
        elif isinstance(data, np.ndarray):
            values = data
        else:
            raise TypeError('data must be a DataBatch or numpy array')
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if values.ndim > 1:
            values = values.ravel()
        if values.size == 0:
            raise ValueError('Input data is empty')
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError('No finite values in input data')
        (figure, axes) = plt.subplots(figsize=(8, 6))
        (counts, bin_edges, _) = axes.hist(finite_values, bins=self.bins, density=self.density, color=self.color, alpha=self.alpha, edgecolor=self.edgecolor, linewidth=self.linewidth, **kwargs)
        axes.set_xlabel('Value')
        axes.set_ylabel('Density' if self.density else 'Frequency')
        axes.grid(True, alpha=0.3)
        return {'figure': figure, 'axes': axes, 'bins': bin_edges, 'counts': counts}