from typing import Optional, Union, List
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.data_batch import DataBatch
import numpy as np

class KernelDensityEstimatePlotter(BasePipelineComponent):
    """
    A pipeline component for generating kernel density estimate (KDE) plots for data visualization.
    
    This class produces smooth density estimates for numerical data distributions using kernel
    smoothing techniques. It supports various kernel functions and bandwidth selection methods
    to visualize underlying probability density functions of datasets.
    
    Attributes:
        name (Optional[str]): Name identifier for the component.
        bandwidth (Union[float, str]): Bandwidth for KDE. Can be a float value or method name
            ('scott', 'silverman'). Defaults to 'scott'.
        kernel (str): Type of kernel function to use. Supported values include 'gaussian',
            'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'. Defaults to 'gaussian'.
        grid_points (int): Number of points in the evaluation grid for density estimation.
            Defaults to 1000.
        fill_area (bool): Whether to fill the area under the density curve. Defaults to True.
        colors (Optional[List[str]]): List of colors for plotting multiple distributions.
    
    Methods:
        process: Generate KDE plot data from input numerical data.
    """

    def __init__(self, name: Optional[str]=None, bandwidth: Union[float, str]='scott', kernel: str='gaussian', grid_points: int=1000, fill_area: bool=True, colors: Optional[List[str]]=None):
        """
        Initialize the KernelDensityEstimatePlotter.
        
        Args:
            name: Optional name for the component.
            bandwidth: Bandwidth for KDE. Can be a fixed float value or method name
                ('scott', 'silverman').
            kernel: Kernel function type for density estimation.
            grid_points: Number of evaluation points for the density curve.
            fill_area: Whether to fill area under the density curve.
            colors: List of colors for multiple distribution plotting.
        """
        super().__init__(name)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.grid_points = grid_points
        self.fill_area = fill_area
        self.colors = colors
        valid_kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        if self.kernel not in valid_kernels:
            raise ValueError(f"Invalid kernel '{self.kernel}'. Supported kernels: {valid_kernels}")
        if isinstance(self.bandwidth, str) and self.bandwidth not in ['scott', 'silverman']:
            raise ValueError(f"Invalid bandwidth method '{self.bandwidth}'. Supported methods: 'scott', 'silverman'")

    def _get_kernel_function(self):
        """Return the kernel function based on the specified kernel type."""
        if self.kernel == 'gaussian':
            return lambda u: np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        elif self.kernel == 'tophat':
            return lambda u: 0.5 * (np.abs(u) <= 1).astype(float)
        elif self.kernel == 'epanechnikov':
            return lambda u: 0.75 * (1 - u ** 2) * (np.abs(u) <= 1).astype(float)
        elif self.kernel == 'exponential':
            return lambda u: 0.5 * np.exp(-np.abs(u))
        elif self.kernel == 'linear':
            return lambda u: (1 - np.abs(u)) * (np.abs(u) <= 1).astype(float)
        elif self.kernel == 'cosine':
            return lambda u: np.pi / 4 * np.cos(np.pi * u / 2) * (np.abs(u) <= 1).astype(float)
        else:
            raise ValueError(f'Unsupported kernel: {self.kernel}')

    def _compute_bandwidth(self, data: np.ndarray) -> float:
        """Compute bandwidth using specified method."""
        n = len(data)
        d = 1
        if self.bandwidth == 'scott':
            return n ** (-1 / (d + 4))
        elif self.bandwidth == 'silverman':
            return (n * (d + 2) / 4) ** (-1 / (d + 4))
        else:
            return float(self.bandwidth)

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate kernel density estimate plot data from numerical input.
        
        Takes numerical data and computes kernel density estimates for visualization.
        Returns plot-ready data including evaluation points and density values.
        
        Args:
            data: Input numerical data as DataBatch or numpy array.
                  For DataBatch, uses the 'data' attribute.
                  For arrays, expects 1D or 2D with samples in rows.
            **kwargs: Additional processing parameters (reserved for extensions).
            
        Returns:
            dict: Dictionary containing:
                - 'x': Evaluation points for the density curve
                - 'density': Estimated density values at evaluation points
                - 'bandwidth': Bandwidth used for estimation
                - 'kernel': Kernel function used
                - 'fill_area': Whether area filling is enabled
                - 'colors': Color specifications for plotting
                
        Raises:
            ValueError: If data is empty or contains non-numerical values.
            TypeError: If data type is not supported.
        """
        if isinstance(data, DataBatch):
            raw_data = data.data
        elif isinstance(data, np.ndarray):
            raw_data = data
        else:
            raise TypeError('Data must be either a DataBatch or numpy array')
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.array(raw_data)
        if raw_data.size == 0:
            raise ValueError('Input data is empty')
        if raw_data.ndim > 1:
            raw_data = raw_data.flatten()
        if not np.issubdtype(raw_data.dtype, np.number):
            raise ValueError('Data contains non-numerical values')
        clean_data = raw_data[~np.isnan(raw_data)]
        if len(clean_data) == 0:
            raise ValueError('All data values are NaN')
        if isinstance(self.bandwidth, str):
            h = self._compute_bandwidth(clean_data)
        else:
            h = float(self.bandwidth)
        (data_min, data_max) = (np.min(clean_data), np.max(clean_data))
        range_ext = (data_max - data_min) * 0.1
        (x_min, x_max) = (data_min - range_ext, data_max + range_ext)
        x = np.linspace(x_min, x_max, self.grid_points)
        kernel_func = self._get_kernel_function()
        density = np.zeros_like(x)
        n = len(clean_data)
        for data_point in clean_data:
            density += kernel_func((x - data_point) / h)
        density /= n * h
        return {'x': x, 'density': density, 'bandwidth': h, 'kernel': self.kernel, 'fill_area': self.fill_area, 'colors': self.colors}