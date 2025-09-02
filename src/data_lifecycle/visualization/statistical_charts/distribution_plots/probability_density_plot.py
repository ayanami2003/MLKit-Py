import numpy as np
from typing import Optional, Union, List
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.data_batch import DataBatch

class ProbabilityDensityPlotter(BasePipelineComponent):
    """
    A pipeline component for generating probability density plots from numerical data.

    This class creates smooth probability density estimates for numerical distributions
    using kernel density estimation. It supports customization of kernel types,
    bandwidth selection, and visualization styling. The generated plots can be used
    for exploratory data analysis and distribution comparison.

    Attributes:
        name (str): Component name identifier.
        bandwidth (Union[float, str]): Bandwidth for KDE ('scott', 'silverman', or numeric).
        kernel (str): Kernel function type ('gaussian', 'tophat', 'epanechnikov', etc.).
        grid_points (int): Number of points in the evaluation grid.
        fill_area (bool): Whether to fill the area under the density curve.
        colors (Optional[List[str]]): Colors for multiple distribution plotting.
    """

    def __init__(self, name: Optional[str]=None, bandwidth: Union[float, str]='scott', kernel: str='gaussian', grid_points: int=1000, fill_area: bool=True, colors: Optional[List[str]]=None):
        """
        Initialize the ProbabilityDensityPlotter.

        Args:
            name (Optional[str]): Name identifier for the component.
            bandwidth (Union[float, str]): Bandwidth for KDE. Can be 'scott', 'silverman', or a float value.
            kernel (str): Type of kernel function to use for density estimation.
            grid_points (int): Number of points to evaluate the density function.
            fill_area (bool): Whether to fill the area under the density curve.
            colors (Optional[List[str]]): List of colors for plotting multiple distributions.
        """
        super().__init__(name)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.grid_points = grid_points
        self.fill_area = fill_area
        self.colors = colors

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate a probability density plot from the input data.

        This method takes numerical data and produces a probability density plot
        using kernel density estimation. It handles both single and multiple
        distributions in the input data.

        Args:
            data (Union[DataBatch, np.ndarray]): Input numerical data to plot.
                Can be a DataBatch containing numerical features or a numpy array.
            **kwargs: Additional plotting parameters and matplotlib style options.

        Returns:
            dict: Dictionary containing plot objects and metadata for rendering.
                Keys include 'figure', 'axes', and 'plot_data'.

        Raises:
            ValueError: If the input data contains non-numerical values or is empty.
            TypeError: If the data type is not supported.
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
            if self.bandwidth == 'scott':
                h = len(clean_data) ** (-1 / 5)
            elif self.bandwidth == 'silverman':
                h = (len(clean_data) * 3 / 4) ** (-1 / 5)
            else:
                raise ValueError(f"Invalid bandwidth method '{self.bandwidth}'. Supported methods: 'scott', 'silverman'")
        else:
            h = float(self.bandwidth)

        def gaussian_kernel(u):
            return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)

        def tophat_kernel(u):
            return 0.5 * (np.abs(u) <= 1).astype(float)

        def epanechnikov_kernel(u):
            return 0.75 * (1 - u ** 2) * (np.abs(u) <= 1).astype(float)

        def exponential_kernel(u):
            return 0.5 * np.exp(-np.abs(u))

        def linear_kernel(u):
            return (1 - np.abs(u)) * (np.abs(u) <= 1).astype(float)

        def cosine_kernel(u):
            return np.pi / 4 * np.cos(np.pi * u / 2) * (np.abs(u) <= 1).astype(float)
        kernel_functions = {'gaussian': gaussian_kernel, 'tophat': tophat_kernel, 'epanechnikov': epanechnikov_kernel, 'exponential': exponential_kernel, 'linear': linear_kernel, 'cosine': cosine_kernel}
        if self.kernel not in kernel_functions:
            raise ValueError(f"Invalid kernel '{self.kernel}'. Supported kernels: {list(kernel_functions.keys())}")
        kernel_func = kernel_functions[self.kernel]
        (data_min, data_max) = (np.min(clean_data), np.max(clean_data))
        range_ext = (data_max - data_min) * 0.1
        (x_min, x_max) = (data_min - range_ext, data_max + range_ext)
        x = np.linspace(x_min, x_max, self.grid_points)
        density = np.zeros_like(x)
        n = len(clean_data)
        for data_point in clean_data:
            density += kernel_func((x - data_point) / h)
        density /= n * h
        plot_data = {'x': x, 'density': density, 'bandwidth': h, 'kernel': self.kernel, 'fill_area': self.fill_area, 'colors': self.colors}
        return {'plot_data': plot_data, 'figure': None, 'axes': None}