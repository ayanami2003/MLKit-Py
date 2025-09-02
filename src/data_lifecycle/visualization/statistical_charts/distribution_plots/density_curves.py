from typing import Optional, Union, List
import numpy as np
from scipy.stats import gaussian_kde
from general.structures.data_batch import DataBatch
from general.base_classes.pipeline_base import BasePipelineComponent

class DensityCurvePlotter(BasePipelineComponent):

    def __init__(self, name: Optional[str]=None, bandwidth: Union[float, str]='scott', kernel: str='gaussian', grid_points: int=1000, fill_area: bool=True, colors: Optional[List[str]]=None):
        """
        Initialize the DensityCurvePlotter.

        Args:
            name (Optional[str]): Name for the component instance.
            bandwidth (Union[float, str]): Method for bandwidth selection or fixed bandwidth value.
            kernel (str): Type of kernel to use for density estimation.
            grid_points (int): Number of points to evaluate the density function.
            fill_area (bool): Whether to fill the area under the curve.
            colors (Optional[List[str]]): List of colors for multiple distributions.
        """
        super().__init__(name)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.grid_points = grid_points
        self.fill_area = fill_area
        self.colors = colors

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate density curve visualization data from input data.

        This method computes the density estimation and prepares visualization-ready
        outputs that can be consumed by plotting libraries. It handles both single
        and multiple feature inputs.

        Args:
            data (Union[DataBatch, np.ndarray]): Input data to visualize. If DataBatch,
                                                 uses the data attribute.
            **kwargs: Additional parameters for processing (e.g., feature_names, labels).

        Returns:
            dict: Dictionary containing:
                - 'x_grid': Evaluation points for the density function
                - 'densities': Density values at each grid point (per feature)
                - 'feature_names': Names of features processed
                - 'config': Configuration used for generation
        """
        if isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
        else:
            X = np.array(data)
            feature_names = kwargs.get('feature_names', None)
        if X.size == 0:
            if feature_names is None or len(feature_names) == 0:
                feature_names = ['feature_0']
            return {'x_grid': np.array([]), 'densities': np.empty((len(feature_names), 0)), 'feature_names': feature_names, 'config': {'bandwidth': self.bandwidth, 'kernel': self.kernel, 'grid_points': self.grid_points, 'fill_area': self.fill_area, 'colors': self.colors}}
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(f'Number of feature names ({len(feature_names)}) must match number of features ({n_features})')
        x_min = np.min(X) if X.size > 0 else 0
        x_max = np.max(X) if X.size > 0 else 1
        x_range = x_max - x_min
        if x_range > 0:
            x_min -= 0.1 * x_range
            x_max += 0.1 * x_range
        else:
            x_min -= 0.1
            x_max += 0.1
        x_grid = np.linspace(x_min, x_max, self.grid_points)
        densities = []
        for i in range(n_features):
            feature_data = X[:, i]
            valid_data = feature_data[np.isfinite(feature_data)]
            if len(valid_data) == 0:
                density = np.zeros_like(x_grid)
            elif np.all(valid_data == valid_data[0]):
                mean_val = valid_data[0]
                bandwidth = (x_max - x_min) / 1000 if x_max != x_min else 0.01
                density = np.exp(-0.5 * ((x_grid - mean_val) / bandwidth) ** 2)
                density = density / (np.sum(density) * (x_grid[1] - x_grid[0]))
            else:
                try:
                    kde = gaussian_kde(valid_data, bw_method=self.bandwidth)
                    density = kde(x_grid)
                    density = np.maximum(density, 0)
                except Exception:
                    density = np.zeros_like(x_grid)
            densities.append(density)
        densities = np.array(densities)
        result = {'x_grid': x_grid, 'densities': densities, 'feature_names': feature_names, 'config': {'bandwidth': self.bandwidth, 'kernel': self.kernel, 'grid_points': self.grid_points, 'fill_area': self.fill_area, 'colors': self.colors}}
        return result