import numpy as np
from typing import Optional, List, Union
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.data_batch import DataBatch
from scipy.stats import gaussian_kde

class ViolinPlotGenerator(BasePipelineComponent):

    def __init__(self, name: Optional[str]=None, showmeans: bool=False, showmedians: bool=True, showextrema: bool=True, bandwidth: Union[float, str]='scott', widths: Union[float, List[float]]=0.8, colors: Optional[List[str]]=None, grid_points: int=100):
        """
        Initialize the ViolinPlotGenerator.
        
        Args:
            name: Optional name for the component.
            showmeans: Whether to show mean markers on violins.
            showmedians: Whether to show median markers on violins.
            showextrema: Whether to show extrema points on violins.
            bandwidth: Method for calculating bandwidth ('scott', 'silverman') or a fixed float value.
            widths: Width(s) of violins relative to space allocated.
            colors: List of colors for violin groups.
            grid_points: Number of points in the grid for density estimation.
        """
        super().__init__(name=name)
        self.showmeans = showmeans
        self.showmedians = showmedians
        self.showextrema = showextrema
        self.bandwidth = bandwidth
        self.widths = widths
        self.colors = colors
        self.grid_points = grid_points

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate a violin plot from the input data.
        
        Args:
            data: Input data as DataBatch or numpy array containing numerical values.
                  For grouped data, expects a DataBatch with categorical metadata.
            **kwargs: Additional parameters for plot customization.
            
        Returns:
            dict: Dictionary containing plot data and configuration for rendering.
                 Keys include 'violins' (density data), 'positions', 'labels', and plot metadata.
                 
        Raises:
            ValueError: If data format is incompatible or required fields are missing.
        """
        if not (isinstance(self.bandwidth, (float, int)) or self.bandwidth in ['scott', 'silverman']):
            raise ValueError("Bandwidth must be a float or one of 'scott', 'silverman'")
        if isinstance(data, DataBatch):
            X = np.asarray(data.data)
            X_flat = X.flatten() if X.size > 0 else np.array([])
            if X.size == 0:
                grouped_data = [np.array([])]
                labels = ['Group 1']
            elif 'categorical' in data.metadata and data.metadata['categorical'] is not None:
                groups = data.metadata['categorical']
                if len(groups) != len(X_flat):
                    raise ValueError('Length of categorical metadata must match data length')
                if len(groups) == 0:
                    grouped_data = [X_flat]
                    labels = ['Group 1']
                else:
                    unique_groups = []
                    seen = set()
                    for group in groups:
                        if group not in seen:
                            unique_groups.append(group)
                            seen.add(group)
                    labels = unique_groups
                    grouped_data = []
                    for group in unique_groups:
                        mask = np.array([g == group for g in groups])
                        grouped_data.append(X_flat[mask])
            else:
                grouped_data = [X_flat]
                labels = ['Group 1']
        elif isinstance(data, np.ndarray):
            if data.size == 0:
                grouped_data = [np.array([])]
                labels = ['Group 1']
            else:
                grouped_data = [data.flatten()]
                labels = ['Group 1']
        else:
            raise ValueError('Data must be either DataBatch or numpy.ndarray')
        n_groups = len(grouped_data)
        positions = list(range(1, n_groups + 1))
        if isinstance(self.widths, (int, float)):
            widths = [float(self.widths)] * n_groups
        else:
            if len(self.widths) != n_groups:
                raise ValueError('Length of widths must match number of groups')
            widths = [float(w) for w in self.widths]
        if self.colors is not None:
            if len(self.colors) != n_groups:
                raise ValueError('Length of colors must match number of groups')
            colors = self.colors
        else:
            colors = [None] * n_groups
        violins = []
        for (i, group_data) in enumerate(grouped_data):
            valid_data = group_data[np.isfinite(group_data)] if group_data.size > 0 else np.array([])
            if len(valid_data) == 0:
                violin = {'vals': np.array([]), 'x': np.array([]), 'density': np.array([]), 'width': widths[i], 'color': colors[i]}
            else:
                violin = {'vals': valid_data, 'width': widths[i], 'color': colors[i]}
                if self.showmeans:
                    violin['mean'] = np.mean(valid_data)
                if self.showmedians:
                    violin['median'] = np.median(valid_data)
                if self.showextrema:
                    violin['min'] = np.min(valid_data)
                    violin['max'] = np.max(valid_data)
                data_range = np.max(valid_data) - np.min(valid_data)
                if data_range > 0:
                    extension = 0.1 * data_range
                    x_min = np.min(valid_data) - extension
                    x_max = np.max(valid_data) + extension
                else:
                    x_min = np.min(valid_data) - 0.1
                    x_max = np.max(valid_data) + 0.1
                x_grid = np.linspace(x_min, x_max, self.grid_points)
                if len(valid_data) > 1 and data_range > 0:
                    try:
                        if self.bandwidth in ['scott', 'silverman']:
                            kde = gaussian_kde(valid_data, bw_method=self.bandwidth)
                        else:
                            kde = gaussian_kde(valid_data, bw_method=float(self.bandwidth))
                        density = kde(x_grid)
                        density = np.maximum(density, 0)
                    except Exception:
                        density = np.ones_like(x_grid) / (x_max - x_min)
                elif len(valid_data) == 1 or data_range == 0:
                    density = np.zeros_like(x_grid)
                    center_idx = len(x_grid) // 2
                    if center_idx < len(density):
                        density[center_idx] = 1.0
                else:
                    density = np.zeros_like(x_grid)
                max_density = np.max(density)
                if max_density > 0:
                    density = density / max_density
                violin['x'] = x_grid
                violin['density'] = density
            violins.append(violin)
        result = {'violins': violins, 'positions': positions, 'labels': labels, 'config': {'showmeans': self.showmeans, 'showmedians': self.showmedians, 'showextrema': self.showextrema, 'bandwidth': self.bandwidth, 'grid_points': self.grid_points}}
        return result