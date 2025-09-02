from typing import Optional, List, Union
import numpy as np
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.data_batch import DataBatch


class RaincloudPlotter(BasePipelineComponent):

    def __init__(self, name: Optional[str]=None, bandwidth: Union[float, str]='scott', jitter: float=0.2, point_size: int=20, colors: Optional[List[str]]=None, show_violin: bool=True, show_boxplot: bool=True):
        """
        Initialize the RaincloudPlotter component.

        Parameters
        ----------
        name : Optional[str]
            Name identifier for the component
        bandwidth : Union[float, str]
            Bandwidth for kernel density estimation ('scott', 'silverman', or numeric value)
        jitter : float
            Amount of jitter to apply to raw data points (0 to 1)
        point_size : int
            Size of individual data points in the scatter plot
        colors : Optional[List[str]]
            List of colors for different groups in the plot
        show_violin : bool
            Whether to include violin plot component
        show_boxplot : bool
            Whether to include boxplot component
        """
        super().__init__(name)
        self.bandwidth = bandwidth
        self.jitter = jitter
        self.point_size = point_size
        self.colors = colors
        self.show_violin = show_violin
        self.show_boxplot = show_boxplot

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate a raincloud plot specification from input data.

        Processes numerical data to create a combined visualization that includes:
        - Raw data points (jittered)
        - Violin plot showing density
        - Box plot showing summary statistics

        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data to visualize. If DataBatch, uses the data attribute.
            Should be a 1D or 2D array where each column represents a group.
        **kwargs : dict
            Additional parameters for plot customization

        Returns
        -------
        dict
            Plot specification containing:
            - 'data_points': coordinates for raw data points
            - 'violin_data': KDE curves for violin plots
            - 'boxplot_data': quartiles and outliers for box plots
            - 'layout': plot configuration details
            - 'annotations': statistical summaries

        Raises
        ------
        ValueError
            If data dimensions are incompatible or contain non-numeric values
        """
        if isinstance(data, DataBatch):
            raw_data = data.data
        else:
            raw_data = data
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.asarray(raw_data)
        if not np.issubdtype(raw_data.dtype, np.number):
            raise ValueError('Data must contain only numeric values')
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, 1)
        elif raw_data.ndim != 2:
            raise ValueError('Data must be 1D or 2D')
        n_groups = raw_data.shape[1]
        data_points = []
        violin_data = [] if self.show_violin else None
        boxplot_data = [] if self.show_boxplot else None
        annotations = []
        for i in range(n_groups):
            group_data = raw_data[:, i]
            valid_data = group_data[~np.isnan(group_data)]
            if len(valid_data) == 0:
                continue
            q1 = np.percentile(valid_data, 25)
            median = np.median(valid_data)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outliers = valid_data[(valid_data < lower_fence) | (valid_data > upper_fence)]
            annotations.append({'group': i, 'count': len(valid_data), 'mean': np.mean(valid_data), 'median': median, 'std': np.std(valid_data), 'min': np.min(valid_data), 'max': np.max(valid_data), 'q1': q1, 'q3': q3, 'outliers': outliers.tolist()})
            jitter_offsets = np.random.uniform(-0.2, 0.2, size=len(valid_data)) * self.jitter
            x_coords = np.full(len(valid_data), i) + jitter_offsets
            data_points.extend(list(zip(x_coords, valid_data)))
            if self.show_violin:
                if self.bandwidth == 'scott':
                    bw = len(valid_data) ** (-1 / 5) * np.std(valid_data)
                elif self.bandwidth == 'silverman':
                    bw = (len(valid_data) * 3 / 4) ** (-1 / 5) * np.std(valid_data)
                else:
                    bw = float(self.bandwidth)
                (y_min, y_max) = (np.min(valid_data), np.max(valid_data))
                y_range = y_max - y_min
                y_vals = np.linspace(y_min - 0.1 * y_range, y_max + 0.1 * y_range, 200)
                density = np.zeros_like(y_vals)
                if bw > 0 and len(valid_data) > 0:
                    for point in valid_data:
                        density += np.exp(-0.5 * ((y_vals - point) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
                    density /= len(valid_data)
                max_density = np.max(density) if np.max(density) > 0 else 1
                left_side = i - density / max_density * 0.4
                right_side = i + density / max_density * 0.4
                violin_data.append({'group': i, 'y_values': y_vals.tolist(), 'left_side': left_side.tolist(), 'right_side': right_side.tolist()})
            if self.show_boxplot:
                boxplot_data.append({'group': i, 'min': np.min(valid_data), 'q1': q1, 'median': median, 'q3': q3, 'max': np.max(valid_data), 'outliers': outliers.tolist(), 'lower_fence': lower_fence, 'upper_fence': upper_fence})
        layout = {'point_size': self.point_size, 'colors': self.colors, 'n_groups': n_groups, 'show_violin': self.show_violin, 'show_boxplot': self.show_boxplot}
        return {'data_points': data_points, 'violin_data': violin_data, 'boxplot_data': boxplot_data, 'layout': layout, 'annotations': annotations}