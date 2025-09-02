from typing import Optional, Union, List
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.data_batch import DataBatch
import numpy as np
import matplotlib.pyplot as plt

class SwarmPlotGenerator(BasePipelineComponent):

    def __init__(self, name: Optional[str]=None, orientation: str='vertical', size: Union[int, float]=20, color_palette: Optional[List[str]]=None, alpha: float=0.7, jitter: bool=True):
        """
        Initialize the SwarmPlotGenerator.
        
        Args:
            name (Optional[str]): Name identifier for the component.
            orientation (str): Orientation of the plot ('vertical' or 'horizontal'). Defaults to 'vertical'.
            size (Union[int, float]): Size of the markers in the plot. Defaults to 20.
            color_palette (Optional[List[str]]): Colors to use for different categories. If None, defaults will be used.
            alpha (float): Transparency level of the markers (0.0 to 1.0). Defaults to 0.7.
            jitter (bool): Whether to apply jitter to the points for better visibility. Defaults to True.
        """
        super().__init__(name)
        self.orientation = orientation
        self.size = size
        self.color_palette = color_palette
        self.alpha = alpha
        self.jitter = jitter

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate a swarm plot visualization from the input data.
        
        This method takes in data and produces a structured representation of a swarm plot
        that can be consumed by visualization rendering engines.
        
        Args:
            data (Union[DataBatch, np.ndarray]): Input data to visualize. Expected to contain
                categorical data for grouping and numerical data for positioning.
            **kwargs: Additional parameters for plot customization.
            
        Returns:
            dict: A dictionary containing the processed visualization data, including:
                - 'points': Coordinates of the plotted points
                - 'categories': Category labels for the groups
                - 'colors': Color values for each point
                - 'metadata': Additional plot metadata
                
        Raises:
            ValueError: If the input data format is not supported or invalid.
        """
        if isinstance(data, DataBatch):
            raw_data = data.data
        elif isinstance(data, np.ndarray):
            raw_data = data
        else:
            raise ValueError('Input data must be a DataBatch or numpy array')
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.array(raw_data)
        if raw_data.ndim != 2 or raw_data.shape[1] < 2:
            raise ValueError('Data must be a 2D array with at least two columns (categorical and numerical)')
        categorical_data = raw_data[:, 0]
        numerical_data = raw_data[:, 1].astype(float)
        categories = np.unique(categorical_data)
        n_categories = len(categories)
        if self.color_palette is None:
            default_colors = plt.cm.tab10.colors if n_categories <= 10 else plt.cm.Set3.colors
            color_palette = ['#%02x%02x%02x' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in default_colors]
        else:
            color_palette = self.color_palette
        category_to_color = {cat: color_palette[i % len(color_palette)] for (i, cat) in enumerate(categories)}
        x_coords = []
        y_coords = []
        point_colors = []
        category_labels = []
        for (i, category) in enumerate(categories):
            category_mask = categorical_data == category
            category_values = numerical_data[category_mask]
            sorted_indices = np.argsort(category_values)
            sorted_values = category_values[sorted_indices]
            if self.orientation == 'vertical':
                base_coord = i
                category_positions = self._calculate_swarm_positions(sorted_values, base_coord, self.jitter)
                x_coords.extend(category_positions)
                y_coords.extend(sorted_values)
            else:
                base_coord = i
                category_positions = self._calculate_swarm_positions(sorted_values, base_coord, self.jitter)
                x_coords.extend(sorted_values)
                y_coords.extend(category_positions)
            n_points = len(sorted_values)
            point_colors.extend([category_to_color[category]] * n_points)
            category_labels.extend([category] * n_points)
        points = np.column_stack([x_coords, y_coords])
        metadata = {'orientation': self.orientation, 'size': self.size, 'alpha': self.alpha, 'jitter_applied': self.jitter, 'n_categories': n_categories, 'total_points': len(points)}
        return {'points': points, 'categories': categories.tolist(), 'colors': point_colors, 'metadata': metadata}

    def _calculate_swarm_positions(self, values: np.ndarray, base_coord: float, jitter: bool) -> np.ndarray:
        """
        Calculate positions along the categorical axis to avoid overlap.
        
        Args:
            values: Sorted numerical values for a category
            base_coord: Base coordinate for the category axis
            jitter: Whether to apply random jitter
            
        Returns:
            Array of positions along the categorical axis
        """
        n_points = len(values)
        if n_points == 0:
            return np.array([])
        positions = np.full(n_points, base_coord, dtype=float)
        if jitter and n_points > 1:
            jitter_scale = min(0.1, 0.5 / np.sqrt(n_points))
            noise = np.random.normal(0, jitter_scale, n_points)
            positions += noise
        if n_points > 1:
            spread_factor = min(0.2, 1.0 / np.sqrt(n_points))
            positions += np.linspace(-spread_factor / 2, spread_factor / 2, n_points)
        return positions