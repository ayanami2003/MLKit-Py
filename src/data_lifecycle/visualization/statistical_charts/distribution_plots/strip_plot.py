from typing import Optional, List, Union, Dict, Any
import numpy as np
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.data_batch import DataBatch

class StripPlotGenerator(BasePipelineComponent):

    def __init__(self, name: Optional[str]=None, jitter: Union[bool, float]=True, palette: Optional[List[str]]=None, size: float=5.0, alpha: float=0.7):
        super().__init__(name)
        self.jitter = jitter
        self.palette = palette
        self.size = size
        self.alpha = alpha

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate a strip plot visualization from the input data.
        
        Args:
            data (Union[DataBatch, np.ndarray]): Input data to visualize. If DataBatch, 
                                                expects numerical features for plotting.
            **kwargs: Additional parameters for plot customization (e.g., group labels).
            
        Returns:
            dict: A dictionary containing plot artifacts and metadata.
                  Example keys might include 'figure', 'axes', or 'plot_data'.
                  
        Raises:
            ValueError: If data is empty or improperly formatted.
        """
        import matplotlib.pyplot as plt
        if isinstance(data, DataBatch):
            plot_data = data.data
            if data.labels is not None:
                groups = data.labels
            else:
                groups = kwargs.get('groups', None)
        else:
            plot_data = data
            groups = kwargs.get('groups', None)
        if not isinstance(plot_data, np.ndarray):
            plot_data = np.array(plot_data)
        if plot_data.size == 0:
            raise ValueError('Input data is empty')
        if plot_data.ndim == 1:
            y_data = plot_data
            x_data = np.zeros_like(plot_data)
        elif plot_data.ndim == 2:
            y_data = plot_data.flatten()
            x_data = np.repeat(np.arange(plot_data.shape[1]), plot_data.shape[0])
        else:
            raise ValueError('Data must be 1D or 2D array')
        if groups is not None:
            if not isinstance(groups, (list, np.ndarray)):
                groups = np.array(groups)
            if len(groups) != len(y_data):
                if len(groups) == plot_data.shape[1] and plot_data.ndim == 2:
                    groups = np.repeat(groups, plot_data.shape[0])
                else:
                    groups = np.tile(groups, len(y_data) // len(groups) + 1)[:len(y_data)]
        else:
            groups = np.zeros(len(y_data))
        if self.jitter:
            if isinstance(self.jitter, bool):
                jitter_amount = 0.1
            else:
                jitter_amount = float(self.jitter)
            x_data = x_data + np.random.uniform(-jitter_amount, jitter_amount, size=len(x_data))
        (fig, ax) = plt.subplots(figsize=(10, 6))
        unique_groups = np.unique(groups)
        num_groups = len(unique_groups)
        if self.palette and len(self.palette) >= num_groups:
            colors = self.palette[:num_groups]
        elif num_groups <= 10:
            cmap = plt.cm.get_cmap('tab10', num_groups)
            colors = [cmap(i) for i in range(num_groups)]
        else:
            colors = [plt.cm.tab20(i / num_groups) for i in range(num_groups)]
        collections = []
        for (i, group) in enumerate(unique_groups):
            mask = groups == group
            collection = ax.scatter(x_data[mask], y_data[mask], s=self.size, alpha=self.alpha, color=colors[i], label=f'Group {group}' if num_groups > 1 else None)
            collections.append(collection)
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        if num_groups > 1:
            ax.legend()
        ax.grid(True, alpha=0.3)
        return {'figure': fig, 'axes': ax, 'plot_data': {'x': x_data, 'y': y_data, 'groups': groups}, 'collections': collections}