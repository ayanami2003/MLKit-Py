from typing import Optional, Union, List, Dict, Any
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.data_batch import DataBatch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kstest

class ProbabilityDistributionPlotter(BasePipelineComponent):

    def __init__(self, name: Optional[str]=None, distribution_type: str='normal', figsize: tuple=(10, 6), color_scheme: str='blue', overlay_theoretical: bool=True, bins: Union[int, str]='auto', alpha: float=0.7):
        """
        Initialize the ProbabilityDistributionPlotter.

        Args:
            name (Optional[str]): Name identifier for the component.
            distribution_type (str): Theoretical distribution type to visualize ('normal', 'exponential', 'uniform', etc.).
            figsize (tuple): Figure size as (width, height) in inches.
            color_scheme (str): Color scheme for the visualization.
            overlay_theoretical (bool): Flag to overlay theoretical distribution curve.
            bins (Union[int, str]): Number of bins or binning strategy ('auto', 'fd', 'scott', etc.).
            alpha (float): Transparency level for plot elements (0-1).

        Raises:
            ValueError: If distribution_type is not supported.
        """
        super().__init__(name)
        self.distribution_type = distribution_type
        self.figsize = figsize
        self.color_scheme = color_scheme
        self.overlay_theoretical = overlay_theoretical
        self.bins = bins
        self.alpha = alpha

    def process(self, data: Union[DataBatch, np.ndarray], **kwargs) -> dict:
        """
        Generate probability distribution plots from input data.

        Creates visualization showing empirical data distribution alongside theoretical distribution
        for comparison. Returns plot objects and statistical metrics.

        Args:
            data (Union[DataBatch, np.ndarray]): Input data to visualize. If DataBatch, uses the data attribute.
            **kwargs: Additional processing parameters (not currently used).

        Returns:
            dict: Dictionary containing:
                - 'figure': Matplotlib figure object
                - 'axes': Matplotlib axes object
                - 'statistics': Dictionary with goodness-of-fit metrics
                - 'theoretical_params': Fitted parameters for theoretical distribution

        Raises:
            ValueError: If data is empty or contains non-numeric values.
            TypeError: If data is not in expected format.
        """
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
        (figure, axes) = plt.subplots(figsize=self.figsize)
        (n, bins, patches) = axes.hist(finite_values, bins=self.bins, density=True, alpha=self.alpha, color=self.color_scheme, edgecolor='black', linewidth=0.5, label='Empirical Data')
        theoretical_params = {}
        statistics = {}
        try:
            if self.distribution_type == 'normal':
                params = stats.norm.fit(finite_values)
                dist_func = stats.norm.pdf
                param_names = ['loc', 'scale']
            elif self.distribution_type == 'exponential':
                params = stats.expon.fit(finite_values)
                dist_func = stats.expon.pdf
                param_names = ['loc', 'scale']
            elif self.distribution_type == 'uniform':
                params = stats.uniform.fit(finite_values)
                dist_func = stats.uniform.pdf
                param_names = ['loc', 'scale']
            elif self.distribution_type == 'gamma':
                params = stats.gamma.fit(finite_values)
                dist_func = stats.gamma.pdf
                param_names = ['a', 'loc', 'scale']
            elif self.distribution_type == 'beta':
                min_val = np.min(finite_values)
                max_val = np.max(finite_values)
                if max_val > min_val:
                    scaled_values = (finite_values - min_val) / (max_val - min_val)
                    params = stats.beta.fit(scaled_values, floc=0, fscale=1)
                    dist_func = stats.beta.pdf
                    param_names = ['a', 'b', 'loc', 'scale']
                else:
                    raise ValueError('Cannot fit beta distribution: all values are identical')
            else:
                raise ValueError(f'Unsupported distribution type: {self.distribution_type}')
            theoretical_params = dict(zip(param_names, params))
            if self.distribution_type == 'normal':
                (ks_statistic, p_value) = kstest(finite_values, lambda x: stats.norm.cdf(x, *params))
            elif self.distribution_type == 'exponential':
                (ks_statistic, p_value) = kstest(finite_values, lambda x: stats.expon.cdf(x, *params))
            elif self.distribution_type == 'uniform':
                (ks_statistic, p_value) = kstest(finite_values, lambda x: stats.uniform.cdf(x, *params))
            elif self.distribution_type == 'gamma':
                (ks_statistic, p_value) = kstest(finite_values, lambda x: stats.gamma.cdf(x, *params))
            elif self.distribution_type == 'beta':
                (ks_statistic, p_value) = kstest(scaled_values, lambda x: stats.beta.cdf(x, *params))
            statistics = {'ks_statistic': ks_statistic, 'p_value': p_value, 'significant': p_value < 0.05}
            if self.overlay_theoretical:
                (x_min, x_max) = axes.get_xlim()
                x = np.linspace(x_min, x_max, 1000)
                if self.distribution_type == 'beta':
                    scaled_x = (x - min_val) / (max_val - min_val)
                    scaled_x = np.clip(scaled_x, 0, 1)
                    y = dist_func(scaled_x, *params)
                else:
                    y = dist_func(x, *params)
                axes.plot(x, y, 'r-', linewidth=2, label=f'Theoretical {self.distribution_type.capitalize()}')
        except Exception as e:
            print(f'Warning: Could not fit theoretical distribution: {e}')
        axes.set_xlabel('Value')
        axes.set_ylabel('Density')
        axes.set_title(f'Probability Distribution ({self.distribution_type.capitalize()})')
        axes.legend()
        axes.grid(True, alpha=0.3)
        return {'figure': figure, 'axes': axes, 'statistics': statistics, 'theoretical_params': theoretical_params}