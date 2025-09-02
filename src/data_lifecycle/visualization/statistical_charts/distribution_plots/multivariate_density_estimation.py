from typing import Optional, Union, Dict, Any
from general.structures.data_batch import DataBatch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

def multivariate_density_estimation(data: Union[DataBatch, np.ndarray], method: str='kde', dimensions: Optional[list]=None, bandwidth: Optional[float]=None, figsize: tuple=(10, 8), title: Optional[str]=None, ax: Optional[plt.Axes]=None, **method_params: Dict[str, Any]) -> plt.Figure:
    """
    Estimate and visualize multivariate probability density functions.

    This function computes and displays the joint probability density of multiple variables
    using various estimation techniques. It supports both parametric and non-parametric approaches
    to model complex multivariate distributions.

    Args:
        data (Union[DataBatch, np.ndarray]): Input data containing multiple variables for density estimation.
                                            If DataBatch, uses the data attribute.
        method (str): Density estimation technique to use. Options include:
                     - 'kde': Kernel Density Estimation (default)
                     - 'histogram': Multivariate histogram
                     - 'gaussian_mixture': Gaussian Mixture Model
        dimensions (Optional[list]): List of column indices or feature names to include in the analysis.
                                   If None, uses all available dimensions.
        bandwidth (Optional[float]): Bandwidth parameter for KDE methods. If None, uses automatic selection.
        figsize (tuple): Figure size for the plot (width, height).
        title (Optional[str]): Custom title for the plot. If None, uses a default title.
        ax (Optional[plt.Axes]): Matplotlib axes object to plot on. If None, creates new figure.
        **method_params (Dict[str, Any]): Additional parameters specific to the chosen method.

    Returns:
        plt.Figure: Matplotlib figure object containing the density estimation visualization.

    Raises:
        ValueError: If specified dimensions are invalid or unsupported method is selected.
        TypeError: If data type is not supported for density estimation.
    """
    if isinstance(data, np.ndarray):
        raw_data = data
        is_databatch = False
    elif hasattr(data, 'data'):
        raw_data = data.data
        is_databatch = True
    else:
        raise TypeError('Data must be either a DataBatch or numpy array.')
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.array(raw_data)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(-1, 1)
    if raw_data.ndim != 2:
        raise ValueError('Data must be 2-dimensional.')
    (n_samples, n_features) = raw_data.shape
    if dimensions is not None and len(dimensions) > 0:
        if isinstance(dimensions[0], str):
            if is_databatch and hasattr(data, 'feature_names') and (data.feature_names is not None):
                try:
                    dim_indices = [data.feature_names.index(name) for name in dimensions]
                except ValueError as e:
                    raise ValueError(f'Invalid dimension name: {e}')
            else:
                raise ValueError('String dimension names require DataBatch with feature_names attribute.')
        else:
            dim_indices = dimensions
        if any((idx < 0 or idx >= n_features for idx in dim_indices)):
            raise ValueError('Dimension indices out of bounds.')
        selected_data = raw_data[:, dim_indices]
    else:
        selected_data = raw_data
        dim_indices = list(range(n_features))
    if ax is None:
        (fig, ax) = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    if method == 'kde':
        if selected_data.shape[1] > 1:
            n_dims = selected_data.shape[1]
            if n_dims > 5:
                raise ValueError('KDE visualization limited to 5 dimensions for clarity.')
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    x = selected_data[:, i]
                    y = selected_data[:, j]
                    if bandwidth is None:
                        bw_method = (4 / (3 * n_samples)) ** (1 / 5)
                    else:
                        bw_method = bandwidth
                    (xx, yy) = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    values = np.vstack([x, y])
                    kernel = gaussian_kde(values, bw_method=bw_method)
                    f = np.reshape(kernel(positions).T, xx.shape)
                    ax.contour(xx, yy, f, cmap='Blues')
                    ax.scatter(x, y, s=1, alpha=0.5)
                    ax.set_xlabel(f'Dimension {dim_indices[i]}')
                    ax.set_ylabel(f'Dimension {dim_indices[j]}')
        else:
            x = selected_data.flatten()
            if bandwidth is None:
                bandwidth = np.std(x) * (4 / (3 * n_samples)) ** (1 / 5)
            kde = gaussian_kde(x, bw_method=bandwidth)
            x_range = np.linspace(x.min(), x.max(), 1000)
            density = kde(x_range)
            ax.plot(x_range, density)
            ax.fill_between(x_range, density, alpha=0.3)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
    elif method == 'histogram':
        if selected_data.shape[1] > 2:
            raise ValueError('Histogram visualization limited to 1D or 2D.')
        if selected_data.shape[1] == 1:
            ax.hist(selected_data.flatten(), bins=method_params.get('bins', 30), density=True)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        else:
            x = selected_data[:, 0]
            y = selected_data[:, 1]
            bins = method_params.get('bins', 30)
            ax.hist2d(x, y, bins=bins, cmap='Blues')
            ax.set_xlabel(f'Dimension {dim_indices[0]}')
            ax.set_ylabel(f'Dimension {dim_indices[1]}')
            plt.colorbar(ax.collections[0], ax=ax)
    elif method == 'gaussian_mixture':
        n_components = method_params.get('n_components', min(5, n_samples // 10))
        covariance_type = method_params.get('covariance_type', 'full')
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
        gmm.fit(selected_data)
        if selected_data.shape[1] > 2:
            raise ValueError('GMM visualization limited to 1D or 2D.')
        if selected_data.shape[1] == 1:
            x = selected_data.flatten()
            x_range = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
            log_prob = gmm.score_samples(x_range)
            prob = np.exp(log_prob)
            ax.plot(x_range, prob)
            ax.fill_between(x_range.flatten(), prob, alpha=0.3)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        else:
            x = selected_data[:, 0]
            y = selected_data[:, 1]
            (xx, yy) = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            positions = np.dstack((xx, yy))
            log_prob = gmm.score_samples(positions.reshape(-1, 2))
            prob = np.exp(log_prob).reshape(xx.shape)
            ax.contour(xx, yy, prob, cmap='Blues')
            ax.scatter(x, y, s=1, alpha=0.5)
            ax.set_xlabel(f'Dimension {dim_indices[0]}')
            ax.set_ylabel(f'Dimension {dim_indices[1]}')
    else:
        raise ValueError(f'Unsupported method: {method}')
    if title is None:
        title = f'{method.upper()} Density Estimation'
    ax.set_title(title)
    return fig