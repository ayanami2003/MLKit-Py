from typing import Union, Optional, Dict, Any, List
from general.structures.data_batch import DataBatch
import matplotlib.pyplot as plt
import numpy as np

def quantile_regression_plots(data: Union[DataBatch, np.ndarray], target_column: Union[str, int], feature_column: Union[str, int], quantiles: Optional[List[float]]=None, figsize: tuple=(10, 6), title: Optional[str]=None, ax: Optional[plt.Axes]=None, **plot_params) -> plt.Figure:
    """
    Generate quantile regression plots for a specified feature-target pair.
    
    This function visualizes how different quantiles of the target variable change with respect to a feature.
    It fits quantile regression models for specified quantiles and plots the resulting regression lines
    along with the scatter plot of the data points.
    
    Args:
        data (Union[DataBatch, np.ndarray]): Input data containing features and target values.
                                            If DataBatch, it should contain the relevant columns.
        target_column (Union[str, int]): Column identifier for the target variable.
                                        String for named columns, integer for positional indexing.
        feature_column (Union[str, int]): Column identifier for the feature variable to regress against.
                                         String for named columns, integer for positional indexing.
        quantiles (Optional[list]): List of quantiles to fit and plot (e.g., [0.1, 0.5, 0.9]).
                                   Defaults to [0.05, 0.25, 0.5, 0.75, 0.95] if not provided.
        figsize (tuple): Figure size for the plot. Defaults to (10, 6).
        title (Optional[str]): Title for the plot. If None, a default title will be generated.
        ax (Optional[plt.Axes]): Matplotlib axes object to plot on. If None, a new figure will be created.
        **plot_params (Dict[str, Any]): Additional parameters for customizing the plot appearance.
        
    Returns:
        plt.Figure: Matplotlib figure object containing the quantile regression plot.
        
    Raises:
        ValueError: If specified columns are not found in the data or if quantiles are invalid.
        TypeError: If data is not in a supported format.
    """
    if quantiles is None or len(quantiles) == 0:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    if not all((0 < q < 1 for q in quantiles)):
        raise ValueError('All quantiles must be between 0 and 1 (exclusive).')
    if isinstance(data, DataBatch):
        X = np.asarray(data.data)
        feature_names = data.feature_names
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if isinstance(target_column, str):
            if feature_names is None:
                raise ValueError('Feature names not available in DataBatch. Use integer indices for column identification.')
            if target_column not in feature_names:
                raise ValueError(f"Target column '{target_column}' not found in DataBatch")
            target_idx = feature_names.index(target_column)
            y = X[:, target_idx]
        elif isinstance(target_column, int):
            if target_column < 0 or target_column >= X.shape[1]:
                raise IndexError(f'Target column index {target_column} is out of bounds for data with {X.shape[1]} columns')
            y = X[:, target_column]
        else:
            raise TypeError('Target column must be either a string or integer')
        if isinstance(feature_column, str):
            if feature_names is None:
                raise ValueError('Feature names not available in DataBatch. Use integer indices for column identification.')
            if feature_column not in feature_names:
                raise ValueError(f"Feature column '{feature_column}' not found in DataBatch")
            feature_idx = feature_names.index(feature_column)
            x = X[:, feature_idx]
        elif isinstance(feature_column, int):
            if feature_column < 0 or feature_column >= X.shape[1]:
                raise IndexError(f'Feature column index {feature_column} is out of bounds for data with {X.shape[1]} columns')
            x = X[:, feature_column]
        else:
            raise TypeError('Feature column must be either a string or integer')
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('Numpy array data must be 2-dimensional.')
        if not isinstance(target_column, int):
            raise TypeError('For numpy array data, column identifiers must be integers.')
        if target_column < 0 or target_column >= data.shape[1]:
            raise IndexError(f'Target column index {target_column} is out of bounds for array with {data.shape[1]} columns')
        y = data[:, target_column]
        if not isinstance(feature_column, int):
            raise TypeError('For numpy array data, column identifiers must be integers.')
        if feature_column < 0 or feature_column >= data.shape[1]:
            raise IndexError(f'Feature column index {feature_column} is out of bounds for array with {data.shape[1]} columns')
        x = data[:, feature_column]
    else:
        raise TypeError('Data must be either a DataBatch or a numpy array.')
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if len(x) == 0 or len(y) == 0:
        raise ValueError('No valid data points after removing NaN values')
    if ax is None:
        (fig, ax) = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    scatter_params = {k: v for (k, v) in plot_params.items() if k.startswith('scatter_')}
    other_params = {k: v for (k, v) in plot_params.items() if not k.startswith('scatter_')}
    ax.scatter(x, y, alpha=0.6, label='Data Points', **scatter_params)
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)))
    for (i, q) in enumerate(quantiles):
        if len(x) < 2:
            continue
        X_design = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
            for iteration in range(20):
                residuals = y - X_design @ beta
                weights = np.where(residuals >= 0, q, 1 - q)
                weights = np.maximum(weights, 1e-10)
                sqrt_weights = np.sqrt(weights)
                X_weighted = sqrt_weights[:, np.newaxis] * X_design
                y_weighted = sqrt_weights * y
                try:
                    beta_new = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]
                    if np.allclose(beta, beta_new, atol=1e-06, rtol=1e-05):
                        beta = beta_new
                        break
                    beta = beta_new
                except np.linalg.LinAlgError:
                    break
            x_sorted = np.sort(x)
            y_pred = beta[0] + beta[1] * x_sorted
            ax.plot(x_sorted, y_pred, color=colors[i], label=f'Q{q:.2f}', linewidth=2, **other_params)
        except (np.linalg.LinAlgError, ValueError):
            try:
                (x_unique, x_indices) = np.unique(x, return_inverse=True)
                y_quantile_per_x = []
                for j in range(len(x_unique)):
                    y_values = y[x_indices == j]
                    if len(y_values) > 0:
                        y_quantile_per_x.append(np.quantile(y_values, q))
                    else:
                        y_quantile_per_x.append(np.nan)
                valid_indices = ~np.isnan(y_quantile_per_x)
                if np.sum(valid_indices) > 1:
                    ax.plot(x_unique[valid_indices], np.array(y_quantile_per_x)[valid_indices], color=colors[i], label=f'Q{q:.2f}', linewidth=2, linestyle='--', **other_params)
            except:
                continue
    if isinstance(data, DataBatch):
        if isinstance(feature_column, str):
            x_label = feature_column
        else:
            x_label = f'Column {feature_column}' if feature_names is None else feature_names[feature_column] if 0 <= feature_column < len(feature_names) else f'Column {feature_column}'
        if isinstance(target_column, str):
            y_label = target_column
        else:
            y_label = f'Column {target_column}' if feature_names is None else feature_names[target_column] if 0 <= target_column < len(feature_names) else f'Column {target_column}'
    else:
        x_label = f'Feature Column {feature_column}'
        y_label = f'Target Column {target_column}'
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is None:
        title = f'Quantile Regression: {y_label} vs {x_label}'
    ax.set_title(title)
    ax.legend()
    if 'grid' in other_params:
        ax.grid(other_params['grid'])
    return fig