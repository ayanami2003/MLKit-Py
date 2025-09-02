from general.structures.data_batch import DataBatch
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional, Dict, Any, List

def conditional_quantile_plots(data: Union[DataBatch, np.ndarray], target_column: Union[str, int], feature_column: Union[str, int], conditioning_vars: Optional[List[Union[str, int]]]=None, quantiles: Optional[List[float]]=None, figsize: tuple=(10, 6), title: Optional[str]=None, ax: Optional[plt.Axes]=None, **plot_params: Dict[str, Any]) -> plt.Figure:
    """
    Generate conditional quantile plots to visualize the relationship between a target variable and a feature variable,
    conditioned on additional variables.

    This function creates plots showing how the quantiles of a target variable change with respect to a feature variable,
    optionally conditioning on other variables. These plots are useful for understanding heterogeneous effects and
    identifying subgroups where relationships differ.

    Args:
        data (Union[DataBatch, np.ndarray]): Input data containing the variables of interest.
        target_column (Union[str, int]): Column identifier for the target variable (y-axis).
        feature_column (Union[str, int]): Column identifier for the feature variable (x-axis).
        conditioning_vars (Optional[List[Union[str, int]]]): Optional list of column identifiers for variables to condition on.
        quantiles (Optional[List[float]]): List of quantiles to compute and display (e.g., [0.1, 0.5, 0.9]).
                                          Defaults to [0.25, 0.5, 0.75] if not specified.
        figsize (tuple): Figure size as (width, height). Defaults to (10, 6).
        title (Optional[str]): Optional title for the plot.
        ax (Optional[plt.Axes]): Pre-existing axes to draw on. If None, a new figure and axes are created.
        **plot_params (Dict[str, Any]): Additional keyword arguments passed to the underlying plotting functions.

    Returns:
        plt.Figure: Matplotlib figure object containing the generated plot.

    Raises:
        ValueError: If specified columns are not found in the data or if quantiles are outside [0, 1].
        TypeError: If data is not in a supported format.
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    if not all((0 <= q <= 1 for q in quantiles)):
        raise ValueError('All quantiles must be between 0 and 1 inclusive.')
    raw_data = None
    target_idx = None
    feature_idx = None
    conditioning_indices = []
    if isinstance(data, DataBatch):
        raw_data = data.data
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.array(raw_data)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, 1)
        column_names = data.feature_names if data.feature_names is not None else [f'col_{i}' for i in range(raw_data.shape[1])]
        col_to_idx = {name: idx for (idx, name) in enumerate(column_names)}
        if isinstance(target_column, str):
            if target_column not in col_to_idx:
                raise ValueError(f"Target column '{target_column}' not found in DataBatch.")
            target_idx = col_to_idx[target_column]
        else:
            if not 0 <= target_column < len(column_names):
                raise ValueError(f'Target column index {target_column} out of bounds for DataBatch with {len(column_names)} columns.')
            target_idx = target_column
        if isinstance(feature_column, str):
            if feature_column not in col_to_idx:
                raise ValueError(f"Feature column '{feature_column}' not found in DataBatch.")
            feature_idx = col_to_idx[feature_column]
        else:
            if not 0 <= feature_column < len(column_names):
                raise ValueError(f'Feature column index {feature_column} out of bounds for DataBatch with {len(column_names)} columns.')
            feature_idx = feature_column
        conditioning_indices = []
        if conditioning_vars:
            for var in conditioning_vars:
                if isinstance(var, str):
                    if var not in col_to_idx:
                        raise ValueError(f"Conditioning variable '{var}' not found in DataBatch.")
                    conditioning_indices.append(col_to_idx[var])
                else:
                    if not 0 <= var < len(column_names):
                        raise ValueError(f'Conditioning variable index {var} out of bounds for DataBatch with {len(column_names)} columns.')
                    conditioning_indices.append(var)
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            raw_data = data.reshape(-1, 1)
        elif data.ndim == 2:
            raw_data = data
        else:
            raise TypeError('NumPy array data must be 1 or 2-dimensional.')
        n_cols = raw_data.shape[1]
        if isinstance(target_column, str):
            raise ValueError('Column names cannot be used with NumPy array data.')
        if isinstance(feature_column, str):
            raise ValueError('Column names cannot be used with NumPy array data.')
        if not 0 <= target_column < n_cols:
            raise ValueError(f'Target column index {target_column} out of bounds for array with {n_cols} columns.')
        if not 0 <= feature_column < n_cols:
            raise ValueError(f'Feature column index {feature_column} out of bounds for array with {n_cols} columns.')
        target_idx = target_column
        feature_idx = feature_column
        conditioning_indices = []
        if conditioning_vars:
            for var in conditioning_vars:
                if isinstance(var, str):
                    raise ValueError('Column names cannot be used with NumPy array data.')
                if not 0 <= var < n_cols:
                    raise ValueError(f'Conditioning variable index {var} out of bounds for array with {n_cols} columns.')
                conditioning_indices.append(var)
    else:
        raise TypeError('Data must be either a DataBatch or a NumPy array.')
    target_values = raw_data[:, target_idx]
    feature_values = raw_data[:, feature_idx]
    try:
        target_values = np.asarray(target_values, dtype=float)
        feature_values = np.asarray(feature_values, dtype=float)
    except (ValueError, TypeError):
        raise ValueError('Target and feature columns must contain numerical data.')
    if not np.issubdtype(target_values.dtype, np.number) or np.all(np.isnan(target_values)):
        raise ValueError('Target column must contain numerical data.')
    if not np.issubdtype(feature_values.dtype, np.number) or np.all(np.isnan(feature_values)):
        raise ValueError('Feature column must contain numerical data.')
    unique_groups = [()]
    group_labels = ['All Data']
    conditioning_data = None
    if conditioning_indices:
        conditioning_data = raw_data[:, conditioning_indices]
        if conditioning_data.size > 0:
            if conditioning_data.dtype.kind in ['U', 'S', 'O']:
                conditioning_data = conditioning_data.astype(object)
            try:
                unique_groups = np.unique(conditioning_data, axis=0)
                if len(conditioning_indices) == 1:
                    group_labels = [str(group[0]) if hasattr(group, '__getitem__') else str(group) for group in unique_groups]
                else:
                    group_labels = [', '.join(map(str, group)) for group in unique_groups]
            except Exception:
                unique_groups = [()]
                group_labels = ['All Data']
    if ax is None:
        (fig, ax) = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_groups))))
    for (i, (group, label)) in enumerate(zip(unique_groups, group_labels)):
        color = colors[i % len(colors)] if len(colors) > 0 else 'blue'
        if len(group) > 0 and conditioning_data is not None:
            mask = np.ones(len(feature_values), dtype=bool)
            for (j, val) in enumerate(group):
                if isinstance(val, str):
                    mask &= conditioning_data[:, j].astype(str) == val
                else:
                    mask &= conditioning_data[:, j] == val
            group_features = feature_values[mask]
            group_targets = target_values[mask]
        else:
            group_features = feature_values
            group_targets = target_values
        if len(group_features) == 0:
            continue
        group_features = np.asarray(group_features, dtype=float)
        group_targets = np.asarray(group_targets, dtype=float)
        if not (np.issubdtype(group_features.dtype, np.number) and np.issubdtype(group_targets.dtype, np.number)):
            continue
        valid_mask = ~(np.isnan(group_features) | np.isnan(group_targets))
        group_features = group_features[valid_mask]
        group_targets = group_targets[valid_mask]
        if len(group_features) == 0:
            continue
        sort_idx = np.argsort(group_features)
        sorted_features = group_features[sort_idx]
        sorted_targets = group_targets[sort_idx]
        for q in quantiles:
            if len(sorted_features) > 10:
                n_bins = min(50, len(sorted_features) // 5)
                if np.issubdtype(sorted_features.dtype, np.number) and len(sorted_features) > 0 and np.isfinite(sorted_features).any() and (sorted_features.min() != sorted_features.max()):
                    (feat_min, feat_max) = (sorted_features.min(), sorted_features.max())
                    bins = np.linspace(feat_min, feat_max, n_bins + 1)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    bin_quantiles = []
                    for j in range(n_bins):
                        bin_mask = (sorted_features >= bins[j]) & (sorted_features <= bins[j + 1])
                        bin_targets = sorted_targets[bin_mask]
                        if len(bin_targets) > 0:
                            bin_quantiles.append(np.percentile(bin_targets, q * 100))
                        else:
                            bin_quantiles.append(np.nan)
                    valid_idx = ~np.isnan(bin_quantiles)
                    if np.sum(valid_idx) > 1:
                        ax.plot(bin_centers[valid_idx], np.array(bin_quantiles)[valid_idx], color=color, linestyle='-', alpha=0.7, label=f'{label} (q={q:.2f})' if i == 0 else '', **plot_params)
                elif len(sorted_targets) > 0 and np.isfinite(sorted_targets).any():
                    finite_targets = sorted_targets[np.isfinite(sorted_targets)]
                    if len(finite_targets) > 0:
                        overall_quantile = np.percentile(finite_targets, q * 100)
                        ax.axhline(y=overall_quantile, color=color, linestyle='-', alpha=0.7, label=f'{label} (q={q:.2f})' if i == 0 else '')
            elif len(group_targets) > 0 and np.isfinite(group_targets).any():
                finite_targets = group_targets[np.isfinite(group_targets)]
                if len(finite_targets) > 0:
                    overall_quantile = np.percentile(finite_targets, q * 100)
                    ax.axhline(y=overall_quantile, color=color, linestyle='-', alpha=0.7, label=f'{label} (q={q:.2f})' if i == 0 else '')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Target Quantiles')
    if title:
        ax.set_title(title)
    if len(group_labels) > 1 or len(quantiles) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return fig