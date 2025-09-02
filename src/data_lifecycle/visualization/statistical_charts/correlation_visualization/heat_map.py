from typing import Optional, Union, Dict, Any
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.feature_set import FeatureSet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

class HeatMapVisualizer(BasePipelineComponent):

    def __init__(self, method: str='pearson', figsize: tuple=(10, 8), annot: bool=True, cmap: str='coolwarm', title: Optional[str]=None, name: Optional[str]=None):
        """
        Initialize the HeatMapVisualizer with configuration parameters.
        
        Args:
            method (str): Method for correlation computation. Defaults to 'pearson'.
            figsize (tuple): Size of the figure in inches. Defaults to (10, 8).
            annot (bool): Flag to enable/disable cell annotations. Defaults to True.
            cmap (str): Color map for visualization. Defaults to 'coolwarm'.
            title (Optional[str]): Optional title for the heat map.
            name (Optional[str]): Name identifier for the component.
        """
        super().__init__(name=name)
        self.method = method
        self.figsize = figsize
        self.annot = annot
        self.cmap = cmap
        self.title = title

    def process(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Generate a heat map visualization from input data.
        
        If input is a FeatureSet, computes the correlation matrix according to the specified method.
        If input is a numpy array, assumes it's already a 2D matrix suitable for visualization.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to visualize. Can be a FeatureSet
                containing features or a precomputed 2D correlation matrix.
            **kwargs: Additional processing parameters (not currently used).
                
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'figure': Matplotlib figure object
                - 'axes': Matplotlib axes object
                - 'correlation_matrix': Computed correlation matrix (if applicable)
        """
        if isinstance(data, FeatureSet):
            features = data.features
            feature_names = data.feature_names
            if features.size == 0:
                raise ValueError('Input FeatureSet is empty')
            n_features = features.shape[1]
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(n_features)]
            elif len(feature_names) != n_features:
                raise ValueError("Number of feature names doesn't match number of features")
            if n_features == 1:
                corr_matrix = np.array([[1.0]])
            elif self.method == 'pearson':
                corr_matrix = np.corrcoef(features, rowvar=False)
                if np.any(np.isnan(corr_matrix)):
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                    np.fill_diagonal(corr_matrix, 1.0)
            elif self.method == 'spearman':
                corr_result = spearmanr(features, axis=0)
                corr_matrix = corr_result.correlation if hasattr(corr_result, 'correlation') else corr_result[0]
                if corr_matrix.ndim == 0:
                    corr_matrix = np.array([[1.0]])
            elif self.method == 'kendall':
                corr_matrix = np.ones((n_features, n_features))
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        try:
                            (corr, _) = kendalltau(features[:, i], features[:, j])
                            corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                            corr_matrix[j, i] = corr if not np.isnan(corr) else 0.0
                        except Exception:
                            corr_matrix[i, j] = 0.0
                            corr_matrix[j, i] = 0.0
            else:
                raise ValueError(f'Unsupported correlation method: {self.method}')
        elif isinstance(data, np.ndarray):
            if data.size == 0:
                raise ValueError('Input array is empty')
            if data.ndim != 2:
                raise ValueError('Input array must be 2-dimensional')
            corr_matrix = data
            feature_names = None
        else:
            raise TypeError('Data must be either FeatureSet or numpy array')
        (fig, ax) = plt.subplots(figsize=self.figsize)
        sns.heatmap(corr_matrix, annot=self.annot, cmap=self.cmap, xticklabels=feature_names or [f'Var{i}' for i in range(corr_matrix.shape[1])], yticklabels=feature_names or [f'Var{i}' for i in range(corr_matrix.shape[0])], ax=ax, vmin=-1, vmax=1, center=0)
        if self.title:
            ax.set_title(self.title)
        result = {'figure': fig, 'axes': ax, 'correlation_matrix': corr_matrix}
        return result