from typing import Optional, Union, Dict, Any
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.feature_set import FeatureSet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

class CorrelationHeatmapVisualizer(BasePipelineComponent):

    def __init__(self, method: str='pearson', figsize: tuple=(10, 8), annot: bool=True, cmap: str='coolwarm', vmin: Optional[float]=-1.0, vmax: Optional[float]=1.0, title: Optional[str]=None, name: Optional[str]=None):
        """
        Initialize the correlation heatmap visualizer.
        
        Parameters
        ----------
        method : str, optional
            Correlation method to use ('pearson', 'spearman', 'kendall'), by default 'pearson'
        figsize : tuple, optional
            Figure size as (width, height), by default (10, 8)
        annot : bool, optional
            Whether to annotate heatmap cells with values, by default True
        cmap : str, optional
            Matplotlib colormap name, by default 'coolwarm'
        vmin : float, optional
            Minimum value for colormap, by default -1.0
        vmax : float, optional
            Maximum value for colormap, by default 1.0
        title : str, optional
            Title for the heatmap, by default None
        name : str, optional
            Component name, by default None
        """
        super().__init__(name)
        self.method = method
        self.figsize = figsize
        self.annot = annot
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.title = title

    def process(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Generate a correlation heatmap from input feature data.
        
        This method computes the correlation matrix from the input data and creates
        a heatmap visualization. It returns a dictionary containing the figure,
        axes, and correlation matrix for further use or customization.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input feature data as FeatureSet or numpy array
        **kwargs : dict
            Additional visualization parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'figure': matplotlib figure object
            - 'axes': matplotlib axes object
            - 'correlation_matrix': computed correlation matrix
            - 'feature_names': names of features (if available)
            
        Raises
        ------
        ValueError
            If data is empty or correlation computation fails
        TypeError
            If data is not in a supported format
        """
        if isinstance(data, FeatureSet):
            features = data.features
            feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            features = data
            feature_names = None
        else:
            raise TypeError('Data must be either FeatureSet or numpy array')
        if features.size == 0:
            raise ValueError('Input data is empty')
        if features.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
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
            from scipy.stats import spearmanr
            corr_result = spearmanr(features, axis=0)
            corr_matrix = corr_result.correlation if hasattr(corr_result, 'correlation') else corr_result[0]
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
            elif corr_matrix.shape == (2,) and n_features == 2:
                corr_matrix = np.array([[1.0, corr_matrix[0]], [corr_matrix[0], 1.0]])
        elif self.method == 'kendall':
            from scipy.stats import kendalltau
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
        (fig, ax) = plt.subplots(figsize=self.figsize)
        sns.heatmap(corr_matrix, annot=self.annot, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, xticklabels=feature_names, yticklabels=feature_names, ax=ax)
        title = self.title if self.title is not None else f'{self.method.capitalize()} Correlation Heatmap'
        ax.set_title(title)
        return {'figure': fig, 'axes': ax, 'correlation_matrix': corr_matrix, 'feature_names': feature_names}