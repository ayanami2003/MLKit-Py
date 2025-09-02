from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.feature_set import FeatureSet
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, Any
from scipy.stats import spearmanr, kendalltau

class CorrelationMatrixVisualizer(BasePipelineComponent):

    def __init__(self, method: str='pearson', plot_type: str='heatmap', figsize: tuple=(10, 8), annot: bool=True, cmap: str='coolwarm', name: Optional[str]=None):
        """
        Initialize the correlation matrix visualizer.
        
        Parameters
        ----------
        method : str, default='pearson'
            Method for computing correlations ('pearson', 'spearman', 'kendall')
        plot_type : str, default='heatmap'
            Type of plot to generate ('heatmap', 'clustermap')
        figsize : tuple, default=(10, 8)
            Figure size as (width, height)
        annot : bool, default=True
            Whether to annotate cells with correlation values
        cmap : str, default='coolwarm'
            Colormap for the visualization
        name : str, optional
            Name for the component
        """
        super().__init__(name)
        self.method = method
        self.plot_type = plot_type
        self.figsize = figsize
        self.annot = annot
        self.cmap = cmap

    def process(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Generate correlation matrix visualization from input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input feature data to visualize correlations for
        **kwargs : dict
            Additional visualization parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing figure object and computed correlation matrix
            
        Raises
        ------
        ValueError
            If data is empty or has incompatible dimensions
        TypeError
            If data is not a supported type
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
            (corr_matrix, _) = spearmanr(features, axis=0)
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0]])
        elif self.method == 'kendall':
            n_features = features.shape[1]
            corr_matrix = np.ones((n_features, n_features))
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    (corr, _) = kendalltau(features[:, i], features[:, j])
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    corr_matrix[j, i] = corr if not np.isnan(corr) else 0.0
        else:
            raise ValueError(f'Unsupported correlation method: {self.method}')
        (fig, ax) = plt.subplots(figsize=self.figsize)
        if self.plot_type == 'heatmap':
            sns.heatmap(corr_matrix, annot=self.annot, cmap=self.cmap, xticklabels=feature_names, yticklabels=feature_names, ax=ax)
        elif self.plot_type == 'clustermap':
            cg = sns.clustermap(corr_matrix, annot=self.annot, cmap=self.cmap, xticklabels=feature_names, yticklabels=feature_names, figsize=self.figsize)
            fig = cg.fig
        else:
            raise ValueError(f'Unsupported plot type: {self.plot_type}')
        ax.set_title(f'{self.method.capitalize()} Correlation Matrix')
        return {'figure': fig, 'correlation_matrix': corr_matrix, 'feature_names': feature_names}