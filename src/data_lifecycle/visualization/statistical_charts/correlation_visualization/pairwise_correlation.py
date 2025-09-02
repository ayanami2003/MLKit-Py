from typing import Union, Optional, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.feature_set import FeatureSet
from scipy.stats import pearsonr, spearmanr, kendalltau

class PairwiseCorrelationVisualizer(BasePipelineComponent):

    def __init__(self, method: str='pearson', plot_type: str='scatter_matrix', figsize: tuple=(12, 10), annot: bool=True, cmap: str='viridis', feature_names: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the PairwiseCorrelationVisualizer.
        
        Parameters
        ----------
        method : str, default='pearson'
            Correlation method to compute ('pearson', 'spearman', 'kendall')
        plot_type : str, default='scatter_matrix'
            Type of visualization to generate ('scatter_matrix', 'heatmap', 'network')
        figsize : tuple, default=(12, 10)
            Size of the figure in inches
        annot : bool, default=True
            Whether to annotate correlation values on heatmap
        cmap : str, default='viridis'
            Colormap for heatmap visualization
        feature_names : Optional[List[str]], optional
            Custom feature names for labeling axes
        name : Optional[str], optional
            Name for the pipeline component
        """
        super().__init__(name=name)
        self.method = method
        self.plot_type = plot_type
        self.figsize = figsize
        self.annot = annot
        self.cmap = cmap
        self.feature_names = feature_names

    def process(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Generate pairwise correlation visualization for the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to visualize correlations for. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for visualization customization
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'figure': matplotlib figure object
            - 'correlation_matrix': computed correlation matrix
            - 'ax': matplotlib axes objects
            
        Raises
        ------
        ValueError
            If data is empty or has incompatible dimensions
        TypeError
            If data is not of supported type
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
        (n_samples, n_features) = features.shape
        if self.feature_names is not None:
            if len(self.feature_names) != n_features:
                raise ValueError("Number of custom feature names doesn't match number of features")
            feature_names = self.feature_names
        elif feature_names is None:
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
            corr_matrix = np.ones((n_features, n_features))
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    (corr, _) = kendalltau(features[:, i], features[:, j])
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    corr_matrix[j, i] = corr if not np.isnan(corr) else 0.0
        else:
            raise ValueError(f'Unsupported correlation method: {self.method}')
        if self.plot_type == 'scatter_matrix':
            df = pd.DataFrame(features, columns=feature_names)
            (fig, ax) = plt.subplots(figsize=self.figsize)
            g = sns.pairplot(df, height=self.figsize[0] / len(feature_names), aspect=1)
            if n_features > 1:
                for (i, var1) in enumerate(feature_names):
                    for (j, var2) in enumerate(feature_names):
                        if i < j:
                            corr_val = corr_matrix[i, j]
                            g.axes[i, j].annotate(f'r={corr_val:.2f}', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            fig = g.fig
            ax = g.axes
        elif self.plot_type == 'heatmap':
            (fig, ax) = plt.subplots(figsize=self.figsize)
            sns.heatmap(corr_matrix, annot=self.annot, cmap=self.cmap, xticklabels=feature_names, yticklabels=feature_names, ax=ax, vmin=-1, vmax=1, center=0)
            ax.set_title(f'{self.method.capitalize()} Correlation Heatmap')
        elif self.plot_type == 'network':
            try:
                import networkx as nx
                G = nx.Graph()
                for (i, name) in enumerate(feature_names):
                    G.add_node(i, label=name)
                threshold = kwargs.get('threshold', 0.5)
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        corr_val = corr_matrix[i, j]
                        if abs(corr_val) > threshold:
                            G.add_edge(i, j, weight=abs(corr_val), correlation=corr_val)
                (fig, ax) = plt.subplots(figsize=self.figsize)
                pos = nx.spring_layout(G, seed=42)
                nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000, ax=ax)
                edges = G.edges()
                widths = [G[u][v]['weight'] * 3 for (u, v) in edges]
                colors = ['red' if G[u][v]['correlation'] < 0 else 'green' for (u, v) in edges]
                nx.draw_networkx_edges(G, pos, width=widths, edge_color=colors, ax=ax)
                labels = {i: name for (i, name) in enumerate(feature_names)}
                nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
                ax.set_title(f'Feature Correlation Network (threshold={threshold})')
                ax.axis('off')
            except ImportError:
                raise ImportError("NetworkX is required for network visualization. Please install it with 'pip install networkx'")
        else:
            raise ValueError(f'Unsupported plot type: {self.plot_type}')
        return {'figure': fig, 'correlation_matrix': corr_matrix, 'ax': ax}