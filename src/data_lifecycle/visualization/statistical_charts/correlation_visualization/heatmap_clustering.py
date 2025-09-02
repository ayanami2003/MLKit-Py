from typing import Optional, Union, Dict, Any
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.feature_set import FeatureSet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, kendalltau

class HeatmapClusteringVisualizer(BasePipelineComponent):

    def __init__(self, method: str='pearson', plot_type: str='clustermap', figsize: tuple=(12, 10), annot: bool=True, cmap: str='coolwarm', linkage_method: str='ward', distance_metric: str='euclidean', name: Optional[str]=None):
        """
        Initialize the heatmap clustering visualizer.
        
        Parameters
        ----------
        method : str, default='pearson'
            Correlation method to compute ('pearson', 'spearman', 'kendall')
        plot_type : str, default='clustermap'
            Type of plot to generate ('clustermap', 'heatmap')
        figsize : tuple, default=(12, 10)
            Figure size as (width, height)
        annot : bool, default=True
            Whether to annotate cells with correlation values
        cmap : str, default='coolwarm'
            Colormap for the heatmap
        linkage_method : str, default='ward'
            Linkage method for hierarchical clustering ('ward', 'complete', 'average', 'single')
        distance_metric : str, default='euclidean'
            Distance metric for clustering ('euclidean', 'manhattan', 'cosine')
        name : str, optional
            Name for the component
        """
        super().__init__(name)
        self.method = method
        self.plot_type = plot_type
        self.figsize = figsize
        self.annot = annot
        self.cmap = cmap
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric

    def process(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Generate a clustered heatmap visualization of feature correlations.
        
        This method computes the correlation matrix from input features and applies
        hierarchical clustering to reorder the matrix for better visualization.
        The resulting plot groups highly correlated features together.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to visualize. If FeatureSet, uses the features attribute.
            If ndarray, assumes samples as rows and features as columns.
        **kwargs : dict
            Additional parameters for visualization customization
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'figure': matplotlib figure object
            - 'axes': matplotlib axes object
            - 'clustered_matrix': reordered correlation matrix
            - 'row_linkage': linkage matrix for rows
            - 'col_linkage': linkage matrix for columns
            - 'feature_order': order of features after clustering
            
        Raises
        ------
        ValueError
            If input data is invalid or incompatible
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
            corr_matrix = np.ones((n_features, n_features))
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    (corr, _) = kendalltau(features[:, i], features[:, j])
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    corr_matrix[j, i] = corr if not np.isnan(corr) else 0.0
        else:
            raise ValueError(f'Unsupported correlation method: {self.method}')
        distance_matrix = 1 - np.abs(corr_matrix)
        np.fill_diagonal(distance_matrix, 0)
        if self.linkage_method == 'ward' and self.distance_metric != 'euclidean':
            linkage_method = 'ward'
            distance_metric = 'euclidean'
        else:
            linkage_method = self.linkage_method
            distance_metric = self.distance_metric
        condensed_distances = squareform(distance_matrix)
        row_linkage = linkage(condensed_distances, method=linkage_method, metric=distance_metric)
        col_linkage = row_linkage
        from scipy.cluster.hierarchy import leaves_list
        feature_order = leaves_list(row_linkage)
        clustered_matrix = corr_matrix[feature_order, :][:, feature_order]
        reordered_feature_names = [feature_names[i] for i in feature_order]
        (fig, ax) = plt.subplots(figsize=self.figsize)
        if self.plot_type == 'clustermap':
            cg = sns.clustermap(corr_matrix, annot=self.annot, cmap=self.cmap, xticklabels=feature_names, yticklabels=feature_names, figsize=self.figsize, row_linkage=row_linkage, col_linkage=col_linkage)
            fig = cg.fig
            ax = cg.ax_heatmap
        elif self.plot_type == 'heatmap':
            sns.heatmap(clustered_matrix, annot=self.annot, cmap=self.cmap, xticklabels=reordered_feature_names, yticklabels=reordered_feature_names, ax=ax)
        else:
            raise ValueError(f'Unsupported plot type: {self.plot_type}')
        ax.set_title(f'{self.method.capitalize()} Correlation Clustered Heatmap')
        return {'figure': fig, 'axes': ax, 'clustered_matrix': clustered_matrix, 'row_linkage': row_linkage, 'col_linkage': col_linkage, 'feature_order': feature_order}