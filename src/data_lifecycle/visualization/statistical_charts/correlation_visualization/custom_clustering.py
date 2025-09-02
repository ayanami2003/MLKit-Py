from typing import Optional, Union, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.feature_set import FeatureSet
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, kendalltau
from src.data_lifecycle.modeling.clustering.k_means.kmeans_algorithm import KMeansClusteringModel
from src.data_lifecycle.modeling.clustering.density_based.optimized_dbscan_algorithm import OptimizedDBSCANModel

class CustomClusteringVisualizer(BasePipelineComponent):

    def __init__(self, method: str='pearson', cluster_method: str='hierarchical', n_clusters: Optional[int]=None, plot_type: str='clustermap', figsize: tuple=(12, 10), annot: bool=True, cmap: str='viridis', name: Optional[str]=None):
        """
        Initialize the CustomClusteringVisualizer.
        
        Args:
            method (str): Method for computing correlation. Defaults to 'pearson'.
            cluster_method (str): Clustering method to apply. Defaults to 'hierarchical'.
            n_clusters (Optional[int]): Number of clusters. Required for some clustering methods.
            plot_type (str): Type of visualization to produce. Defaults to 'clustermap'.
            figsize (tuple): Size of the figure. Defaults to (12, 10).
            annot (bool): Annotate correlation values on plot. Defaults to True.
            cmap (str): Color map for visualization. Defaults to 'viridis'.
            name (Optional[str]): Name for the component. Defaults to class name.
        """
        super().__init__(name)
        self.method = method
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.plot_type = plot_type
        self.figsize = figsize
        self.annot = annot
        self.cmap = cmap

    def _compute_correlation_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix using the specified method.
        
        Args:
            features (np.ndarray): Input features of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Correlation matrix of shape (n_features, n_features).
        """
        n_features = features.shape[1]
        if n_features == 1:
            return np.array([[1.0]])
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
        return corr_matrix

    def _apply_clustering(self, correlation_matrix: np.ndarray, n_features: int) -> np.ndarray:
        """
        Apply clustering to the correlation matrix.
        
        Args:
            correlation_matrix (np.ndarray): Correlation matrix of shape (n_features, n_features).
            n_features (int): Number of features.
            
        Returns:
            np.ndarray: Cluster labels for each feature.
        """
        distance_matrix = 1 - np.abs(correlation_matrix)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        if self.cluster_method == 'hierarchical':
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='ward')
            if self.n_clusters is None:
                n_clusters = max(2, int(np.sqrt(n_features)))
            else:
                n_clusters = min(self.n_clusters, n_features)
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        elif self.cluster_method == 'kmeans':
            if self.n_clusters is None:
                n_clusters = max(2, int(np.sqrt(n_features)))
            else:
                n_clusters = min(self.n_clusters, n_features)
            kmeans = KMeansClusteringModel(n_clusters=n_clusters)
            feature_set = FeatureSet(features=correlation_matrix)
            clusters = kmeans.fit_predict(feature_set)
        elif self.cluster_method == 'dbscan':
            dbscan = OptimizedDBSCANModel(eps=0.5, min_samples=2)
            feature_set = FeatureSet(features=correlation_matrix)
            clusters = dbscan.fit_predict(feature_set)
        else:
            raise ValueError(f'Unsupported clustering method: {self.cluster_method}')
        return clusters

    def _generate_visualization(self, correlation_matrix: np.ndarray, clusters: np.ndarray, feature_names: List[str]) -> plt.Figure:
        """
        Generate visualization based on plot type.
        
        Args:
            correlation_matrix (np.ndarray): Correlation matrix.
            clusters (np.ndarray): Cluster labels for features.
            feature_names (List[str]): Names of features.
            
        Returns:
            plt.Figure: Generated figure object.
        """
        if self.plot_type == 'clustermap':
            cluster_colors = [f'C{cluster}' for cluster in clusters]
            cg = sns.clustermap(correlation_matrix, annot=self.annot, cmap=self.cmap, xticklabels=feature_names, yticklabels=feature_names, figsize=self.figsize, row_colors=cluster_colors, col_colors=cluster_colors)
            fig = cg.fig
        elif self.plot_type == 'dendrogram':
            distance_matrix = 1 - np.abs(correlation_matrix)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='ward')
            (fig, ax) = plt.subplots(figsize=self.figsize)
            dendrogram(linkage_matrix, labels=feature_names, ax=ax)
            ax.set_title(f'Dendrogram ({self.method.capitalize()} correlation)')
        elif self.plot_type == 'matrix':
            (fig, ax) = plt.subplots(figsize=self.figsize)
            sns.heatmap(correlation_matrix, annot=self.annot, cmap=self.cmap, xticklabels=feature_names, yticklabels=feature_names, ax=ax)
            ax.set_title(f'{self.method.capitalize()} Correlation Matrix')
        else:
            raise ValueError(f'Unsupported plot type: {self.plot_type}')
        return fig

    def process(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Perform clustering on correlated features and generate visualization.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input features to visualize.
            **kwargs: Additional parameters for clustering or plotting.
            
        Returns:
            Dict[str, Any]: Contains generated plot objects and cluster assignments.
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
        if n_features < 2:
            raise ValueError('At least 2 features are required for clustering visualization')
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError("Number of feature names doesn't match number of features")
        correlation_matrix = self._compute_correlation_matrix(features)
        clusters = self._apply_clustering(correlation_matrix, n_features)
        fig = self._generate_visualization(correlation_matrix, clusters, feature_names)
        unique_clusters = np.unique(clusters)
        cluster_labels = {i: f'Cluster_{i}' for i in unique_clusters if i != -1}
        if -1 in unique_clusters:
            cluster_labels[-1] = 'Noise'
        return {'figure': fig, 'clusters': clusters, 'correlation_matrix': correlation_matrix, 'cluster_labels': cluster_labels}