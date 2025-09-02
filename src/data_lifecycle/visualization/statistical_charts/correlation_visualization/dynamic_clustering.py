from typing import Optional, Union, Dict, Any
import numpy as np
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.feature_set import FeatureSet
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, kendalltau
from src.data_lifecycle.modeling.clustering.k_means.kmeans_algorithm import KMeansClusteringModel
from src.data_lifecycle.modeling.clustering.density_based.optimized_dbscan_algorithm import OptimizedDBSCANModel

class DynamicClusteringVisualizer(BasePipelineComponent):
    """
    Visualizes dynamic clustering of features based on correlation structures.
    
    This component performs clustering on features using their correlation matrix
    and visualizes the resulting clusters dynamically. It supports various clustering
    methods and visualization techniques to help identify groups of highly correlated features.
    
    The visualization helps in understanding feature relationships and can be useful
    for feature selection, dimensionality reduction, or identifying multicollinearity.
    
    Attributes
    ----------
    method : str
        Clustering method to use ('hierarchical', 'kmeans', 'dbscan')
    n_clusters : Optional[int]
        Number of clusters for methods that require it
    correlation_method : str
        Method for computing correlation matrix ('pearson', 'spearman', 'kendall')
    plot_type : str
        Type of plot to generate ('dendrogram', 'clustermap', 'network')
    figsize : tuple
        Figure size for the plot
    threshold : float
        Threshold for correlation-based clustering
        
    Methods
    -------
    process(data, **kwargs)
        Performs dynamic clustering visualization on input data
    """

    def __init__(self, method: str='hierarchical', n_clusters: Optional[int]=None, correlation_method: str='pearson', plot_type: str='clustermap', figsize: tuple=(12, 10), threshold: float=0.5, name: Optional[str]=None):
        """
        Initialize the DynamicClusteringVisualizer.
        
        Parameters
        ----------
        method : str, default='hierarchical'
            Clustering method to use. Options are 'hierarchical', 'kmeans', 'dbscan'.
        n_clusters : int, optional
            Number of clusters for methods that require it (e.g., kmeans).
        correlation_method : str, default='pearson'
            Method for computing correlation matrix. Options are 'pearson', 'spearman', 'kendall'.
        plot_type : str, default='clustermap'
            Type of plot to generate. Options are 'dendrogram', 'clustermap', 'network'.
        figsize : tuple, default=(12, 10)
            Figure size for the plot as (width, height).
        threshold : float, default=0.5
            Threshold for correlation-based clustering. Used to determine significant correlations.
        name : str, optional
            Name of the component. If None, uses class name.
        """
        super().__init__(name)
        self.method = method
        self.n_clusters = n_clusters
        self.correlation_method = correlation_method
        self.plot_type = plot_type
        self.figsize = figsize
        self.threshold = threshold
        if self.method not in ['hierarchical', 'kmeans', 'dbscan']:
            raise ValueError("method must be one of 'hierarchical', 'kmeans', 'dbscan'")
        if self.correlation_method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("correlation_method must be one of 'pearson', 'spearman', 'kendall'")
        if self.plot_type not in ['dendrogram', 'clustermap', 'network']:
            raise ValueError("plot_type must be one of 'dendrogram', 'clustermap', 'network'")
        if not 0 <= self.threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')

    def process(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Perform dynamic clustering visualization on input data.
        
        This method computes the correlation matrix of the input features,
        applies the specified clustering algorithm, and generates a visualization
        showing the identified clusters.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to visualize. If FeatureSet, uses the features attribute.
            If ndarray, assumes it's a 2D array where rows are samples and columns are features.
        **kwargs : dict
            Additional parameters for clustering or plotting.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'figure': matplotlib figure object
            - 'clusters': cluster assignments for each feature
            - 'correlation_matrix': computed correlation matrix
            - 'cluster_labels': labels for each cluster
            
        Raises
        ------
        ValueError
            If data is empty or has incompatible dimensions
        NotImplementedError
            If specified method or plot_type is not implemented
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

    def _compute_correlation_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix using the specified method.
        
        Parameters
        ----------
        features : np.ndarray
            Input features array of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Correlation matrix of shape (n_features, n_features)
        """
        n_features = features.shape[1]
        if n_features == 1:
            return np.array([[1.0]])
        if self.correlation_method == 'pearson':
            corr_matrix = np.corrcoef(features, rowvar=False)
            if np.any(np.isnan(corr_matrix)):
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                np.fill_diagonal(corr_matrix, 1.0)
        elif self.correlation_method == 'spearman':
            (corr_matrix, _) = spearmanr(features, axis=0)
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0]])
        elif self.correlation_method == 'kendall':
            corr_matrix = np.ones((n_features, n_features))
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    (corr, _) = kendalltau(features[:, i], features[:, j])
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    corr_matrix[j, i] = corr if not np.isnan(corr) else 0.0
        return corr_matrix

    def _apply_clustering(self, correlation_matrix: np.ndarray, n_features: int) -> np.ndarray:
        """
        Apply the specified clustering algorithm to the correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape (n_features, n_features)
        n_features : int
            Number of features
            
        Returns
        -------
        np.ndarray
            Cluster assignments for each feature
        """
        if self.method == 'hierarchical':
            return self._hierarchical_clustering(correlation_matrix, n_features)
        elif self.method == 'kmeans':
            return self._kmeans_clustering(correlation_matrix, n_features)
        elif self.method == 'dbscan':
            return self._dbscan_clustering(correlation_matrix, n_features)
        else:
            raise NotImplementedError(f"Clustering method '{self.method}' is not implemented")

    def _hierarchical_clustering(self, correlation_matrix: np.ndarray, n_features: int) -> np.ndarray:
        """
        Apply hierarchical clustering to the correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape (n_features, n_features)
        n_features : int
            Number of features
            
        Returns
        -------
        np.ndarray
            Cluster assignments for each feature
        """
        distance_matrix = 1 - np.abs(correlation_matrix)
        np.fill_diagonal(distance_matrix, 0)
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        if self.n_clusters is not None:
            n_clusters = self.n_clusters
        else:
            n_clusters = max(1, int(n_features * (1 - self.threshold)))
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        return clusters - 1

    def _kmeans_clustering(self, correlation_matrix: np.ndarray, n_features: int) -> np.ndarray:
        """
        Apply K-means clustering to the correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape (n_features, n_features)
        n_features : int
            Number of features
            
        Returns
        -------
        np.ndarray
            Cluster assignments for each feature
        """
        if self.n_clusters is None:
            n_clusters = max(1, int(n_features * (1 - self.threshold)))
        else:
            n_clusters = self.n_clusters
        if n_clusters >= n_features:
            return np.arange(n_features)
        kmeans_model = KMeansClusteringModel(n_clusters=n_clusters, random_state=42, max_iter=300)
        features_for_clustering = correlation_matrix.T

        class MockFeatureSet:

            def __init__(self, features):
                self.features = features
        mock_fs = MockFeatureSet(features_for_clustering)
        try:
            clusters = kmeans_model.fit_predict(mock_fs)
            return clusters
        except Exception:
            return np.zeros(n_features, dtype=int)

    def _dbscan_clustering(self, correlation_matrix: np.ndarray, n_features: int) -> np.ndarray:
        """
        Apply DBSCAN clustering to the correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape (n_features, n_features)
        n_features : int
            Number of features
            
        Returns
        -------
        np.ndarray
            Cluster assignments for each feature
        """
        distance_matrix = 1 - np.abs(correlation_matrix)
        dbscan_model = OptimizedDBSCANModel(eps=1 - self.threshold, min_samples=2)

        class MockFeatureSet:

            def __init__(self, features):
                self.features = features
        mock_fs = MockFeatureSet(distance_matrix)
        try:
            clusters = dbscan_model.fit_predict(mock_fs)
            return clusters
        except Exception:
            return np.zeros(n_features, dtype=int)

    def _generate_visualization(self, correlation_matrix: np.ndarray, clusters: np.ndarray, feature_names: list) -> plt.Figure:
        """
        Generate the visualization based on the specified plot type.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape (n_features, n_features)
        clusters : np.ndarray
            Cluster assignments for each feature
        feature_names : list
            Names of the features
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        (fig, ax) = plt.subplots(figsize=self.figsize)
        if self.plot_type == 'dendrogram':
            return self._generate_dendrogram(correlation_matrix, feature_names, fig, ax)
        elif self.plot_type == 'clustermap':
            return self._generate_clustermap(correlation_matrix, clusters, feature_names, fig)
        elif self.plot_type == 'network':
            return self._generate_network(correlation_matrix, clusters, feature_names, fig, ax)
        else:
            raise NotImplementedError(f"Plot type '{self.plot_type}' is not implemented")

    def _generate_dendrogram(self, correlation_matrix: np.ndarray, feature_names: list, fig: plt.Figure, ax: plt.Axes) -> plt.Figure:
        """
        Generate a dendrogram visualization.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape (n_features, n_features)
        feature_names : list
            Names of the features
        fig : plt.Figure
            Matplotlib figure object
        ax : plt.Axes
            Matplotlib axes object
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        distance_matrix = 1 - np.abs(correlation_matrix)
        np.fill_diagonal(distance_matrix, 0)
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        dendrogram(linkage_matrix, labels=feature_names, ax=ax, leaf_rotation=90)
        ax.set_title(f'Hierarchical Clustering Dendrogram ({self.correlation_method.capitalize()} Correlation)')
        plt.tight_layout()
        return fig

    def _generate_clustermap(self, correlation_matrix: np.ndarray, clusters: np.ndarray, feature_names: list, fig: plt.Figure) -> plt.Figure:
        """
        Generate a clustermap visualization.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape (n_features, n_features)
        clusters : np.ndarray
            Cluster assignments for each feature
        feature_names : list
            Names of the features
        fig : plt.Figure
            Matplotlib figure object
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        cluster_colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(clusters))))
        row_colors = [cluster_colors[cluster] for cluster in clusters]
        cg = sns.clustermap(correlation_matrix, annot=True, cmap='coolwarm', center=0, xticklabels=feature_names, yticklabels=feature_names, figsize=self.figsize, row_colors=row_colors, col_colors=row_colors)
        cg.fig.suptitle(f'Feature Clustering Map ({self.correlation_method.capitalize()} Correlation)', y=1.02)
        return cg.fig

    def _generate_network(self, correlation_matrix: np.ndarray, clusters: np.ndarray, feature_names: list, fig: plt.Figure, ax: plt.Axes) -> plt.Figure:
        """
        Generate a network visualization.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix of shape (n_features, n_features)
        clusters : np.ndarray
            Cluster assignments for each feature
        feature_names : list
            Names of the features
        fig : plt.Figure
            Matplotlib figure object
        ax : plt.Axes
            Matplotlib axes object
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        try:
            import networkx as nx
            G = nx.Graph()
            unique_clusters = np.unique(clusters)
            cluster_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
            color_map = {cluster: cluster_colors[i] for (i, cluster) in enumerate(unique_clusters)}
            for (i, feature_name) in enumerate(feature_names):
                G.add_node(feature_name, cluster=clusters[i], color=color_map[clusters[i]])
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    corr_val = abs(correlation_matrix[i, j])
                    if corr_val >= self.threshold:
                        G.add_edge(feature_names[i], feature_names[j], weight=corr_val)
            pos = nx.spring_layout(G, seed=42)
            node_colors = [G.nodes[node]['color'] for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax, node_size=500)
            edges = G.edges()
            weights = [G[u][v]['weight'] for (u, v) in edges]
            nx.draw_networkx_edges(G, pos, width=np.array(weights) * 2, ax=ax, alpha=0.7)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            ax.set_title(f'Feature Correlation Network ({self.correlation_method.capitalize()} Correlation)')
            ax.axis('off')
        except ImportError:
            ax.text(0.5, 0.5, 'Network visualization requires networkx package', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Network Visualization Unavailable')
        return fig