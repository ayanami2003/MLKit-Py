from typing import Optional, Union, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from general.structures.data_batch import DataBatch

def visualize_networks(adjacency_data: Union[np.ndarray, DataBatch], node_labels: Optional[list]=None, edge_weights: Optional[np.ndarray]=None, layout: str='spring', node_size: Union[int, list]=300, node_color: Union[str, list]='blue', edge_color: Union[str, list]='gray', figsize: tuple=(8, 6), title: Optional[str]=None, ax: Optional[plt.Axes]=None, **layout_params: Dict[str, Any]) -> plt.Figure:
    """
    Visualize network graphs from adjacency data or edge lists.

    This function creates a graphical representation of network data using various layout algorithms
    and customization options. It supports both dense adjacency matrices and sparse representations
    for efficient rendering of complex networks.

    Args:
        adjacency_data (Union[np.ndarray, DataBatch]): 
            Network data represented as an adjacency matrix (2D array) or DataBatch containing edges.
            If DataBatch, expects data attribute to contain edge list as [source, target] pairs.
        node_labels (Optional[list]): 
            Labels for each node in the network. Length must match number of nodes.
        edge_weights (Optional[np.ndarray]): 
            Weights for edges to control thickness or opacity in visualization.
        layout (str): 
            Layout algorithm for node positioning ('spring', 'circular', 'random', 'shell', 'spectral').
        node_size (Union[int, list]): 
            Size of nodes in the visualization. Can be uniform or per-node.
        node_color (Union[str, list]): 
            Color of nodes. Can be uniform or specified per node.
        edge_color (Union[str, list]): 
            Color of edges. Can be uniform or specified per edge.
        figsize (tuple): 
            Figure size as (width, height) in inches.
        title (Optional[str]): 
            Title for the network visualization plot.
        ax (Optional[plt.Axes]): 
            Matplotlib axes object to draw on. If None, creates new figure.
        **layout_params (Dict[str, Any]): 
            Additional parameters passed to the layout algorithm (e.g., k for spring layout).

    Returns:
        plt.Figure: 
            Matplotlib figure object containing the network visualization.

    Raises:
        ValueError: 
            If adjacency_data dimensions don't match node_labels or if layout is unsupported.
        TypeError: 
            If adjacency_data is not a valid type for network representation.
    """
    G = nx.Graph()
    if isinstance(adjacency_data, np.ndarray):
        if adjacency_data.ndim != 2 or adjacency_data.shape[0] != adjacency_data.shape[1]:
            raise ValueError('Adjacency matrix must be a square 2D array')
        n_nodes = adjacency_data.shape[0]
        G.add_nodes_from(range(n_nodes))
        edges_added = []
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                if adjacency_data[i, j] != 0:
                    G.add_edge(i, j, weight=adjacency_data[i, j])
                    edges_added.append((i, j))
        if edge_weights is not None:
            if edge_weights.ndim == 1:
                if len(edge_weights) != len(edges_added):
                    raise ValueError(f'Length of edge_weights ({len(edge_weights)}) must match number of edges ({len(edges_added)})')
                for (idx, (u, v)) in enumerate(edges_added):
                    G[u][v]['weight'] = edge_weights[idx]
            elif edge_weights.ndim == 2:
                if edge_weights.shape != adjacency_data.shape:
                    raise ValueError(f'Shape of 2D edge_weights {edge_weights.shape} must match adjacency matrix shape {adjacency_data.shape}')
                for (u, v) in edges_added:
                    G[u][v]['weight'] = edge_weights[u, v]
            else:
                raise ValueError('edge_weights must be 1D or 2D array')
    elif isinstance(adjacency_data, DataBatch):
        if not hasattr(adjacency_data, 'data') or adjacency_data.data is None:
            raise TypeError('DataBatch must contain data attribute')
        edge_data = adjacency_data.data
        if isinstance(edge_data, np.ndarray):
            if edge_data.ndim != 2 or edge_data.shape[1] < 2:
                raise ValueError('Edge list in DataBatch must be a 2D array with at least 2 columns')
            flat_edges = edge_data.flatten()
            n_nodes = int(np.max(flat_edges)) + 1 if len(flat_edges) > 0 else 0
            G.add_nodes_from(range(n_nodes))
            if edge_weights is not None and len(edge_weights) != len(edge_data):
                raise ValueError(f'Length of edge_weights ({len(edge_weights)}) must match number of edges ({len(edge_data)})')
            for (idx, edge) in enumerate(edge_data):
                (source, target) = (int(edge[0]), int(edge[1]))
                if source >= n_nodes or target >= n_nodes or source < 0 or (target < 0):
                    raise ValueError(f'Edge ({source}, {target}) contains invalid node indices')
                weight = 1.0
                if edge_weights is not None and len(edge_weights) > idx:
                    try:
                        weight = edge_weights[idx]
                    except (IndexError, TypeError):
                        weight = 1.0
                G.add_edge(source, target, weight=weight)
        elif isinstance(edge_data, list):
            if len(edge_data) == 0:
                n_nodes = 0
            else:
                try:
                    edge_array = np.array(edge_data)
                    if edge_array.ndim != 2 or edge_array.shape[1] < 2:
                        raise ValueError('Edge list in DataBatch must be a 2D array with at least 2 columns')
                    flat_edges = edge_array.flatten()
                    n_nodes = int(np.max(flat_edges)) + 1 if len(flat_edges) > 0 else 0
                    G.add_nodes_from(range(n_nodes))
                    if edge_weights is not None and len(edge_weights) != len(edge_array):
                        raise ValueError(f'Length of edge_weights ({len(edge_weights)}) must match number of edges ({len(edge_array)})')
                    for (idx, edge) in enumerate(edge_array):
                        (source, target) = (int(edge[0]), int(edge[1]))
                        if source >= n_nodes or target >= n_nodes or source < 0 or (target < 0):
                            raise ValueError(f'Edge ({source}, {target}) contains invalid node indices')
                        weight = 1.0
                        if edge_weights is not None and len(edge_weights) > idx:
                            try:
                                weight = edge_weights[idx]
                            except (IndexError, TypeError):
                                weight = 1.0
                        G.add_edge(source, target, weight=weight)
                except (ValueError, TypeError):
                    raise ValueError('Edge list in DataBatch must be convertible to a 2D array with at least 2 columns')
        else:
            raise TypeError('DataBatch data must be a numpy array or list')
    else:
        raise TypeError('adjacency_data must be either a numpy array or DataBatch')
    if node_labels is not None:
        if len(node_labels) != G.number_of_nodes():
            raise ValueError(f'Length of node_labels ({len(node_labels)}) must match number of nodes ({G.number_of_nodes()})')
        label_mapping = dict(zip(G.nodes(), node_labels))
    else:
        label_mapping = None
    layout_algorithms = {'spring': nx.spring_layout, 'circular': nx.circular_layout, 'random': nx.random_layout, 'shell': nx.shell_layout, 'spectral': nx.spectral_layout}
    if layout not in layout_algorithms:
        raise ValueError(f'Unsupported layout: {layout}. Supported layouts are: {list(layout_algorithms.keys())}')
    pos = layout_algorithms[layout](G, **layout_params)
    if ax is None:
        (fig, ax) = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    edges = G.edges()
    weights = [G[u][v].get('weight', 1.0) for (u, v) in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color=edge_color, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, ax=ax)
    if label_mapping is not None:
        nx.draw_networkx_labels(G, pos, labels=label_mapping, ax=ax)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    return fig