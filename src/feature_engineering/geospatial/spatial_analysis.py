from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, Union, Tuple, List
from general.base_classes.transformer_base import BaseTransformer
from scipy.spatial import Voronoi
import warnings
import pandas as pd


# ...(code omitted)...


class VoronoiDiagramTransformer(BaseTransformer):

    def __init__(self, compute_area: bool=True, compute_perimeter: bool=True, compute_vertex_count: bool=True, compute_centroid_distances: bool=True, name: Optional[str]=None):
        super().__init__(name)
        self.compute_area = compute_area
        self.compute_perimeter = compute_perimeter
        self.compute_vertex_count = compute_vertex_count
        self.compute_centroid_distances = compute_centroid_distances

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'VoronoiDiagramTransformer':
        """
        Fit the Voronoi diagram transformer to the input data.

        This method constructs the Voronoi diagram from the provided coordinates
        and prepares for feature extraction.

        Args:
            data (Union[FeatureSet, np.ndarray]): Input data containing latitude and longitude features.
                Expected shape is (n_samples, 2) where columns represent [latitude, longitude].
            **kwargs: Additional keyword arguments for fitting.

        Returns:
            VoronoiDiagramTransformer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            coords = data.features
            if hasattr(coords, 'values'):
                coords = coords.values
        elif isinstance(data, np.ndarray):
            coords = data
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError('Input data must have shape (n_samples, 2)')
        if len(coords) < 3:
            raise ValueError('At least 3 points are required to compute a Voronoi diagram')
        self.coords_ = coords.copy()
        self._voronoi_diagram = Voronoi(coords)
        self._compute_cell_centroids()
        return self

    def _compute_cell_centroids(self):
        """Compute centroids of all Voronoi cells."""
        centroids = []
        for region_idx in self._voronoi_diagram.point_region:
            region = self._voronoi_diagram.regions[region_idx]
            if not region or -1 in region:
                centroids.append([np.nan, np.nan])
            else:
                vertices = self._voronoi_diagram.vertices[region]
                if len(vertices) > 0:
                    centroid = np.mean(vertices, axis=0)
                    centroids.append(centroid)
                else:
                    centroids.append([np.nan, np.nan])
        self.cell_centroids_ = np.array(centroids)
        return self.cell_centroids_

    def _get_finite_vertices(self, point_idx: int) -> np.ndarray:
        """Get finite vertices of a Voronoi cell, handling infinite ridges."""
        region_idx = self._voronoi_diagram.point_region[point_idx]
        region = self._voronoi_diagram.regions[region_idx]
        if not region:
            return np.empty((0, 2))
        if -1 in region:
            return self._approximate_infinite_region(point_idx, region)
        return self._voronoi_diagram.vertices[region]

    def _approximate_infinite_region(self, point_idx: int, region: List[int]) -> np.ndarray:
        """Approximate infinite Voronoi regions with a bounding box."""
        margin = 0.2
        min_coords = np.min(self.coords_, axis=0)
        max_coords = np.max(self.coords_, axis=0)
        range_coords = max_coords - min_coords
        min_coords -= range_coords * margin
        max_coords += range_coords * margin
        finite_vertices = []
        for vertex_idx in region:
            if vertex_idx != -1:
                finite_vertices.append(self._voronoi_diagram.vertices[vertex_idx])
        if not finite_vertices:
            return np.empty((0, 2))
        vertices = np.array(finite_vertices)
        clipped_vertices = np.clip(vertices, min_coords, max_coords)
        return clipped_vertices

    def _compute_polygon_area(self, vertices: np.ndarray) -> float:
        """Compute area of a polygon given its vertices using the shoelace formula."""
        if len(vertices) < 3:
            return 0.0
        vertices = np.vstack([vertices, vertices[0]])
        x = vertices[:, 0]
        y = vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        return area

    def _compute_polygon_perimeter(self, vertices: np.ndarray) -> float:
        """Compute perimeter of a polygon given its vertices."""
        if len(vertices) < 2:
            return 0.0
        vertices = np.vstack([vertices, vertices[0]])
        diffs = np.diff(vertices, axis=0)
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.sum(distances)

    def _find_neighbors(self, point_idx: int) -> List[int]:
        """Find neighboring points in the Voronoi diagram."""
        neighbors = []
        for ridge_points in self._voronoi_diagram.ridge_points:
            if point_idx in ridge_points:
                neighbor_idx = ridge_points[1] if ridge_points[0] == point_idx else ridge_points[0]
                neighbors.append(neighbor_idx)
        return neighbors

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Generate Voronoi diagram-based features.

        Computes geometric properties of Voronoi cells corresponding to each input point.
        Features are appended to the original dataset, preserving sample alignment.

        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform. Must have same
                format as in `fit`.
            **kwargs: Additional keyword arguments for transformation.

        Returns:
            Union[FeatureSet, np.ndarray]: Transformed data with added Voronoi-based features including:
                - voronoi_area: Area of the Voronoi cell (if enabled)
                - voronoi_perimeter: Perimeter of the Voronoi cell (if enabled)
                - vertex_count: Number of vertices defining the cell (if enabled)
                - avg_centroid_distance: Average distance to neighboring cell centroids (if enabled)
        """
        if isinstance(data, FeatureSet):
            coords = data.features
            feature_names = list(data.feature_names) if data.feature_names else [f'feature_{i}' for i in range(coords.shape[1])]
            is_feature_set = True
        else:
            coords = data
            feature_names = [f'feature_{i}' for i in range(coords.shape[1])]
            is_feature_set = False
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError('Input data must have shape (n_samples, 2)')
        n_samples = coords.shape[0]
        features_to_add = []
        feature_names_to_add = []
        if self.compute_area:
            areas = np.zeros(n_samples)
            feature_names_to_add.append('voronoi_area')
        if self.compute_perimeter:
            perimeters = np.zeros(n_samples)
            feature_names_to_add.append('voronoi_perimeter')
        if self.compute_vertex_count:
            vertex_counts = np.zeros(n_samples)
            feature_names_to_add.append('vertex_count')
        if self.compute_centroid_distances:
            avg_distances = np.full(n_samples, np.nan)
            feature_names_to_add.append('avg_centroid_distance')
        for i in range(n_samples):
            vertices = self._get_finite_vertices(i)
            if self.compute_area:
                areas[i] = self._compute_polygon_area(vertices)
            if self.compute_perimeter:
                perimeters[i] = self._compute_polygon_perimeter(vertices)
            if self.compute_vertex_count:
                vertex_counts[i] = len(vertices)
            if self.compute_centroid_distances:
                neighbors = self._find_neighbors(i)
                if neighbors and (not np.isnan(self.cell_centroids_[i]).any()):
                    neighbor_centroids = self.cell_centroids_[neighbors]
                    valid_centroids = neighbor_centroids[~np.isnan(neighbor_centroids).any(axis=1)]
                    if len(valid_centroids) > 0:
                        distances = np.sqrt(np.sum((valid_centroids - self.cell_centroids_[i]) ** 2, axis=1))
                        avg_distances[i] = np.mean(distances)
        computed_features = []
        if self.compute_area:
            computed_features.append(areas.reshape(-1, 1))
        if self.compute_perimeter:
            computed_features.append(perimeters.reshape(-1, 1))
        if self.compute_vertex_count:
            computed_features.append(vertex_counts.reshape(-1, 1))
        if self.compute_centroid_distances:
            computed_features.append(avg_distances.reshape(-1, 1))
        if computed_features:
            computed_features = np.hstack(computed_features)
        else:
            computed_features = np.empty((n_samples, 0))
        combined_features = np.hstack([coords, computed_features])
        combined_feature_names = feature_names + feature_names_to_add
        if is_feature_set:
            return FeatureSet(features=combined_features, feature_names=combined_feature_names, feature_types=data.feature_types + ['numeric'] * len(feature_names_to_add) if data.feature_types else None, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        else:
            return FeatureSet(features=combined_features, feature_names=combined_feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for Voronoi features.

        Voronoi diagram features are derived geometric properties and cannot be
        meaningfully inverted to recover original coordinates.

        Args:
            data (Union[FeatureSet, np.ndarray]): Data to inverse transform.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[FeatureSet, np.ndarray]: Returns the input data unchanged.
        """
        warnings.warn('Inverse transform is not supported for Voronoi diagram features. Returning input unchanged.')
        return data


# ...(code omitted)...


def compute_spatial_autocorrelation(coordinates: Union[np.ndarray, FeatureSet], values: Union[np.ndarray, str], method: str='moran', k: int=5, binary_weights: bool=True) -> float:
    """
    Compute a spatial autocorrelation statistic for geographic data.

    This function calculates a measure of spatial autocorrelation, quantifying
    the degree to which nearby locations resemble each other in terms of a variable of interest.
    Supported statistics include Moran's I and Geary's C, which help identify
    spatial patterns such as clustering or dispersion.

    The computation involves constructing spatial weights based on k-nearest neighbors
    and comparing the variable's spatial distribution to random permutations.

    Args:
        coordinates (Union[np.ndarray, FeatureSet]): Location data for observations.
            If ndarray, expected shape is (n_samples, 2) representing [latitude, longitude].
            If FeatureSet, must contain coordinate features.
        values (Union[np.ndarray, str]): Target values for autocorrelation analysis.
            If ndarray, must align with coordinates' sample count.
            If string, refers to a column name in the FeatureSet.
        method (str): The autocorrelation measure to compute ('moran', 'geary').
            Defaults to 'moran'.
        k (int): Number of nearest neighbors to consider for weight matrix construction.
            Defaults to 5.
        binary_weights (bool): Whether to use binary (0/1) or continuous distance-based weights.
            Defaults to True.

    Returns:
        float: Computed autocorrelation statistic:
            - Moran's I: Ranges from approximately -1 (dispersion) to +1 (clustering)
            - Geary's C: Ranges from 0 (positive autocorrelation) to 2 (negative autocorrelation)

    Raises:
        ValueError: If input dimensions mismatch or unsupported method is specified.
        TypeError: If inputs are not of expected types.
    """
    if isinstance(coordinates, FeatureSet):
        coord_data = coordinates.features
        if coord_data.shape[1] != 2:
            raise ValueError('Coordinates in FeatureSet must have exactly 2 columns (lat, lon).')
    elif isinstance(coordinates, np.ndarray):
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError('Coordinates array must have shape (n_samples, 2).')
        coord_data = coordinates
    else:
        raise TypeError('Coordinates must be either a numpy array or a FeatureSet.')
    n_samples = coord_data.shape[0]
    if isinstance(values, str):
        if not isinstance(coordinates, FeatureSet):
            raise TypeError('Column name for values can only be used with FeatureSet coordinates.')
        if values not in coordinates.targets.columns:
            raise ValueError(f"Target column '{values}' not found in FeatureSet.")
        value_data = coordinates.targets[values].values
    elif isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError('Values array must be 1-dimensional.')
        if len(values) != n_samples:
            raise ValueError('Length of values array must match number of coordinate samples.')
        value_data = values
    else:
        raise TypeError('Values must be either a numpy array or a string column name.')
    if method not in ['moran', 'geary']:
        raise ValueError("Method must be either 'moran' or 'geary'.")
    if k >= n_samples:
        raise ValueError('k must be less than the number of samples.')
    diff = coord_data[:, np.newaxis, :] - coord_data[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    knn_indices = np.argpartition(distances, k, axis=1)[:, :k]
    W = np.zeros((n_samples, n_samples))
    if binary_weights:
        for i in range(n_samples):
            W[i, knn_indices[i]] = 1
    else:
        for i in range(n_samples):
            neighbor_distances = distances[i, knn_indices[i]]
            weights = 1.0 / (neighbor_distances + 1e-10)
            W[i, knn_indices[i]] = weights
    z = value_data - np.mean(value_data)
    S0 = np.sum(W)
    if method == 'moran':
        numerator = np.sum(W * np.outer(z, z))
        denominator = np.sum(z ** 2)
        morans_i = n_samples / S0 * (numerator / denominator)
        return morans_i
    elif method == 'geary':
        diff_matrix = z[:, np.newaxis] - z[np.newaxis, :]
        numerator = np.sum(W * diff_matrix ** 2)
        denominator = np.sum(z ** 2)
        gearys_c = (n_samples - 1) / (2 * S0) * (numerator / denominator)
        return gearys_c