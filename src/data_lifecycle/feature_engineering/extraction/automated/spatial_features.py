from typing import Optional, List, Union
import numpy as np
from scipy.spatial.distance import cdist
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet


class SpatialFeatureExtractor(BaseTransformer):

    def __init__(self, coordinate_columns: Optional[List[Union[str, int]]]=None, distance_metrics: List[str]=None, geometric_features: bool=True, spatial_relations: bool=True, name: Optional[str]=None):
        """
        Initialize the SpatialFeatureExtractor.
        
        Parameters
        ----------
        coordinate_columns : Optional[List[Union[str, int]]]
            Names or indices of columns containing coordinate data. If None, assumes first 2 or 3 columns.
        distance_metrics : List[str]
            Distance metrics to compute. Defaults to ['euclidean'].
        geometric_features : bool
            Whether to compute geometric properties of point sets.
        spatial_relations : bool
            Whether to compute spatial relationship features.
        name : Optional[str]
            Name of the transformer instance.
        """
        super().__init__(name)
        self.coordinate_columns = coordinate_columns
        self.distance_metrics = distance_metrics or ['euclidean']
        self.geometric_features = geometric_features
        self.spatial_relations = spatial_relations

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'SpatialFeatureExtractor':
        """
        Fit the transformer to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing coordinate information
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        SpatialFeatureExtractor
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            features = data.features
            feature_names = data.feature_names
        else:
            features = data
            feature_names = None
        if features.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        n_features = features.shape[1]
        if self.coordinate_columns is None:
            if n_features < 2:
                raise ValueError('At least 2 features required for spatial analysis')
            coord_dim = min(3, n_features)
            if feature_names:
                self.coordinate_columns = feature_names[:coord_dim]
            else:
                self.coordinate_columns = list(range(coord_dim))
        else:
            self.coordinate_columns = self.coordinate_columns
        supported_metrics = {'euclidean', 'manhattan', 'haversine', 'cosine'}
        for metric in self.distance_metrics:
            if metric not in supported_metrics:
                raise ValueError(f'Unsupported distance metric: {metric}. Supported metrics are: {supported_metrics}')
        if isinstance(self.coordinate_columns, list):
            self.n_coordinates = len(self.coordinate_columns)
        else:
            self.n_coordinates = 2
        if self.n_coordinates < 2 or self.n_coordinates > 3:
            raise ValueError('Number of coordinate dimensions must be 2 or 3')
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Extract spatial features from the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing coordinate information
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            FeatureSet with extracted spatial features
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError('Transformer must be fitted before transform')
        if isinstance(data, FeatureSet):
            features = data.features
            feature_names = data.feature_names
        else:
            features = data
            feature_names = [f'feature_{i}' for i in range(features.shape[1])] if features.ndim == 2 else []
        if isinstance(self.coordinate_columns[0], str):
            coord_indices = [feature_names.index(col) for col in self.coordinate_columns]
        else:
            coord_indices = self.coordinate_columns
        coordinates = features[:, coord_indices]
        new_features = []
        new_feature_names = []
        if self.distance_metrics:
            for metric in self.distance_metrics:
                if metric == 'euclidean':
                    dists = cdist(coordinates, coordinates, metric='euclidean')
                elif metric == 'manhattan':
                    dists = cdist(coordinates, coordinates, metric='cityblock')
                elif metric == 'haversine':
                    if self.n_coordinates != 2:
                        raise ValueError('Haversine distance requires 2D coordinates (lat/lon)')
                    coords_rad = np.radians(coordinates)
                    dists = cdist(coords_rad, coords_rad, metric='haversine')
                elif metric == 'cosine':
                    dists = cdist(coordinates, coordinates, metric='cosine')
                np.fill_diagonal(dists, np.inf)
                new_features.append(np.min(dists, axis=1))
                new_feature_names.append(f'{metric}_min_distance')
                new_features.append(np.max(dists, axis=1))
                new_feature_names.append(f'{metric}_max_distance')
                new_features.append(np.mean(dists, axis=1))
                new_feature_names.append(f'{metric}_mean_distance')
        if self.geometric_features:
            if self.n_coordinates == 2:
                centroid = np.mean(coordinates, axis=0)
                centroid_dists = np.sqrt(np.sum((coordinates - centroid) ** 2, axis=1))
                new_features.append(centroid_dists)
                new_feature_names.append('centroid_distance')
                x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
                y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
                bbox_areas = np.full(coordinates.shape[0], x_range * y_range)
                new_features.append(bbox_areas)
                new_feature_names.append('bounding_box_area')
            elif self.n_coordinates == 3:
                centroid = np.mean(coordinates, axis=0)
                centroid_dists = np.sqrt(np.sum((coordinates - centroid) ** 2, axis=1))
                new_features.append(centroid_dists)
                new_feature_names.append('centroid_distance')
                x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
                y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
                z_range = np.max(coordinates[:, 2]) - np.min(coordinates[:, 2])
                bbox_volumes = np.full(coordinates.shape[0], x_range * y_range * z_range)
                new_features.append(bbox_volumes)
                new_feature_names.append('bounding_box_volume')
        if self.spatial_relations and coordinates.shape[0] > 1:
            if self.n_coordinates >= 2:
                ref_point = np.mean(coordinates, axis=0)
                north = (coordinates[:, 1] > ref_point[1]).astype(float)
                east = (coordinates[:, 0] > ref_point[0]).astype(float)
                new_features.append(north)
                new_feature_names.append('is_north')
                new_features.append(east)
                new_feature_names.append('is_east')
                if self.n_coordinates == 3:
                    above = (coordinates[:, 2] > ref_point[2]).astype(float)
                    new_features.append(above)
                    new_feature_names.append('is_above')
        if new_features:
            new_features_array = np.column_stack(new_features)
            all_features = np.hstack([features, new_features_array])
            all_feature_names = feature_names + new_feature_names
        else:
            all_features = features
            all_feature_names = feature_names
        return FeatureSet(all_features, all_feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation (not applicable for feature extraction).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Original data format (identity transformation)
        """
        if isinstance(data, FeatureSet):
            return data
        else:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])] if data.ndim == 2 else []
            return FeatureSet(data, feature_names)

    def get_feature_names(self) -> List[str]:
        """
        Get names of generated features.
        
        Returns
        -------
        List[str]
            List of feature names that will be generated
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            return []
        if not self.distance_metrics and (not self.geometric_features) and (not self.spatial_relations):
            return []
        feature_names = []
        for metric in self.distance_metrics:
            feature_names.extend([f'{metric}_min_distance', f'{metric}_max_distance', f'{metric}_mean_distance'])
        if self.geometric_features:
            if self.n_coordinates == 2:
                feature_names.extend(['centroid_distance', 'bounding_box_area'])
            elif self.n_coordinates == 3:
                feature_names.extend(['centroid_distance', 'bounding_box_volume'])
        if self.spatial_relations:
            if self.n_coordinates >= 2:
                feature_names.extend(['is_north', 'is_east'])
            if self.n_coordinates == 3:
                feature_names.append('is_above')
        return feature_names