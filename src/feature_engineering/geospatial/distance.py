from typing import Union, Optional
from general.structures.data_batch import DataBatch
import numpy as np

def calculate_haversine_distance(lat1: Union[float, np.ndarray], lon1: Union[float, np.ndarray], lat2: Union[float, np.ndarray], lon2: Union[float, np.ndarray], earth_radius: float=6371.0) -> Union[float, np.ndarray]:
    """
    Calculate the great-circle distance between two points on the Earth's surface using the Haversine formula.

    This function computes the shortest distance over the Earth's surface between pairs of coordinates,
    accounting for the spherical geometry of the planet. Inputs can be scalar values for a single pair
    of coordinates or arrays for batch computations.

    Args:
        lat1 (Union[float, np.ndarray]): Latitude of the first point(s) in decimal degrees.
        lon1 (Union[float, np.ndarray]): Longitude of the first point(s) in decimal degrees.
        lat2 (Union[float, np.ndarray]): Latitude of the second point(s) in decimal degrees.
        lon2 (Union[float, np.ndarray]): Longitude of the second point(s) in decimal degrees.
        earth_radius (float): Radius of the Earth in kilometers. Defaults to 6371.0 km.

    Returns:
        Union[float, np.ndarray]: Distance between the points in kilometers. Returns a scalar if inputs
                                  were scalars, or an array of distances if inputs were arrays.

    Raises:
        ValueError: If input arrays have mismatched shapes.
    """
    all_scalars = all((np.isscalar(x) for x in [lat1, lon1, lat2, lon2]))
    lat1_arr = np.asarray(lat1)
    lon1_arr = np.asarray(lon1)
    lat2_arr = np.asarray(lat2)
    lon2_arr = np.asarray(lon2)
    try:
        (lat1_broadcast, lon1_broadcast, lat2_broadcast, lon2_broadcast) = np.broadcast_arrays(lat1_arr, lon1_arr, lat2_arr, lon2_arr)
    except ValueError:
        raise ValueError('Input arrays must have compatible shapes for broadcasting')
    lat1_rad = np.radians(lat1_broadcast)
    lon1_rad = np.radians(lon1_broadcast)
    lat2_rad = np.radians(lat2_broadcast)
    lon2_rad = np.radians(lon2_broadcast)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = earth_radius * c
    if all_scalars:
        return float(distance)
    return distance


# ...(code omitted)...


def calculate_distance_to_centroid(points: Union[np.ndarray, DataBatch], centroid: Optional[np.ndarray]=None, method: str='haversine') -> Union[np.ndarray, DataBatch]:
    """
    Calculate the distance from each point in a dataset to a specified centroid.

    This function computes distances from a collection of geographic points to a central reference point.
    If no centroid is provided, the geographic center of the input points is computed and used. The method
    parameter allows selecting the distance calculation approach, with 'haversine' as the default for
    geographic coordinates.

    Args:
        points (Union[np.ndarray, DataBatch]): Array of shape (n_samples, 2) with longitude and latitude
                                               coordinates, or a DataBatch containing such data.
        centroid (Optional[np.ndarray]): Centroid coordinates as [longitude, latitude]. If None, the
                                        geographic centroid of the points is calculated.
        method (str): Distance calculation method ('haversine', 'euclidean'). Defaults to 'haversine'.

    Returns:
        Union[np.ndarray, DataBatch]: Array of distances for each point, or a DataBatch with distances
                                      added to metadata if input was a DataBatch.

    Raises:
        ValueError: If points array does not have the correct shape or if an unsupported method is specified.
    """
    is_databatch = isinstance(points, DataBatch)
    if is_databatch:
        data = points.data
    else:
        data = points
    if not isinstance(data, np.ndarray):
        raise ValueError('Input data must be a numpy array')
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError('Points array must have shape (n_samples, 2)')
    if data.shape[0] == 0:
        distances = np.array([])
        if is_databatch:
            result_batch = DataBatch(data=points.data, labels=points.labels, metadata=points.metadata.copy() if points.metadata else {}, sample_ids=points.sample_ids, feature_names=points.feature_names, batch_id=points.batch_id)
            result_batch.metadata['distances_to_centroid'] = distances
            return result_batch
        return distances
    lon_points = data[:, 0]
    lat_points = data[:, 1]
    if centroid is None:
        if method == 'haversine':
            centroid_lon = np.mean(lon_points)
            centroid_lat = np.mean(lat_points)
            centroid = np.array([centroid_lon, centroid_lat])
        else:
            centroid_lon = np.mean(lon_points)
            centroid_lat = np.mean(lat_points)
            centroid = np.array([centroid_lon, centroid_lat])
    if not isinstance(centroid, np.ndarray) or centroid.shape != (2,):
        raise ValueError('Centroid must be a numpy array of shape (2,)')
    (centroid_lon, centroid_lat) = (centroid[0], centroid[1])
    if method not in ['haversine', 'euclidean']:
        raise ValueError("Method must be either 'haversine' or 'euclidean'")
    if method == 'haversine':
        distances = calculate_haversine_distance(lat_points, lon_points, centroid_lat, centroid_lon)
    else:
        points_array = data
        centroid_array = centroid.reshape(1, -1)
        distances = np.sqrt(np.sum((points_array - centroid_array) ** 2, axis=1))
    if is_databatch:
        result_batch = DataBatch(data=points.data, labels=points.labels, metadata=points.metadata.copy() if points.metadata else {}, sample_ids=points.sample_ids, feature_names=points.feature_names, batch_id=points.batch_id)
        result_batch.metadata['distances_to_centroid'] = distances
        return result_batch
    else:
        return distances

def calculate_elevation_differences(elevations: Union[np.ndarray, DataBatch], reference_elevation: Optional[float]=None, method: str='absolute') -> Union[np.ndarray, DataBatch]:
    """
    Calculate elevation differences between points and a reference elevation.

    This function computes vertical differences for a set of elevation values compared to either a
    specified reference elevation or a computed statistic (e.g., mean) of the input elevations.
    Different methods allow for absolute differences or relative comparisons.

    Args:
        elevations (Union[np.ndarray, DataBatch]): Array of elevation values in meters, or a DataBatch
                                                  containing elevation data.
        reference_elevation (Optional[float]): Reference elevation for comparison. If None, a reference
                                              is computed based on the method parameter.
        method (str): Method for calculating differences ('absolute', 'relative', 'mean_diff').
                     Defaults to 'absolute'.

    Returns:
        Union[np.ndarray, DataBatch]: Array of elevation differences, or a DataBatch with differences
                                      added to metadata if input was a DataBatch.

    Raises:
        ValueError: If elevations array is empty or if an unsupported method is specified.
    """
    supported_methods = {'absolute', 'relative', 'mean_diff'}
    if method not in supported_methods:
        raise ValueError(f"Unsupported method '{method}'. Supported methods are: {supported_methods}")
    is_databatch = isinstance(elevations, DataBatch)
    if is_databatch:
        data = elevations.data
    else:
        data = elevations
    if not isinstance(data, np.ndarray):
        raise ValueError('Input data must be a numpy array')
    if data.ndim != 1:
        raise ValueError('Elevations array must be 1-dimensional')
    if data.size == 0:
        differences = np.array([])
        if is_databatch:
            result_batch = DataBatch(data=elevations.data, labels=elevations.labels, metadata=elevations.metadata.copy() if elevations.metadata else {}, sample_ids=elevations.sample_ids, feature_names=elevations.feature_names, batch_id=elevations.batch_id)
            result_batch.metadata['elevation_differences'] = differences
            return result_batch
        else:
            return differences
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        differences = np.full_like(data, np.nan, dtype=float)
        if is_databatch:
            result_batch = DataBatch(data=elevations.data, labels=elevations.labels, metadata=elevations.metadata.copy() if elevations.metadata else {}, sample_ids=elevations.sample_ids, feature_names=elevations.feature_names, batch_id=elevations.batch_id)
            result_batch.metadata['elevation_differences'] = differences
            return result_batch
        else:
            return differences
    if reference_elevation is None:
        valid_data = data[valid_mask]
        if method == 'absolute':
            ref_elevation = 0.0
        elif method == 'relative':
            ref_elevation = np.min(valid_data) if valid_data.size > 0 else 0.0
        elif method == 'mean_diff':
            ref_elevation = np.mean(valid_data) if valid_data.size > 0 else 0.0
    else:
        ref_elevation = reference_elevation
    differences = data.astype(float) - ref_elevation
    if is_databatch:
        result_batch = DataBatch(data=elevations.data, labels=elevations.labels, metadata=elevations.metadata.copy() if elevations.metadata else {}, sample_ids=elevations.sample_ids, feature_names=elevations.feature_names, batch_id=elevations.batch_id)
        result_batch.metadata['elevation_differences'] = differences
        return result_batch
    else:
        return differences