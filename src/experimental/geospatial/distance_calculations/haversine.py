import numpy as np
from typing import Union, Optional
from general.structures.data_batch import DataBatch

def compute_haversine_distances(lat1: Union[float, np.ndarray, DataBatch], lon1: Union[float, np.ndarray, DataBatch], lat2: Union[float, np.ndarray, DataBatch], lon2: Union[float, np.ndarray, DataBatch], earth_radius_km: float=6371.0, in_batch: Optional[DataBatch]=None) -> Union[float, np.ndarray, DataBatch]:
    """
    Compute the Haversine distance between two sets of geographic coordinates.

    The Haversine formula calculates the great-circle distance between two points on the Earth's surface
    given their latitude and longitude in decimal degrees. This function supports both scalar and array inputs,
    as well as DataBatch objects for batch processing.

    Args:
        lat1 (Union[float, np.ndarray, DataBatch]): Latitude of the first point(s) in decimal degrees.
        lon1 (Union[float, np.ndarray, DataBatch]): Longitude of the first point(s) in decimal degrees.
        lat2 (Union[float, np.ndarray, DataBatch]): Latitude of the second point(s) in decimal degrees.
        lon2 (Union[float, np.ndarray, DataBatch]): Longitude of the second point(s) in decimal degrees.
        earth_radius_km (float): Radius of the Earth in kilometers. Defaults to 6371.0 km.
        in_batch (Optional[DataBatch]): Optional input DataBatch to extract coordinates from. If provided,
                                       lat/lon arguments are treated as keys or indices to access coordinate data.

    Returns:
        Union[float, np.ndarray, DataBatch]: Distance between points in kilometers. Type matches input type:
                                            - float for scalar inputs
                                            - np.ndarray for array inputs
                                            - DataBatch for batch inputs

    Raises:
        ValueError: If input dimensions do not match or invalid coordinates are provided.
        TypeError: If unsupported input types are provided.
    """

    def extract_coord(coord_key, in_batch_obj):
        """Extract coordinate data from various input types."""
        if isinstance(coord_key, str):
            if in_batch_obj.metadata and coord_key in in_batch_obj.metadata:
                return in_batch_obj.metadata[coord_key]
            elif hasattr(in_batch_obj.data, '__getitem__'):
                return in_batch_obj.data[coord_key]
            else:
                raise ValueError(f"Cannot extract '{coord_key}' from DataBatch")
        elif isinstance(coord_key, int):
            if hasattr(in_batch_obj.data, '__getitem__'):
                return in_batch_obj.data[coord_key]
            else:
                raise ValueError(f'Cannot extract index {coord_key} from DataBatch')
        else:
            return coord_key
    if in_batch is not None:
        lat1_vals = extract_coord(lat1, in_batch)
        lon1_vals = extract_coord(lon1, in_batch)
        lat2_vals = extract_coord(lat2, in_batch)
        lon2_vals = extract_coord(lon2, in_batch)
    else:
        lat1_vals = lat1
        lon1_vals = lon1
        lat2_vals = lat2
        lon2_vals = lon2
    is_output_databatch = isinstance(lat1_vals, DataBatch) or isinstance(lon1_vals, DataBatch) or isinstance(lat2_vals, DataBatch) or isinstance(lon2_vals, DataBatch)
    original_input_type = None
    if not is_output_databatch:
        if not isinstance(lat1_vals, DataBatch):
            original_input_type = type(lat1_vals)
        elif not isinstance(lon1_vals, DataBatch):
            original_input_type = type(lon1_vals)
        elif not isinstance(lat2_vals, DataBatch):
            original_input_type = type(lat2_vals)
        elif not isinstance(lon2_vals, DataBatch):
            original_input_type = type(lon2_vals)
    if isinstance(lat1_vals, DataBatch):
        lat1_vals = lat1_vals.data
        if original_input_type is None:
            original_input_type = type(lat1_vals)
    if isinstance(lon1_vals, DataBatch):
        lon1_vals = lon1_vals.data
        if original_input_type is None:
            original_input_type = type(lon1_vals)
    if isinstance(lat2_vals, DataBatch):
        lat2_vals = lat2_vals.data
        if original_input_type is None:
            original_input_type = type(lat2_vals)
    if isinstance(lon2_vals, DataBatch):
        lon2_vals = lon2_vals.data
        if original_input_type is None:
            original_input_type = type(lon2_vals)
    try:
        lat1_arr = np.asarray(lat1_vals, dtype=float)
        lon1_arr = np.asarray(lon1_vals, dtype=float)
        lat2_arr = np.asarray(lat2_vals, dtype=float)
        lon2_arr = np.asarray(lon2_vals, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError('All coordinate values must be convertible to float')
    if np.any(np.abs(lat1_arr) > 90) or np.any(np.abs(lat2_arr) > 90):
        raise ValueError('Latitude values must be between -90 and 90 degrees')
    if np.any(np.abs(lon1_arr) > 180) or np.any(np.abs(lon2_arr) > 180):
        raise ValueError('Longitude values must be between -180 and 180 degrees')
    try:
        (lat1_broadcast, lon1_broadcast, lat2_broadcast, lon2_broadcast) = np.broadcast_arrays(lat1_arr, lon1_arr, lat2_arr, lon2_arr)
    except ValueError as e:
        raise ValueError('Coordinate arrays cannot be broadcast to compatible shapes')
    lat1_rad = np.radians(lat1_broadcast)
    lon1_rad = np.radians(lon1_broadcast)
    lat2_rad = np.radians(lat2_broadcast)
    lon2_rad = np.radians(lon2_broadcast)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = earth_radius_km * c
    if is_output_databatch:
        return DataBatch(data=distances)
    elif original_input_type == float and distances.ndim == 0:
        return float(distances)
    else:
        return distances