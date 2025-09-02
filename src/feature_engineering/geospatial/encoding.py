import numpy as np
from typing import Union, List, Optional
from general.structures.data_batch import DataBatch
import re
from typing import Union, List, Optional, Tuple

class GeohashEncoder:

    def __init__(self, precision: int=8):
        """
        Initialize the GeohashEncoder.

        Args:
            precision (int): The length of the geohash string, which determines
                             the granularity of the encoded location.
                             Higher values mean more precise locations.
                             Defaults to 8.
        """
        self.precision = precision
        self._base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
        self._decode_map = {char: i for (i, char) in enumerate(self._base32)}

    def encode(self, latitudes: Union[float, List[float], np.ndarray], longitudes: Union[float, List[float], np.ndarray]) -> Union[str, List[str]]:
        """
        Encode latitude and longitude coordinates into geohash strings.

        Args:
            latitudes (Union[float, List[float], np.ndarray]): Latitude value(s) in decimal degrees.
            longitudes (Union[float, List[float], np.ndarray]): Longitude value(s) in decimal degrees.

        Returns:
            Union[str, List[str]]: Geohash string(s) representing the input coordinates.
                                   Returns a single string if inputs are scalars,
                                   otherwise returns a list of strings.

        Raises:
            ValueError: If latitude or longitude values are out of valid ranges
                        or if input arrays have mismatched lengths.
        """
        if np.isscalar(latitudes):
            latitudes = np.array([latitudes])
            longitudes = np.array([longitudes])
            is_scalar = True
        else:
            latitudes = np.asarray(latitudes)
            longitudes = np.asarray(longitudes)
            is_scalar = False
        if len(latitudes) != len(longitudes):
            raise ValueError('Latitude and longitude arrays must have the same length')
        if np.any(latitudes < -90) or np.any(latitudes > 90):
            raise ValueError('All latitude values must be between -90 and 90 degrees')
        if np.any(longitudes < -180) or np.any(longitudes > 180):
            raise ValueError('All longitude values must be between -180 and 180 degrees')
        geohashes = []
        for (lat, lon) in zip(latitudes, longitudes):
            geohash = self._encode_single(lat, lon, self.precision)
            geohashes.append(geohash)
        return geohashes[0] if is_scalar else geohashes

    def _encode_single(self, latitude: float, longitude: float, precision: int) -> str:
        """
        Encode a single latitude/longitude pair into a geohash string.

        Args:
            latitude (float): Latitude in decimal degrees.
            longitude (float): Longitude in decimal degrees.
            precision (int): Length of the geohash string.

        Returns:
            str: Geohash string of specified precision.
        """
        (lat_min, lat_max) = (-90.0, 90.0)
        (lon_min, lon_max) = (-180.0, 180.0)
        geohash = ''
        bits = 0
        ch = 0
        even = True
        for i in range(precision * 5):
            if even:
                mid = (lon_min + lon_max) / 2
                if longitude >= mid:
                    ch = ch << 1 | 1
                    lon_min = mid
                else:
                    ch = ch << 1 | 0
                    lon_max = mid
            else:
                mid = (lat_min + lat_max) / 2
                if latitude >= mid:
                    ch = ch << 1 | 1
                    lat_min = mid
                else:
                    ch = ch << 1 | 0
                    lat_max = mid
            even = not even
            bits += 1
            if bits == 5:
                geohash += self._base32[ch]
                bits = 0
                ch = 0
        return geohash

    def decode(self, geohashes: Union[str, List[str]]) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """
        Decode geohash strings back into approximate latitude and longitude coordinates.

        Args:
            geohashes (Union[str, List[str]]): Geohash string(s) to decode.

        Returns:
            Union[tuple, List[tuple]]: Tuple of (latitude, longitude) for single input,
                                       or list of tuples for multiple inputs.
                                       Coordinates are approximate center points of the geohash areas.

        Raises:
            ValueError: If geohash string(s) are invalid.
        """
        if isinstance(geohashes, str):
            is_scalar = True
            geohashes = [geohashes]
        else:
            is_scalar = False
        for gh in geohashes:
            if not re.match(f'^[{self._base32}]+$', gh):
                raise ValueError(f'Invalid geohash string: {gh}')
        coordinates = []
        for geohash in geohashes:
            (lat, lon) = self._decode_single(geohash)
            coordinates.append((lat, lon))
        return coordinates[0] if is_scalar else coordinates

    def _decode_single(self, geohash: str) -> Tuple[float, float]:
        """
        Decode a single geohash string into approximate latitude and longitude.

        Args:
            geohash (str): Geohash string to decode.

        Returns:
            Tuple[float, float]: Approximate (latitude, longitude) at center of geohash area.
        """
        (lat_min, lat_max) = (-90.0, 90.0)
        (lon_min, lon_max) = (-180.0, 180.0)
        even = True
        for char in geohash:
            if char not in self._decode_map:
                raise ValueError(f'Invalid character in geohash: {char}')
            cd = self._decode_map[char]
            for i in range(4, -1, -1):
                bit = cd >> i & 1
                if even:
                    mid = (lon_min + lon_max) / 2
                    if bit:
                        lon_min = mid
                    else:
                        lon_max = mid
                else:
                    mid = (lat_min + lat_max) / 2
                    if bit:
                        lat_min = mid
                    else:
                        lat_max = mid
                even = not even
        lat = (lat_min + lat_max) / 2
        lon = (lon_min + lon_max) / 2
        return (lat, lon)

    def fit_transform(self, data: Union[DataBatch, np.ndarray]) -> Union[DataBatch, np.ndarray]:
        """
        Fit the encoder (if needed) and transform coordinates in a DataBatch or array.

        This method assumes the input contains columns for latitude and longitude.
        New columns with geohash encodings will be added to the output.

        Args:
            data (Union[DataBatch, np.ndarray]): Input data containing latitude and longitude columns.

        Returns:
            Union[DataBatch, np.ndarray]: Transformed data with added geohash columns.
        """
        if isinstance(data, DataBatch):
            return self._fit_transform_databatch(data)
        elif isinstance(data, np.ndarray):
            return self._fit_transform_array(data)
        else:
            raise TypeError('Input data must be either a DataBatch or numpy array')

    def _fit_transform_databatch(self, data: DataBatch) -> DataBatch:
        """
        Transform DataBatch with latitude/longitude columns.

        Args:
            data (DataBatch): Input DataBatch.

        Returns:
            DataBatch: Transformed DataBatch with geohash columns.
        """
        arr_data = data.data
        (lat_col_idx, lon_col_idx) = (None, None)
        (lat_col_name, lon_col_name) = (None, None)
        geohash_col_name = f'geohash_{self.precision}'
        if data.feature_names:
            for (i, name) in enumerate(data.feature_names):
                lower_name = name.lower()
                if lower_name == 'latitude' or lower_name == 'lat':
                    lat_col_idx = i
                    lat_col_name = name
                elif lower_name == 'longitude' or lower_name == 'lon' or lower_name == 'lng':
                    lon_col_idx = i
                    lon_col_name = name
        if lat_col_idx is None or lon_col_idx is None:
            if arr_data.shape[1] < 2:
                raise ValueError('Data must contain at least two columns for latitude and longitude')
            (lat_col_idx, lon_col_idx) = (0, 1)
            lat_col_name = data.feature_names[0] if data.feature_names else 'latitude'
            lon_col_name = data.feature_names[1] if data.feature_names else 'longitude'
        latitudes = arr_data[:, lat_col_idx].astype(float)
        longitudes = arr_data[:, lon_col_idx].astype(float)
        geohashes = self.encode(latitudes, longitudes)
        geohash_col = np.array(geohashes, dtype=object).reshape(-1, 1)
        if arr_data.dtype.kind in ['U', 'S', 'O']:
            new_data = np.hstack([arr_data, geohash_col])
        else:
            arr_data_obj = np.array(arr_data, dtype=object)
            new_data = np.hstack([arr_data_obj, geohash_col])
        new_feature_names = None
        if data.feature_names:
            new_feature_names = data.feature_names + [geohash_col_name]
        return DataBatch(data=new_data, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=new_feature_names, batch_id=data.batch_id)

    def _fit_transform_array(self, data: np.ndarray) -> np.ndarray:
        """
        Transform numpy array with latitude/longitude columns.

        Args:
            data (np.ndarray): Input array with latitude and longitude columns.

        Returns:
            np.ndarray: Transformed array with geohash column.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            raise ValueError('Data must contain at least two columns for latitude and longitude')
        latitudes = data[:, 0].astype(float)
        longitudes = data[:, 1].astype(float)
        geohashes = self.encode(latitudes, longitudes)
        geohash_col = np.array(geohashes, dtype=object).reshape(-1, 1)
        if data.dtype.kind in ['U', 'S', 'O']:
            new_data = np.hstack([data, geohash_col])
        else:
            data_as_object = np.array(data, dtype=object)
            new_data = np.hstack([data_as_object, geohash_col])
        return new_data