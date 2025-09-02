import numpy as np
from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class HashingEncoder(BaseTransformer):

    def __init__(self, n_components: int=100, hash_method: str='md5', alternate_sign: bool=True, name: Optional[str]=None):
        """
        Initialize the HashingEncoder.
        
        Parameters
        ----------
        n_components : int, default=100
            Number of dimensions in the encoded output space. Larger values
            reduce the probability of collisions but increase memory usage.
        hash_method : str, default='md5'
            Hash function to use. Supported values depend on the system's
            hashlib implementation.
        alternate_sign : bool, default=True
            Whether to use alternative sign for hash values to reduce collisions.
            If True, the sign of the hash value alternates based on the hash.
        name : Optional[str], default=None
            Name of the transformer instance. If None, uses class name.
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.hash_method = hash_method
        self.alternate_sign = alternate_sign

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'HashingEncoder':
        """
        Fit the encoder to the input data.
        
        For hashing encoding, this is a no-op since no statistics need to be computed.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        HashingEncoder
            Self instance for method chaining
        """
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical features using hashing encoding.
        
        Applies the hashing trick to convert categorical features into
        a fixed-dimensional numerical representation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Encoded features with hashed representations
            
        Raises
        ------
        ValueError
            If input data format is not supported
        """
        import hashlib
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        (n_samples, n_features) = X.shape
        encoded_features = np.zeros((n_samples, self.n_components))
        for feature_idx in range(n_features):
            feature_column = X[:, feature_idx].astype(str)
            for (sample_idx, category) in enumerate(feature_column):
                if category == 'nan':
                    continue
                hash_obj = hashlib.new(self.hash_method)
                hash_obj.update(category.encode('utf-8'))
                hash_digest = hash_obj.hexdigest()
                hash_int = int(hash_digest, 16)
                position = hash_int % self.n_components
                if self.alternate_sign:
                    value = 1 if hash_int % 2 == 0 else -1
                else:
                    value = 1
                encoded_features[sample_idx, position] += value
        new_feature_names = [f'hashed_feature_{i}' for i in range(self.n_components)]
        metadata['encoding_type'] = 'hashing_encoding'
        metadata['n_components'] = self.n_components
        metadata['hash_method'] = self.hash_method
        metadata['alternate_sign'] = self.alternate_sign
        if feature_names:
            metadata['original_feature_names'] = feature_names
        return FeatureSet(features=encoded_features, feature_names=new_feature_names, feature_types=['numeric'] * self.n_components, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation (not supported for hashing encoder).
        
        Hashing encoding is a one-way transformation due to potential hash collisions,
        so the inverse transformation is not possible.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Encoded data to invert
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Original data format
            
        Raises
        ------
        NotImplementedError
            Always raised since inverse transformation is not supported
        """
        raise NotImplementedError('Hashing encoding does not support inverse transformation.')