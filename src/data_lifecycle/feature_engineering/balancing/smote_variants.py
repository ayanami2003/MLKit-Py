from typing import Optional, Union
import numpy as np
from sklearn.neighbors import NearestNeighbors
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class BorderlineSMOTETransformer(BaseTransformer):

    def __init__(self, k_neighbors: int=5, sampling_strategy: Union[float, str]='auto', random_state: Optional[int]=None, m_neighbors: int=10, kind: str='borderline-1'):
        super().__init__(name='BorderlineSMOTETransformer')
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.m_neighbors = m_neighbors
        self.kind = kind
        self.fitted_ = False
        self.sampling_indices_ = []

    def fit(self, data: FeatureSet, **kwargs) -> 'BorderlineSMOTETransformer':
        """
        Identify borderline samples in the minority class for synthetic generation.
        
        This method analyzes the feature space to identify minority class samples
        that are close to the decision boundary. These samples will be used
        for generating synthetic instances during the transform step.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with features and labels. Labels are required
            to identify class distributions.
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        BorderlineSMOTETransformer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If input data does not contain labels or has insufficient samples
        """
        if 'labels' not in data.metadata or data.metadata['labels'] is None:
            raise ValueError('Input data must contain labels in metadata to perform Borderline-SMOTE')
        X = data.features
        y = np.asarray(data.metadata['labels'])
        if len(X) != len(y):
            raise ValueError('Number of samples in features and labels must match')
        if len(np.unique(y)) < 2:
            raise ValueError('At least two classes are required for Borderline-SMOTE')
        np.random.seed(self.random_state)
        (unique_classes, counts) = np.unique(y, return_counts=True)
        min_class = unique_classes[np.argmin(counts)]
        maj_class = unique_classes[np.argmax(counts)]
        min_indices = np.where(y == min_class)[0]
        maj_indices = np.where(y == maj_class)[0]
        if len(min_indices) < self.m_neighbors + 1:
            raise ValueError(f'Not enough minority class samples for m_neighbors={self.m_neighbors}. Found {len(min_indices)} minority samples.')
        if len(min_indices) < self.k_neighbors + 1:
            raise ValueError(f'Not enough minority class samples for k_neighbors={self.k_neighbors}. Found {len(min_indices)} minority samples.')
        nn_m = NearestNeighbors(n_neighbors=self.m_neighbors + 1)
        nn_m.fit(X)
        (distances_m, indices_m) = nn_m.kneighbors(X[min_indices])
        borderline_indices = []
        for (i, min_idx) in enumerate(min_indices):
            neighbor_labels = y[indices_m[i][1:]]
            maj_count = np.sum(neighbor_labels == maj_class)
            if maj_count > self.m_neighbors / 2:
                borderline_indices.append(min_idx)
        self.borderline_indices_ = np.array(borderline_indices)
        self.min_class_ = min_class
        self.maj_class_ = maj_class
        self.classes_ = unique_classes
        self.fitted_ = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply Borderline-SMOTE oversampling to generate synthetic samples.
        
        Generates synthetic samples for minority class borderline samples
        identified during the fit phase. The synthetic samples are created
        by interpolating between borderline samples and their neighbors.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform. Must have same feature structure
            as the data used in fit.
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed feature set with synthetic samples added for minority class
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or input data structure mismatch
        """
        pass

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Remove synthetic samples to recover original data structure (approximate).
        
        Since SMOTE generates synthetic samples, exact inverse transformation
        is not possible. This method attempts to identify and remove synthetic
        samples based on metadata markers.
        
        Parameters
        ----------
        data : FeatureSet
            Transformed feature set with potential synthetic samples
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Feature set with synthetic samples removed (best effort)
        """
        pass