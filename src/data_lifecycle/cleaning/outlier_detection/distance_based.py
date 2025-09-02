from typing import Union, Optional
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy.stats import chi2

class MahalanobisDistanceOutlierDetector(BaseTransformer):
    """
    Detects outliers using the Mahalanobis distance metric.
    
    This transformer computes the Mahalanobis distance for each sample in the input data
    relative to the distribution of the training data. Samples with distances exceeding
    a specified threshold are flagged as outliers.
    
    The Mahalanobis distance accounts for feature correlations and scales appropriately
    by using the inverse covariance matrix, making it more effective than Euclidean distance
    for multivariate outlier detection.
    
    Parameters
    ----------
    threshold : float, default=3.0
        The threshold value for flagging outliers. Observations with Mahalanobis distance
        greater than this value are considered outliers.
    compute_threshold_from_chi_square : bool, default=False
        If True, automatically compute threshold based on chi-square distribution
        with degrees of freedom equal to number of features.
    chi_square_probability : float, default=0.99
        Probability level for chi-square threshold when compute_threshold_from_chi_square is True.
        
    Attributes
    ----------
    mean_ : np.ndarray
        Mean of the training data.
    cov_inv_ : np.ndarray
        Inverse of the covariance matrix of the training data.
    threshold_ : float
        Actual threshold used for outlier detection.
    """

    def __init__(self, threshold: float=3.0, compute_threshold_from_chi_square: bool=False, chi_square_probability: float=0.99):
        self.threshold = threshold
        self.compute_threshold_from_chi_square = compute_threshold_from_chi_square
        self.chi_square_probability = chi_square_probability

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'MahalanobisDistanceOutlierDetector':
        """
        Fit the detector by computing the mean and inverse covariance matrix of the training data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Training data. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional fitting parameters (not used).
            
        Returns
        -------
        MahalanobisDistanceOutlierDetector
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the covariance matrix is singular and cannot be inverted.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.mean_ = np.mean(X, axis=0)
        cov_matrix = np.cov(X, rowvar=False)
        if cov_matrix.ndim == 0:
            cov_matrix = cov_matrix.reshape(1, 1)
        try:
            self.cov_inv_ = np.linalg.pinv(cov_matrix)
            cond_num = np.linalg.cond(cov_matrix)
            if cond_num > 1000000000000.0:
                raise np.linalg.LinAlgError('Covariance matrix is singular or near-singular')
        except np.linalg.LinAlgError as e:
            raise ValueError(f'Cannot invert covariance matrix: {str(e)}')
        n_features = X.shape[1]
        if self.compute_threshold_from_chi_square:
            self.threshold_ = chi2.ppf(self.chi_square_probability, df=n_features)
        else:
            self.threshold_ = self.threshold
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the data by adding outlier flags based on Mahalanobis distance.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to detect outliers in.
        **kwargs : dict
            Additional transformation parameters (not used).
            
        Returns
        -------
        FeatureSet
            FeatureSet with added metadata containing outlier flags and distances.
        """
        if not hasattr(self, 'mean_') or not hasattr(self, 'cov_inv_'):
            raise RuntimeError("Detector has not been fitted yet. Call 'fit' first.")
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
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f'Input data has {X.shape[1]} features, but detector was fitted on {self.mean_.shape[0]} features')
        distances = self.compute_mahalanobis_distances(X)
        outliers = distances > self.threshold_
        metadata['mahalanobis_distances'] = distances
        metadata['mahalanobis_outliers'] = outliers
        metadata['mahalanobis_threshold'] = self.threshold_
        return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transform is not applicable for this outlier detector.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data to inverse transform (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Unmodified input data.
        """
        if isinstance(data, np.ndarray):
            return FeatureSet(features=data)
        return data

    def predict_outliers(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Predict which samples are outliers.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to predict outliers for.
            
        Returns
        -------
        np.ndarray
            Boolean array indicating which samples are outliers.
        """
        if not hasattr(self, 'mean_') or not hasattr(self, 'cov_inv_'):
            raise RuntimeError("Detector has not been fitted yet. Call 'fit' first.")
        distances = self.compute_mahalanobis_distances(data)
        return distances > self.threshold_

    def compute_mahalanobis_distances(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Compute Mahalanobis distances for the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to compute distances for.
            
        Returns
        -------
        np.ndarray
            Array of Mahalanobis distances for each sample.
        """
        if not hasattr(self, 'mean_') or not hasattr(self, 'cov_inv_'):
            raise RuntimeError("Detector has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f'Input data has {X.shape[1]} features, but detector was fitted on {self.mean_.shape[0]} features')
        diff = X - self.mean_
        distances_squared = np.einsum('ni,ij,nj->n', diff, self.cov_inv_, diff)
        distances = np.sqrt(np.maximum(distances_squared, 0))
        return distances

    def decision_function(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Compute the decision function values (negative Mahalanobis distances).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to compute decision function for.
            
        Returns
        -------
        np.ndarray
            Array of decision function values for each sample.
        """
        if not hasattr(self, 'mean_') or not hasattr(self, 'cov_inv_'):
            raise RuntimeError("Detector has not been fitted yet. Call 'fit' first.")
        distances = self.compute_mahalanobis_distances(data)
        return -distances