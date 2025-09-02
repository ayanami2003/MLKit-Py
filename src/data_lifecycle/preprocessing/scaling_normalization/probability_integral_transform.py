from typing import Optional, Union
import numpy as np
from scipy import stats
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ProbabilityIntegralTransformer(BaseTransformer):

    def __init__(self, method: str='empirical', distribution_family: Optional[str]=None, name: Optional[str]=None):
        """
        Initialize the ProbabilityIntegralTransformer.
        
        Parameters
        ----------
        method : str, default='empirical'
            Method for estimating the distribution. Options:
            - 'empirical': Use empirical CDF
            - 'parametric': Fit a parametric distribution
        distribution_family : str, optional
            Family of distribution to fit when method='parametric'.
            Examples: 'norm', 'beta', 'gamma', 'expon'.
            Required when method='parametric'.
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.method = method
        self.distribution_family = distribution_family
        if method == 'parametric' and distribution_family is None:
            raise ValueError("distribution_family must be specified when method='parametric'")

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ProbabilityIntegralTransformer':
        """
        Fit the transformer to the input data by estimating the distribution.
        
        For empirical method, computes the empirical CDF for each feature.
        For parametric method, fits the specified distribution to each feature.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to fit the transformer on. Should be a 2D array-like
            where rows are samples and columns are features.
        **kwargs : dict
            Additional parameters for fitting (e.g., optimization settings)
            
        Returns
        -------
        ProbabilityIntegralTransformer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data is not 2D or if fitting fails
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        self.n_features_ = n_features
        if self.method == 'empirical':
            self.ecdfs_ = []
            for i in range(n_features):
                feature_data = X[:, i]
                if np.all(feature_data == feature_data[0]):
                    self.ecdfs_.append((np.array([feature_data[0]]), np.array([0.5])))
                else:
                    sorted_data = np.sort(feature_data)
                    ranks = np.argsort(np.argsort(feature_data)) + 1
                    probabilities = ranks / (n_samples + 1)
                    self.ecdfs_.append((sorted_data, probabilities))
        elif self.method == 'parametric':
            self.distributions_ = {}
            try:
                dist_class = getattr(stats, self.distribution_family)
            except AttributeError:
                raise ValueError(f"Distribution '{self.distribution_family}' not found in scipy.stats")
            for i in range(n_features):
                feature_data = X[:, i]
                if np.all(feature_data == feature_data[0]):
                    if self.distribution_family in ['norm', 'expon']:
                        self.distributions_[i] = dist_class(loc=feature_data[0], scale=0)
                    else:
                        self.distributions_[i] = self._create_degenerate_distribution(feature_data[0])
                else:
                    try:
                        params = dist_class.fit(feature_data)
                        self.distributions_[i] = dist_class(*params)
                    except Exception:
                        sorted_data = np.sort(feature_data)
                        ranks = np.argsort(np.argsort(feature_data)) + 1
                        probabilities = ranks / (n_samples + 1)
                        self.distributions_[i] = self._create_empirical_distribution(feature_data, probabilities)
        else:
            raise ValueError(f'Unsupported method: {self.method}')
        return self

    def _create_degenerate_distribution(self, value):
        """
        Create a degenerate distribution for constant data.
        """

        class DegenerateDistribution:

            def __init__(self, val):
                self.value = val
                self.a = val
                self.b = val

            def cdf(self, x):
                return np.where(np.asarray(x) >= self.value, 1.0, 0.0)

            def ppf(self, q):
                return np.full_like(q, self.value, dtype=float)

            def pdf(self, x):
                return np.where(np.asarray(x) == self.value, np.inf, 0.0)

            def rvs(self, size=1):
                return np.full(size, self.value, dtype=float)
        return DegenerateDistribution(value)

    def _create_empirical_distribution(self, data, probabilities):
        """
        Create a custom distribution that mimics the behavior of a scipy distribution
        but uses empirical CDF/PPF.
        """

        class EmpiricalDistribution:

            def __init__(self, data_vals, probs):
                self.data = data_vals.copy()
                self.probs = probs.copy()
                sort_idx = np.argsort(self.data)
                self.sorted_data = self.data[sort_idx]
                self.sorted_probs = self.probs[sort_idx]
                self.a = self.sorted_data[0]
                self.b = self.sorted_data[-1]

            def cdf(self, x):
                return np.interp(x, self.sorted_data, self.sorted_probs, left=0.0, right=1.0)

            def ppf(self, q):
                return np.interp(q, self.sorted_probs, self.sorted_data, left=self.sorted_data[0], right=self.sorted_data[-1])

            def pdf(self, x):
                return np.gradient(self.cdf(x), x, edge_order=2) if len(np.unique(x)) > 1 else np.zeros_like(x)

            def rvs(self, size=1):
                uniform_samples = np.random.uniform(0, 1, size)
                return self.ppf(uniform_samples)
        return EmpiricalDistribution(data, probabilities)

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the probability integral transform to the input data.
        
        Transforms each feature by applying its fitted CDF, resulting in
        uniformly distributed values in [0, 1].
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to transform. Must have the same number of features
            as the data used for fitting.
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed data with uniform distribution
            
        Raises
        ------
        ValueError
            If transformer is not fitted or data dimensions don't match
        """
        if self.method == 'empirical' and (not hasattr(self, 'ecdfs_')):
            raise ValueError('Transformer has not been fitted yet.')
        if self.method == 'parametric' and (not hasattr(self, 'distributions_')):
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
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
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features, but got {X.shape[1]}')
        transformed_features = np.zeros_like(X)
        if self.method == 'empirical':
            for i in range(self.n_features_):
                feature_data = X[:, i]
                (sorted_data, probabilities) = self.ecdfs_[i]
                if len(sorted_data) == 1:
                    transformed_features[:, i] = 0.5
                else:
                    transformed_features[:, i] = np.interp(feature_data, sorted_data, probabilities)
        elif self.method == 'parametric':
            for i in range(self.n_features_):
                feature_data = X[:, i]
                dist = self.distributions_[i]
                if hasattr(dist, 'scale') and dist.scale == 0:
                    transformed_features[:, i] = 0.5
                else:
                    transformed_features[:, i] = dist.cdf(feature_data)
        return FeatureSet(features=transformed_features, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse probability integral transform.
        
        Maps uniform values back to the original distribution space using
        the inverse CDF (quantile function).
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Uniformly distributed data to inverse transform
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        FeatureSet
            Data mapped back to original distribution space
            
        Raises
        ------
        ValueError
            If transformer is not fitted or inverse transform is not possible
        NotImplementedError
            If inverse transform is not implemented for the chosen method
        """
        if self.method == 'empirical' and (not hasattr(self, 'ecdfs_')):
            raise ValueError('Transformer has not been fitted yet.')
        if self.method == 'parametric' and (not hasattr(self, 'distributions_')):
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
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
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features, but got {X.shape[1]}')
        inverted_features = np.zeros_like(X)
        if self.method == 'empirical':
            for i in range(self.n_features_):
                uniform_data = X[:, i]
                (sorted_data, probabilities) = self.ecdfs_[i]
                uniform_data = np.clip(uniform_data, 0, 1)
                if len(sorted_data) == 1:
                    inverted_features[:, i] = sorted_data[0]
                else:
                    sort_idx = np.argsort(probabilities)
                    sorted_probs = probabilities[sort_idx]
                    sorted_vals = sorted_data[sort_idx]
                    inverted_features[:, i] = np.interp(uniform_data, sorted_probs, sorted_vals, left=sorted_vals[0], right=sorted_vals[-1])
        elif self.method == 'parametric':
            for i in range(self.n_features_):
                uniform_data = X[:, i]
                uniform_data = np.clip(uniform_data, 1e-10, 1 - 1e-10)
                dist = self.distributions_[i]
                if hasattr(dist, 'scale') and dist.scale == 0:
                    if hasattr(dist, 'loc'):
                        inverted_features[:, i] = dist.loc
                    elif hasattr(dist, 'value'):
                        inverted_features[:, i] = dist.value
                    else:
                        inverted_features[:, i] = 0.0
                else:
                    inverted_features[:, i] = dist.ppf(uniform_data)
        return FeatureSet(features=inverted_features, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)