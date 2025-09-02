import numpy as np
from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class PrincipalCurveExtractor(BaseTransformer):
    """
    Extract features using principal curves and surfaces method for non-linear dimensionality reduction.
    
    Principal curves and surfaces provide a non-linear generalization of principal components
    by finding smooth one-dimensional curves that pass through the "middle" of a data cloud.
    This transformer fits a principal curve to the data and projects samples onto it to create
    lower-dimensional representations while preserving non-linear structures.
    
    Attributes
    ----------
    n_components : int, optional
        Number of principal curve dimensions to extract (default is 2)
    max_iterations : int
        Maximum number of iterations for curve fitting (default is 50)
    tolerance : float
        Convergence tolerance for the fitting algorithm (default is 1e-4)
    smoothing_factor : float
        Regularization parameter controlling curve smoothness (default is 0.1)
    random_state : int, optional
        Random seed for reproducible results
        
    Methods
    -------
    fit(X, y=None) -> 'PrincipalCurveExtractor'
        Fit the principal curve to training data
    transform(X) -> FeatureSet
        Project data onto the fitted principal curve
    fit_transform(X, y=None) -> FeatureSet
        Fit and transform in one step
    inverse_transform(X) -> FeatureSet
        Map points back to original space (approximate)
    """

    def __init__(self, n_components: Optional[int]=2, max_iterations: int=50, tolerance: float=0.0001, smoothing_factor: float=0.1, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the PrincipalCurveExtractor.
        
        Parameters
        ----------
        n_components : int, optional
            Number of principal curve dimensions to extract (default is 2)
        max_iterations : int
            Maximum number of iterations for curve fitting (default is 50)
        tolerance : float
            Convergence tolerance for the fitting algorithm (default is 1e-4)
        smoothing_factor : float
            Regularization parameter controlling curve smoothness (default is 0.1)
        random_state : int, optional
            Random seed for reproducible results
        name : str, optional
            Name for the transformer instance
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.smoothing_factor = smoothing_factor
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'PrincipalCurveExtractor':
        """
        Fit the principal curve to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the principal curve on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional fitting parameters (ignored)
            
        Returns
        -------
        PrincipalCurveExtractor
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data has insufficient samples or dimensions
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(data, FeatureSet):
            X = data.features.copy()
        elif isinstance(data, np.ndarray):
            X = data.copy()
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        (n_samples, n_features) = X.shape
        if n_samples < 2:
            raise ValueError('Need at least 2 samples to fit a principal curve')
        if self.n_components >= n_features:
            raise ValueError('n_components must be less than the number of input features')
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        cov_matrix = np.cov(X_centered, rowvar=False)
        (eigenvals, eigenvecs) = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        pc1 = eigenvecs[:, 0]
        projections = X_centered @ pc1
        (t_min, t_max) = np.percentile(projections, [5, 95])
        n_points = max(50, n_samples // 5)
        t_values = np.linspace(t_min, t_max, n_points)
        self.curve_points_ = np.outer(t_values, pc1)
        prev_error = np.inf
        for iteration in range(self.max_iterations):
            projected_indices = self._project_onto_curve(X_centered, self.curve_points_)
            projected_points = self.curve_points_[projected_indices]
            error = np.mean(np.sum((X_centered - projected_points) ** 2, axis=1))
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error
            new_curve_points = np.zeros_like(self.curve_points_)
            for i in range(len(self.curve_points_)):
                distances = np.linalg.norm(self.curve_points_ - self.curve_points_[i], axis=1)
                weights = np.exp(-distances ** 2 / (2 * (np.std(distances) * 0.5) ** 2))
                relevant_mask = projected_indices == i
                if np.sum(relevant_mask) > 0:
                    X_local = X_centered[relevant_mask]
                    weights_local = weights[projected_indices[relevant_mask]]
                    if np.sum(weights_local) > 0:
                        weighted_mean = np.average(X_local, axis=0, weights=weights_local)
                        new_curve_points[i] = (1 - self.smoothing_factor) * weighted_mean + self.smoothing_factor * self.curve_points_[i]
                    else:
                        new_curve_points[i] = self.curve_points_[i]
                else:
                    new_curve_points[i] = self.curve_points_[i]
            self.curve_points_ = new_curve_points
        self._compute_tangent_spaces(X_centered)
        self._is_fitted = True
        self.n_features_in_ = n_features
        self.feature_names_ = [f'principal_curve_component_{i}' for i in range(self.n_components)]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Project data onto the fitted principal curve.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Must have same number of features as training data.
        **kwargs : dict
            Additional transformation parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed data with reduced dimensions
            
        Raises
        ------
        ValueError
            If transformer is not fitted or data dimensions don't match
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
        elif isinstance(data, np.ndarray):
            X = data.copy()
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data must have {self.n_features_in_} features, got {X.shape[1]}')
        X_centered = X - self.mean_
        projected_indices = self._project_onto_curve(X_centered, self.curve_points_)
        coordinates = projected_indices.reshape(-1, 1).astype(float)
        result = np.zeros((X.shape[0], self.n_components))
        result[:, 0] = coordinates[:, 0]
        transformed_fs = FeatureSet(features=result, feature_names=self.get_feature_names())
        return transformed_fs

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Map points from principal curve space back to original space (approximate reconstruction).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data in principal curve space to map back to original space
        **kwargs : dict
            Additional inverse transformation parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Approximated data in original space
            
        Raises
        ------
        ValueError
            If transformer is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
        elif isinstance(data, np.ndarray):
            X = data.copy()
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if X.shape[1] != self.n_components:
            raise ValueError(f'Input data must have {self.n_components} features, got {X.shape[1]}')
        positions = X[:, 0].astype(int)
        positions = np.clip(positions, 0, len(self.curve_points_) - 1)
        reconstructed_points = self.curve_points_[positions]
        reconstructed_original = reconstructed_points + self.mean_
        reconstructed_fs = FeatureSet(features=reconstructed_original, feature_names=[f'reconstructed_feature_{i}' for i in range(self.n_features_in_)])
        return reconstructed_fs

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names for transformed data.
        
        Parameters
        ----------
        input_features : List[str], optional
            Input feature names (ignored for this transformer)
            
        Returns
        -------
        List[str]
            Output feature names for transformed data
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        return self.feature_names_.copy()

    def _project_onto_curve(self, X: np.ndarray, curve_points: np.ndarray) -> np.ndarray:
        """
        Project points onto the nearest point on the curve.
        
        Parameters
        ----------
        X : np.ndarray
            Data points to project
        curve_points : np.ndarray
            Points defining the curve
            
        Returns
        -------
        np.ndarray
            Indices of nearest curve points for each data point
        """
        distances = np.linalg.norm(X[:, np.newaxis, :] - curve_points[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _compute_tangent_spaces(self, X: np.ndarray):
        """
        Compute tangent spaces at curve points for inverse transformation.
        
        Parameters
        ----------
        X : np.ndarray
            Centered training data
        """
        n_points = len(self.curve_points_)
        self.tangents_ = np.zeros_like(self.curve_points_)
        if n_points > 1:
            self.tangents_[0] = self.curve_points_[1] - self.curve_points_[0]
            for i in range(1, n_points - 1):
                self.tangents_[i] = self.curve_points_[i + 1] - self.curve_points_[i - 1]
            self.tangents_[-1] = self.curve_points_[-1] - self.curve_points_[-2]
            norms = np.linalg.norm(self.tangents_, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.tangents_ /= norms