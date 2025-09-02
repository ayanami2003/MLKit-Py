from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class AffinityPropagationClustering(BaseModel):

    def __init__(self, damping: float=0.5, max_iter: int=200, convergence_iter: int=15, preference: Optional[Union[str, float, np.ndarray]]=None, verbose: bool=False, name: Optional[str]=None):
        super().__init__(name=name)
        if not 0.5 <= damping <= 1.0:
            raise ValueError('damping must be between 0.5 and 1.0')
        if max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if convergence_iter <= 0:
            raise ValueError('convergence_iter must be positive')
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.verbose = verbose
        self.cluster_centers_indices_ = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.affinity_matrix_ = None
        self.n_iter_ = 0
        self._fitted_with_affinity_matrix = False
        self.is_fitted = False

    def _convert_input(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, FeatureSet):
            if hasattr(X.features, 'values'):
                return X.features.values
            else:
                return np.asarray(X.features)
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError('Input must be a FeatureSet or numpy array')

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'AffinityPropagationClustering':
        """
        Fit the affinity propagation clustering model according to the given training data.

        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training instances to cluster. If FeatureSet, uses the features attribute.
        y : Optional[np.ndarray], default=None
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional parameters (e.g., affinity_matrix for precomputed similarities).

        Returns
        -------
        AffinityPropagationClustering
            Fitted estimator.
        """
        if 'affinity_matrix' in kwargs:
            S = kwargs['affinity_matrix']
            if not isinstance(S, np.ndarray):
                raise TypeError('affinity_matrix must be a numpy array')
            if S.ndim != 2 or S.shape[0] != S.shape[1]:
                raise ValueError('affinity_matrix must be a square 2D array')
            X_array = None
            self._fitted_with_affinity_matrix = True
        else:
            X_array = self._convert_input(X)
            if X_array.ndim != 2:
                raise ValueError('Input data must be a 2D array')
            n_samples = X_array.shape[0]
            if n_samples == 1:
                S = np.array([[0.0]])
            else:
                S = -np.sum((X_array[:, np.newaxis] - X_array[np.newaxis, :]) ** 2, axis=2)
            self._fitted_with_affinity_matrix = False
        S = S.astype(float)
        n_samples = S.shape[0]
        if self.preference is None or (isinstance(self.preference, str) and self.preference == 'median'):
            preference = np.median(S)
        elif np.isscalar(self.preference):
            preference = float(self.preference)
        elif isinstance(self.preference, np.ndarray):
            if self.preference.shape[0] != n_samples:
                raise ValueError('preference array must have the same length as the number of samples')
            preference = self.preference.astype(float)
        else:
            raise ValueError("preference must be None, 'median', a scalar, or an array of length n_samples")
        S.flat[::n_samples + 1] = preference if np.isscalar(preference) else preference
        R = np.zeros_like(S, dtype=float)
        A = np.zeros_like(S, dtype=float)
        if n_samples == 1:
            E = np.array([True])
            current_labels = np.array([0])
            self.n_iter_ = 0
        else:
            last_labels = np.full(n_samples, -1, dtype=int)
            changes_count = 0
            for it in range(self.max_iter):
                AS = A + S
                I = np.arange(n_samples)
                idx_max = np.argmax(AS, axis=1)
                AS_max = AS[I, idx_max]
                AS_temp = AS.copy()
                AS_temp[I, idx_max] = -np.inf
                idx_second_max = np.argmax(AS_temp, axis=1)
                AS_second_max = AS_temp[I, idx_second_max]
                new_R = np.zeros_like(R)
                new_R[:] = S - AS_second_max[:, np.newaxis]
                new_R[I, idx_max] = S[I, idx_max] - AS_max
                R = (1 - self.damping) * new_R + self.damping * R
                Rp = np.maximum(R, 0)
                Rp.flat[::n_samples + 1] = R.flat[::n_samples + 1]
                A_old = A.copy()
                A_sum = np.sum(Rp, axis=0)
                A = np.zeros_like(A)
                A[:] = A_sum - Rp
                A = np.minimum(A, 0)
                A.flat[::n_samples + 1] = A_sum.flat[::n_samples + 1] - Rp.flat[::n_samples + 1]
                A = (1 - self.damping) * A + self.damping * A_old
                E = (A + R).diagonal() > 0
                current_labels = np.argmax(S + A, axis=1)
                if it > 0:
                    changed = np.sum(last_labels != current_labels)
                    if changed == 0:
                        changes_count += 1
                    else:
                        changes_count = 0
                    if changes_count >= self.convergence_iter:
                        if self.verbose:
                            print(f'Converged after {it + 1} iterations.')
                        break
                last_labels[:] = current_labels
            self.n_iter_ = it + 1
        exemplars = np.where(E)[0]
        if len(exemplars) == 0:
            if np.isscalar(preference):
                exemplars = np.array([np.argmax(np.diag(S))])
            else:
                exemplars = np.array([np.argmax(preference)])
        if n_samples == 1:
            labels = np.array([0])
        else:
            labels = np.argmax(S[:, exemplars], axis=1)
        self.labels_ = exemplars[labels]
        self.cluster_centers_indices_ = exemplars
        if X_array is not None:
            self.cluster_centers_ = X_array[exemplars]
        else:
            self.cluster_centers_ = np.zeros((len(exemplars), 1)) if len(exemplars) > 0 else np.array([]).reshape(0, 1)
        self.affinity_matrix_ = S
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Note: Affinity Propagation doesn't naturally support predicting clusters for new data.
        This would require a separate assignment step based on learned exemplars.

        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            New data to predict cluster labels for.

        Returns
        -------
        np.ndarray
            Index of the cluster each sample belongs to.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        if self._fitted_with_affinity_matrix or self.cluster_centers_ is None:
            raise ValueError('Cannot predict for new data when the model was fitted with a precomputed affinity matrix.')
        X_array = self._convert_input(X)
        if X_array.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X_array.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError('New data must have the same number of features as training data')
        distances = -np.sum((X_array[:, np.newaxis] - self.cluster_centers_[np.newaxis, :]) ** 2, axis=2)
        labels = np.argmax(distances, axis=1)
        return labels

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Return the negative inertia (sum of squared distances to nearest cluster center).

        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Data to compute score for.
        y : Optional[np.ndarray], default=None
            Not used.

        Returns
        -------
        float
            Negative sum of squared distances to nearest cluster center (higher is better).
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        if self._fitted_with_affinity_matrix or self.cluster_centers_ is None:
            raise ValueError('Cannot compute score when the model was fitted with a precomputed affinity matrix.')
        X_array = self._convert_input(X)
        if X_array.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X_array.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError('New data must have the same number of features as training data')
        total_distance = 0.0
        for i in range(X_array.shape[0]):
            min_dist = np.inf
            for center in self.cluster_centers_:
                dist = np.sum((X_array[i] - center) ** 2)
                if dist < min_dist:
                    min_dist = dist
            total_distance += min_dist
        return -total_distance