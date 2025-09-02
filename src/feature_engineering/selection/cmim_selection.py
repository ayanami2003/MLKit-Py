from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class CMIMFeatureSelector(BaseTransformer):

    def __init__(self, k: Optional[int]=None, threshold: Optional[float]=None, discrete_target: bool=False, max_candidates: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        if (k is None) == (threshold is None):
            raise ValueError('Either k or threshold must be specified, but not both.')
        if k is not None and k <= 0:
            raise ValueError('k must be a positive integer.')
        if threshold is not None and threshold < 0:
            raise ValueError('threshold must be non-negative.')
        self.k = k
        self.threshold = threshold
        self.discrete_target = discrete_target
        self.max_candidates = max_candidates
        self._selected_features: List[int] = []
        self._feature_scores: np.ndarray = np.array([])

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'CMIMFeatureSelector':
        """
        Fit the CMIM feature selector to the data.
        
        Computes the CMIM scores for all features and determines which features
        to select based on the specified criteria (number of features or threshold).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        **kwargs : dict
            Additional fitting parameters (ignored).
            
        Returns
        -------
        CMIMFeatureSelector
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If neither k nor threshold is specified, or if both are specified.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        (n_samples, n_features) = X.shape
        if not hasattr(self, '_feature_scores') or self._feature_scores is None or len(self._feature_scores) != n_features:
            self._feature_scores = np.zeros(n_features)
        mi_scores = np.array([self._mutual_information(X[:, i], y) for i in range(n_features)])
        candidate_indices = np.argsort(mi_scores)[::-1]
        if self.max_candidates is not None:
            candidate_indices = candidate_indices[:self.max_candidates]
        self._selected_features = []
        if self.k is not None:
            stop_criterion = lambda : len(self._selected_features) >= self.k
        elif self.threshold is not None:
            stop_criterion = lambda : False
        else:
            raise ValueError('Either k or threshold must be specified.')
        while not stop_criterion() and len(candidate_indices) > 0:
            best_score = -np.inf
            best_idx = -1
            for idx in candidate_indices:
                if idx in self._selected_features:
                    continue
                if len(self._selected_features) == 0:
                    cmim_score = mi_scores[idx]
                else:
                    min_cmim = np.inf
                    for selected_idx in self._selected_features:
                        cmim = self._conditional_mutual_information(X[:, idx], y, X[:, selected_idx])
                        min_cmim = min(min_cmim, cmim)
                    cmim_score = min_cmim
                if cmim_score > best_score:
                    best_score = cmim_score
                    best_idx = idx
            if self.threshold is not None and best_score < self.threshold:
                break
            if best_idx != -1:
                self._selected_features.append(best_idx)
                self._feature_scores[best_idx] = best_score
                candidate_indices = candidate_indices[candidate_indices != best_idx]
            else:
                break
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply feature selection to the input data.
        
        Reduces the feature set to only those features selected during fitting.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to transform.
        **kwargs : dict
            Additional transformation parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with only selected features.
        """
        if len(self._selected_features) == 0:
            raise ValueError('No features have been selected. Please call fit() first.')
        if isinstance(data, FeatureSet):
            X_transformed = data.features[:, self._selected_features]
            return FeatureSet(features=X_transformed, feature_names=[data.feature_names[i] for i in self._selected_features] if data.feature_names else None)
        else:
            return data[:, self._selected_features]

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for feature selection methods.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data to inverse transform (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Original data structure without modification.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for feature selection.
        """
        raise NotImplementedError('Inverse transformation is not supported for feature selection methods.')

    def get_selected_features(self) -> List[int]:
        """
        Get indices of selected features.
        
        Returns
        -------
        List[int]
            Indices of selected features.
        """
        return self._selected_features.copy()

    def get_feature_scores(self) -> np.ndarray:
        """
        Get CMIM scores for all features.
        
        Returns
        -------
        np.ndarray
            Array of CMIM scores for each feature.
        """
        return self._feature_scores.copy()

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information between two variables.
        
        Parameters
        ----------
        x : np.ndarray
            First variable.
        y : np.ndarray
            Second variable.
            
        Returns
        -------
        float
            Mutual information between x and y.
        """
        if self.discrete_target:
            return self._discrete_mutual_information(x, y)
        else:
            x_discrete = self._discretize_continuous(x)
            y_discrete = self._discretize_continuous(y)
            return self._discrete_mutual_information(x_discrete, y_discrete)

    def _discretize_continuous(self, x: np.ndarray, bins: int=10) -> np.ndarray:
        """
        Discretize a continuous variable using equal-frequency binning.
        
        Parameters
        ----------
        x : np.ndarray
            Continuous variable to discretize.
        bins : int, default=50
            Number of bins to use.
            
        Returns
        -------
        np.ndarray
            Discretized variable.
        """
        if np.all(x == x[0]):
            return np.ones_like(x, dtype=int)
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.quantile(x, quantiles)
        bin_edges = np.unique(bin_edges)
        return np.digitize(x, bin_edges[:-1], right=False)

    def _discrete_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information between two discrete variables.
        
        Parameters
        ----------
        x : np.ndarray
            First discrete variable.
        y : np.ndarray
            Second discrete variable.
            
        Returns
        -------
        float
            Mutual information between x and y.
        """
        (x_vals, x_counts) = np.unique(x, return_counts=True)
        (y_vals, y_counts) = np.unique(y, return_counts=True)
        x_probs = x_counts / len(x)
        y_probs = y_counts / len(y)
        joint_counts = np.zeros((len(x_vals), len(y_vals)))
        for (i, x_val) in enumerate(x_vals):
            for (j, y_val) in enumerate(y_vals):
                joint_counts[i, j] = np.sum((x == x_val) & (y == y_val))
        joint_probs = joint_counts / len(x)
        mi = 0.0
        for i in range(len(x_vals)):
            for j in range(len(y_vals)):
                if joint_probs[i, j] > 0:
                    mi += joint_probs[i, j] * np.log(joint_probs[i, j] / (x_probs[i] * y_probs[j]))
        return mi

    def _conditional_mutual_information(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """
        Compute conditional mutual information I(X;Y|Z) between three variables.
        
        Parameters
        ----------
        x : np.ndarray
            First variable.
        y : np.ndarray
            Second variable.
        z : np.ndarray
            Conditioning variable.
            
        Returns
        -------
        float
            Conditional mutual information I(X;Y|Z).
        """
        if not self.discrete_target:
            x = self._discretize_continuous(x)
            y = self._discretize_continuous(y)
            z = self._discretize_continuous(z)
        z_vals = np.unique(z)
        cmi = 0.0
        for z_val in z_vals:
            mask = z == z_val
            if np.sum(mask) == 0:
                continue
            p_z = np.mean(mask)
            mi_cond = self._discrete_mutual_information(x[mask], y[mask])
            cmi += p_z * mi_cond
        return cmi