from general.base_classes.transformer_base import BaseTransformer
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from typing import Optional, Union, List
import numpy as np

class LinearDiscriminantAnalysisTransformer(BaseTransformer):
    """
    Linear Discriminant Analysis (LDA) transformer for dimensionality reduction.
    
    This transformer implements LDA for unsupervised dimensionality reduction by projecting
    data onto linear discriminants that maximize the separation between classes. It can be
    used to reduce the dimensionality of feature spaces while preserving class-discriminatory
    information.
    
    Attributes
    ----------
    n_components : int, optional
        Number of components to keep. If None, defaults to min(n_classes-1, n_features).
    solver : str, default='svd'
        Solver to use ('svd', 'lsqr', or 'eigen').
    shrinkage : str or float, optional
        Shrinkage parameter ('auto', float between 0 and 1, or None).
    
    Methods
    -------
    fit(X, y) : Fit the LDA model
    transform(X) : Apply dimensionality reduction
    inverse_transform(X) : Transform data back to original space (approximate)
    """

    def __init__(self, n_components: Optional[int]=None, solver: str='svd', shrinkage: Optional[Union[str, float]]=None, name: Optional[str]=None):
        """
        Initialize the LinearDiscriminantAnalysisTransformer.
        
        Parameters
        ----------
        n_components : int, optional
            Number of components to keep. If None, defaults to min(n_classes-1, n_features).
        solver : str, default='svd'
            Solver to use ('svd', 'lsqr', or 'eigen').
        shrinkage : str or float, optional
            Shrinkage parameter ('auto', float between 0 and 1, or None).
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage

    def fit(self, data: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'LinearDiscriminantAnalysisTransformer':
        """
        Fit the LDA model according to the given training data and parameters.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray, optional
            Target values of shape (n_samples,). Required for supervised LDA.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        LinearDiscriminantAnalysisTransformer
            Fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = np.asarray(data)
        if y is None:
            raise ValueError('y is required for LDA transformation')
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if len(X) != len(y):
            raise ValueError('X and y must have the same number of samples')
        (self.n_samples_, self.n_features_) = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError('LDA requires at least 2 classes')
        if self.n_components is None:
            self.n_components_ = min(self.n_classes_ - 1, self.n_features_)
        else:
            if self.n_components > min(self.n_classes_ - 1, self.n_features_):
                raise ValueError(f'n_components cannot be larger than min(n_classes-1, n_features) = {min(self.n_classes_ - 1, self.n_features_)}')
            self.n_components_ = self.n_components
        valid_solvers = ['svd', 'lsqr', 'eigen']
        if self.solver not in valid_solvers:
            raise ValueError(f'solver must be one of {valid_solvers}')
        if self.shrinkage is not None:
            if isinstance(self.shrinkage, str):
                if self.shrinkage != 'auto':
                    raise ValueError("shrinkage can only be 'auto' or a float in [0,1]")
            elif not isinstance(self.shrinkage, (int, float)) or not 0 <= self.shrinkage <= 1:
                raise ValueError("shrinkage must be 'auto' or a float in [0,1]")
        self.means_ = np.zeros((self.n_classes_, self.n_features_))
        for (i, cls) in enumerate(self.classes_):
            self.means_[i] = np.mean(X[y == cls], axis=0)
        self.xbar_ = np.mean(X, axis=0)
        if self.shrinkage is None:
            self.covariance_ = self._estimate_covariance(X, y)
        else:
            cov = self._estimate_covariance(X, y)
            if self.shrinkage == 'auto':
                shrinkage_val = 0.1
            else:
                shrinkage_val = self.shrinkage
            target = np.diag(np.diag(cov))
            self.covariance_ = (1 - shrinkage_val) * cov + shrinkage_val * target
        self._solve_eigensystem()
        self.components_ = self.scalings_[:, :self.n_components_].T
        self.explained_variance_ratio_ = self.discriminant_values_[:self.n_components_] / np.sum(self.discriminant_values_)
        self.is_fitted = True
        return self

    def _estimate_covariance(self, X, y):
        """Estimate the covariance matrix."""
        cov = np.zeros((self.n_features_, self.n_features_))
        for (i, cls) in enumerate(self.classes_):
            X_cls = X[y == cls]
            diff = X_cls - self.means_[i]
            cov += (X_cls.shape[0] - 1) * np.dot(diff.T, diff)
        return cov / (X.shape[0] - self.n_classes_)

    def _solve_eigensystem(self):
        """Solve the eigenvalue problem for LDA."""
        St = self.covariance_ * self.n_samples_
        Sw = np.zeros_like(St)
        for (i, cls) in enumerate(self.classes_):
            ni = np.sum(y == cls)
            diff = (self.means_[i] - self.xbar_).reshape(-1, 1)
            Sw += ni * np.dot(diff, diff.T)
        try:
            Sw_reg = Sw + 1e-09 * np.eye(Sw.shape[0])
            (eigs, eigv) = np.linalg.eigh(np.dot(np.linalg.inv(Sw_reg), St))
        except np.linalg.LinAlgError:
            Sw_reg = Sw + 1e-06 * np.eye(Sw.shape[0])
            (eigs, eigv) = np.linalg.eigh(np.dot(np.linalg.pinv(Sw_reg), St))
        idx = np.argsort(eigs)[::-1]
        self.discriminant_values_ = eigs[idx]
        self.scalings_ = eigv[:, idx]

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply dimensionality reduction to the input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Data to transform of shape (n_samples, n_features).
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet or np.ndarray
            Transformed data of shape (n_samples, n_components).
        """
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            is_feature_set = True
        else:
            X = np.asarray(data)
            is_feature_set = False
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X.shape[1] != self.n_features_:
            raise ValueError(f'X has {X.shape[1]} features, but LDA was fitted with {self.n_features_} features')
        Xc = X - self.xbar_
        X_transformed = np.dot(Xc, self.components_.T)
        if is_feature_set:
            transformed_fs = FeatureSet(features=X_transformed, feature_names=[f'LD{i + 1}' for i in range(X_transformed.shape[1])], metadata={'source': 'LDA transformation'})
            return transformed_fs
        else:
            return X_transformed

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Transform data back to its original space (approximation).
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Data to inverse transform of shape (n_samples, n_components).
        **kwargs : dict
            Additional inverse transformation parameters.
            
        Returns
        -------
        FeatureSet or np.ndarray
            Approximated data in original space of shape (n_samples, n_features).
        """
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            is_feature_set = True
        else:
            X = np.asarray(data)
            is_feature_set = False
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X.shape[1] != self.n_components_:
            raise ValueError(f'X has {X.shape[1]} components, but LDA was fitted with {self.n_components_} components')
        X_original = np.dot(X, self.components_) + self.xbar_
        if is_feature_set:
            reconstructed_fs = FeatureSet(features=X_original, feature_names=[f'feature_{i}' for i in range(X_original.shape[1])] if not hasattr(data, 'feature_names') else data.feature_names, metadata={'source': 'LDA inverse transformation'})
            return reconstructed_fs
        else:
            return X_original

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the amount of variance explained by each of the selected components.
        
        Returns
        -------
        np.ndarray
            Explained variance ratio for each component.
        """
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        return self.explained_variance_ratio_

    def get_components(self) -> np.ndarray:
        """
        Get the linear discriminant components.
        
        Returns
        -------
        np.ndarray
            Linear discriminant components of shape (n_components, n_features).
        """
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        return self.components_

class LinearDiscriminantAnalysisClassifier(BaseModel):
    """
    Linear Discriminant Analysis (LDA) classifier for supervised classification tasks.
    
    This classifier implements LDA for classification by modeling class-conditional densities
    and applying Bayes' theorem to compute posterior probabilities. It assumes that data
    follows a Gaussian distribution within each class and that all classes share the same
    covariance matrix. This makes it particularly effective when classes are well-separated
    and the distribution is approximately normal.
    
    The classifier computes linear discriminant functions that define decision boundaries
    between classes, making predictions by assigning samples to the class with the highest
    discriminant score.
    """

    def __init__(self, solver: str='svd', shrinkage: Optional[Union[str, float]]=None, priors: Optional[np.ndarray]=None, name: Optional[str]=None):
        """
        Initialize the LinearDiscriminantAnalysisClassifier.
        
        This classifier uses Linear Discriminant Analysis to find a linear combination
        of features that best separates classes. It's particularly useful for multi-class
        problems and works well with relatively small training sets.
        
        Parameters
        ----------
        solver : str, default='svd'
            Solver to use ('svd', 'lsqr', or 'eigen'). 'svd' is recommended for better
            numerical stability.
        shrinkage : str or float, optional
            Shrinkage parameter ('auto', float between 0 and 1, or None). Shrinks
            the empirical covariance matrix towards a diagonal matrix to improve
            estimates with limited samples.
        priors : np.ndarray, optional
            Prior probabilities of the classes. If specified, the priors are not
            adjusted according to the data.
        name : str, optional
            Name of the classifier instance.
            
        Raises
        ------
        ValueError
            If parameters are incompatible or invalid.
        """
        super().__init__(name=name)
        if solver not in ('svd', 'lsqr', 'eigen'):
            raise ValueError("solver must be one of 'svd', 'lsqr', or 'eigen'")
        if shrinkage is not None:
            if isinstance(shrinkage, str):
                if shrinkage != 'auto':
                    raise ValueError("shrinkage can only be 'auto' or a float")
            elif not 0 <= shrinkage <= 1:
                raise ValueError('shrinkage must be between 0 and 1')
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self._is_fitted = False

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, List]]=None, **kwargs) -> 'LinearDiscriminantAnalysisClassifier':
        """
        Fit the LDA classifier according to the given training data.
        
        Computes the class means, overall covariance matrix, and class priors from
        the training data. These statistics are then used to define the decision
        boundaries for classification.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray or list, optional
            Target values of shape (n_samples,). Must be provided for supervised learning.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        LinearDiscriminantAnalysisClassifier
            Fitted classifier instance for method chaining.
            
        Raises
        ------
        ValueError
            If target values are not provided or shapes are inconsistent.
        """
        if y is None:
            raise ValueError('Target values (y) must be provided for supervised learning')
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y_array.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        (self.n_samples_, self.n_features_) = X_array.shape
        self.classes_ = np.unique(y_array)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError('The number of classes must be at least 2')
        if self.n_classes_ > self.n_samples_:
            raise ValueError('The number of classes cannot exceed the number of samples')
        if self.priors is not None:
            if len(self.priors) != self.n_classes_:
                raise ValueError('Number of priors must match number of classes')
            if np.any(self.priors < 0) or abs(np.sum(self.priors) - 1.0) > 1e-05:
                raise ValueError('Priors must be non-negative and sum to 1')
            self.class_priors_ = self.priors
        else:
            (_, counts) = np.unique(y_array, return_counts=True)
            self.class_priors_ = counts / float(self.n_samples_)
        self.class_means_ = np.array([np.mean(X_array[y_array == cls], axis=0) for cls in self.classes_])
        self.overall_mean_ = np.average(self.class_means_, axis=0, weights=self.class_priors_)
        if self.shrinkage is None:
            self.covariance_ = self._compute_covariance(X_array, y_array)
        else:
            self.covariance_ = self._compute_shrinkage_covariance(X_array, y_array)
        self._compute_coef_intercept()
        self._is_fitted = True
        return self

    def _compute_covariance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the pooled within-class covariance matrix."""
        cov = np.zeros((self.n_features_, self.n_features_))
        for (idx, cls) in enumerate(self.classes_):
            Xi = X[y == cls]
            diff = Xi - self.class_means_[idx]
            cov += diff.T @ diff
        return cov / (self.n_samples_ - self.n_classes_)

    def _compute_shrinkage_covariance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute shrunk covariance matrix."""
        emp_cov = self._compute_covariance(X, y)
        if self.shrinkage == 'auto':
            shrinkage_param = self._ledoit_wolf_shrinkage(emp_cov, X, y)
        else:
            shrinkage_param = self.shrinkage
        mu = np.trace(emp_cov) / self.n_features_
        shrunk_cov = (1 - shrinkage_param) * emp_cov + shrinkage_param * mu * np.eye(self.n_features_)
        return shrunk_cov

    def _ledoit_wolf_shrinkage(self, emp_cov: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate optimal shrinkage parameter using Ledoit-Wolf lemma."""
        (n_samples, n_features) = X.shape
        X_centered = X - np.mean(X, axis=0)
        emp_cov_sq = emp_cov @ emp_cov
        pi_hat = np.sum(X_centered ** 4) / n_samples - np.sum(emp_cov_sq)
        gamma_hat = np.linalg.norm(emp_cov - np.diag(np.diag(emp_cov))) ** 2
        kappa_hat = pi_hat / gamma_hat
        shrinkage = max(0, min(1, kappa_hat / n_samples))
        return shrinkage

    def _compute_coef_intercept(self):
        """Compute coefficients for decision function."""
        if self.solver == 'svd':
            try:
                (U, S, Vt) = np.linalg.svd(self.covariance_, full_matrices=False)
                tol = S.max() * max(self.covariance_.shape) * np.finfo(S.dtype).eps
                rank = np.sum(S > tol)
                S_inv = np.zeros_like(S)
                S_inv[:rank] = 1 / S[:rank]
                cov_inv = Vt.T * S_inv @ U.T
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(self.covariance_)
        else:
            cov_inv = np.linalg.inv(self.covariance_)
        self.coef_ = (self.class_means_ - self.overall_mean_) @ cov_inv
        self.intercept_ = 0.5 * np.diag(self.class_means_ @ cov_inv @ self.class_means_.T) + np.log(self.class_priors_)

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Uses the learned model parameters to compute the posterior probabilities
        for each class and assigns each sample to the class with the highest
        probability.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Input data of shape (n_samples, n_features).
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).
            
        Raises
        ------
        RuntimeError
            If the classifier has not been fitted yet.
        ValueError
            If the input data dimensions don't match the fitted model.
        """
        if not self._is_fitted:
            raise RuntimeError("This LinearDiscriminantAnalysisClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[1] != self.n_features_:
            raise ValueError(f'X has {X_array.shape[1]} features, but LDA was fitted with {self.n_features_} features.')
        scores = X_array @ self.coef_.T + self.intercept_
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Computes the posterior probability of each class for each sample using
        Bayes' theorem with the assumption of Gaussian class-conditional densities.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Input data of shape (n_samples, n_features).
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (n_samples, n_classes).
            
        Raises
        ------
        RuntimeError
            If the classifier has not been fitted yet.
        ValueError
            If the input data dimensions don't match the fitted model.
        """
        if not self._is_fitted:
            raise RuntimeError("This LinearDiscriminantAnalysisClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[1] != self.n_features_:
            raise ValueError(f'X has {X_array.shape[1]} features, but LDA was fitted with {self.n_features_} features.')
        scores = X_array @ self.coef_.T + self.intercept_
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List], **kwargs) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Computes the fraction of correctly classified samples by comparing predictions
        with true labels.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Test samples of shape (n_samples, n_features).
        y : np.ndarray or list
            True labels for X of shape (n_samples,).
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            Mean accuracy of self.predict(X) wrt. y, in range [0, 1].
            
        Raises
        ------
        RuntimeError
            If the classifier has not been fitted yet.
        ValueError
            If the input data dimensions don't match or labels are inconsistent.
        """
        if not self._is_fitted:
            raise RuntimeError("This LinearDiscriminantAnalysisClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        y_true = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y_true.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X_array.shape[0] != y_true.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        y_pred = self.predict(X_array)
        return np.mean(y_pred == y_true)

    def get_class_means(self) -> np.ndarray:
        """
        Get the class means.
        
        Returns the mean feature vector for each class computed during training.
        
        Returns
        -------
        np.ndarray
            Class means of shape (n_classes, n_features).
            
        Raises
        ------
        RuntimeError
            If the classifier has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("This LinearDiscriminantAnalysisClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return self.class_means_

    def get_covariance(self) -> np.ndarray:
        """
        Get the overall covariance matrix.
        
        Returns the pooled within-class covariance matrix computed during training.
        
        Returns
        -------
        np.ndarray
            Overall covariance matrix of shape (n_features, n_features).
            
        Raises
        ------
        RuntimeError
            If the classifier has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("This LinearDiscriminantAnalysisClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return self.covariance_