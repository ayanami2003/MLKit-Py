from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from sklearn.neighbors import LocalOutlierFactor
from typing import Union, Optional
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN


# ...(code omitted)...


class LocalOutlierFactorDetector(BaseTransformer):
    """
    Detects outliers using the Local Outlier Factor (LOF) algorithm.
    
    LOF measures the local density deviation of a given data point with respect to its neighbors.
    Points with significantly lower density than their neighbors are considered outliers.
    
    Args:
        n_neighbors (int): Number of neighbors to use for LOF calculation.
        algorithm (str): Algorithm to use for nearest neighbor search ('auto', 'ball_tree', 'kd_tree', 'brute').
        leaf_size (int): Leaf size for tree-based algorithms.
        metric (str): Distance metric to use for calculating neighbors.
        contamination (float): Expected proportion of outliers in the data (used for threshold determination).
        novelty (bool): If True, allows predicting outliers for new data; if False, only fits on training data.
    """

    def __init__(self, n_neighbors: int=20, contamination: float=0.1, novelty: bool=False, n_jobs: Optional[int]=None):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs
        self._lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=novelty, n_jobs=n_jobs)

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'LocalOutlierFactorDetector':
        """
        Fit the LOF model to the input data.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self._lof_model.fit(X)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the data and flag outliers based on LOF scores.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            FeatureSet with outlier flags in metadata under 'lof_outliers' key.
        """
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
        if self.novelty:
            outlier_flags = self._lof_model.predict(X) == -1
        else:
            outlier_flags = self._lof_model.fit_predict(X) == -1
        metadata['lof_outliers'] = outlier_flags
        return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        LOF does not support inverse transformation.
        
        Args:
            data: Transformed data (ignored).
            
        Raises:
            NotImplementedError: Always raised as LOF cannot perform inverse transformation.
        """
        raise NotImplementedError('LocalOutlierFactorDetector does not support inverse transformation')

    def predict_outliers(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Predict which samples are outliers.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Boolean array indicating outliers (True) or inliers (False).
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if self.novelty:
            return self._lof_model.predict(X) == -1
        else:
            return self._lof_model.fit_predict(X) == -1

    def get_lof_scores(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Get the LOF scores for each sample.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Array of LOF scores (higher = more outlier-like).
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if self.novelty:
            return -self._lof_model.decision_function(X)
        else:
            temp_model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=False, n_jobs=self.n_jobs)
            temp_model.fit(X)
            return -temp_model.negative_outlier_factor_

    def get_negative_outlier_factor(self) -> np.ndarray:
        """
        Get the opposite of the LOF scores computed during fitting.
        
        Returns:
            Array of negative LOF factors (lower = more outlier-like).
        """
        if hasattr(self._lof_model, 'negative_outlier_factor_'):
            return self._lof_model.negative_outlier_factor_
        else:
            raise RuntimeError('Negative outlier factor not available. Model may not be fitted or novelty mode is enabled.')

    def get_threshold(self) -> float:
        """
        Get the threshold used to determine outliers.
        
        Returns:
            Threshold value for outlier detection.
        """
        if hasattr(self._lof_model, 'offset_'):
            return self._lof_model.offset_
        else:
            raise RuntimeError('Threshold not available. Model may not be fitted or novelty mode is enabled.')

class IsolationForestDetector(BaseTransformer):
    """
    Detects outliers using the Isolation Forest algorithm.
    
    Isolation Forest isolates observations by randomly selecting a feature and then
    randomly selecting a split value between the maximum and minimum values of the selected feature.
    Outliers are easier to isolate and thus have shorter paths in the isolation trees.
    
    Args:
        n_estimators (int): Number of isolation trees in the forest.
        max_samples (Union[int, float, str]): Number of samples to draw for each tree ('auto', int, or float).
        contamination (float): Expected proportion of outliers in the data.
        max_features (float): Proportion of features to consider for each tree.
        bootstrap (bool): If True, samples are drawn with replacement.
        n_jobs (Optional[int]): Number of jobs to run in parallel.
        random_state (Optional[int]): Random seed for reproducibility.
        verbose (int): Controls verbosity of the tree building process.
    """

    def __init__(self, n_estimators: int=100, max_samples: Union[int, float, str]='auto', contamination: float=0.1, max_features: float=1.0, bootstrap: bool=False, n_jobs: Optional[int]=None, random_state: Optional[int]=None, verbose: int=0):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self._isolation_forest = None
        self._fitted = False

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'IsolationForestDetector':
        """
        Fit the isolation forest model to the input data.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        self._isolation_forest = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, contamination=self.contamination, max_features=self.max_features, bootstrap=self.bootstrap, n_jobs=self.n_jobs, random_state=self.random_state, verbose=self.verbose)
        self._isolation_forest.fit(X)
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the data and flag outliers based on isolation forest predictions.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            FeatureSet with outlier flags in metadata.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before transform. Call fit() first.')
        if isinstance(data, FeatureSet):
            result = FeatureSet(features=data.features.copy(), feature_names=data.feature_names.copy() if data.feature_names else None, feature_types=data.feature_types.copy() if data.feature_types else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
            X = data.features
        else:
            X = data
            result = FeatureSet(features=X.copy())
        outlier_flags = self.predict_outliers(X)
        if result.metadata is None:
            result.metadata = {}
        result.metadata['outlier_flags'] = outlier_flags
        return result

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Not implemented for Isolation Forest as it's not a reversible transformation.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Raises:
            NotImplementedError: Always raised as inverse transform is not supported.
        """
        raise NotImplementedError('Isolation Forest does not support inverse transformation.')

    def predict_outliers(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Predict which samples are outliers.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Boolean array indicating outliers (True) or inliers (False).
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before predicting outliers. Call fit() first.')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        predictions = self._isolation_forest.predict(X)
        return predictions == -1

    def get_anomaly_scores(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Calculate anomaly scores for each sample.
        
        Lower scores indicate more abnormal samples.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Array of anomaly scores for each sample.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before getting anomaly scores. Call fit() first.')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        return self._isolation_forest.decision_function(X)

    def get_threshold(self) -> float:
        """
        Get the threshold used for determining outliers.
        
        Returns:
            Threshold value for anomaly scores.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before getting threshold. Call fit() first.')
        return self._isolation_forest.offset_

    def get_feature_importances(self) -> np.ndarray:
        """
        Get the feature importances from the isolation forest.
        
        Returns:
            Array of feature importances.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before getting feature importances. Call fit() first.')
        importances = np.mean([tree.feature_importances_ for tree in self._isolation_forest.estimators_], axis=0)
        return importances

class OneClassSVMDetector(BaseTransformer):
    """
    Detects outliers using the One-Class Support Vector Machine algorithm.
    
    One-Class SVM learns a decision function for novelty detection: 
    classifying new data as similar or different to the training set.
    
    Args:
        kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed').
        degree (int): Degree for polynomial kernel.
        gamma (Union[float, str]): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        coef0 (float): Independent term in kernel function.
        tol (float): Tolerance for stopping criterion.
        nu (float): Upper bound on the fraction of training errors and lower bound on the fraction of support vectors.
        shrinking (bool): Whether to use the shrinking heuristic.
        cache_size (float): Size of the kernel cache (in MB).
        verbose (bool): Enable verbose output.
        max_iter (int): Hard limit on iterations within solver, or -1 for no limit.
    """

    def __init__(self, kernel: str='rbf', degree: int=3, gamma: Union[str, float]='scale', coef0: float=0.0, tol: float=0.001, nu: float=0.5, shrinking: bool=True, cache_size: float=200, verbose: bool=False, max_iter: int=-1):
        super().__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self._svm_model = OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu, shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)
        self._fitted = False

    def _validate_input_data(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Validate input data for One-Class SVM compatibility.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Validated numpy array of shape (n_samples, n_features).
            
        Raises:
            ValueError: If data contains invalid values or wrong dimensions.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if not np.all(np.isfinite(X)):
            raise ValueError('Input data contains non-finite values')
        return X

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'OneClassSVMDetector':
        """
        Fit the One-Class SVM model to the input data.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            Self instance for method chaining.
        """
        X = self._validate_input_data(data)
        self._svm_model.fit(X)
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the data and flag outliers based on One-Class SVM predictions.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            FeatureSet with outlier flags in metadata under 'ocsvm_outliers' key.
        """
        if not self._fitted:
            raise RuntimeError('Model must be fitted before transforming data')
        X = self._validate_input_data(data)
        if isinstance(data, FeatureSet):
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        else:
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        outlier_flags = self.predict_outliers(data)
        metadata['ocsvm_outliers'] = outlier_flags
        return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        One-Class SVM does not support inverse transformation.
        
        Args:
            data: Transformed data (ignored).
            
        Raises:
            NotImplementedError: Always raised as One-Class SVM cannot perform inverse transformation.
        """
        raise NotImplementedError('OneClassSVMDetector does not support inverse transformation')

    def predict_outliers(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Predict which samples are outliers.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Boolean array indicating outliers (True) or inliers (False).
        """
        if not self._fitted:
            raise RuntimeError('Model must be fitted before predicting outliers')
        X = self._validate_input_data(data)
        return self._svm_model.predict(X) == -1

    def get_decision_function_scores(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Get the decision function scores for each sample.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Array of decision function scores (negative = outlier, positive = inlier).
        """
        if not self._fitted:
            raise RuntimeError('Model must be fitted before getting decision function scores')
        X = self._validate_input_data(data)
        return self._svm_model.decision_function(X)

    def get_support_vectors(self) -> np.ndarray:
        """
        Get the support vectors from the fitted model.
        
        Returns:
            Array of support vectors of shape (n_support_vectors, n_features).
        """
        if not self._fitted:
            raise RuntimeError('Model must be fitted before accessing support vectors')
        return self._svm_model.support_vectors_

    def get_support_indices(self) -> np.ndarray:
        """
        Get the indices of support vectors in the training data.
        
        Returns:
            Array of indices indicating positions of support vectors.
        """
        if not self._fitted:
            raise RuntimeError('Model must be fitted before accessing support indices')
        return self._svm_model.support_

class AutoencoderOutlierDetector(BaseTransformer):

    def __init__(self, hidden_layers: List[int]=[64, 32, 64], activation: str='relu', optimizer: str='adam', epochs: int=100, batch_size: int=32, validation_split: float=0.1, dropout_rate: float=0.0, learning_rate: float=0.001, threshold_percentile: float=95.0, random_state: Optional[int]=None):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self._weights = []
        self._biases = []
        self._threshold = None
        self._n_features = None
        self._fitted = False
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _validate_input_data(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Validate input data for autoencoder compatibility.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Validated numpy array of shape (n_samples, n_features).
            
        Raises:
            ValueError: If data contains invalid values or wrong dimensions.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if not np.all(np.isfinite(X)):
            raise ValueError('Input data contains non-finite values')
        return X

    def _initialize_weights(self, input_dim: int) -> None:
        """Initialize weights and biases for the autoencoder."""
        self._weights = []
        self._biases = []
        layers = [input_dim] + self.hidden_layers + [input_dim]
        for i in range(len(layers) - 1):
            limit = np.sqrt(6.0 / (layers[i] + layers[i + 1]))
            weight = np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            bias = np.zeros((1, layers[i + 1]))
            self._weights.append(weight)
            self._biases.append(bias)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            x = np.clip(x, -500, 500)
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'linear':
            return x
        else:
            raise ValueError(f'Unsupported activation function: {self.activation}')

    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f'Unsupported activation function: {self.activation}')

    def _forward_pass(self, X: np.ndarray, training: bool=True) -> tuple:
        """Perform forward pass through the network."""
        activations = [X]
        current = X
        for i in range(len(self._weights)):
            z = np.dot(current, self._weights[i]) + self._biases[i]
            if training and self.dropout_rate > 0 and (i < len(self._weights) - 1):
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, z.shape) / (1 - self.dropout_rate)
                z *= dropout_mask
            if i < len(self._weights) - 1:
                current = self._activate(z)
            else:
                current = z
            activations.append(current)
        return activations

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error loss."""
        return np.mean((y_true - y_pred) ** 2)

    def _backward_pass(self, activations: List[np.ndarray], X: np.ndarray) -> tuple:
        """Perform backward pass and compute gradients."""
        dW = [np.zeros_like(w) for w in self._weights]
        db = [np.zeros_like(b) for b in self._biases]
        m = X.shape[0]
        dz = activations[-1] - X
        for i in reversed(range(len(self._weights))):
            dW[i] = np.dot(activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
            if i > 0:
                dz = np.dot(dz, self._weights[i].T) * self._activate_derivative(activations[i])
        return (dW, db)

    def _update_parameters(self, dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        """Update weights and biases using gradient descent."""
        if self.optimizer == 'adam':
            for i in range(len(self._weights)):
                self._weights[i] -= self.learning_rate * dW[i]
                self._biases[i] -= self.learning_rate * db[i]
        else:
            for i in range(len(self._weights)):
                self._weights[i] -= self.learning_rate * dW[i]
                self._biases[i] -= self.learning_rate * db[i]

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'AutoencoderOutlierDetector':
        """
        Fit the autoencoder model to the input data.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            Self instance for method chaining.
        """
        X = self._validate_input_data(data)
        self._n_features = X.shape[1]
        self._initialize_weights(self._n_features)
        if self.validation_split > 0:
            n_val = int(X.shape[0] * self.validation_split)
            indices = np.random.permutation(X.shape[0])
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            X_train = X[train_indices]
            X_val = X[val_indices]
        else:
            X_train = X
            X_val = None
        reconstruction_errors = []
        for epoch in range(self.epochs):
            indices = np.random.permutation(X_train.shape[0])
            for start_idx in range(0, X_train.shape[0], self.batch_size):
                end_idx = min(start_idx + self.batch_size, X_train.shape[0])
                batch_indices = indices[start_idx:end_idx]
                X_batch = X_train[batch_indices]
                activations = self._forward_pass(X_batch, training=True)
                (dW, db) = self._backward_pass(activations, X_batch)
                self._update_parameters(dW, db)
            if epoch == self.epochs - 1:
                train_activations = self._forward_pass(X_train, training=False)
                train_reconstructions = train_activations[-1]
                reconstruction_errors = np.mean((X_train - train_reconstructions) ** 2, axis=1)
        self._threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the data and flag outliers based on reconstruction errors.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            FeatureSet with outlier flags in metadata.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before transform. Call fit() first.')
        X = self._validate_input_data(data)
        if isinstance(data, FeatureSet):
            result = FeatureSet(features=data.features.copy(), feature_names=data.feature_names.copy() if data.feature_names else None, feature_types=data.feature_types.copy() if data.feature_types else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        else:
            result = FeatureSet(features=X.copy())
        outlier_flags = self.predict_outliers(data)
        if result.metadata is None:
            result.metadata = {}
        result.metadata['outlier_flags'] = outlier_flags
        return result

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reconstruct data from the encoded representation.
        
        Args:
            data: Encoded data as FeatureSet or numpy array.
            
        Returns:
            FeatureSet with reconstructed data.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before inverse transform. Call fit() first.')
        X = self._validate_input_data(data)
        if isinstance(data, FeatureSet):
            result = FeatureSet(features=data.features.copy(), feature_names=data.feature_names.copy() if data.feature_names else None, feature_types=data.feature_types.copy() if data.feature_types else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        else:
            result = FeatureSet(features=X.copy())
        activations = self._forward_pass(X, training=False)
        reconstructed = activations[-1]
        result.features = reconstructed
        return result

    def predict_outliers(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Predict which samples are outliers based on reconstruction errors.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Boolean array indicating outliers (True) or inliers (False).
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before predicting outliers. Call fit() first.')
        X = self._validate_input_data(data)
        reconstruction_errors = self.get_reconstruction_error(data)
        return reconstruction_errors > self._threshold

    def get_reconstruction_error(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Calculate reconstruction errors for each sample.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Array of reconstruction errors for each sample.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before calculating reconstruction errors. Call fit() first.')
        X = self._validate_input_data(data)
        activations = self._forward_pass(X, training=False)
        reconstructions = activations[-1]
        return np.mean((X - reconstructions) ** 2, axis=1)

    def get_encoded_representation(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Get the encoded representation of the input data.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Array of encoded representations.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before getting encoded representations. Call fit() first.')
        X = self._validate_input_data(data)
        activations = self._forward_pass(X, training=False)
        mid_layer_idx = len(self.hidden_layers) // 2
        return activations[mid_layer_idx + 1]

    def get_threshold(self) -> float:
        """
        Get the threshold used for determining outliers.
        
        Returns:
            Threshold value for reconstruction errors.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before getting threshold. Call fit() first.')
        return self._threshold

class KernelDensityEstimationOutlierDetector(BaseTransformer):
    """
    Detects outliers using Kernel Density Estimation.
    
    KDE estimates the probability density function of the data. 
    Samples with low density are considered outliers.
    
    Args:
        bandwidth (Union[float, str]): Bandwidth for the kernel ('scott', 'silverman', or float).
        kernel (str): Kernel type ('gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine').
        algorithm (str): Algorithm to use for KDE ('auto', 'ball_tree', 'kd_tree').
        leaf_size (int): Leaf size for tree-based algorithms.
        metric (str): Distance metric to use.
        threshold_percentile (float): Percentile of density scores to use as threshold for outlier detection.
    """

    def __init__(self, bandwidth: Union[float, str]='scott', kernel: str='gaussian', algorithm: str='auto', leaf_size: int=30, metric: str='euclidean', threshold_percentile: float=5.0):
        super().__init__()
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.threshold_percentile = threshold_percentile
        from sklearn.neighbors import KernelDensity
        self._kde_model = KernelDensity(bandwidth=bandwidth, kernel=kernel, algorithm=algorithm, leaf_size=leaf_size, metric=metric)
        self._fitted = False
        self._threshold = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'KernelDensityEstimationOutlierDetector':
        """
        Fit the KDE model to the input data.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if not np.all(np.isfinite(X)):
            raise ValueError('Input data contains non-finite values')
        self._kde_model.fit(X)
        self._fitted = True
        log_density_scores = self._kde_model.score_samples(X)
        self._threshold = np.percentile(log_density_scores, self.threshold_percentile)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the data and flag outliers based on KDE density scores.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            FeatureSet with outlier flags in metadata.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before transform. Call fit() first.')
        if isinstance(data, FeatureSet):
            result = FeatureSet(features=data.features.copy(), feature_names=data.feature_names.copy() if data.feature_names else None, feature_types=data.feature_types.copy() if data.feature_types else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
            X = data.features
        else:
            X = data
            result = FeatureSet(features=X.copy())
        outlier_flags = self.predict_outliers(X)
        if result.metadata is None:
            result.metadata = {}
        result.metadata['outlier_flags'] = outlier_flags
        return result

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Not applicable for KDE. Raises NotImplementedError.
        
        Args:
            data: Any input data.
            
        Returns:
            Never returns - raises exception.
            
        Raises:
            NotImplementedError: KDE does not support inverse transformation.
        """
        raise NotImplementedError('Kernel Density Estimation does not support inverse transformation.')

    def predict_outliers(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Predict which samples are outliers using KDE density scores.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Boolean array indicating outliers (True) or inliers (False).
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before predicting outliers. Call fit() first.')
        log_density_scores = self.get_log_density_scores(data)
        return log_density_scores < self._threshold

    def get_log_density_scores(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Calculate log density scores for each sample.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Array of log density scores for each sample (more negative scores indicate lower density).
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before getting density scores. Call fit() first.')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        return self._kde_model.score_samples(X)

    def get_threshold(self) -> float:
        """
        Get the threshold used for outlier detection based on training data density scores.
        
        Returns:
            Threshold value for determining outliers.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before getting threshold. Call fit() first.')
        return self._threshold

    def sample(self, n_samples: int=1000, random_state: Optional[int]=None) -> np.ndarray:
        """
        Generate random samples from the fitted KDE model.
        
        Args:
            n_samples: Number of samples to generate.
            random_state: Random seed for reproducibility.
            
        Returns:
            Array of generated samples of shape (n_samples, n_features).
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before sampling. Call fit() first.')
        return self._kde_model.sample(n_samples=n_samples, random_state=random_state)


# ...(code omitted)...


class DBSCANOutlierDetector(BaseTransformer):
    """
    Detects outliers using the DBSCAN clustering algorithm.
    
    DBSCAN groups together points that are closely packed together, marking points in low-density regions as outliers.
    Points labeled as noise by DBSCAN are considered outliers.
    
    Args:
        eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): Number of samples in a neighborhood for a point to be considered as a core point.
        metric (str): Metric to use for distance computation.
        algorithm (str): Algorithm to use for computing nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
        leaf_size (int): Leaf size for tree-based algorithms.
        p (float): Power parameter for Minkowski metric.
        n_jobs (Optional[int]): Number of parallel jobs to run.
    """

    def __init__(self, eps: float=0.5, min_samples: int=5, metric: str='euclidean', metric_params: Optional[dict]=None, algorithm: str='auto', leaf_size: int=30, p: Optional[float]=None, n_jobs: Optional[int]=None):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self._dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs)
        self._fitted = False

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'DBSCANOutlierDetector':
        """
        Fit the DBSCAN model to the input data.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            self: Returns the instance itself.
        """
        X = self._validate_input_data(data)
        if X.shape[0] == 0:
            raise ValueError('Cannot fit DBSCAN model with empty data.')
        if X.shape[0] == 1:
            self._labels_ = np.array([-1])
            self._core_sample_indices_ = np.array([], dtype=int)
            self._n_clusters_ = 0
        else:
            self._dbscan_model.fit(X)
            self._labels_ = self._dbscan_model.labels_
            self._core_sample_indices_ = self._dbscan_model.core_sample_indices_
            self._n_clusters_ = len(set(self._labels_)) - (1 if -1 in self._labels_ else 0)
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the data and flag outliers detected by DBSCAN.
        
        Args:
            data: Input data as FeatureSet or numpy array of shape (n_samples, n_features).
            
        Returns:
            FeatureSet with DBSCAN outlier flags in metadata.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before transform. Call fit() first.')
        X = self._validate_input_data(data)
        if isinstance(data, FeatureSet):
            result = FeatureSet(features=data.features.copy(), feature_names=data.feature_names.copy() if data.feature_names else None, feature_types=data.feature_types.copy() if data.feature_types else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        else:
            result = FeatureSet(features=X.copy())
        outlier_flags = self.predict_outliers(data)
        if result.metadata is None:
            result.metadata = {}
        result.metadata['dbscan_outlier_flags'] = outlier_flags
        return result

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        DBSCAN does not support inverse transformation.
        
        Args:
            data: Input data.
            
        Raises:
            NotImplementedError: Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('DBSCAN does not support inverse transformation.')

    def predict_outliers(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Predict which samples are outliers based on DBSCAN clustering results.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Boolean array indicating outliers (True) or inliers (False).
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before predicting outliers. Call fit() first.')
        X = self._validate_input_data(data)
        if X.shape[0] == 0:
            return np.array([], dtype=bool)
        if X.shape[0] == 1:
            return np.array([True])
        if hasattr(self, '_labels_') and X.shape[0] == len(self._labels_):
            outlier_flags = self._labels_ == -1
        else:
            temp_model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, metric_params=self.metric_params, algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, n_jobs=self.n_jobs)
            temp_labels = temp_model.fit_predict(X)
            outlier_flags = temp_labels == -1
        return outlier_flags.astype(bool)

    def get_cluster_labels(self) -> np.ndarray:
        """
        Get the cluster labels assigned by DBSCAN.
        
        Returns:
            Array of cluster labels for each point. -1 indicates noise/outliers.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before accessing cluster labels. Call fit() first.')
        return self._labels_.copy()

    def get_core_sample_indices(self) -> np.ndarray:
        """
        Get the indices of core samples.
        
        Returns:
            Array of indices of core samples.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before accessing core sample indices. Call fit() first.')
        return self._core_sample_indices_.copy()

    def get_number_of_clusters(self) -> int:
        """
        Get the number of clusters found (excluding noise).
        
        Returns:
            Number of clusters.
        """
        if not self._fitted:
            raise ValueError('Model must be fitted before accessing number of clusters. Call fit() first.')
        return self._n_clusters_

    def _validate_input_data(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Validate and extract features from input data.
        
        Args:
            data: Input data as FeatureSet or numpy array.
            
        Returns:
            Numpy array of features.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError(f'Input data must be FeatureSet or numpy array, got {type(data)}')
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 0:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D array.')
        return X