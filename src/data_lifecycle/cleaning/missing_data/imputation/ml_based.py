from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial.distance import cdist


class KNNImputerTransformer(BaseTransformer):

    def __init__(self, n_neighbors: int=5, weights: str='uniform', metric: str='nan_euclidean', name: Optional[str]=None):
        super().__init__(name=name)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'KNNImputerTransformer':
        """
        Fit the KNN imputer on the input data.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to fit the imputer on.
        **kwargs : dict
            Additional parameters for fitting.

        Returns
        -------
        KNNImputerTransformer
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            X = data.data
        elif isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a DataBatch, FeatureSet, or numpy array')
        self._training_data = X.copy()
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Impute missing values in the input data using the fitted KNN imputer.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to impute.
        **kwargs : dict
            Additional parameters for transformation.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Data with imputed values.
        """
        original_type = type(data)
        if isinstance(data, DataBatch):
            X = data.data.copy()
            original_data = data.data
        elif isinstance(data, FeatureSet):
            X = data.features.copy()
            original_data = data.features
        elif isinstance(data, np.ndarray):
            X = data.copy()
            original_data = data
        else:
            raise ValueError('Input data must be either a DataBatch, FeatureSet, or numpy array')
        imputed_X = self._knn_impute(X)
        if isinstance(data, DataBatch):
            return DataBatch(data=imputed_X, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        elif isinstance(data, FeatureSet):
            return FeatureSet(features=imputed_X, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        else:
            return imputed_X

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for imputation transformers.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Transformed data.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Original data format (identity operation).
        """
        return data

    def _knn_impute(self, X: np.ndarray) -> np.ndarray:
        """
        Perform KNN imputation on the given data.

        Parameters
        ----------
        X : np.ndarray
            Input data with missing values.

        Returns
        -------
        np.ndarray
            Data with imputed values.
        """
        result = X.copy()
        train_data = self._training_data
        missing_mask = np.isnan(X)
        missing_indices = np.where(missing_mask)
        for (i, j) in zip(missing_indices[0], missing_indices[1]):
            distances = self._calculate_distances(X[i], train_data)
            n_neighbors = min(self.n_neighbors, len(train_data))
            if n_neighbors == 0:
                column_mean = np.nanmean(train_data[:, j])
                result[i, j] = column_mean if not np.isnan(column_mean) else 0
                continue
            kth = min(n_neighbors - 1, len(distances) - 1)
            if kth < 0:
                column_mean = np.nanmean(train_data[:, j])
                result[i, j] = column_mean if not np.isnan(column_mean) else 0
                continue
            nearest_indices = np.argpartition(distances, kth)[:n_neighbors]
            neighbor_values = train_data[nearest_indices, j]
            valid_mask = ~np.isnan(neighbor_values)
            if np.sum(valid_mask) == 0:
                column_mean = np.nanmean(train_data[:, j])
                result[i, j] = column_mean if not np.isnan(column_mean) else 0
                continue
            valid_neighbor_values = neighbor_values[valid_mask]
            valid_distances = distances[nearest_indices][valid_mask]
            if self.weights == 'uniform':
                result[i, j] = np.mean(valid_neighbor_values)
            elif self.weights == 'distance':
                if np.any(valid_distances == 0):
                    result[i, j] = valid_neighbor_values[valid_distances == 0][0]
                else:
                    weights = 1.0 / (valid_distances + 1e-08)
                    result[i, j] = np.average(valid_neighbor_values, weights=weights)
            else:
                raise ValueError(f'Unsupported weights: {self.weights}')
        return result

    def _calculate_distances(self, sample: np.ndarray, train_data: np.ndarray) -> np.ndarray:
        """
        Calculate distances between a sample and training data.

        Parameters
        ----------
        sample : np.ndarray
            Sample to calculate distances for.
        train_data : np.ndarray
            Training data to calculate distances to.

        Returns
        -------
        np.ndarray
            Array of distances.
        """
        if self.metric == 'nan_euclidean':
            return self._nan_euclidean_distance(sample, train_data)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(sample, train_data)
        elif self.metric == 'cosine':
            return self._cosine_distance(sample, train_data)
        else:
            sample_2d = sample.reshape(1, -1)
            return cdist(sample_2d, train_data, metric=self.metric).flatten()

    def _nan_euclidean_distance(self, sample: np.ndarray, train_data: np.ndarray) -> np.ndarray:
        """Calculate nan-aware euclidean distance."""
        sample_valid = ~np.isnan(sample)
        train_valid = ~np.isnan(train_data)
        distances = np.zeros(len(train_data))
        for i in range(len(train_data)):
            common_valid = sample_valid & train_valid[i]
            if np.sum(common_valid) == 0:
                distances[i] = np.inf
            else:
                diff = sample[common_valid] - train_data[i, common_valid]
                distances[i] = np.sqrt(np.sum(diff ** 2))
        return distances

    def _manhattan_distance(self, sample: np.ndarray, train_data: np.ndarray) -> np.ndarray:
        """Calculate manhattan distance."""
        sample_valid = ~np.isnan(sample)
        train_valid = ~np.isnan(train_data)
        distances = np.zeros(len(train_data))
        for i in range(len(train_data)):
            common_valid = sample_valid & train_valid[i]
            if np.sum(common_valid) == 0:
                distances[i] = np.inf
            else:
                diff = np.abs(sample[common_valid] - train_data[i, common_valid])
                distances[i] = np.sum(diff)
        return distances

    def _cosine_distance(self, sample: np.ndarray, train_data: np.ndarray) -> np.ndarray:
        """Calculate cosine distance."""
        sample_valid = ~np.isnan(sample)
        train_valid = ~np.isnan(train_data)
        distances = np.zeros(len(train_data))
        for i in range(len(train_data)):
            common_valid = sample_valid & train_valid[i]
            if np.sum(common_valid) == 0:
                distances[i] = np.inf
            else:
                u = sample[common_valid]
                v = train_data[i, common_valid]
                dot_product = np.dot(u, v)
                norm_u = np.linalg.norm(u)
                norm_v = np.linalg.norm(v)
                if norm_u == 0 or norm_v == 0:
                    similarities = 0
                else:
                    similarities = dot_product / (norm_u * norm_v)
                distances[i] = 1 - similarities
        return distances


# ...(code omitted)...


class IterativeImputerTransformer(BaseTransformer):
    """
    Impute missing values using iterative regression models for each feature.

    This transformer uses a round-robin approach where each feature with missing values is modeled
    as a function of other features in a round-robin fashion. It supports various regression models
    for the imputation process and is particularly effective for complex missing data patterns.

    The imputation is performed iteratively, updating missing values with predictions from models
    trained on other features until convergence or a maximum number of iterations is reached.

    Attributes
    ----------
    max_iter : int, default=10
        Maximum number of imputation rounds to perform.
    initial_strategy : str, default='mean'
        Strategy to initialize missing values ('mean', 'median', 'most_frequent', 'constant').
    imputation_order : str, default='ascending'
        Order in which features are imputed ('ascending', 'descending', 'roman', 'arabic', 'random').
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    """

    def __init__(self, max_iter: int=10, initial_strategy: str='mean', imputation_order: str='ascending', random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.random_state = random_state

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'IterativeImputerTransformer':
        """
        Fit the iterative imputer on the input data.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to fit the imputer on.
        **kwargs : dict
            Additional parameters for fitting.

        Returns
        -------
        IterativeImputerTransformer
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            X = data.X.values if hasattr(data.X, 'values') else np.array(data.X)
        elif isinstance(data, FeatureSet):
            X = data.features.values if hasattr(data.features, 'values') else np.array(data.features)
        else:
            X = np.array(data)
        self._original_shape = X.shape
        self._original_dtype = X.dtype
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._statistics = {}
        for i in range(X.shape[1]):
            col_data = X[:, i]
            non_missing = col_data[~np.isnan(col_data)]
            if len(non_missing) == 0:
                self._statistics[i] = 0.0
            elif self.initial_strategy == 'mean':
                self._statistics[i] = np.mean(non_missing)
            elif self.initial_strategy == 'median':
                self._statistics[i] = np.median(non_missing)
            elif self.initial_strategy == 'most_frequent':
                (values, counts) = np.unique(non_missing, return_counts=True)
                self._statistics[i] = values[np.argmax(counts)]
            elif self.initial_strategy == 'constant':
                self._statistics[i] = 0.0
            else:
                raise ValueError(f'Unknown initial_strategy: {self.initial_strategy}')
        n_features = X.shape[1]
        if self.imputation_order == 'ascending':
            self._imputation_sequence = list(range(n_features))
        elif self.imputation_order == 'descending':
            self._imputation_sequence = list(range(n_features - 1, -1, -1))
        elif self.imputation_order == 'roman':
            self._imputation_sequence = []
            asc = list(range(n_features))
            desc = list(range(n_features - 1, -1, -1))
            for i in range(max(len(asc), len(desc))):
                if i < len(asc):
                    self._imputation_sequence.append(asc[i])
                if i < len(desc):
                    self._imputation_sequence.append(desc[i])
            seen = set()
            self._imputation_sequence = [x for x in self._imputation_sequence if not (x in seen or seen.add(x))]
        elif self.imputation_order == 'arabic':
            self._imputation_sequence = list(range(n_features))
            self._imputation_sequence.reverse()
        elif self.imputation_order == 'random':
            self._imputation_sequence = list(range(n_features))
            np.random.shuffle(self._imputation_sequence)
        else:
            raise ValueError(f'Unknown imputation_order: {self.imputation_order}')
        self.is_fitted = True
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Impute missing values in the input data using the fitted iterative imputer.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to impute.
        **kwargs : dict
            Additional parameters for transformation.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Data with imputed values.
        """
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        original_type = type(data)
        if isinstance(data, DataBatch):
            X = data.X.values if hasattr(data.X, 'values') else np.array(data.X)
        elif isinstance(data, FeatureSet):
            X = data.features.values if hasattr(data.features, 'values') else np.array(data.features)
        else:
            X = np.array(data)
        X_imputed = X.copy()
        for i in range(X_imputed.shape[1]):
            col_data = X_imputed[:, i]
            missing_mask = np.isnan(col_data)
            if np.any(missing_mask):
                X_imputed[missing_mask, i] = self._statistics[i]
        for iteration in range(self.max_iter):
            X_previous = X_imputed.copy()
            for feature_idx in self._imputation_sequence:
                missing_mask = np.isnan(X[:, feature_idx]) | np.isnan(X_imputed[:, feature_idx])
                if not np.any(missing_mask):
                    continue
                X_train = X_imputed[~missing_mask, :]
                y_train = X_imputed[~missing_mask, feature_idx]
                X_test = X_imputed[missing_mask, :]
                if len(y_train) == 0:
                    continue
                X_train_other = np.delete(X_train, feature_idx, axis=1)
                X_test_other = np.delete(X_test, feature_idx, axis=1)
                if X_train_other.shape[1] == 0:
                    X_imputed[missing_mask, feature_idx] = self._statistics[feature_idx]
                    continue
                X_train_with_intercept = np.column_stack([np.ones(X_train_other.shape[0]), X_train_other])
                X_test_with_intercept = np.column_stack([np.ones(X_test_other.shape[0]), X_test_other])
                try:
                    XtX = X_train_with_intercept.T @ X_train_with_intercept
                    XtX += np.eye(XtX.shape[0]) * 1e-10
                    coefficients = np.linalg.solve(XtX, X_train_with_intercept.T @ y_train)
                    y_pred = X_test_with_intercept @ coefficients
                    X_imputed[missing_mask, feature_idx] = y_pred
                except np.linalg.LinAlgError:
                    X_imputed[missing_mask, feature_idx] = self._statistics[feature_idx]
            change = np.nansum(np.abs(X_imputed - X_previous))
            if change < 1e-06:
                break
        if original_type == DataBatch:
            if isinstance(data, DataBatch):
                result = DataBatch(X=X_imputed, y=data.y, feature_names=data.feature_names if hasattr(data, 'feature_names') else None)
            else:
                result = DataBatch(X=X_imputed)
            return result
        elif original_type == FeatureSet:
            if isinstance(data, FeatureSet):
                result = FeatureSet(features=X_imputed, feature_names=data.feature_names if hasattr(data, 'feature_names') else None)
            else:
                result = FeatureSet(features=X_imputed)
            return result
        else:
            return X_imputed

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for imputation transformers.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Transformed data.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Original data format (identity operation).

        Raises
        ------
        NotImplementedError
            Always raised as imputation is not invertible.
        """
        raise NotImplementedError('Inverse transform is not supported for imputation transformers.')

class MissForestImputerTransformer(BaseTransformer):

    def __init__(self, n_estimators: int=100, max_depth: Optional[int]=None, min_samples_split: int=2, min_samples_leaf: int=1, max_features: Union[str, int]='sqrt', criterion: str='squared_error', random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'MissForestImputerTransformer':
        """
        Fit the MissForest imputer on the input data.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data containing missing values to fit the imputer on.
        **kwargs : dict
            Additional parameters (ignored).

        Returns
        -------
        MissForestImputerTransformer
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            self._original_type = DataBatch
            X = data.X.values if hasattr(data.X, 'values') else np.array(data.X)
        elif isinstance(data, FeatureSet):
            self._original_type = FeatureSet
            X = data.features.values if hasattr(data.features, 'values') else np.array(data.features)
        else:
            self._original_type = np.ndarray
            X = np.array(data)
        self._n_features = X.shape[1]
        self._categorical_features = []
        for i in range(X.shape[1]):
            col_data = X[:, i]
            unique_vals = np.unique(col_data[~np.isnan(col_data)] if np.issubdtype(col_data.dtype, np.number) else col_data)
            if not np.issubdtype(col_data.dtype, np.number) or len(unique_vals) <= min(10, len(col_data) // 10):
                self._categorical_features.append(i)
        self._initial_statistics = {}
        for i in range(X.shape[1]):
            col_data = X[:, i]
            if np.issubdtype(col_data.dtype, np.number):
                self._initial_statistics[i] = np.nanmean(col_data)
            else:
                (unique_vals, counts) = np.unique(col_data[~pd.isna(col_data)], return_counts=True)
                self._initial_statistics[i] = unique_vals[np.argmax(counts)] if len(unique_vals) > 0 else None
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Impute missing values using the MissForest algorithm.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data containing missing values to impute.
        **kwargs : dict
            Additional parameters (ignored).

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Data with missing values imputed.
        """
        if isinstance(data, DataBatch):
            original_type = DataBatch
            X = data.X.values if hasattr(data.X, 'values') else np.array(data.X)
        elif isinstance(data, FeatureSet):
            original_type = FeatureSet
            X = data.features.values if hasattr(data.features, 'values') else np.array(data.features)
        else:
            original_type = np.ndarray
            X = np.array(data)
        X_imputed = X.copy()
        for i in range(X_imputed.shape[1]):
            col_data = X_imputed[:, i]
            missing_mask = np.isnan(col_data) if np.issubdtype(col_data.dtype, np.number) else pd.isna(col_data)
            if np.any(missing_mask):
                if i in self._categorical_features:
                    X_imputed[missing_mask, i] = self._initial_statistics[i] if self._initial_statistics[i] is not None else 'missing'
                else:
                    X_imputed[missing_mask, i] = self._initial_statistics[i] if not np.isnan(self._initial_statistics[i]) else 0.0
        max_iter = 10
        tol = 1e-06
        for iteration in range(max_iter):
            X_previous = X_imputed.copy()
            for feature_idx in range(self._n_features):
                missing_mask = np.isnan(X[:, feature_idx]) if np.issubdtype(X[:, feature_idx].dtype, np.number) else pd.isna(X[:, feature_idx])
                if not np.any(missing_mask):
                    continue
                X_train = X_imputed[~missing_mask, :]
                y_train = X_imputed[~missing_mask, feature_idx]
                X_test = X_imputed[missing_mask, :]
                if len(y_train) == 0:
                    continue
                if feature_idx in self._categorical_features:
                    model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, random_state=self.random_state)
                    y_train_encoded = pd.Categorical(y_train).codes
                    valid_mask = y_train_encoded != -1
                    if np.sum(valid_mask) == 0:
                        continue
                    X_train_filtered = X_train[valid_mask, :]
                    y_train_filtered = y_train_encoded[valid_mask]
                    if X_train_filtered.shape[0] > 0:
                        model.fit(X_train_filtered, y_train_filtered)
                        if X_test.shape[0] > 0:
                            y_pred_codes = model.predict(X_test)
                            y_pred = pd.Categorical(y_train).categories[y_pred_codes]
                            X_imputed[missing_mask, feature_idx] = y_pred
                else:
                    model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, criterion=self.criterion, random_state=self.random_state)
                    if X_train.shape[0] > 0:
                        model.fit(X_train, y_train)
                        if X_test.shape[0] > 0:
                            y_pred = model.predict(X_test)
                            X_imputed[missing_mask, feature_idx] = y_pred
            change = np.nansum(np.abs(X_imputed - X_previous))
            if change < tol:
                break
        if original_type == DataBatch:
            if isinstance(data, DataBatch):
                result = DataBatch(X=X_imputed, y=data.y, feature_names=data.feature_names if hasattr(data, 'feature_names') else None)
            else:
                result = DataBatch(X=X_imputed)
            return result
        elif original_type == FeatureSet:
            if isinstance(data, FeatureSet):
                result = FeatureSet(features=X_imputed, feature_names=data.feature_names if hasattr(data, 'feature_names') else None)
            else:
                result = FeatureSet(features=X_imputed)
            return result
        else:
            return X_imputed

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for imputation transformers.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Transformed data.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Original data format (identity operation).

        Raises
        ------
        NotImplementedError
            Always raised as imputation is not invertible.
        """
        raise NotImplementedError('Inverse transform is not supported for imputation transformers.')

class GaussianProcessImputerTransformer(BaseTransformer):

    def __init__(self, kernel: Optional[object]=None, alpha: float=1e-10, optimizer: str='fmin_l_bfgs_b', n_restarts_optimizer: int=0, normalize_y: bool=False, copy_X_train: bool=True, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'GaussianProcessImputerTransformer':
        """
        Fit the Gaussian Process imputer on the input data.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to fit the imputer on.
        **kwargs : dict
            Additional parameters for fitting.

        Returns
        -------
        GaussianProcessImputerTransformer
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            X = data.data
        elif isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a DataBatch, FeatureSet, or numpy array')
        self._original_type = type(data)
        if isinstance(data, DataBatch):
            self._feature_names = getattr(data, 'feature_names', None)
            self._labels = getattr(data, 'labels', None)
        elif isinstance(data, FeatureSet):
            self._feature_names = getattr(data, 'feature_names', None)
        else:
            self._feature_names = None
            self._labels = None
        self._training_data = X.copy()
        self._models = {}
        self._feature_means = {}
        missing_mask = np.isnan(X)
        features_with_missing = np.where(np.any(missing_mask, axis=0))[0]
        if self.kernel is None:
            kernel = C(1.0, (0.001, 1000.0)) * RBF(1.0, (0.01, 100.0))
        else:
            kernel = self.kernel
        for feature_idx in features_with_missing:
            non_missing_values = X[~missing_mask[:, feature_idx], feature_idx]
            if len(non_missing_values) > 0:
                self._feature_means[feature_idx] = np.mean(non_missing_values)
            else:
                self._feature_means[feature_idx] = 0.0
            missing_indices = missing_mask[:, feature_idx]
            non_missing_indices = ~missing_indices
            if np.sum(non_missing_indices) == 0:
                continue
            X_train = X[non_missing_indices, :]
            y_train = X[non_missing_indices, feature_idx]
            X_train_features = np.delete(X_train, feature_idx, axis=1)
            if X_train_features.shape[1] > 0 and X_train_features.shape[0] > 0:
                if np.all(X_train_features == X_train_features[0, :], axis=0).all():
                    continue
                gp = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, optimizer=self.optimizer, n_restarts_optimizer=self.n_restarts_optimizer, normalize_y=self.normalize_y, copy_X_train=self.copy_X_train, random_state=self.random_state)
                try:
                    gp.fit(X_train_features, y_train)
                    self._models[feature_idx] = gp
                except Exception:
                    pass
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Impute missing values in the input data using fitted Gaussian Process models.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to impute.
        **kwargs : dict
            Additional parameters for transformation.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Data with missing values imputed.
        """
        if isinstance(data, DataBatch):
            X = data.data
        elif isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a DataBatch, FeatureSet, or numpy array')
        X_imputed = X.copy()
        missing_mask = np.isnan(X)
        features_with_missing = np.where(np.any(missing_mask, axis=0))[0]
        for feature_idx in features_with_missing:
            if feature_idx not in self._models:
                X_imputed[missing_mask[:, feature_idx], feature_idx] = self._feature_means.get(feature_idx, 0.0)
                continue
            missing_indices = missing_mask[:, feature_idx]
            non_missing_indices = ~missing_indices
            if np.sum(non_missing_indices) == 0:
                X_imputed[missing_mask[:, feature_idx], feature_idx] = self._feature_means.get(feature_idx, 0.0)
                continue
            X_train = X_imputed[non_missing_indices, :][:, [i for i in range(X_imputed.shape[1]) if i != feature_idx]]
            y_train = X_imputed[non_missing_indices, feature_idx]
            if X_train.shape[0] == 0:
                X_imputed[missing_mask[:, feature_idx], feature_idx] = self._feature_means.get(feature_idx, 0.0)
                continue
            X_test = X_imputed[missing_indices, :][:, [i for i in range(X_imputed.shape[1]) if i != feature_idx]]
            if X_test.shape[0] > 0:
                try:
                    (y_pred, _) = self._models[feature_idx].predict(X_test, return_std=True)
                    X_imputed[missing_mask[:, feature_idx], feature_idx] = y_pred
                except:
                    X_imputed[missing_mask[:, feature_idx], feature_idx] = self._feature_means.get(feature_idx, 0.0)
        if isinstance(data, DataBatch):
            return DataBatch(data=X_imputed, labels=getattr(data, 'labels', None), metadata=getattr(data, 'metadata', None), sample_ids=getattr(data, 'sample_ids', None), feature_names=getattr(data, 'feature_names', None), batch_id=getattr(data, 'batch_id', None))
        elif isinstance(data, FeatureSet):
            return FeatureSet(features=X_imputed, feature_names=getattr(data, 'feature_names', None), feature_types=getattr(data, 'feature_types', None), sample_ids=getattr(data, 'sample_ids', None), metadata=getattr(data, 'metadata', None), quality_scores=getattr(data, 'quality_scores', None))
        else:
            return X_imputed

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for imputation transformers.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Transformed data.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Original data format (identity operation).

        Raises
        ------
        NotImplementedError
            Always raised as imputation is not invertible.
        """
        return data

class PerformGaussianProcessImputerTransformer(BaseTransformer):

    def __init__(self, kernel: Optional[object]=None, alpha: float=1e-10, optimizer: str='fmin_l_bfgs_b', n_restarts_optimizer: int=0, normalize_y: bool=False, copy_X_train: bool=True, random_state: Optional[int]=None, n_targets: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.n_targets = n_targets

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'PerformGaussianProcessImputerTransformer':
        """
        Fit the enhanced Gaussian Process imputer on the input data.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to fit the imputer on.
        **kwargs : dict
            Additional parameters for fitting.

        Returns
        -------
        PerformGaussianProcessImputerTransformer
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            X = data.data
        elif isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a DataBatch, FeatureSet, or numpy array')
        self._original_type = type(data)
        if isinstance(data, DataBatch):
            self._feature_names = getattr(data, 'feature_names', None)
            self._labels = getattr(data, 'labels', None)
        elif isinstance(data, FeatureSet):
            self._feature_names = getattr(data, 'feature_names', None)
        else:
            self._feature_names = None
            self._labels = None
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self._training_data = X.copy()
        self._models = {}
        self._feature_means = {}
        missing_mask = np.isnan(X)
        features_with_missing = np.where(np.any(missing_mask, axis=0))[0]
        if self.kernel is None:
            kernel = C(1.0, (0.001, 1000.0)) * RBF(1.0, (0.01, 100.0))
        else:
            kernel = self.kernel
        for feature_idx in features_with_missing:
            non_missing_values = X[~missing_mask[:, feature_idx], feature_idx]
            if len(non_missing_values) > 0:
                self._feature_means[feature_idx] = np.mean(non_missing_values)
            else:
                self._feature_means[feature_idx] = 0.0
            missing_indices = missing_mask[:, feature_idx]
            non_missing_indices = ~missing_indices
            if np.sum(non_missing_indices) == 0:
                continue
            X_train = X[non_missing_indices, :]
            y_train = X[non_missing_indices, feature_idx]
            X_train_features = np.delete(X_train, feature_idx, axis=1)
            if X_train_features.shape[1] > 0 and X_train_features.shape[0] > 0:
                if not np.all(X_train_features == X_train_features[0, :], axis=0).all():
                    gp = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, optimizer=self.optimizer, n_restarts_optimizer=self.n_restarts_optimizer, normalize_y=self.normalize_y, copy_X_train=self.copy_X_train, random_state=self.random_state)
                    try:
                        gp.fit(X_train_features, y_train)
                        self._models[feature_idx] = gp
                    except Exception:
                        pass
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Impute missing values in the input data using the fitted enhanced Gaussian Process imputer.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to impute.
        **kwargs : dict
            Additional parameters for transformation.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Data with imputed values.
        """
        if isinstance(data, DataBatch):
            X = data.data
        elif isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a DataBatch, FeatureSet, or numpy array')
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X_imputed = X.copy()
        missing_mask = np.isnan(X)
        features_with_missing = np.where(np.any(missing_mask, axis=0))[0]
        for feature_idx in features_with_missing:
            if feature_idx not in self._models:
                X_imputed[missing_mask[:, feature_idx], feature_idx] = self._feature_means.get(feature_idx, 0.0)
                continue
            missing_indices = missing_mask[:, feature_idx]
            non_missing_indices = ~missing_indices
            if np.sum(non_missing_indices) == 0:
                X_imputed[missing_mask[:, feature_idx], feature_idx] = self._feature_means.get(feature_idx, 0.0)
                continue
            X_train = X_imputed[non_missing_indices, :]
            X_test = X_imputed[missing_indices, :]
            if X_train.shape[0] > 0 and X_test.shape[0] > 0:
                try:
                    X_train_features = np.delete(X_train, feature_idx, axis=1)
                    y_train = X_train[:, feature_idx]
                    X_test_features = np.delete(X_test, feature_idx, axis=1)
                    if X_train_features.shape[1] > 0 and (not np.all(X_train_features == X_train_features[0, :], axis=0).all()):
                        (y_pred, _) = self._models[feature_idx].predict(X_test_features, return_std=True)
                        X_imputed[missing_mask[:, feature_idx], feature_idx] = y_pred
                    else:
                        X_imputed[missing_mask[:, feature_idx], feature_idx] = self._feature_means.get(feature_idx, 0.0)
                except:
                    X_imputed[missing_mask[:, feature_idx], feature_idx] = self._feature_means.get(feature_idx, 0.0)
            else:
                X_imputed[missing_mask[:, feature_idx], feature_idx] = self._feature_means.get(feature_idx, 0.0)
        if isinstance(data, DataBatch):
            return DataBatch(data=X_imputed, labels=getattr(data, 'labels', self._labels), metadata=getattr(data, 'metadata', None), sample_ids=getattr(data, 'sample_ids', None), feature_names=getattr(data, 'feature_names', self._feature_names), batch_id=getattr(data, 'batch_id', None))
        elif isinstance(data, FeatureSet):
            return FeatureSet(features=X_imputed, feature_names=getattr(data, 'feature_names', self._feature_names), feature_types=getattr(data, 'feature_types', None), sample_ids=getattr(data, 'sample_ids', None), metadata=getattr(data, 'metadata', None), quality_scores=getattr(data, 'quality_scores', None))
        else:
            return X_imputed

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for imputation transformers.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Transformed data.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Original data format (identity operation).
        """
        return data

class ExpectationMaximizationImputerTransformer(BaseTransformer):
    """
    Impute missing values using the Expectation-Maximization (EM) algorithm.

    This transformer uses the EM algorithm to estimate missing values under the assumption
    that data follows a Gaussian distribution. It iteratively performs E-step (estimate
    missing values based on current parameters) and M-step (update parameters based on
    complete data) until convergence.

    The method is particularly effective when data is assumed to be normally distributed
    and when there are moderate amounts of missing data.

    Attributes
    ----------
    max_iter : int, default=100
        Maximum number of EM iterations to perform.
    tol : float, default=1e-3
        Tolerance for convergence checking.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance matrices.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    """

    def __init__(self, max_iter: int=100, tol: float=0.001, reg_covar: float=1e-06, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'ExpectationMaximizationImputerTransformer':
        """
        Fit the EM imputer on the input data.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to fit the imputer on.
        **kwargs : dict
            Additional parameters for fitting.

        Returns
        -------
        ExpectationMaximizationImputerTransformer
            Self instance for method chaining.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(data, DataBatch):
            X = np.array(data.data)
        elif isinstance(data, FeatureSet):
            X = data.features.copy()
        else:
            X = data.copy()
        self.original_shape_ = X.shape
        missing_mask = np.isnan(X)
        X_observed = X.copy()
        observed_mask = ~missing_mask
        self.mean_ = np.full(X.shape[1], np.nan)
        for j in range(X.shape[1]):
            observed_values = X_observed[:, j][observed_mask[:, j]]
            if len(observed_values) > 0:
                self.mean_[j] = np.mean(observed_values)
            else:
                self.mean_[j] = 0.0
        self.covariance_ = np.eye(X.shape[1])
        n_observed_features = np.sum(observed_mask, axis=0)
        valid_features = n_observed_features > 1
        if np.any(valid_features):
            X_centered = X_observed - self.mean_
            X_centered[missing_mask] = 0
            cov = np.zeros((X.shape[1], X.shape[1]))
            for i in range(X.shape[1]):
                for j in range(X.shape[1]):
                    if valid_features[i] and valid_features[j]:
                        both_observed = observed_mask[:, i] & observed_mask[:, j]
                        if np.sum(both_observed) > 1:
                            cov[i, j] = np.sum(X_centered[both_observed, i] * X_centered[both_observed, j]) / np.sum(both_observed)
                        else:
                            cov[i, j] = 0 if i != j else 1
                    else:
                        cov[i, j] = 0 if i != j else 1
            self.covariance_ = cov
        self.covariance_ += self.reg_covar * np.eye(self.covariance_.shape[0])
        prev_mean = self.mean_.copy()
        prev_cov = self.covariance_.copy()
        for iteration in range(self.max_iter):
            X_imputed = self._impute_missing_values(X_observed, missing_mask)
            self.mean_ = np.mean(X_imputed, axis=0)
            X_centered = X_imputed - self.mean_
            self.covariance_ = np.dot(X_centered.T, X_centered) / X.shape[0]
            self.covariance_ += self.reg_covar * np.eye(self.covariance_.shape[0])
            mean_diff = np.linalg.norm(self.mean_ - prev_mean)
            cov_diff = np.linalg.norm(self.covariance_ - prev_cov)
            if mean_diff < self.tol and cov_diff < self.tol:
                break
            prev_mean = self.mean_.copy()
            prev_cov = self.covariance_.copy()
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Impute missing values in the input data using the fitted EM imputer.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Input data with missing values to impute.
        **kwargs : dict
            Additional parameters for transformation.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Data with imputed values.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("This transformer has not been fitted yet. Call 'fit' before using this method.")
        original_type = type(data)
        if isinstance(data, DataBatch):
            X = np.array(data.data)
            original_data_ref = data.data
        elif isinstance(data, FeatureSet):
            X = data.features.copy()
            original_features_ref = data.features
        else:
            X = data.copy()
        if X.shape[1] != self.original_shape_[1]:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.original_shape_[1]} features.')
        missing_mask = np.isnan(X)
        X_imputed = self._impute_missing_values(X, missing_mask)
        if original_type == DataBatch:
            result_data = data.__class__(data=X_imputed, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        elif original_type == FeatureSet:
            result_data = data.__class__(features=X_imputed, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        else:
            result_data = X_imputed
        return result_data

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Union[DataBatch, FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for imputation transformers.

        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Transformed data.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        Union[DataBatch, FeatureSet, np.ndarray]
            Original data format (identity operation).
        """
        raise NotImplementedError('Inverse transformation is not applicable for imputation transformers.')

    def _impute_missing_values(self, X: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Impute missing values using the current Gaussian parameters.

        Parameters
        ----------
        X : np.ndarray
            Input data with missing values.
        missing_mask : np.ndarray
            Boolean mask indicating missing values.

        Returns
        -------
        np.ndarray
            Data with imputed values.
        """
        X_imputed = X.copy()
        for i in range(X.shape[0]):
            missing_indices = np.where(missing_mask[i])[0]
            observed_indices = np.where(~missing_mask[i])[0]
            if len(missing_indices) == 0:
                continue
            if len(observed_indices) == 0:
                X_imputed[i, missing_indices] = self.mean_[missing_indices]
                continue
            mu_obs = self.mean_[observed_indices]
            mu_miss = self.mean_[missing_indices]
            sigma_obs_obs = self.covariance_[np.ix_(observed_indices, observed_indices)]
            sigma_miss_obs = self.covariance_[np.ix_(missing_indices, observed_indices)]
            try:
                sigma_obs_obs_inv = np.linalg.inv(sigma_obs_obs)
            except np.linalg.LinAlgError:
                sigma_obs_obs_inv = np.linalg.pinv(sigma_obs_obs)
            x_obs = X_imputed[i, observed_indices]
            cond_mean = mu_miss + np.dot(np.dot(sigma_miss_obs, sigma_obs_obs_inv), x_obs - mu_obs)
            X_imputed[i, missing_indices] = cond_mean
        return X_imputed