from typing import Optional, List, Union, Callable, Dict, Any
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


# ...(code omitted)...


class NearestNeighborsImputationTransformer(BaseTransformer):

    def __init__(self, n_neighbors: int=5, distance_metric: str='euclidean', columns: Optional[List[str]]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.columns = columns
        self._training_data = None
        self._fitted_columns = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'NearestNeighborsImputationTransformer':
        """
        Stores complete training samples for neighbor lookup during imputation.

        Args:
            data (Union[FeatureSet, np.ndarray]): Complete training data to use for neighbor lookup.
            **kwargs: Additional fitting parameters.

        Returns:
            NearestNeighborsImputationTransformer: Fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._fitted_columns = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            self._fitted_columns = None
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        supported_metrics = ['euclidean', 'manhattan']
        if self.distance_metric not in supported_metrics:
            raise ValueError(f"Unsupported distance metric '{self.distance_metric}'. Supported metrics: {supported_metrics}")
        if isinstance(X, np.ndarray):
            complete_mask = ~np.isnan(X).any(axis=1)
            self._training_data = X[complete_mask].copy()
        else:
            raise TypeError('Data must be convertible to numpy array')
        if self.columns is not None and self._fitted_columns is not None:
            missing_cols = set(self.columns) - set(self._fitted_columns)
            if missing_cols:
                raise ValueError(f'Specified columns not found in data: {missing_cols}')
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Applies nearest neighbors imputation to fill missing values in the data.

        Args:
            data (Union[FeatureSet, np.ndarray]): Data with missing values to impute.
            **kwargs: Additional transformation parameters.

        Returns:
            Union[FeatureSet, np.ndarray]: Data with missing values filled using nearest neighbors.
        """
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, np.ndarray):
            X = data.copy()
            feature_names = None
            sample_ids = None
            metadata = None
            quality_scores = None
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if self.columns is not None and feature_names is not None:
            column_indices = [feature_names.index(col) for col in self.columns if col in feature_names]
        else:
            column_indices = list(range(X.shape[1]))
        result = self._impute_missing_values(X, column_indices)
        if isinstance(data, FeatureSet):
            return FeatureSet(features=result, feature_names=feature_names, feature_types=data.feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return result

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation is not supported for imputation methods.

        Args:
            data (Union[FeatureSet, np.ndarray]): Data to inversely transform.
            **kwargs: Additional parameters.

        Returns:
            Union[FeatureSet, np.ndarray]: Returns the input data unchanged.
        """
        pass

    def _impute_missing_values(self, X: np.ndarray, column_indices: List[int]) -> np.ndarray:
        """
        Impute missing values using nearest neighbors.

        Args:
            X (np.ndarray): Data array with missing values.
            column_indices (List[int]): Indices of columns to impute.

        Returns:
            np.ndarray: Data with imputed values.
        """
        result = X.copy()
        missing_mask = np.isnan(X)
        rows_with_missing = np.where(np.any(missing_mask[:, column_indices], axis=1))[0]
        for row_idx in rows_with_missing:
            for col_idx in column_indices:
                if missing_mask[row_idx, col_idx]:
                    neighbors = self._find_nearest_neighbors(X[row_idx], self.n_neighbors)
                    if len(neighbors) > 0:
                        neighbor_values = self._training_data[neighbors, col_idx]
                        valid_values = neighbor_values[~np.isnan(neighbor_values)]
                        if len(valid_values) > 0:
                            result[row_idx, col_idx] = np.mean(valid_values)
                        else:
                            column_mean = np.nanmean(self._training_data[:, col_idx])
                            result[row_idx, col_idx] = column_mean if not np.isnan(column_mean) else 0
                    else:
                        column_mean = np.nanmean(self._training_data[:, col_idx])
                        result[row_idx, col_idx] = column_mean if not np.isnan(column_mean) else 0
        return result

    def _find_nearest_neighbors(self, sample: np.ndarray, k: int) -> np.ndarray:
        """
        Find k nearest neighbors for a given sample.

        Args:
            sample (np.ndarray): Sample to find neighbors for.
            k (int): Number of neighbors to find.

        Returns:
            np.ndarray: Indices of nearest neighbors.
        """
        if len(self._training_data) == 0:
            return np.array([])
        distances = self._calculate_distances(sample, self._training_data)
        if len(distances) < k:
            k = len(distances)
        neighbor_indices = np.argpartition(distances, k - 1)[:k]
        sorted_indices = neighbor_indices[np.argsort(distances[neighbor_indices])]
        return sorted_indices

    def _calculate_distances(self, sample: np.ndarray, train_data: np.ndarray) -> np.ndarray:
        """
        Calculate distances between a sample and training data.

        Args:
            sample (np.ndarray): Sample to calculate distances for.
            train_data (np.ndarray): Training data to calculate distances to.

        Returns:
            np.ndarray: Array of distances.
        """
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(sample, train_data)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(sample, train_data)
        else:
            raise ValueError(f'Unsupported distance metric: {self.distance_metric}')

    def _euclidean_distance(self, sample: np.ndarray, train_data: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance between a sample and training data,
        handling missing values appropriately.

        Args:
            sample (np.ndarray): Sample to calculate distances for.
            train_data (np.ndarray): Training data to calculate distances to.

        Returns:
            np.ndarray: Array of Euclidean distances.
        """
        sample_valid = ~np.isnan(sample)
        train_valid = ~np.isnan(train_data)
        distances = np.full(len(train_data), np.inf)
        for i in range(len(train_data)):
            common_valid = sample_valid & train_valid[i]
            if np.sum(common_valid) > 0:
                diff = sample[common_valid] - train_data[i, common_valid]
                distances[i] = np.sqrt(np.sum(diff ** 2))
        return distances

    def _manhattan_distance(self, sample: np.ndarray, train_data: np.ndarray) -> np.ndarray:
        """
        Calculate Manhattan distance between a sample and training data,
        handling missing values appropriately.

        Args:
            sample (np.ndarray): Sample to calculate distances for.
            train_data (np.ndarray): Training data to calculate distances to.

        Returns:
            np.ndarray: Array of Manhattan distances.
        """
        sample_valid = ~np.isnan(sample)
        train_valid = ~np.isnan(train_data)
        distances = np.full(len(train_data), np.inf)
        for i in range(len(train_data)):
            common_valid = sample_valid & train_valid[i]
            if np.sum(common_valid) > 0:
                diff = np.abs(sample[common_valid] - train_data[i, common_valid])
                distances[i] = np.sum(diff)
        return distances


# ...(code omitted)...


class SequentialImputationTransformer(BaseTransformer):
    """
    Implements sequential imputation to fill missing values considering temporal or sequential dependencies.

    This transformer fills missing values by leveraging the sequential nature of the data, using
    previous observations to inform imputations for later time points. It supports both forward
    and bidirectional imputation strategies and can incorporate temporal patterns in the data.

    Attributes:
        direction (str): Direction of imputation ('forward', 'backward', 'both').
        max_gap (int): Maximum gap size to interpolate across.
        temporal_columns (Optional[List[str]]): Columns representing temporal indices.
        columns (Optional[List[str]]): Specific columns to apply imputation to.
        name (Optional[str]): Name identifier for the transformer.

    Methods:
        fit: Prepares for sequential imputation by analyzing temporal structure.
        transform: Applies sequential imputation to fill missing values.
        inverse_transform: Not supported for imputation methods.
    """

    def __init__(self, direction: str='forward', max_gap: int=5, temporal_columns: Optional[List[str]]=None, columns: Optional[List[str]]=None, name: Optional[str]=None):
        super().__init__(name=name)
        if direction not in ['forward', 'backward', 'both']:
            raise ValueError("direction must be one of 'forward', 'backward', or 'both'")
        if max_gap <= 0:
            raise ValueError('max_gap must be a positive integer')
        self.direction = direction
        self.max_gap = max_gap
        self.temporal_columns = temporal_columns
        self.columns = columns
        self._temporal_structure = None
        self._fitted_columns = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'SequentialImputationTransformer':
        """
        Analyzes temporal structure and prepares for sequential imputation.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Sequential input data to analyze.
            **kwargs: Additional fitting parameters.
            
        Returns:
            SequentialImputationTransformer: Fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = [f'col_{i}' for i in range(X.shape[1])] if X.ndim > 1 else ['col_0']
        else:
            raise TypeError('data must be either FeatureSet or numpy array')
        if self.columns is None:
            self._fitted_columns = list(range(X.shape[1])) if X.ndim > 1 else [0]
        elif isinstance(data, FeatureSet):
            col_to_idx = {name: i for (i, name) in enumerate(feature_names)}
            try:
                self._fitted_columns = [col_to_idx[col] for col in self.columns]
            except KeyError as e:
                raise ValueError(f'Column {e} not found in data') from e
        else:
            self._fitted_columns = self.columns
        max_col_idx = X.shape[1] - 1 if X.ndim > 1 else 0
        for col_idx in self._fitted_columns:
            if not 0 <= col_idx <= max_col_idx:
                raise ValueError(f'Column index {col_idx} out of range [0, {max_col_idx}]')
        if self.temporal_columns is not None and isinstance(data, FeatureSet):
            col_to_idx = {name: i for (i, name) in enumerate(feature_names)}
            try:
                self._temporal_structure = [col_to_idx[col] for col in self.temporal_columns]
            except KeyError as e:
                raise ValueError(f'Temporal column {e} not found in data') from e
        else:
            self._temporal_structure = None
        self.is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Applies sequential imputation to fill missing values respecting temporal dependencies.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Sequential data with missing values to impute.
            **kwargs: Additional transformation parameters.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Sequential data with missing values filled using temporal patterns.
        """
        if not self.is_fitted:
            raise ValueError('Transformer must be fitted before transform')
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            is_feature_set = True
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, np.ndarray):
            X = data.copy()
            is_feature_set = False
        else:
            raise TypeError('data must be either FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for col_idx in self._fitted_columns:
            col_data = X[:, col_idx]
            if not np.isnan(col_data).any():
                continue
            if self.direction == 'forward':
                X[:, col_idx] = self._forward_fill(col_data, self.max_gap)
            elif self.direction == 'backward':
                X[:, col_idx] = self._backward_fill(col_data, self.max_gap)
            elif self.direction == 'both':
                forward_filled = self._forward_fill(col_data, self.max_gap)
                backward_filled = self._backward_fill(col_data, self.max_gap)
                X[:, col_idx] = self._combine_directions(col_data, forward_filled, backward_filled)
        if is_feature_set:
            return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return X.reshape(-1) if data.ndim == 1 else X

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation is not supported for imputation methods.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Data to inversely transform.
            **kwargs: Additional parameters.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Returns the input data unchanged.
        """
        import warnings
        warnings.warn('inverse_transform is not supported for imputation methods. Returning input unchanged.', UserWarning, stacklevel=2)
        return data

    def _forward_fill(self, data: np.ndarray, max_gap: int) -> np.ndarray:
        """Apply forward fill with maximum gap constraint."""
        result = data.copy()
        last_valid_idx = None
        for i in range(len(data)):
            if not np.isnan(data[i]):
                last_valid_idx = i
            elif last_valid_idx is not None and i - last_valid_idx <= max_gap:
                result[i] = result[last_valid_idx]
        return result

    def _backward_fill(self, data: np.ndarray, max_gap: int) -> np.ndarray:
        """Apply backward fill with maximum gap constraint."""
        result = data.copy()
        next_valid_idx = None
        for i in range(len(data) - 1, -1, -1):
            if not np.isnan(data[i]):
                next_valid_idx = i
            elif next_valid_idx is not None and next_valid_idx - i <= max_gap:
                result[i] = result[next_valid_idx]
        return result

    def _combine_directions(self, original: np.ndarray, forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
        """Combine forward and backward filled arrays, preferring closer neighbors."""
        result = original.copy()
        for i in range(len(original)):
            if np.isnan(original[i]):
                if not np.isnan(forward[i]) and (not np.isnan(backward[i])):
                    result[i] = (forward[i] + backward[i]) / 2
                elif not np.isnan(forward[i]):
                    result[i] = forward[i]
                elif not np.isnan(backward[i]):
                    result[i] = backward[i]
        return result


# ...(code omitted)...


class MultipleImputationTransformer(BaseTransformer):
    """
    Implements multiple imputation to handle missing data by generating several complete datasets.
    
    This transformer creates multiple imputed datasets by applying a specified single imputation
    method with added randomness to account for uncertainty in the imputation process.
    
    Attributes:
        n_imputations (int): Number of imputed datasets to generate.
        imputation_method (str): Method used for single imputation ('mean', 'median', 'regression', 'knn').
        columns (Optional[List[str]]): Specific columns to apply imputation to.
        name (Optional[str]): Name identifier for the transformer.
        _fitted_params (Dict): Parameters fitted from training data.
        _feature_names (List[str]): Names of features in the dataset.
        _column_indices (List[int]): Indices of columns to impute.
        
    Methods:
        fit: Computes parameters needed for imputation.
        transform: Generates multiple imputed datasets.
        inverse_transform: Returns input data unchanged.
    """

    def __init__(self, n_imputations: int=5, imputation_method: str='mean', columns: Optional[List[str]]=None, name: Optional[str]=None):
        super().__init__(name=name)
        if n_imputations <= 0:
            raise ValueError('n_imputations must be a positive integer')
        if imputation_method not in ['mean', 'median', 'regression', 'knn']:
            raise ValueError("imputation_method must be one of 'mean', 'median', 'regression', 'knn'")
        self.n_imputations = n_imputations
        self.imputation_method = imputation_method
        self.columns = columns
        self._fitted_params = {}
        self._feature_names = []
        self._column_indices = []

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'MultipleImputationTransformer':
        """
        Compute and store parameters needed for multiple imputation.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit parameters on.
            **kwargs: Additional fitting parameters.
            
        Returns:
            MultipleImputationTransformer: Fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            self._feature_names = [f'col_{i}' for i in range(X.shape[1])] if X.ndim > 1 else ['col_0']
        else:
            raise TypeError('data must be either FeatureSet or numpy array')
        if self.columns is None:
            self._column_indices = list(range(X.shape[1])) if X.ndim > 1 else [0]
        elif isinstance(data, FeatureSet):
            col_to_idx = {name: i for (i, name) in enumerate(self._feature_names)}
            try:
                self._column_indices = [col_to_idx[col] for col in self.columns]
            except KeyError as e:
                raise ValueError(f'Column {e} not found in data') from e
        else:
            self._column_indices = self.columns
        max_col_idx = X.shape[1] - 1 if X.ndim > 1 else 0
        for col_idx in self._column_indices:
            if not 0 <= col_idx <= max_col_idx:
                raise ValueError(f'Column index {col_idx} out of range [0, {max_col_idx}]')
        self._fitted_params = {}
        for col_idx in self._column_indices:
            col_data = X[:, col_idx] if X.ndim > 1 else X
            if self.imputation_method == 'mean':
                non_missing = col_data[~np.isnan(col_data)]
                if len(non_missing) > 0:
                    self._fitted_params[col_idx] = {'mean': np.mean(non_missing), 'std': np.std(non_missing) if len(non_missing) > 1 else 0}
                else:
                    self._fitted_params[col_idx] = {'mean': 0, 'std': 1}
            elif self.imputation_method == 'median':
                non_missing = col_data[~np.isnan(col_data)]
                if len(non_missing) > 0:
                    self._fitted_params[col_idx] = {'median': np.median(non_missing), 'mad': np.median(np.abs(non_missing - np.median(non_missing))) if len(non_missing) > 1 else 0}
                else:
                    self._fitted_params[col_idx] = {'median': 0, 'mad': 1}
            elif self.imputation_method == 'regression':
                self._fitted_params[col_idx] = {'method': 'regression'}
            elif self.imputation_method == 'knn':
                self._fitted_params[col_idx] = {'method': 'knn'}
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> List[Union[FeatureSet, np.ndarray]]:
        """
        Generate multiple imputed datasets.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to impute.
            **kwargs: Additional transformation parameters.
            
        Returns:
            List[Union[FeatureSet, np.ndarray]]: List of imputed datasets.
        """
        is_feature_set = isinstance(data, FeatureSet)
        if is_feature_set:
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, np.ndarray):
            X = data.copy()
            feature_names = [f'col_{i}' for i in range(X.shape[1])] if X.ndim > 1 else ['col_0']
        else:
            raise TypeError('data must be either FeatureSet or numpy array')
        imputed_datasets = []
        for _ in range(self.n_imputations):
            X_imp = X.copy()
            for col_idx in self._column_indices:
                if X.ndim > 1:
                    col_data = X_imp[:, col_idx]
                else:
                    col_data = X_imp
                missing_mask = np.isnan(col_data)
                if not np.any(missing_mask):
                    continue
                if self.imputation_method == 'mean':
                    params = self._fitted_params[col_idx]
                    mean_val = params['mean']
                    std_val = params['std']
                    noise = np.random.normal(0, std_val * 0.1, size=np.sum(missing_mask))
                    imputed_values = mean_val + noise
                    col_data[missing_mask] = imputed_values
                elif self.imputation_method == 'median':
                    params = self._fitted_params[col_idx]
                    median_val = params['median']
                    mad_val = params['mad']
                    noise = np.random.normal(0, mad_val * 0.1, size=np.sum(missing_mask))
                    imputed_values = median_val + noise
                    col_data[missing_mask] = imputed_values
                elif self.imputation_method == 'regression':
                    if X.ndim > 1 and X.shape[1] > 1:
                        complete_rows = ~np.isnan(X_imp).any(axis=1)
                        if np.sum(complete_rows) > 1:
                            predictor_cols = [i for i in range(X.shape[1]) if i != col_idx]
                            X_predictors = X_imp[np.ix_(complete_rows, predictor_cols)]
                            y_target = X_imp[complete_rows, col_idx]
                            model = LinearRegression()
                            model.fit(X_predictors, y_target)
                            missing_rows = np.isnan(X_imp[:, col_idx])
                            if np.any(missing_rows):
                                X_missing_predictors = X_imp[np.ix_(missing_rows, predictor_cols)]
                                predicted_values = model.predict(X_missing_predictors)
                                residuals = y_target - model.predict(X_predictors)
                                residual_std = np.std(residuals) if len(residuals) > 1 else 0
                                noise = np.random.normal(0, residual_std * 0.1, size=len(predicted_values))
                                predicted_values += noise
                                col_data[missing_rows] = predicted_values
                elif self.imputation_method == 'knn':
                    if X.ndim > 1 and X.shape[1] > 1:
                        complete_rows = ~np.isnan(X_imp).any(axis=1)
                        if np.sum(complete_rows) > 0:
                            missing_rows = np.isnan(X_imp[:, col_idx])
                            if np.any(missing_rows):
                                k = min(5, np.sum(complete_rows))
                                if k > 0:
                                    knn = KNeighborsRegressor(n_neighbors=k)
                                    predictor_cols = [i for i in range(X.shape[1]) if i != col_idx]
                                    X_complete = X_imp[np.ix_(complete_rows, predictor_cols)]
                                    y_complete = X_imp[complete_rows, col_idx]
                                    knn.fit(X_complete, y_complete)
                                    X_missing = X_imp[np.ix_(missing_rows, predictor_cols)]
                                    predicted_values = knn.predict(X_missing)
                                    noise = np.random.normal(0, np.std(y_complete) * 0.1, size=len(predicted_values))
                                    predicted_values += noise
                                    col_data[missing_rows] = predicted_values
                if X.ndim > 1:
                    X_imp[:, col_idx] = col_data
                else:
                    X_imp = col_data
            if is_feature_set:
                imputed_dataset = FeatureSet(features=X_imp, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata.copy() if metadata else {}, quality_scores=quality_scores.copy() if quality_scores else {})
                imputed_datasets.append(imputed_dataset)
            else:
                imputed_datasets.append(X_imp)
        return imputed_datasets

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Return input data unchanged (inverse imputation not supported).
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Data to inversely transform.
            **kwargs: Additional parameters.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Returns the input data unchanged.
        """
        return data

class MatrixFactorizationImputationTransformer(BaseTransformer):
    """
    Implements matrix factorization imputation to fill missing values using low-rank matrix completion.

    This transformer treats the data matrix as a partially observed matrix and uses matrix factorization
    techniques to estimate missing entries. It assumes that the underlying data matrix has a low-rank
    structure and finds latent factors that can reconstruct the observed entries while predicting missing ones.

    Attributes:
        n_components (int): Number of latent components for matrix factorization.
        regularization (float): Regularization parameter to prevent overfitting.
        max_iter (int): Maximum number of iterations for optimization.
        columns (Optional[List[str]]): Specific columns to apply imputation to.
        name (Optional[str]): Name identifier for the transformer.

    Methods:
        fit: Learns latent factors from observed data entries.
        transform: Applies matrix factorization to impute missing values.
        inverse_transform: Not supported for imputation methods.
    """

    def __init__(self, n_components: int=10, max_iter: int=100, reg_param: float=0.01, columns: Optional[List[str]]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.columns = columns
        self._latent_factors = None
        self._fitted_columns = None
        self._column_indices = None
        self._feature_names = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'MatrixFactorizationImputationTransformer':
        """
        Fits the matrix factorization model by learning latent factors from observed data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit on.
            **kwargs: Additional fitting parameters.
            
        Returns:
            MatrixFactorizationImputationTransformer: Fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            self._feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data.copy()
            self._feature_names = [f'col_{i}' for i in range(X.shape[1])] if X.ndim > 1 else ['col_0']
        else:
            raise TypeError('data must be either FeatureSet or numpy array')
        if self.columns is None:
            self._column_indices = list(range(X.shape[1])) if X.ndim > 1 else [0]
            self._fitted_columns = self._feature_names
        else:
            if isinstance(data, FeatureSet):
                col_to_idx = {name: i for (i, name) in enumerate(self._feature_names)}
                try:
                    self._column_indices = [col_to_idx[col] for col in self.columns]
                except KeyError as e:
                    raise ValueError(f'Column {e} not found in data') from e
            else:
                self._column_indices = self.columns
            self._fitted_columns = self.columns
        max_col_idx = X.shape[1] - 1 if X.ndim > 1 else 0
        for col_idx in self._column_indices:
            if not 0 <= col_idx <= max_col_idx:
                raise ValueError(f'Column index {col_idx} out of range [0, {max_col_idx}]')
        if X.ndim > 1:
            X_subset = X[:, self._column_indices]
        else:
            X_subset = X.reshape(-1, 1)
        (n_samples, n_features) = X_subset.shape
        U = np.random.normal(0, 0.1, (n_samples, self.n_components))
        V = np.random.normal(0, 0.1, (self.n_components, n_features))
        mask = ~np.isnan(X_subset)
        for iteration in range(self.max_iter):
            for i in range(n_samples):
                observed_indices = np.where(mask[i, :])[0]
                if len(observed_indices) > 0:
                    V_obs = V[:, observed_indices]
                    X_obs = X_subset[i, observed_indices]
                    A = V_obs @ V_obs.T + self.reg_param * np.eye(self.n_components)
                    b = V_obs @ X_obs
                    try:
                        U[i, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        U[i, :] = np.linalg.pinv(A) @ b
            for j in range(n_features):
                observed_indices = np.where(mask[:, j])[0]
                if len(observed_indices) > 0:
                    U_obs = U[observed_indices, :]
                    X_obs = X_subset[observed_indices, j]
                    A = U_obs.T @ U_obs + self.reg_param * np.eye(self.n_components)
                    b = U_obs.T @ X_obs
                    try:
                        V[:, j] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        V[:, j] = np.linalg.pinv(A) @ b
        self._latent_factors = (U, V)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Applies matrix factorization imputation to fill missing values in the data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Data with missing values to impute.
            **kwargs: Additional transformation parameters.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Data with missing values filled using matrix factorization.
        """
        if self._latent_factors is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        is_feature_set = isinstance(data, FeatureSet)
        if is_feature_set:
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, np.ndarray):
            X = data.copy()
            feature_names = [f'col_{i}' for i in range(X.shape[1])] if X.ndim > 1 else ['col_0']
        else:
            raise TypeError('data must be either FeatureSet or numpy array')
        if self._column_indices is None:
            column_indices = list(range(X.shape[1])) if X.ndim > 1 else [0]
        else:
            column_indices = self._column_indices
        max_col_idx = X.shape[1] - 1 if X.ndim > 1 else 0
        for col_idx in column_indices:
            if not 0 <= col_idx <= max_col_idx:
                raise ValueError(f'Column index {col_idx} out of range [0, {max_col_idx}]')
        X_result = X.copy()
        (U, V) = self._latent_factors
        if X_result.ndim == 1:
            X_result = X_result.reshape(-1, 1)
        if X_result.shape[0] != U.shape[0]:
            n_new_samples = X_result.shape[0]
            n_features = len(column_indices)
            X_subset = X_result[:, column_indices] if X_result.ndim > 1 else X_result.reshape(-1, 1)
            new_U = np.random.normal(0, 0.1, (n_new_samples, self.n_components))
            mask = ~np.isnan(X_subset)
            for i in range(n_new_samples):
                observed_indices = np.where(mask[i, :])[0]
                if len(observed_indices) > 0:
                    V_obs = V[:, observed_indices]
                    X_obs = X_subset[i, observed_indices]
                    A = V_obs @ V_obs.T + self.reg_param * np.eye(self.n_components)
                    b = V_obs @ X_obs
                    try:
                        new_U[i, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        new_U[i, :] = np.linalg.pinv(A) @ b
            reconstructed = new_U @ V
            for (j, col_idx) in enumerate(column_indices):
                missing_mask = np.isnan(X_result[:, col_idx])
                if np.any(missing_mask):
                    if X_result.ndim > 1:
                        X_result[missing_mask, col_idx] = reconstructed[missing_mask, j]
                    else:
                        X_result[missing_mask] = reconstructed[missing_mask, j]
        else:
            X_subset = X_result[:, column_indices] if X_result.ndim > 1 else X_result.reshape(-1, 1)
            reconstructed = U @ V
            for (j, col_idx) in enumerate(column_indices):
                missing_mask = np.isnan(X_subset[:, j] if X_subset.ndim > 1 else X_subset)
                if np.any(missing_mask):
                    if X_result.ndim > 1:
                        X_result[missing_mask, col_idx] = reconstructed[missing_mask, j]
                    else:
                        X_result[missing_mask] = reconstructed[missing_mask, j]
        if is_feature_set:
            return FeatureSet(features=X_result, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return X_result if data.ndim > 1 else X_result.flatten()

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation is not supported for imputation methods.
        Returns the input data unchanged.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Data to inversely transform.
            **kwargs: Additional parameters.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Returns the input data unchanged.
        """
        return data

class AdvancedImputationTransformer(BaseTransformer):
    """
    Implements advanced imputation strategies combining multiple techniques for complex missing data patterns.

    This transformer provides a flexible framework for applying sophisticated imputation methods that
    can adapt to different missing data mechanisms (MCAR, MAR, MNAR). It supports ensemble approaches,
    hybrid methods, and can automatically select the best technique based on data characteristics.

    Attributes:
        strategy (str): Advanced imputation strategy ('ensemble', 'hybrid', 'adaptive').
        ensemble_methods (List[str]): List of methods to ensemble when strategy='ensemble'.
        adaptive_criteria (Dict[str, Any]): Criteria for adaptive method selection.
        columns (Optional[List[str]]): Specific columns to apply imputation to.
        name (Optional[str]): Name identifier for the transformer.

    Methods:
        fit: Analyzes data characteristics and configures advanced imputation strategy.
        transform: Applies advanced imputation techniques to fill missing values.
        inverse_transform: Not supported for imputation methods.
    """

    def __init__(self, adaptive_threshold: float=0.3, ensemble_threshold: float=0.5, hybrid_threshold: float=0.7, random_state: Optional[int]=None, columns: Optional[List[str]]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.adaptive_threshold = adaptive_threshold
        self.ensemble_threshold = ensemble_threshold
        self.hybrid_threshold = hybrid_threshold
        self.random_state = random_state
        self.columns = columns
        self._strategy = None
        self._fitted_stats = {}
        self._column_indices = None
        self._feature_names = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'AdvancedImputationTransformer':
        """
        Analyze data characteristics and select optimal imputation strategy.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data with missing values.
            **kwargs: Additional fitting parameters.
            
        Returns:
            AdvancedImputationTransformer: Fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            self._feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data.copy()
            self._feature_names = [f'col_{i}' for i in range(X.shape[1])] if X.ndim > 1 else ['col_0']
        else:
            raise TypeError('data must be either FeatureSet or numpy array')
        if self.columns is None:
            self._column_indices = list(range(X.shape[1])) if X.ndim > 1 else [0]
        elif isinstance(data, FeatureSet):
            col_to_idx = {name: i for (i, name) in enumerate(self._feature_names)}
            try:
                self._column_indices = [col_to_idx[col] for col in self.columns]
            except KeyError as e:
                raise ValueError(f'Column {e} not found in data') from e
        else:
            self._column_indices = self.columns
        max_col_idx = X.shape[1] - 1 if X.ndim > 1 else 0
        for col_idx in self._column_indices:
            if not 0 <= col_idx <= max_col_idx:
                raise ValueError(f'Column index {col_idx} out of range [0, {max_col_idx}]')
        if X.ndim > 1:
            X_subset = X[:, self._column_indices]
        else:
            X_subset = X.reshape(-1, 1)
        missing_ratio = np.isnan(X_subset).sum() / X_subset.size
        if missing_ratio <= self.adaptive_threshold:
            self._strategy = 'adaptive'
        elif missing_ratio <= self.ensemble_threshold:
            self._strategy = 'ensemble'
        elif missing_ratio <= self.hybrid_threshold:
            self._strategy = 'hybrid'
        else:
            self._strategy = 'simple'
        for (i, col_idx) in enumerate(self._column_indices):
            col_data = X_subset[:, i] if X_subset.ndim > 1 else X_subset
            non_nan_data = col_data[~np.isnan(col_data)]
            if len(non_nan_data) > 0:
                self._fitted_stats[col_idx] = {'mean': np.mean(non_nan_data), 'median': np.median(non_nan_data), 'std': np.std(non_nan_data) if len(non_nan_data) > 1 else 0}
            else:
                self._fitted_stats[col_idx] = {'mean': 0, 'median': 0, 'std': 0}
        self.is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply selected imputation strategy to fill missing values.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Data with missing values to impute.
            **kwargs: Additional transformation parameters.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Data with missing values filled.
        """
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        is_feature_set = isinstance(data, FeatureSet)
        if is_feature_set:
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, np.ndarray):
            X = data.copy()
            feature_names = [f'col_{i}' for i in range(X.shape[1])] if X.ndim > 1 else ['col_0']
        else:
            raise TypeError('data must be either FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_result = X.copy()
        if self._strategy == 'adaptive':
            X_result = self._select_adaptive_strategy(X_result)
        elif self._strategy == 'ensemble':
            X_result = self._ensemble_impute(X_result)
        elif self._strategy == 'hybrid':
            X_result = self._hybrid_impute(X_result)
        else:
            X_result = self._simple_impute(X_result)
        if is_feature_set:
            return FeatureSet(features=X_result, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return X_result.reshape(-1) if data.ndim == 1 else X_result

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation is not supported for imputation methods.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Data to inversely transform.
            **kwargs: Additional parameters.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Returns the input data unchanged.
            
        Raises:
            NotImplementedError: Always raised as imputation is not invertible.
        """
        raise NotImplementedError('Inverse transform is not supported for imputation transformers.')

    def _select_adaptive_strategy(self, X: np.ndarray) -> np.ndarray:
        """Select and apply adaptive imputation strategy based on column characteristics."""
        result = X.copy()
        for (i, col_idx) in enumerate(self._column_indices):
            col_data = result[:, col_idx]
            if not np.isnan(col_data).any():
                continue
            stats = self._fitted_stats.get(col_idx, {'std': 0})
            if stats['std'] > 1.0:
                fill_value = stats.get('median', 0)
            else:
                fill_value = stats.get('mean', 0)
            col_data[np.isnan(col_data)] = fill_value
            result[:, col_idx] = col_data
        return result

    def _ensemble_impute(self, X: np.ndarray) -> np.ndarray:
        """Apply ensemble imputation combining multiple strategies."""
        result = X.copy()
        for (i, col_idx) in enumerate(self._column_indices):
            col_data = result[:, col_idx]
            if not np.isnan(col_data).any():
                continue
            stats = self._fitted_stats.get(col_idx, {'mean': 0, 'median': 0})
            fill_value = (stats.get('mean', 0) + stats.get('median', 0)) / 2
            col_data[np.isnan(col_data)] = fill_value
            result[:, col_idx] = col_data
        return result

    def _hybrid_impute(self, X: np.ndarray) -> np.ndarray:
        """Apply hybrid imputation combining simple and adaptive strategies."""
        result = X.copy()
        for (i, col_idx) in enumerate(self._column_indices):
            col_data = result[:, col_idx]
            if not np.isnan(col_data).any():
                continue
            stats = self._fitted_stats.get(col_idx, {'mean': 0, 'median': 0, 'std': 0})
            std = stats.get('std', 0)
            weight = min(std / 2.0, 1.0) if std > 0 else 0.5
            fill_value = weight * stats.get('median', 0) + (1 - weight) * stats.get('mean', 0)
            col_data[np.isnan(col_data)] = fill_value
            result[:, col_idx] = col_data
        return result

    def _simple_impute(self, X: np.ndarray) -> np.ndarray:
        """Apply simple mean imputation."""
        result = X.copy()
        for (i, col_idx) in enumerate(self._column_indices):
            col_data = result[:, col_idx]
            if not np.isnan(col_data).any():
                continue
            fill_value = self._fitted_stats.get(col_idx, {'mean': 0}).get('mean', 0)
            col_data[np.isnan(col_data)] = fill_value
            result[:, col_idx] = col_data
        return result