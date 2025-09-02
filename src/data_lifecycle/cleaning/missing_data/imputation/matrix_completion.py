from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class MatrixCompletionTransformer(BaseTransformer):

    def __init__(self, method: str='nuclear_norm', max_iterations: int=100, tolerance: float=1e-06, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.fitted_matrices_ = {}

    def fit(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> 'MatrixCompletionTransformer':
        """
        Fit the matrix completion transformer to the input data.
        
        This method analyzes the input data structure and prepares the transformer
        for completing missing values. It does not modify the input data.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data with missing values to be completed. Can be:
            - numpy array: 2D array with missing values represented as NaN
            - DataBatch: Batched data structure with potential missing values
            - FeatureSet: Structured feature set with potential missing values
        **kwargs : dict
            Additional parameters for fitting (reserved for future extensions)
            
        Returns
        -------
        MatrixCompletionTransformer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the input data format is not supported or contains invalid values
        """
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError('Input numpy array must be 2-dimensional')
            X = data.copy()
        elif isinstance(data, DataBatch):
            if hasattr(data, 'data') and isinstance(data.data, np.ndarray):
                if data.data.ndim != 2:
                    raise ValueError('DataBatch data must be 2-dimensional')
                X = data.data.copy()
            else:
                raise ValueError('DataBatch must contain a 2D numpy array')
        elif isinstance(data, FeatureSet):
            if data.features.ndim != 2:
                raise ValueError('FeatureSet features must be 2-dimensional')
            X = data.features.copy()
        else:
            raise ValueError(f'Unsupported data type: {type(data)}. Expected numpy.ndarray, DataBatch, or FeatureSet')
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('Input data must be numeric')
        self._original_shape = X.shape
        self._missing_mask = np.isnan(X)
        if not np.any(self._missing_mask):
            self.fitted_matrices_['completed_data'] = X.copy()
        else:
            self.fitted_matrices_['missing_mask'] = self._missing_mask.copy()
            if self.method == 'nuclear_norm':
                pass
            elif self.method == 'soft_threshold':
                pass
            else:
                raise ValueError(f'Unsupported method: {self.method}')
        self._is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> Union[np.ndarray, DataBatch, FeatureSet]:
        """
        Apply matrix completion to fill missing values in the input data.
        
        This method uses the fitted parameters to estimate and fill missing entries
        in the input data matrix using the specified completion method.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data with missing values to be completed. Must have the same
            structure as the data used in fit().
        **kwargs : dict
            Additional parameters for transformation (reserved for future extensions)
            
        Returns
        -------
        Union[np.ndarray, DataBatch, FeatureSet]
            Data with missing values filled using matrix completion. The return
            type matches the input data type.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or input data is incompatible
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError('Input numpy array must be 2-dimensional')
            X = data.copy()
            input_type = 'numpy'
        elif isinstance(data, DataBatch):
            if hasattr(data, 'data') and isinstance(data.data, np.ndarray):
                if data.data.ndim != 2:
                    raise ValueError('DataBatch data must be 2-dimensional')
                X = data.data.copy()
            else:
                raise ValueError('DataBatch must contain a 2D numpy array')
            input_type = 'databatch'
        elif isinstance(data, FeatureSet):
            if data.features.ndim != 2:
                raise ValueError('FeatureSet features must be 2-dimensional')
            X = data.features.copy()
            input_type = 'featureset'
        else:
            raise ValueError(f'Unsupported data type: {type(data)}. Expected numpy.ndarray, DataBatch, or FeatureSet')
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('Input data must be numeric')
        if X.shape != self._original_shape:
            raise ValueError(f'Input data shape {X.shape} does not match the shape used in fit {self._original_shape}')
        if 'completed_data' in self.fitted_matrices_:
            result = self.fitted_matrices_['completed_data'].copy()
        else:
            missing_mask = self.fitted_matrices_['missing_mask']
            if self.method == 'nuclear_norm':
                result = self._nuclear_norm_minimization(X, missing_mask)
            elif self.method == 'soft_threshold':
                result = self._soft_threshold_completion(X, missing_mask)
            else:
                raise ValueError(f'Unsupported method: {self.method}')
        if input_type == 'numpy':
            return result
        elif input_type == 'databatch':
            if isinstance(data, DataBatch):
                new_data = data.__class__(data=result)
                return new_data
            else:
                return DataBatch(data=result)
        elif input_type == 'featureset':
            if isinstance(data, FeatureSet):
                new_features = data.__class__(features=result, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
                return new_features
            else:
                return FeatureSet(features=result)

    def inverse_transform(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> Union[np.ndarray, DataBatch, FeatureSet]:
        """
        Reverse the matrix completion transformation (not supported).
        
        Matrix completion is a lossy transformation that estimates missing values,
        so an exact inverse transformation is not possible. This method raises
        a NotImplementedError.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[np.ndarray, DataBatch, FeatureSet]
            Never returns successfully
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported
        """
        pass

    def _nuclear_norm_minimization(self, X: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """Apply nuclear norm minimization to complete the matrix."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_completed = X.copy()
        X_completed[missing_mask] = 0
        for iteration in range(self.max_iterations):
            (U, s, Vt) = np.linalg.svd(X_completed, full_matrices=False)
            s_thresholded = np.maximum(s - 1.0 / np.sqrt(X.shape[0]), 0)
            X_reconstructed = U @ np.diag(s_thresholded) @ Vt
            X_completed[missing_mask] = X_reconstructed[missing_mask]
            diff = np.linalg.norm(X_reconstructed - X_completed, 'fro')
            if diff < self.tolerance:
                break
        return X_completed

    def _soft_threshold_completion(self, X: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """Apply soft thresholding based completion."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_completed = X.copy()
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X_completed[np.isnan(X_completed[:, i]), i] = col_means[i]
        for iteration in range(self.max_iterations):
            X_old = X_completed.copy()
            (U, s, Vt) = np.linalg.svd(X_completed, full_matrices=False)
            threshold = np.median(s) * 0.1
            s_thresholded = np.maximum(s - threshold, 0)
            X_completed = U @ np.diag(s_thresholded) @ Vt
            X_completed[~missing_mask] = X[~missing_mask]
            diff = np.linalg.norm(X_completed - X_old, 'fro') / np.linalg.norm(X_old, 'fro')
            if diff < self.tolerance:
                break
        return X_completed

class SoftThresholdMatrixCompletionTransformer(BaseTransformer):
    """
    Transformer for completing missing values using soft thresholding matrix completion.
    
    This transformer applies soft thresholding singular value decomposition (SVD) to 
    estimate missing entries in a data matrix. It works by iteratively applying a 
    soft thresholding operator to the singular values of the matrix, promoting a 
    low-rank solution.
    
    Parameters
    ----------
    threshold : float, default=0.1
        Threshold value for soft thresholding operation on singular values
    max_iterations : int, default=100
        Maximum number of iterations for the completion algorithm
    tolerance : float, default=1e-6
        Convergence tolerance for the algorithm
    random_state : int, optional
        Random seed for reproducibility
    name : str, optional
        Name of the transformer instance
        
    Attributes
    ----------
    fitted_params_ : dict
        Dictionary storing fitted parameters from the completion process
    """

    def __init__(self, threshold: float=0.1, max_iterations: int=100, tolerance: float=1e-06, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.fitted_params_ = {}

    def fit(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> 'SoftThresholdMatrixCompletionTransformer':
        """
        Fit the soft thresholding matrix completion transformer to the input data.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data with missing values to be completed
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        SoftThresholdMatrixCompletionTransformer
            Self instance for method chaining
        """
        if hasattr(data, 'features'):
            if hasattr(data.features, 'values'):
                X = data.features.values
            else:
                X = data.features
        elif hasattr(data, 'data'):
            X = data.data
        else:
            X = data
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('Input data must contain numeric values')
        if not np.isnan(X).any():
            raise ValueError('No missing values found in input data')
        self.fitted_params_['data_shape'] = X.shape
        self.fitted_params_['is_fitted'] = True
        return self

    def transform(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> Union[np.ndarray, DataBatch, FeatureSet]:
        """
        Apply soft thresholding matrix completion to fill missing values.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data with missing values to be completed
            
        Returns
        -------
        Union[np.ndarray, DataBatch, FeatureSet]
            Data with missing values filled using soft thresholding completion
        """
        if not self.fitted_params_.get('is_fitted', False):
            raise RuntimeError('Transformer must be fitted before transform can be called')
        original_type = type(data)
        if hasattr(data, 'features'):
            if hasattr(data.features, 'values'):
                X = data.features.values
            else:
                X = data.features
        elif hasattr(data, 'data'):
            X = data.data
        else:
            X = data
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.shape != self.fitted_params_['data_shape']:
            raise ValueError(f"Input data shape {X.shape} does not match fitted data shape {self.fitted_params_['data_shape']}")
        if not np.isnan(X).any():
            if original_type == np.ndarray:
                return X
            elif hasattr(data, 'features'):
                data.features = X
                return data
            elif hasattr(data, 'data'):
                data.data = X
                return data
            else:
                return X
        X_completed = self._soft_threshold_svd_completion(X)
        if original_type == np.ndarray:
            return X_completed
        elif hasattr(data, 'features'):
            data.features = X_completed
            return data
        elif hasattr(data, 'data'):
            data.data = X_completed
            return data
        else:
            return X_completed

    def inverse_transform(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> Union[np.ndarray, DataBatch, FeatureSet]:
        """
        Not supported for matrix completion transformers.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data
            
        Returns
        -------
        Union[np.ndarray, DataBatch, FeatureSet]
            Same as input (identity operation)
        """
        return data

    def _soft_threshold_svd_completion(self, X: np.ndarray) -> np.ndarray:
        """
        Apply soft thresholding SVD to complete the matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix with missing values (NaN)
            
        Returns
        -------
        np.ndarray
            Completed matrix with missing values filled
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_completed = X.copy()
        missing_mask = np.isnan(X_completed)
        col_means = np.nanmean(X_completed, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        for i in range(X_completed.shape[1]):
            X_completed[missing_mask[:, i], i] = col_means[i]
        for iteration in range(self.max_iterations):
            X_old = X_completed.copy()
            try:
                (U, s, Vt) = np.linalg.svd(X_completed, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            s_thresholded = np.maximum(s - self.threshold, 0)
            X_completed = U @ np.diag(s_thresholded) @ Vt
            X_completed[~missing_mask] = X[~missing_mask]
            diff_norm = np.linalg.norm(X_completed - X_old, 'fro')
            old_norm = np.linalg.norm(X_old, 'fro')
            if old_norm > 0 and diff_norm / old_norm < self.tolerance:
                break
        return X_completed

class NuclearNormMatrixCompletionTransformer(BaseTransformer):
    """
    Transformer for completing missing values using nuclear norm minimization.
    
    This transformer applies nuclear norm minimization to estimate missing entries 
    in a data matrix. Nuclear norm minimization promotes low-rank solutions by 
    minimizing the sum of singular values, effectively acting as a convex relaxation 
    of rank minimization problems.
    
    Parameters
    ----------
    regularization : float, default=0.1
        Regularization parameter for nuclear norm penalty
    max_iterations : int, default=100
        Maximum number of iterations for the optimization algorithm
    tolerance : float, default=1e-6
        Convergence tolerance for the optimization algorithm
    solver : str, default='svt'
        Solver to use for optimization. Options: 'svt' (singular value thresholding)
    random_state : int, optional
        Random seed for reproducibility
    name : str, optional
        Name of the transformer instance
        
    Attributes
    ----------
    fitted_params_ : dict
        Dictionary storing fitted parameters from the completion process
    """

    def __init__(self, regularization: float=0.1, max_iterations: int=100, tolerance: float=1e-06, solver: str='svt', random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.solver = solver
        self.random_state = random_state
        self.fitted_params_ = {}

    def fit(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> 'NuclearNormMatrixCompletionTransformer':
        """
        Fit the nuclear norm matrix completion transformer to the input data.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data with missing values to be completed
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        NuclearNormMatrixCompletionTransformer
            Self instance for method chaining
        """
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError('Input numpy array must be 2-dimensional')
            X = data.copy()
            self._input_type = 'numpy'
        elif isinstance(data, DataBatch):
            if hasattr(data, 'data') and isinstance(data.data, np.ndarray):
                if data.data.ndim != 2:
                    raise ValueError('DataBatch data must be 2-dimensional')
                X = data.data.copy()
            else:
                raise ValueError('DataBatch must contain a 2D numpy array')
            self._input_type = 'databatch'
        elif isinstance(data, FeatureSet):
            if data.features.ndim != 2:
                raise ValueError('FeatureSet features must be 2-dimensional')
            X = data.features.copy()
            self._input_type = 'featureset'
        else:
            raise ValueError(f'Unsupported data type: {type(data)}. Expected numpy.ndarray, DataBatch, or FeatureSet')
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('Input data must be numeric')
        missing_mask = np.isnan(X)
        self.fitted_params_['data_shape'] = X.shape
        self.fitted_params_['missing_mask'] = missing_mask.copy()
        self.fitted_params_['is_fitted'] = True
        return self

    def transform(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> Union[np.ndarray, DataBatch, FeatureSet]:
        """
        Apply nuclear norm minimization to fill missing values.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data with missing values to be completed
            
        Returns
        -------
        Union[np.ndarray, DataBatch, FeatureSet]
            Data with missing values filled using nuclear norm completion
        """
        if not self.fitted_params_.get('is_fitted', False):
            raise RuntimeError('Transformer must be fitted before transform can be called')
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError('Input numpy array must be 2-dimensional')
            X = data.copy()
            input_type = 'numpy'
        elif isinstance(data, DataBatch):
            if hasattr(data, 'data') and isinstance(data.data, np.ndarray):
                if data.data.ndim != 2:
                    raise ValueError('DataBatch data must be 2-dimensional')
                X = data.data.copy()
            else:
                raise ValueError('DataBatch must contain a 2D numpy array')
            input_type = 'databatch'
        elif isinstance(data, FeatureSet):
            if data.features.ndim != 2:
                raise ValueError('FeatureSet features must be 2-dimensional')
            X = data.features.copy()
            input_type = 'featureset'
        else:
            raise ValueError(f'Unsupported data type: {type(data)}. Expected numpy.ndarray, DataBatch, or FeatureSet')
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('Input data must be numeric')
        if X.shape != self.fitted_params_['data_shape']:
            raise ValueError(f"Input data shape {X.shape} does not match fitted data shape {self.fitted_params_['data_shape']}")
        missing_mask = self.fitted_params_['missing_mask']
        if self.solver not in ['svt']:
            raise ValueError(f"Unsupported solver: {self.solver}. Supported solvers: 'svt'")
        if not np.any(missing_mask):
            if input_type == 'numpy':
                return X
            elif input_type == 'databatch':
                if isinstance(data, DataBatch):
                    new_data = data.__class__(data=X)
                    return new_data
                else:
                    return DataBatch(data=X)
            elif input_type == 'featureset':
                if isinstance(data, FeatureSet):
                    new_features = data.__class__(features=X, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
                    return new_features
                else:
                    return FeatureSet(features=X)
        if self.solver == 'svt':
            X_completed = self._singular_value_thresholding(X, missing_mask)
        else:
            raise ValueError(f'Unsupported solver: {self.solver}')
        if input_type == 'numpy':
            return X_completed
        elif input_type == 'databatch':
            if isinstance(data, DataBatch):
                new_data = data.__class__(data=X_completed)
                return new_data
            else:
                return DataBatch(data=X_completed)
        elif input_type == 'featureset':
            if isinstance(data, FeatureSet):
                new_features = data.__class__(features=X_completed, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
                return new_features
            else:
                return FeatureSet(features=X_completed)

    def inverse_transform(self, data: Union[np.ndarray, DataBatch, FeatureSet], **kwargs) -> Union[np.ndarray, DataBatch, FeatureSet]:
        """
        Not supported for matrix completion transformers.
        
        Parameters
        ----------
        data : Union[np.ndarray, DataBatch, FeatureSet]
            Input data
            
        Returns
        -------
        Union[np.ndarray, DataBatch, FeatureSet]
            Same as input (identity operation)
        """
        return data

    def _singular_value_thresholding(self, X: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Apply singular value thresholding algorithm for nuclear norm minimization.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix with missing values (NaN)
        missing_mask : np.ndarray
            Boolean mask indicating missing values
            
        Returns
        -------
        np.ndarray
            Completed matrix with missing values filled
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_completed = X.copy()
        col_means = np.nanmean(X_completed, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        for i in range(X_completed.shape[1]):
            X_completed[np.isnan(X_completed[:, i]), i] = col_means[i]
        delta = 1.2 * np.prod(X.shape) / np.sum(~missing_mask)
        for iteration in range(self.max_iterations):
            X_old = X_completed.copy()
            try:
                (U, s, Vt) = np.linalg.svd(X_completed, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            s_thresholded = np.maximum(s - self.regularization * delta, 0)
            X_completed = U @ np.diag(s_thresholded) @ Vt
            X_completed[~missing_mask] = X[~missing_mask]
            diff_norm = np.linalg.norm(X_completed - X_old, 'fro')
            old_norm = np.linalg.norm(X_old, 'fro')
            if old_norm > 0 and diff_norm / old_norm < self.tolerance:
                break
        return X_completed

def apply_nuclear_norm_matrix_completion(data: Union[np.ndarray, DataBatch, FeatureSet], regularization: float=0.1, max_iterations: int=100, tolerance: float=1e-06, solver: str='svt', random_state: Optional[int]=None) -> Union[np.ndarray, DataBatch, FeatureSet]:
    """
    Apply matrix completion with nuclear norm minimization to fill missing values.
    
    This function directly applies nuclear norm minimization to estimate missing 
    entries in a data matrix. It uses singular value thresholding to promote 
    low-rank solutions by minimizing the sum of singular values.
    
    Parameters
    ----------
    data : Union[np.ndarray, DataBatch, FeatureSet]
        Input data with missing values to be completed. Missing values should
        be represented as NaN in numpy arrays.
    regularization : float, default=0.1
        Regularization parameter controlling the nuclear norm penalty. Higher
        values lead to lower-rank solutions.
    max_iterations : int, default=100
        Maximum number of iterations for the optimization algorithm.
    tolerance : float, default=1e-6
        Convergence tolerance for the optimization algorithm. The algorithm
        stops when the change between iterations is below this threshold.
    solver : str, default='svt'
        Solver to use for optimization. Currently only supports 'svt' (singular
        value thresholding).
    random_state : int, optional
        Random seed for reproducibility of any randomized components.
        
    Returns
    -------
    Union[np.ndarray, DataBatch, FeatureSet]
        Data with missing values filled using nuclear norm minimization. The
        return type matches the input data type.
        
    Examples
    --------
    >>> import numpy as np
    >>> from src.data_lifecycle.cleaning.missing_data.imputation.matrix_completion import apply_nuclear_norm_matrix_completion
    >>> 
    >>> # Create sample data with missing values
    >>> X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    >>> 
    >>> # Apply matrix completion
    >>> X_completed = apply_nuclear_norm_matrix_completion(X, regularization=0.05)
    >>> print(X_completed)
    """
    if solver not in ['svt']:
        raise ValueError(f"Unsupported solver: {solver}. Supported solvers: 'svt'")
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('Input numpy array must be 2-dimensional')
        X = data.copy()
        input_type = 'numpy'
    elif isinstance(data, DataBatch):
        if hasattr(data, 'data') and isinstance(data.data, np.ndarray):
            if data.data.ndim != 2:
                raise ValueError('DataBatch data must be 2-dimensional')
            X = data.data.copy()
        else:
            raise ValueError('DataBatch must contain a 2D numpy array')
        input_type = 'databatch'
    elif isinstance(data, FeatureSet):
        if data.features.ndim != 2:
            raise ValueError('FeatureSet features must be 2-dimensional')
        X = data.features.copy()
        input_type = 'featureset'
    else:
        raise ValueError(f'Unsupported data type: {type(data)}. Expected numpy.ndarray, DataBatch, or FeatureSet')
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError('Input data must be numeric')
    missing_mask = np.isnan(X)
    if not np.any(missing_mask):
        if input_type == 'numpy':
            return X
        elif input_type == 'databatch':
            if isinstance(data, DataBatch):
                return data
            else:
                return DataBatch(data=X)
        elif input_type == 'featureset':
            if isinstance(data, FeatureSet):
                return data
            else:
                return FeatureSet(features=X)
    if solver == 'svt':
        if random_state is not None:
            np.random.seed(random_state)
        X_completed = X.copy()
        col_means = np.nanmean(X_completed, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        for i in range(X_completed.shape[1]):
            X_completed[np.isnan(X_completed[:, i]), i] = col_means[i]
        delta = 1.2 * np.prod(X.shape) / np.sum(~missing_mask)
        for iteration in range(max_iterations):
            X_old = X_completed.copy()
            try:
                (U, s, Vt) = np.linalg.svd(X_completed, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            s_thresholded = np.maximum(s - regularization * delta, 0)
            X_completed = U @ np.diag(s_thresholded) @ Vt
            X_completed[~missing_mask] = X[~missing_mask]
            diff_norm = np.linalg.norm(X_completed - X_old, 'fro')
            old_norm = np.linalg.norm(X_old, 'fro')
            if old_norm > 0 and diff_norm / old_norm < tolerance:
                break
    else:
        raise ValueError(f'Unsupported solver: {solver}')
    if input_type == 'numpy':
        return X_completed
    elif input_type == 'databatch':
        if isinstance(data, DataBatch):
            new_data = data.__class__(data=X_completed, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
            return new_data
        else:
            return DataBatch(data=X_completed)
    elif input_type == 'featureset':
        if isinstance(data, FeatureSet):
            new_features = data.__class__(features=X_completed, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
            return new_features
        else:
            return FeatureSet(features=X_completed)