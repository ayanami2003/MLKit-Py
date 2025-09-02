import numpy as np
from scipy import sparse
from typing import Union, Tuple, List, Optional, Any
from general.structures.feature_set import FeatureSet

class SpectralClusteringPreprocessor:
    """
    A transformer class for preparing data for spectral clustering algorithms.
    
    This class computes the graph Laplacian from input features and performs
    eigenvalue decomposition to obtain the spectral embedding. It is designed
    to work with the system's FeatureSet structure and integrates with downstream
    clustering components.
    """

    def __init__(self, n_components: int=2, affinity_metric: str='rbf', gamma: float=1.0):
        """
        Initialize the spectral clustering preprocessor.
        
        Args:
            n_components (int): Number of eigenvectors to compute for embedding.
            affinity_metric (str): Method for computing affinity matrix ('rbf', 'nearest_neighbors').
            gamma (float): Parameter for RBF kernel in affinity computation.
        """
        if affinity_metric not in ['rbf', 'nearest_neighbors']:
            raise ValueError("affinity_metric must be either 'rbf' or 'nearest_neighbors'")
        self.n_components = n_components
        self.affinity_metric = affinity_metric
        self.gamma = gamma
        self.eigenvectors_ = None
        self.eigenvalues_ = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'SpectralClusteringPreprocessor':
        """
        Compute the spectral embedding from input data.
        
        This method constructs an affinity matrix, computes the graph Laplacian,
        and performs eigenvalue decomposition to obtain the spectral embedding.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data for spectral clustering.
            **kwargs: Additional parameters for fitting.
            
        Returns:
            SpectralClusteringPreprocessor: Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError('Input data must be either FeatureSet or numpy array')
        if self.affinity_metric == 'rbf':
            pairwise_sq_dists = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2)
            affinity_matrix = np.exp(-self.gamma * pairwise_sq_dists)
        else:
            pairwise_dists = np.sqrt(np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2))
            k = min(self.n_components, X.shape[0] - 1)
            affinity_matrix = np.zeros_like(pairwise_dists)
            for i in range(X.shape[0]):
                knn_indices = np.argpartition(pairwise_dists[i], k)[:k]
                affinity_matrix[i, knn_indices] = 1
                affinity_matrix[knn_indices, i] = 1
        degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
        laplacian_matrix = degree_matrix - affinity_matrix
        (eigenvals, eigenvecs) = np.linalg.eigh(laplacian_matrix)
        idx = np.argsort(eigenvals)
        self.eigenvalues_ = eigenvals[idx][:self.n_components]
        self.eigenvectors_ = eigenvecs[:, idx][:, :self.n_components]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Transform input data to its spectral embedding representation.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional parameters for transformation.
            
        Returns:
            np.ndarray: Spectral embedding of shape (n_samples, n_components).
        """
        if self.eigenvectors_ is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        return self.eigenvectors_

    def fit_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Fit the preprocessor and transform data in one step.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit and transform.
            **kwargs: Additional parameters.
            
        Returns:
            np.ndarray: Spectral embedding of shape (n_samples, n_components).
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)

    def get_spectral_embedding(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve computed eigenvalues and eigenvectors.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - eigenvalues (np.ndarray): Computed eigenvalues.
                - eigenvectors (np.ndarray): Computed eigenvectors (spectral embedding).
        """
        if self.eigenvalues_ is None or self.eigenvectors_ is None:
            raise ValueError("Spectral embedding has not been computed. Call 'fit' first.")
        return (self.eigenvalues_, self.eigenvectors_)

def transpose_matrix(matrix: Union[np.ndarray, list]) -> np.ndarray:
    """
    Transpose a 2D matrix (swap rows and columns).

    This function takes a matrix represented as a NumPy array or a list of lists
    and returns its transpose. The transpose of a matrix is obtained by flipping
    the matrix over its main diagonal, which switches the row and column indices.

    Args:
        matrix (Union[np.ndarray, list]): A 2D matrix to be transposed. If provided
                                          as a list, it will be converted to a NumPy array.

    Returns:
        np.ndarray: The transposed matrix as a NumPy array.

    Raises:
        ValueError: If the input is not a 2D matrix or cannot be converted to one.
    """
    if isinstance(matrix, list):
        try:
            matrix = np.array(matrix)
        except Exception as e:
            raise ValueError(f'Failed to convert input to numpy array: {e}')
    if not isinstance(matrix, np.ndarray):
        raise ValueError('Input must be a numpy array or a list that can be converted to a numpy array')
    if matrix.ndim != 2:
        raise ValueError(f'Input must be a 2D matrix, but got {matrix.ndim} dimensions')
    return matrix.T

def multiply_matrices(A: Union[np.ndarray, list], B: Union[np.ndarray, list]) -> np.ndarray:
    """
    Multiply two matrices using dot product.

    This function performs matrix multiplication (dot product) of two input matrices.
    It supports both dense and sparse matrix representations through NumPy arrays
    or compatible list structures.

    Args:
        A (Union[np.ndarray, list]): First matrix operand of shape (m, n).
        B (Union[np.ndarray, list]): Second matrix operand of shape (n, p).

    Returns:
        np.ndarray: Resultant matrix of shape (m, p) from multiplying A and B.

    Raises:
        ValueError: If matrices have incompatible shapes for multiplication.
        TypeError: If inputs are not valid matrix representations.
    """
    if isinstance(A, list):
        try:
            A = np.array(A)
        except Exception as e:
            raise TypeError(f'Failed to convert first input to numpy array: {str(e)}')
    if isinstance(B, list):
        try:
            B = np.array(B)
        except Exception as e:
            raise TypeError(f'Failed to convert second input to numpy array: {str(e)}')
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError('Inputs must be numpy arrays or list structures that can be converted to numpy arrays')
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError('Both inputs must be at least 2-dimensional matrices')
    shape_a = A.shape
    shape_b = B.shape
    if shape_a[-1] != shape_b[-2]:
        raise ValueError(f'Incompatible shapes for matrix multiplication: {shape_a} and {shape_b}. The number of columns in the first matrix ({shape_a[-1]}) must equal the number of rows in the second matrix ({shape_b[-2]}).')
    try:
        result = np.dot(A, B)
        return result
    except Exception as e:
        raise ValueError(f'Matrix multiplication failed: {str(e)}')

def concatenate_matrices(matrices: List[Union[np.ndarray, list]], axis: int=0) -> np.ndarray:
    """
    Concatenate a sequence of matrices along a specified axis.

    This function joins multiple matrices along a given axis to create a new matrix.
    It supports both NumPy arrays and list-based matrix representations, providing
    flexibility for various data sources.

    Args:
        matrices (List[Union[np.ndarray, list]]): A list of matrices to concatenate.
                                                  All matrices must have matching
                                                  dimensions except along the axis.
        axis (int): The axis along which to concatenate. Default is 0 (row-wise).

    Returns:
        np.ndarray: A new matrix formed by concatenating input matrices along the specified axis.

    Raises:
        ValueError: If matrices have incompatible shapes for concatenation along the axis.
        TypeError: If inputs are not valid matrix representations.
    """
    if not isinstance(matrices, list):
        raise TypeError("Input 'matrices' must be a list of matrix-like objects.")
    if not matrices:
        raise ValueError("Input 'matrices' cannot be empty.")
    try:
        arrays = [np.asarray(matrix) for matrix in matrices]
    except Exception as e:
        raise TypeError("All elements in 'matrices' must be convertible to NumPy arrays.") from e
    for (i, arr) in enumerate(arrays):
        if arr.ndim == 0:
            raise ValueError(f'Element at index {i} is a scalar. All elements must be at least 1-dimensional.')
    ndims = arrays[0].ndim
    for (i, arr) in enumerate(arrays):
        if arr.ndim != ndims:
            raise ValueError(f'All matrices must have the same number of dimensions. Matrix at index 0 has {ndims} dimensions, but matrix at index {i} has {arr.ndim} dimensions.')
    if not isinstance(axis, int):
        raise TypeError('Axis must be an integer.')
    if axis < 0:
        axis += ndims
    if axis < 0 or axis >= ndims:
        raise ValueError(f'Axis {axis} is out of bounds for array with {ndims} dimensions.')
    for (i, arr) in enumerate(arrays):
        for (j, (dim1, dim2)) in enumerate(zip(arrays[0].shape, arr.shape)):
            if j != axis and dim1 != dim2:
                raise ValueError(f'Matrices have incompatible shapes for concatenation along axis {axis}. All dimensions except the concatenation axis must match exactly. Matrix at index 0 has shape {arrays[0].shape}, but matrix at index {i} has shape {arr.shape}.')
    try:
        result = np.concatenate(arrays, axis=axis)
    except Exception as e:
        raise ValueError(f'Failed to concatenate matrices due to an unexpected error: {e}') from e
    return result

def convert_to_sparse(matrix: Union[np.ndarray, list], format: str='csr') -> sparse.spmatrix:
    """
    Convert a dense matrix to a sparse matrix representation.

    This function converts a standard dense matrix into a sparse matrix format
    for efficient storage and computation when dealing with matrices containing
    a significant number of zero elements.

    Args:
        matrix (Union[np.ndarray, list]): A dense matrix to convert to sparse format.
        format (str): Target sparse matrix format ('csr', 'csc', 'coo', 'lil', etc.).
                      Default is 'csr' (Compressed Sparse Row).

    Returns:
        sparse.spmatrix: The matrix in the specified sparse format.

    Raises:
        ValueError: If the requested sparse format is not supported.
        TypeError: If the input is not a valid matrix representation.
    """
    if not isinstance(matrix, (np.ndarray, list)):
        raise TypeError('Input matrix must be a numpy array or a list')
    if isinstance(matrix, list):
        try:
            matrix = np.array(matrix)
        except Exception as e:
            raise TypeError('Input list cannot be converted to numpy array') from e
    if matrix.ndim != 2:
        raise TypeError('Input must be a 2D matrix')
    supported_formats = {'csr', 'csc', 'coo', 'lil'}
    if format not in supported_formats:
        raise ValueError(f"Unsupported sparse format '{format}'. Supported formats are: {supported_formats}")
    if format == 'csr':
        return sparse.csr_matrix(matrix)
    elif format == 'csc':
        return sparse.csc_matrix(matrix)
    elif format == 'coo':
        return sparse.coo_matrix(matrix)
    elif format == 'lil':
        return sparse.lil_matrix(matrix)

def compute_eigenvalues(matrix: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a square matrix.

    This function calculates the eigenvalues and corresponding eigenvectors
    of a given square matrix. It supports both dense and list-based matrix
    representations and is foundational for operations like spectral clustering.

    Args:
        matrix (Union[np.ndarray, list]): A square matrix of shape (n, n) for which
                                          eigenvalues and eigenvectors are computed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - eigenvalues (np.ndarray): 1D array of eigenvalues.
            - eigenvectors (np.ndarray): 2D array where each column is an eigenvector.

    Raises:
        ValueError: If the input matrix is not square or cannot be converted properly.
        np.linalg.LinAlgError: If eigenvalue computation fails to converge.
    """
    if isinstance(matrix, list):
        try:
            matrix = np.array(matrix, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(f'Cannot convert input to numpy array: {e}')
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError('Input must be a 2D array or a list that can be converted to a 2D array.')
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Input matrix must be square (n x n).')
    (eigenvalues, eigenvectors) = np.linalg.eig(matrix)
    return (eigenvalues, eigenvectors)


# ...(code omitted)...


def optimized_eigen_decomposition(matrix: Union[np.ndarray, list], largest_only: bool=True, k: int=10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimized eigenvalue decomposition for large matrices with performance enhancements.

    This function performs efficient eigenvalue decomposition, optimized for large-scale matrices.
    It focuses on computing only the largest eigenvalues/eigenvectors when needed, using iterative
    methods for improved performance over standard approaches.

    Args:
        matrix (Union[np.ndarray, list]): Symmetric matrix of shape (n, n) for decomposition.
        largest_only (bool): If True, compute only the largest eigenvalues. Default is True.
        k (int): Number of largest eigenvalues to compute when largest_only is True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - eigenvalues (np.ndarray): 1D array of computed eigenvalues (sorted descending).
            - eigenvectors (np.ndarray): 2D array where each column is an eigenvector.

    Raises:
        ValueError: If matrix is not square or k exceeds matrix dimensions.
        np.linalg.LinAlgError: If decomposition fails to converge.
    """
    if isinstance(matrix, list):
        try:
            matrix = np.array(matrix, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError('Input list cannot be converted to a numeric numpy array') from e
    if not isinstance(matrix, np.ndarray):
        raise TypeError('Input must be a numpy array or a list that can be converted to numpy array')
    if matrix.ndim != 2:
        raise ValueError('Input must be a 2D matrix')
    (n_rows, n_cols) = matrix.shape
    if n_rows != n_cols:
        raise ValueError('Input matrix must be square')
    n = n_rows
    if not largest_only:
        try:
            (eigenvalues, eigenvectors) = np.linalg.eigh(matrix)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            return (eigenvalues, eigenvectors)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError('Eigenvalue decomposition failed to converge') from e
    if k <= 0:
        raise ValueError('k must be a positive integer')
    if k > n:
        k = n
    try:
        from scipy.sparse.linalg import eigsh
        sparse_matrix = sparse.csr_matrix(matrix)
        (eigenvalues, eigenvectors) = eigsh(sparse_matrix, k=k, which='LM')
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return (eigenvalues, eigenvectors)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError('Eigenvalue decomposition failed to converge') from e
    except Exception as e:
        raise np.linalg.LinAlgError('Eigenvalue decomposition failed to converge') from e