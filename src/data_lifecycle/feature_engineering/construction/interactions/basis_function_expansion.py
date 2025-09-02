import numpy as np
from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class BasisFunctionExpansionTransformer(BaseTransformer):

    def __init__(self, basis_type: str='polynomial', degree: int=2, n_components: int=10, gamma: float=1.0, include_bias: bool=False, name: Optional[str]=None):
        """
        Initialize the BasisFunctionExpansionTransformer.
        
        Parameters
        ----------
        basis_type : str, default='polynomial'
            Type of basis function to apply. Supported values are:
            - 'polynomial': Polynomial features up to specified degree
            - 'fourier': Fourier basis functions (sin/cos)
            - 'rbf': Radial basis functions with Gaussian kernels
        degree : int, default=2
            Degree of polynomial expansion or number of Fourier terms
        n_components : int, default=10
            Number of RBF centers to generate
        gamma : float, default=1.0
            Scaling parameter for RBF and other basis functions
        include_bias : bool, default=False
            Whether to include a bias/intercept term
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        if basis_type not in ['polynomial', 'fourier', 'rbf']:
            raise ValueError(f"Unsupported basis_type: {basis_type}. Supported values are 'polynomial', 'fourier', 'rbf'")
        self.basis_type = basis_type
        self.degree = degree
        self.n_components = n_components
        self.gamma = gamma
        self.include_bias = include_bias
        self.feature_names: List[str] = []
        self._centers: Optional[np.ndarray] = None
        self._input_feature_names: List[str] = []
        self._n_input_features: int = 0

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'BasisFunctionExpansionTransformer':
        """
        Fit the transformer to the input data.
        
        For RBF expansion, this determines the centers and scales.
        For other expansions, this validates the input dimensions.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        BasisFunctionExpansionTransformer
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._input_feature_names = data.feature_names if data.feature_names is not None else [f'x{i}' for i in range(X.shape[1])]
        else:
            X = data
            self._input_feature_names = [f'x{i}' for i in range(X.shape[1])]
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._n_input_features = X.shape[1]
        if self.basis_type == 'rbf':
            n_samples = X.shape[0]
            n_centers = min(self.n_components, n_samples)
            indices = np.random.choice(n_samples, n_centers, replace=False)
            self._centers = X[indices].copy()
        self.feature_names = self.get_feature_names(self._input_feature_names)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the basis function expansion to input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, transforms the features attribute.
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed data with expanded features
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata is not None else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores is not None else {}
        else:
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.basis_type == 'polynomial':
            X_transformed = self._apply_polynomial_expansion(X)
        elif self.basis_type == 'fourier':
            X_transformed = self._apply_fourier_expansion(X)
        elif self.basis_type == 'rbf':
            X_transformed = self._apply_rbf_expansion(X)
        else:
            raise ValueError(f'Unsupported basis_type: {self.basis_type}')
        metadata['basis_type'] = self.basis_type
        metadata['degree'] = self.degree
        metadata['n_components'] = self.n_components
        metadata['gamma'] = self.gamma
        metadata['include_bias'] = self.include_bias
        return FeatureSet(features=X_transformed, feature_names=self.feature_names, feature_types=['numeric'] * len(self.feature_names), sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation if possible.
        
        Note: Inverse transformation is generally not possible for basis function expansions
        unless they are bijective, which is rarely the case.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to invert
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Original data format (approximate if exact inversion is not possible)
            
        Raises
        ------
        NotImplementedError
            If inverse transformation is not supported for the basis type
        """
        raise NotImplementedError('Inverse transformation is not implemented for basis function expansions.')

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get the names of the generated features.
        
        Parameters
        ----------
        input_features : List[str], optional
            Names of the input features
            
        Returns
        -------
        List[str]
            Names of the generated features after expansion
        """
        if input_features is None:
            if hasattr(self, '_input_feature_names') and self._input_feature_names:
                input_features = self._input_feature_names
            else:
                n_features = getattr(self, '_n_input_features', 0)
                input_features = [f'x{i}' for i in range(n_features)]
        feature_names = []
        if self.include_bias:
            feature_names.append('bias')
        if self.basis_type == 'polynomial':
            feature_names.extend(self._generate_polynomial_feature_names(input_features))
        elif self.basis_type == 'fourier':
            feature_names.extend(self._generate_fourier_feature_names(input_features))
        elif self.basis_type == 'rbf':
            feature_names.extend(self._generate_rbf_feature_names())
        return feature_names

    def _apply_polynomial_expansion(self, X: np.ndarray) -> np.ndarray:
        """Apply polynomial feature expansion."""
        (n_samples, n_features) = X.shape
        powers = self._generate_powers(n_features, self.degree)
        features = []
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        for power in powers:
            feature = np.prod([X[:, i:i + 1] ** p for (i, p) in enumerate(power) if p > 0], axis=0)
            features.append(feature)
        return np.hstack(features) if features else np.ones((n_samples, 1))

    def _generate_powers(self, n_features: int, degree: int) -> List[List[int]]:
        """Generate all power combinations for polynomial features."""
        from itertools import combinations_with_replacement
        powers = []
        for d in range(1, degree + 1):
            combos = list(combinations_with_replacement(range(n_features), d))
            for combo in combos:
                power = [0] * n_features
                for i in combo:
                    power[i] += 1
                powers.append(power)
        return powers

    def _generate_polynomial_feature_names(self, input_features: List[str]) -> List[str]:
        """Generate feature names for polynomial expansion."""
        powers = self._generate_powers(len(input_features), self.degree)
        names = []
        for power in powers:
            name_parts = []
            for (i, p) in enumerate(power):
                if p == 1:
                    name_parts.append(input_features[i])
                elif p > 1:
                    name_parts.append(f'{input_features[i]}^{p}')
            names.append(' '.join(name_parts) if name_parts else '1')
        return names

    def _apply_fourier_expansion(self, X: np.ndarray) -> np.ndarray:
        """Apply Fourier feature expansion."""
        (n_samples, n_features) = X.shape
        features = []
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        for i in range(n_features):
            for j in range(1, self.degree + 1):
                features.append(np.sin(j * X[:, i:i + 1]))
                features.append(np.cos(j * X[:, i:i + 1]))
        return np.hstack(features) if features else np.ones((n_samples, 1))

    def _generate_fourier_feature_names(self, input_features: List[str]) -> List[str]:
        """Generate feature names for Fourier expansion."""
        names = []
        for (i, feature) in enumerate(input_features):
            for j in range(1, self.degree + 1):
                if j == 1:
                    names.append(f'sin({feature})')
                    names.append(f'cos({feature})')
                else:
                    names.append(f'sin({j}*{feature})')
                    names.append(f'cos({j}*{feature})')
        return names

    def _apply_rbf_expansion(self, X: np.ndarray) -> np.ndarray:
        """Apply RBF feature expansion."""
        if self._centers is None:
            raise ValueError('RBF centers not initialized. Call fit() first.')
        n_samples = X.shape[0]
        n_centers = self._centers.shape[0]
        features = []
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        for i in range(n_centers):
            diff = X - self._centers[i]
            distances_squared = np.sum(diff ** 2, axis=1, keepdims=True)
            rbf_values = np.exp(-self.gamma * distances_squared)
            features.append(rbf_values)
        return np.hstack(features) if features else np.ones((n_samples, 1))

    def _generate_rbf_feature_names(self) -> List[str]:
        """Generate feature names for RBF expansion."""
        names = []
        if self._centers is not None:
            for i in range(self._centers.shape[0]):
                names.append(f'rbf_{i}')
        return names