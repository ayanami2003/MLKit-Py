import numpy as np
from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet


# ...(code omitted)...


class SymbolicFeatureGenerator(BaseTransformer):
    """
    A transformer for generating new features using symbolic regression and genetic programming techniques.
    
    This class discovers mathematical expressions that best describe relationships in the data,
    creating new features based on these expressions. It combines symbolic regression for
    expression discovery with genetic programming for feature synthesis.
    
    Attributes
    ----------
    population_size : int
        Number of programs in each generation
    generations : int
        Number of generations to evolve programs
    tournament_size : int
        Tournament size for selection
    stopping_criteria : float
        Fitness threshold for early stopping
    const_range : tuple
        Range of constants in programs
    init_depth : tuple
        Initial depth of programs
    init_method : str
        Method to initialize programs ('half and half', 'grow', 'full')
    function_set : list
        Functions to use in programs
    metric : str
        Metric to evaluate program fitness
    parsimony_coefficient : float
        Coefficient to control bloat
    p_crossover : float
        Probability of crossover
    p_subtree_mutation : float
        Probability of subtree mutation
    p_hoist_mutation : float
        Probability of hoist mutation
    p_point_mutation : float
        Probability of point mutation
    max_samples : float
        Fraction of samples to use
    warm_start : bool
        Whether to reuse previous solution
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Controls verbosity
    random_state : int
        Random state for reproducibility
    """

    def __init__(self, population_size: int=1000, generations: int=20, tournament_size: int=20, stopping_criteria: float=0.0, const_range: tuple=(-1.0, 1.0), init_depth: tuple=(2, 6), init_method: str='half and half', function_set: List[str]=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'], metric: str='mean absolute error', parsimony_coefficient: float=0.001, p_crossover: float=0.9, p_subtree_mutation: float=0.01, p_hoist_mutation: float=0.01, p_point_mutation: float=0.01, max_samples: float=1.0, warm_start: bool=False, n_jobs: int=1, verbose: int=0, random_state: Optional[int]=None):
        """
        Initialize the SymbolicFeatureGenerator.
        
        Parameters
        ----------
        population_size : int, default=1000
            Number of programs in each generation.
        generations : int, default=20
            Number of generations to evolve programs.
        tournament_size : int, default=20
            Tournament size for selection.
        stopping_criteria : float, default=0.0
            Fitness threshold for early stopping.
        const_range : tuple, default=(-1.0, 1.0)
            Range of constants in programs.
        init_depth : tuple, default=(2, 6)
            Initial depth of programs.
        init_method : str, default='half and half'
            Method to initialize programs ('half and half', 'grow', 'full').
        function_set : list, default=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min']
            Functions to use in programs.
        metric : str, default='mean absolute error'
            Metric to evaluate program fitness.
        parsimony_coefficient : float, default=0.001
            Coefficient to control bloat.
        p_crossover : float, default=0.9
            Probability of crossover.
        p_subtree_mutation : float, default=0.01
            Probability of subtree mutation.
        p_hoist_mutation : float, default=0.01
            Probability of hoist mutation.
        p_point_mutation : float, default=0.01
            Probability of point mutation.
        max_samples : float, default=1.0
            Fraction of samples to use.
        warm_start : bool, default=False
            Whether to reuse previous solution.
        n_jobs : int, default=1
            Number of parallel jobs.
        verbose : int, default=0
            Controls verbosity.
        random_state : int, optional
            Random state for reproducibility.
        """
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.max_samples = max_samples
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        if not GP_LEARN_AVAILABLE:
            raise ImportError('gplearn is required for SymbolicFeatureGenerator but is not installed. Please install it with: pip install gplearn')

    def fit(self, data: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'SymbolicFeatureGenerator':
        """
        Fit the symbolic feature generator to the data.
        
        Discovers mathematical expressions that describe relationships in the data
        and prepares for feature generation based on these expressions.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing features. If FeatureSet, uses features attribute.
        y : np.ndarray, optional
            Target values for supervised feature generation.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        SymbolicFeatureGenerator
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        else:
            X = data
            self._feature_names = None
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self._n_features = X.shape[1]
        if self._feature_names is None:
            self._feature_names = [f'x{i}' for i in range(self._n_features)]
        if y is None:
            self._programs = []
            for i in range(min(10, self._n_features)):
                regressor = SymbolicRegressor(population_size=self.population_size, generations=self.generations, tournament_size=self.tournament_size, stopping_criteria=self.stopping_criteria, const_range=self.const_range, init_depth=self.init_depth, init_method=self.init_method, function_set=self.function_set, metric=self.metric, parsimony_coefficient=self.parsimony_coefficient, p_crossover=self.p_crossover, p_subtree_mutation=self.p_subtree_mutation, p_hoist_mutation=self.p_hoist_mutation, p_point_mutation=self.p_point_mutation, max_samples=self.max_samples, warm_start=self.warm_start, n_jobs=self.n_jobs, verbose=self.verbose, random_state=self.random_state)
                X_other = np.delete(X, i, axis=1)
                y_target = X[:, i]
                regressor.fit(X_other, y_target)
                self._programs.append(regressor)
        else:
            self._programs = []
            regressor = SymbolicRegressor(population_size=self.population_size, generations=self.generations, tournament_size=self.tournament_size, stopping_criteria=self.stopping_criteria, const_range=self.const_range, init_depth=self.init_depth, init_method=self.init_method, function_set=self.function_set, metric=self.metric, parsimony_coefficient=self.parsimony_coefficient, p_crossover=self.p_crossover, p_subtree_mutation=self.p_subtree_mutation, p_hoist_mutation=self.p_hoist_mutation, p_point_mutation=self.p_point_mutation, max_samples=self.max_samples, warm_start=self.warm_start, n_jobs=self.n_jobs, verbose=self.verbose, random_state=self.random_state)
            regressor.fit(X, y)
            self._programs.append(regressor)
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Generate new features based on discovered mathematical expressions.
        
        Applies the evolved programs to create new features from the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, uses features attribute.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Transformed data with newly generated features.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self._n_features:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted with {self._n_features} features.')
        new_features = []
        new_feature_names = []
        for (i, program) in enumerate(self._programs):
            new_feature = program.predict(X).reshape(-1, 1)
            new_features.append(new_feature)
            new_feature_names.append(f'symbolic_feature_{i}')
        if new_features:
            X_transformed = np.hstack([X] + new_features)
            all_feature_names = self._feature_names + new_feature_names
        else:
            X_transformed = X
            all_feature_names = self._feature_names
        return FeatureSet(features=X_transformed, feature_names=all_feature_names, feature_types=['numeric'] * X_transformed.shape[1], sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply inverse transformation if possible.
        
        For symbolic features, this operation is generally not reversible.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to invert.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original data format (identity transformation).
        """
        if isinstance(data, FeatureSet):
            return data
        else:
            return FeatureSet(features=data, feature_names=[f'x{i}' for i in range(data.shape[1])] if data.ndim > 1 else ['x0'], feature_types=['numeric'] * (data.shape[1] if data.ndim > 1 else 1))

    def get_feature_expressions(self) -> List[str]:
        """
        Get the mathematical expressions for the generated features.
        
        Returns
        -------
        List[str]
            List of string representations of the mathematical expressions.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        expressions = []
        for (i, program) in enumerate(self._programs):
            expressions.append(str(program._program))
        return expressions