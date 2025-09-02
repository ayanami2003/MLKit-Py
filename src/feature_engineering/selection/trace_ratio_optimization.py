from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class TraceRatioOptimizer(BaseTransformer):
    """
    Transformer for performing trace ratio optimization-based feature selection.
    
    This transformer implements the trace ratio optimization criterion for multivariate
    feature selection. It selects features by optimizing the ratio of the trace of
    between-class scatter to the trace of within-class scatter, which is particularly
    effective for discriminative feature subset selection in classification tasks.
    
    The method iteratively searches for the optimal feature subset that maximizes
    the trace ratio criterion, considering both the discriminatory power and redundancy
    among features.
    
    Attributes
    ----------
    n_features_to_select : int, optional
        Number of features to select. If None, it will be determined automatically.
    max_iterations : int, default=100
        Maximum number of iterations for the optimization process.
    tolerance : float, default=1e-6
        Convergence tolerance for the optimization algorithm.
    verbose : bool, default=False
        Whether to print progress messages during optimization.
    """

    def __init__(self, n_features_to_select: Optional[int]=None, max_iterations: int=100, tolerance: float=1e-06, verbose: bool=False, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_features_to_select = n_features_to_select
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

    def fit(self, data: Union[FeatureSet, np.ndarray], labels: Optional[np.ndarray]=None, **kwargs) -> 'TraceRatioOptimizer':
        """
        Fit the trace ratio optimizer to the input data.
        
        This method computes the optimal feature subset by maximizing the trace ratio
        criterion. It analyzes the between-class and within-class scatter matrices
        to determine feature importance.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to select from. If FeatureSet, uses the features attribute.
        labels : Optional[np.ndarray], optional
            Target labels for supervised feature selection. Required for trace ratio optimization.
        **kwargs : dict
            Additional parameters for fitting (not used currently).
            
        Returns
        -------
        TraceRatioOptimizer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If labels are not provided or data dimensions don't match.
        """
        if labels is None:
            raise ValueError('Labels are required for trace ratio optimization.')
        if isinstance(data, FeatureSet):
            X = data.features
            self.feature_names_ = data.feature_names
        else:
            X = data
            self.feature_names_ = None
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array.')
        if len(labels) != X.shape[0]:
            raise ValueError('Number of labels must match number of samples in data.')
        (n_samples, n_features) = X.shape
        if self.n_features_to_select is None:
            self.n_features_to_select = min(int(np.sqrt(n_features)), n_features)
        if self.n_features_to_select > n_features:
            raise ValueError('n_features_to_select cannot be larger than the number of features.')
        (Sb, Sw) = self._compute_scatter_matrices(X, labels)
        selected_indices = self._optimize_trace_ratio(X, Sb, Sw)
        self.selected_indices_ = np.array(selected_indices)
        self._support_mask = np.zeros(n_features, dtype=bool)
        self._support_mask[self.selected_indices_] = True
        return self

    def _compute_scatter_matrices(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Compute between-class and within-class scatter matrices.
        
        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features)
        y : np.ndarray
            Target labels of shape (n_samples,)
            
        Returns
        -------
        tuple
            (Sb, Sw) where Sb is between-class scatter matrix and Sw is within-class scatter matrix
        """
        (n_samples, n_features) = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        mean_global = np.mean(X, axis=0)
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        for cls in classes:
            X_cls = X[y == cls]
            n_cls = X_cls.shape[0]
            mean_cls = np.mean(X_cls, axis=0)
            diff_cls = X_cls - mean_cls
            Sw += diff_cls.T @ diff_cls
            diff_mean = (mean_cls - mean_global).reshape(-1, 1)
            Sb += n_cls * (diff_mean @ diff_mean.T)
        return (Sb, Sw)

    def _optimize_trace_ratio(self, X: np.ndarray, Sb: np.ndarray, Sw: np.ndarray) -> list:
        """
        Optimize trace ratio using iterative algorithm.
        
        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features)
        Sb : np.ndarray
            Between-class scatter matrix
        Sw : np.ndarray
            Within-class scatter matrix
            
        Returns
        -------
        list
            Indices of selected features
        """
        (n_samples, n_features) = X.shape
        fisher_scores = np.zeros(n_features)
        for i in range(n_features):
            sb_i = Sb[i, i]
            sw_i = Sw[i, i]
            fisher_scores[i] = sb_i / sw_i if sw_i != 0 else 0
        initial_indices = np.argsort(fisher_scores)[::-1][:self.n_features_to_select]
        selected_indices = list(initial_indices)
        prev_trace_ratio = -np.inf
        for iteration in range(self.max_iterations):
            if len(selected_indices) == 0:
                break
            Sb_selected = Sb[np.ix_(selected_indices, selected_indices)]
            Sw_selected = Sw[np.ix_(selected_indices, selected_indices)]
            trace_sb = np.trace(Sb_selected)
            trace_sw = np.trace(Sw_selected)
            if trace_sw == 0:
                current_trace_ratio = np.inf
            else:
                current_trace_ratio = trace_sb / trace_sw
            if abs(current_trace_ratio - prev_trace_ratio) < self.tolerance:
                if self.verbose:
                    print(f'Converged at iteration {iteration}')
                break
            prev_trace_ratio = current_trace_ratio
            best_trace_ratio = current_trace_ratio
            best_swap = None
            if len(selected_indices) > self.n_features_to_select:
                for i in selected_indices[:]:
                    temp_indices = [idx for idx in selected_indices if idx != i]
                    if len(temp_indices) > 0:
                        trace_ratio = self._compute_trace_ratio(temp_indices, Sb, Sw)
                        if trace_ratio > best_trace_ratio:
                            best_trace_ratio = trace_ratio
                            best_swap = ('remove', i)
            elif len(selected_indices) < self.n_features_to_select:
                available_features = [i for i in range(n_features) if i not in selected_indices]
                for i in available_features:
                    temp_indices = selected_indices + [i]
                    trace_ratio = self._compute_trace_ratio(temp_indices, Sb, Sw)
                    if trace_ratio > best_trace_ratio:
                        best_trace_ratio = trace_ratio
                        best_swap = ('add', i)
            elif len(selected_indices) == self.n_features_to_select and len(selected_indices) > 0:
                for i in selected_indices[:]:
                    available_features = [j for j in range(n_features) if j not in selected_indices]
                    for j in available_features:
                        temp_indices = [idx if idx != i else j for idx in selected_indices]
                        trace_ratio = self._compute_trace_ratio(temp_indices, Sb, Sw)
                        if trace_ratio > best_trace_ratio:
                            best_trace_ratio = trace_ratio
                            best_swap = ('swap', (i, j))
            if best_swap is not None and best_trace_ratio > current_trace_ratio:
                (action, idx) = best_swap
                if action == 'add':
                    selected_indices.append(idx)
                elif action == 'remove':
                    selected_indices.remove(idx)
                elif action == 'swap':
                    (old_idx, new_idx) = idx
                    selected_indices.remove(old_idx)
                    selected_indices.append(new_idx)
            else:
                if self.verbose:
                    print(f'No improvement at iteration {iteration}')
                break
            if self.verbose:
                print(f'Iteration {iteration}: trace ratio = {current_trace_ratio:.6f}')
        if len(selected_indices) > self.n_features_to_select:
            fisher_scores_selected = [fisher_scores[i] for i in selected_indices]
            top_indices = np.argsort(fisher_scores_selected)[::-1][:self.n_features_to_select]
            selected_indices = [selected_indices[i] for i in top_indices]
        elif len(selected_indices) < self.n_features_to_select and len(selected_indices) > 0:
            available_features = [i for i in range(n_features) if i not in selected_indices]
            fisher_scores_available = [fisher_scores[i] for i in available_features]
            n_needed = self.n_features_to_select - len(selected_indices)
            top_available = np.argsort(fisher_scores_available)[::-1][:n_needed]
            additional_indices = [available_features[i] for i in top_available]
            selected_indices.extend(additional_indices)
        return selected_indices

    def _compute_trace_ratio(self, indices: list, Sb: np.ndarray, Sw: np.ndarray) -> float:
        """
        Compute trace ratio for a given set of feature indices.
        
        Parameters
        ----------
        indices : list
            List of feature indices
        Sb : np.ndarray
            Between-class scatter matrix
        Sw : np.ndarray
            Within-class scatter matrix
            
        Returns
        -------
        float
            Trace ratio value
        """
        if len(indices) == 0:
            return 0.0
        Sb_selected = Sb[np.ix_(indices, indices)]
        Sw_selected = Sw[np.ix_(indices, indices)]
        trace_sb = np.trace(Sb_selected)
        trace_sw = np.trace(Sw_selected)
        if trace_sw == 0:
            return np.inf if trace_sb > 0 else 0.0
        return trace_sb / trace_sw

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply feature selection to the input data.
        
        Transforms the input data by selecting only the features identified as optimal
        during the fitting process based on the trace ratio criterion.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to transform. Must have same number of features as fitted data.
        **kwargs : dict
            Additional parameters for transformation (not used currently).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with selected features only. Type matches input type.
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or data dimensions don't match.
        """
        if not hasattr(self, '_support_mask'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array.')
        if X.shape[1] != len(self._support_mask):
            raise ValueError('Number of features in input data does not match the fitted data.')
        X_selected = X[:, self._support_mask]
        if isinstance(data, FeatureSet):
            selected_feature_names = None
            if feature_names is not None and hasattr(self, 'selected_indices_'):
                selected_feature_names = [feature_names[i] for i in self.selected_indices_]
            return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=None if data.feature_types is None else [data.feature_types[i] for i in self.selected_indices_], sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        else:
            return X_selected

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Reverse the feature selection transformation.
        
        Creates a representation with all original features, filling non-selected
        features with zeros. This is primarily for compatibility with pipelines
        that might require invertible transformations.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Selected features data to inverse transform.
        **kwargs : dict
            Additional parameters for inverse transformation (not used currently).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Data with all original features reconstructed. Non-selected features are zero-filled.
            
        Raises
        ------
        ValueError
            If transformer has not been fitted.
        """
        if not hasattr(self, '_support_mask'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X_selected = data.features
        else:
            X_selected = data
        if X_selected.ndim != 2:
            raise ValueError('Input data must be a 2D array.')
        if X_selected.shape[1] != len(self.selected_indices_):
            raise ValueError('Number of features in input data does not match the number of selected features.')
        n_samples = X_selected.shape[0]
        n_features = len(self._support_mask)
        X_full = np.zeros((n_samples, n_features))
        X_full[:, self._support_mask] = X_selected
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_full, feature_names=self.feature_names_, feature_types=None, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        else:
            return X_full

    def get_support(self, indices: bool=False) -> Union[np.ndarray, list]:
        """
        Get a mask or indices of the selected features.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, returns indices of selected features rather than a boolean mask.
            
        Returns
        -------
        Union[np.ndarray, list]
            Boolean mask array or list of indices indicating selected features.
        """
        if not hasattr(self, '_support_mask'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if indices:
            return self.selected_indices_.tolist()
        else:
            return self._support_mask.copy()

    def get_feature_names(self) -> Optional[list]:
        """
        Get names of the selected features.
        
        Returns
        -------
        Optional[list]
            List of selected feature names, or None if not available.
        """
        if not hasattr(self, '_support_mask'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if self.feature_names_ is None:
            return None
        return [self.feature_names_[i] for i in self.selected_indices_]