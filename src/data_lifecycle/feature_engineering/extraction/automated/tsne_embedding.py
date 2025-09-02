from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
import warnings


# ...(code omitted)...


class TunableTSNEEmbeddingExtractor(BaseTransformer):

    def __init__(self, n_components: int=2, perplexity_range: List[float]=None, n_trials: int=5, early_exaggeration: float=12.0, learning_rate: float=200.0, n_iter: int=1000, random_state: Optional[int]=None, verbose: int=0, name: Optional[str]=None):
        super().__init__(name=name)
        if n_components < 1:
            raise ValueError('n_components must be >= 1')
        if n_iter < 250:
            raise ValueError('n_iter must be >= 250')
        if n_trials < 1:
            raise ValueError('n_trials must be >= 1')
        if perplexity_range is not None and len(perplexity_range) != 2:
            raise ValueError('perplexity_range must be a list of two values [min, max]')
        self.n_components = n_components
        self.perplexity_range = perplexity_range or [5.0, 50.0]
        self.n_trials = n_trials
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, data: FeatureSet, **kwargs) -> 'TunableTSNEEmbeddingExtractor':
        """
        Fit the t-SNE model with automated perplexity tuning.

        This method evaluates multiple perplexity values within the specified range and selects
        the one that produces the best embedding quality according to an internal criterion.
        The optimal perplexity value is then used to compute the final embedding.

        Args:
            data (FeatureSet): Input feature set containing high-dimensional data to embed.
                               Must contain a valid feature matrix.

        Returns:
            TunableTSNEEmbeddingExtractor: Returns self for method chaining.

        Raises:
            ValueError: If the input data is invalid or incompatible.
            RuntimeError: If t-SNE fails to converge for all perplexity values tested.
        """
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet instance.')
        X = data.features
        if X is None or X.shape[0] == 0:
            raise ValueError('Input FeatureSet must contain a non-empty feature matrix.')
        (min_p, max_p) = self.perplexity_range
        if min_p >= max_p:
            raise ValueError('Invalid perplexity_range: min must be less than max.')
        n_samples = X.shape[0]
        max_allowed_perplexity = min(max_p, (n_samples - 1) // 3)
        if max_allowed_perplexity <= min_p:
            perplexity_candidates = [min_p]
        elif self.n_trials == 1:
            perplexity_candidates = [(min_p + max_allowed_perplexity) / 2]
        else:
            perplexity_candidates = np.linspace(min_p, max_allowed_perplexity, self.n_trials).tolist()
        best_perplexity = None
        best_kl_divergence = np.inf
        best_embedding = None
        for (i, perplexity) in enumerate(perplexity_candidates):
            try:
                if self.verbose > 0:
                    print(f'Fitting t-SNE with perplexity={perplexity:.2f} ({i + 1}/{len(perplexity_candidates)})')
                tsne = TSNE(n_components=self.n_components, perplexity=perplexity, early_exaggeration=self.early_exaggeration, learning_rate=self.learning_rate, n_iter=self.n_iter, random_state=self.random_state, verbose=max(0, self.verbose - 1), init='random')
                embedding = tsne.fit_transform(X)
                if tsne.kl_divergence_ < best_kl_divergence:
                    best_kl_divergence = tsne.kl_divergence_
                    best_perplexity = perplexity
                    best_embedding = embedding
            except Exception as e:
                if self.verbose > 0:
                    print(f'Failed to fit t-SNE with perplexity={perplexity}: {e}')
                continue
        if best_perplexity is None:
            raise RuntimeError('Failed to fit t-SNE with any perplexity value in the given range')
        self.best_perplexity_ = best_perplexity
        self.embedding_ = best_embedding
        self.kl_divergence_ = best_kl_divergence
        self.is_fitted_ = True
        if self.verbose > 0:
            print(f'Selected perplexity: {best_perplexity:.2f} with KL divergence: {best_kl_divergence:.4f}')
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform the input data using the fitted t-SNE model.

        Note that t-SNE is a transductive algorithm, so the transform operation simply returns
        the embedding computed during the fit phase.

        Args:
            data (FeatureSet): Input feature set (not actually used since t-SNE is transductive).

        Returns:
            FeatureSet: Low-dimensional embedding of the input data with updated feature names.
            
        Raises:
            RuntimeError: If the transformer has not been fitted yet.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError('Transformer has not been fitted yet.')
        feature_names = [f'tsne_component_{i}' for i in range(self.n_components)]
        sample_ids = data.sample_ids if data.sample_ids is not None else [f'sample_{i}' for i in range(self.embedding_.shape[0])]
        return FeatureSet(features=self.embedding_, feature_names=feature_names, sample_ids=sample_ids)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for t-SNE as it is a non-invertible transformation.

        Args:
            data (FeatureSet): Input low-dimensional embedding.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            FeatureSet: This method raises a NotImplementedError as t-SNE cannot reconstruct original features.
            
        Raises:
            NotImplementedError: Always raised as t-SNE does not support inverse transformation.
        """
        raise NotImplementedError('t-SNE does not support inverse transformation.')