from typing import List, Union, Optional
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np

class ConsensusMatrixGenerator(BaseTransformer):

    def __init__(self, name: Optional[str]=None):
        """
        Initialize the ConsensusMatrixGenerator.
        
        Parameters
        ----------
        name : Optional[str]
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.consensus_matrix_: Optional[np.ndarray] = None
        self.clusterings_: List[np.ndarray] = []
        self.n_samples_: int = 0

    def fit(self, clusterings: List[Union[np.ndarray, List[int]]], **kwargs) -> 'ConsensusMatrixGenerator':
        """
        Compute the consensus matrix from multiple clustering results.
        
        Parameters
        ----------
        clusterings : List[Union[np.ndarray, List[int]]]
            List of cluster assignment arrays. Each array should have the same length,
            representing the cluster labels for each sample in that clustering run.
            Cluster labels should be integers starting from 0.
        **kwargs : dict
            Additional parameters (not used in this implementation)
            
        Returns
        -------
        ConsensusMatrixGenerator
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If clusterings list is empty or if clustering arrays have inconsistent lengths
        """
        if not clusterings:
            raise ValueError('Clusterings list is empty')
        processed_clusterings = []
        n_samples = len(clusterings[0]) if isinstance(clusterings[0], (list, np.ndarray)) else 0
        for (i, clustering) in enumerate(clusterings):
            if isinstance(clustering, list):
                clustering = np.array(clustering)
            if not isinstance(clustering, np.ndarray):
                raise ValueError(f'Clustering {i} must be a numpy array or list')
            if len(clustering) != n_samples:
                raise ValueError('All clustering arrays must have the same length')
            processed_clusterings.append(clustering)
        self.n_samples_ = n_samples
        self.clusterings_ = processed_clusterings
        self.consensus_matrix_ = np.zeros((n_samples, n_samples))
        for clustering in processed_clusterings:
            if clustering.ndim > 1 and clustering.shape[1] > 1:
                predicted_clusters = np.argmax(clustering, axis=1)
            else:
                predicted_clusters = clustering.ravel()
            for i in range(n_samples):
                for j in range(i, n_samples):
                    if predicted_clusters[i] == predicted_clusters[j]:
                        self.consensus_matrix_[i, j] += 1
                        if i != j:
                            self.consensus_matrix_[j, i] += 1
        self.consensus_matrix_ = self.consensus_matrix_ / len(processed_clusterings)
        np.fill_diagonal(self.consensus_matrix_, 1.0)
        return self

    def transform(self, clusterings: List[Union[np.ndarray, List[int]]], **kwargs) -> np.ndarray:
        """
        Generate consensus matrix from clustering results.
        
        This is equivalent to calling fit followed by accessing the consensus_matrix_ attribute.
        
        Parameters
        ----------
        clusterings : List[Union[np.ndarray, List[int]]]
            List of cluster assignment arrays. Each array should have the same length,
            representing the cluster labels for each sample in that clustering run.
        **kwargs : dict
            Additional parameters (not used in this implementation)
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_samples)
            Consensus matrix where element (i,j) represents the fraction of clusterings
            in which samples i and j were assigned to the same cluster
            
        Raises
        ------
        ValueError
            If clusterings list is empty or if clustering arrays have inconsistent lengths
        """
        self.fit(clusterings)
        return self.consensus_matrix_

    def fit_transform(self, clusterings: List[Union[np.ndarray, List[int]]], **kwargs) -> np.ndarray:
        """
        Compute and return the consensus matrix in one step.
        
        Parameters
        ----------
        clusterings : List[Union[np.ndarray, List[int]]]
            List of cluster assignment arrays. Each array should have the same length,
            representing the cluster labels for each sample in that clustering run.
        **kwargs : dict
            Additional parameters (not used in this implementation)
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_samples)
            Consensus matrix where element (i,j) represents the fraction of clusterings
            in which samples i and j were assigned to the same cluster
        """
        return self.fit(clusterings).consensus_matrix_

    def inverse_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Not applicable for consensus matrix generation.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        np.ndarray
            Always raises NotImplementedError
            
        Raises
        ------
        NotImplementedError
            Inverse transformation is not defined for consensus matrices
        """
        raise NotImplementedError('Inverse transformation is not defined for consensus matrices')

    def get_consensus_score(self) -> float:
        """
        Calculate the consensus score (average consensus) of the matrix.
        
        The consensus score measures the overall stability of the clustering ensemble
        by computing the average value of the consensus matrix (excluding diagonal).
        
        Returns
        -------
        float
            Average consensus value ranging from 0 (no agreement) to 1 (perfect agreement)
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if self.consensus_matrix_ is None:
            raise ValueError('ConsensusMatrixGenerator has not been fitted yet.')
        n = self.consensus_matrix_.shape[0]
        if n <= 1:
            return 1.0
        upper_triangle_sum = np.sum(np.triu(self.consensus_matrix_, k=1))
        upper_triangle_count = n * (n - 1) // 2
        return upper_triangle_sum / upper_triangle_count if upper_triangle_count > 0 else 0.0

    def get_cluster_stability(self, cluster_labels: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Calculate stability scores for each cluster in a given clustering.
        
        Cluster stability is measured as the average consensus within each cluster.
        
        Parameters
        ----------
        cluster_labels : Union[np.ndarray, List[int]]
            Cluster labels for each sample
            
        Returns
        -------
        np.ndarray
            Stability score for each cluster
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet or if cluster_labels length
            doesn't match the fitted data
        """
        if self.consensus_matrix_ is None:
            raise ValueError('ConsensusMatrixGenerator has not been fitted yet.')
        if isinstance(cluster_labels, list):
            cluster_labels = np.array(cluster_labels)
        if len(cluster_labels) != self.n_samples_:
            raise ValueError('Cluster labels length must match fitted data length')
        unique_clusters = np.unique(cluster_labels)
        stability_scores = np.zeros(len(unique_clusters))
        for (i, cluster_id) in enumerate(unique_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) <= 1:
                stability_scores[i] = 1.0
                continue
            submatrix = self.consensus_matrix_[np.ix_(cluster_indices, cluster_indices)]
            upper_triangle_sum = np.sum(np.triu(submatrix, k=1))
            num_pairs = len(cluster_indices) * (len(cluster_indices) - 1) // 2
            stability_scores[i] = upper_triangle_sum / num_pairs if num_pairs > 0 else 1.0
        return stability_scores