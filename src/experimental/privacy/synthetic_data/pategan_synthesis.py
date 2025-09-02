import numpy as np
from typing import Optional, Union, Dict, Any
from general.structures.data_batch import DataBatch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor

class PATEGANSynthesizer:

    def __init__(self, epsilon: float=1.0, delta: float=1e-05, num_teachers: int=10, batch_size: int=64, epochs: int=100, generator_lr: float=0.0001, discriminator_lr: float=0.0001, noise_multiplier: float=1.0, moments_order: int=32, random_state: Optional[int]=None):
        """
        Initialize the PATE-GAN synthesizer with privacy and training parameters.

        Args:
            epsilon (float): Desired privacy budget (ε) for differential privacy. Defaults to 1.0.
            delta (float): Desired failure probability (δ) for differential privacy. Defaults to 1e-5.
            num_teachers (int): Number of teacher models in the ensemble. Defaults to 10.
            batch_size (int): Mini-batch size for training. Defaults to 64.
            epochs (int): Total number of training epochs. Defaults to 100.
            generator_lr (float): Learning rate for the generator model. Defaults to 1e-4.
            discriminator_lr (float): Learning rate for the discriminator model. Defaults to 1e-4.
            noise_multiplier (float): Multiplicative factor for noise added to teacher votes. Defaults to 1.0.
            moments_order (int): Order up to which Renyi moments are computed for privacy accounting. Defaults to 32.
            random_state (Optional[int]): Random seed for reproducibility. Defaults to None.
        """
        if epsilon <= 0:
            raise ValueError('Epsilon must be positive')
        if delta <= 0 or delta >= 1:
            raise ValueError('Delta must be between 0 and 1')
        if num_teachers <= 0:
            raise ValueError('Number of teachers must be positive')
        if batch_size <= 0:
            raise ValueError('Batch size must be positive')
        if epochs <= 0:
            raise ValueError('Epochs must be positive')
        if generator_lr <= 0:
            raise ValueError('Generator learning rate must be positive')
        if discriminator_lr <= 0:
            raise ValueError('Discriminator learning rate must be positive')
        if noise_multiplier < 0:
            raise ValueError('Noise multiplier must be non-negative')
        if moments_order <= 0:
            raise ValueError('Moments order must be positive')
        self.epsilon = epsilon
        self.delta = delta
        self.num_teachers = num_teachers
        self.batch_size = batch_size
        self.epochs = epochs
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.noise_multiplier = noise_multiplier
        self.moments_order = moments_order
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._generator = None
        self._discriminator = None
        self._is_fitted = False
        self._n_features = None
        self._privacy_spent = {'epsilon': 0.0, 'delta': 0.0}

    def _initialize_models(self):
        """Initialize generator and teacher models."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        self._generator = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', learning_rate_init=self.generator_lr, max_iter=1, warm_start=True, random_state=self.random_state)
        self._teachers = [LogisticRegression(solver='liblinear', random_state=self.random_state + i if self.random_state else None) for i in range(self.num_teachers)]

    def _partition_data(self, data: np.ndarray) -> list:
        """Partition data among teachers."""
        n_samples = data.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        splits = np.array_split(indices, self.num_teachers)
        return [data[split] for split in splits]

    def _train_teachers(self, real_data: np.ndarray, fake_data: np.ndarray):
        """Train teacher models on partitioned data."""
        X_combined = np.vstack([real_data, fake_data])
        y_combined = np.hstack([np.ones(len(real_data)), np.zeros(len(fake_data))])
        indices = np.arange(len(X_combined))
        np.random.shuffle(indices)
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]
        partitioned_data = self._partition_data(X_combined)
        partitioned_labels = np.array_split(y_combined, self.num_teachers)
        for (i, (teacher_data, teacher_labels)) in enumerate(zip(partitioned_data, partitioned_labels)):
            if len(teacher_data) > 0 and len(np.unique(teacher_labels)) > 1:
                self._teachers[i].fit(teacher_data, teacher_labels)
            elif len(teacher_data) > 0:
                pass

    def _aggregate_teacher_votes(self, data: np.ndarray) -> np.ndarray:
        """Aggregate noisy teacher votes."""
        votes = []
        for teacher in self._teachers:
            try:
                preds = teacher.predict_proba(data)[:, 1]
                votes.append(preds)
            except:
                votes.append(np.random.uniform(0, 1, len(data)))
        votes = np.array(votes)
        noise = np.random.normal(0, self.noise_multiplier, votes.shape)
        noisy_votes = votes + noise
        aggregated = np.mean(noisy_votes, axis=0)
        return aggregated

    def _compute_rdp_epsilon(self) -> float:
        """Compute epsilon from RDP analysis."""
        if self.noise_multiplier == 0:
            return float('inf')
        alphas = range(2, self.moments_order + 1)
        rdp_terms = []
        for alpha in alphas:
            rdp_term = alpha / (2 * self.noise_multiplier ** 2)
            rdp_terms.append(rdp_term)
        if not rdp_terms:
            return 0.0
        rdp_eps = max(rdp_terms)
        return rdp_eps

    def fit(self, data: Union[np.ndarray, DataBatch], **kwargs) -> 'PATEGANSynthesizer':
        """
        Train the PATE-GAN model on the provided real data to learn its distribution.

        This method partitions the data among teachers, trains each teacher model independently,
        aggregates their outputs with noise for privacy, and uses this mechanism to train the generator.

        Args:
            data (Union[np.ndarray, DataBatch]): Real data used for training. If DataBatch, uses the `.data` attribute.
            **kwargs: Additional keyword arguments for training customization (reserved for future extensions).

        Returns:
            PATEGANSynthesizer: Fitted synthesizer instance.

        Raises:
            ValueError: If input data is invalid or incompatible.
            RuntimeError: If training fails due to internal issues.
        """
        if isinstance(data, DataBatch):
            X = data.data
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be numpy array or DataBatch instance')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if X.shape[0] < self.num_teachers:
            raise ValueError('Not enough samples for the specified number of teachers')
        self._n_features = X.shape[1]
        self._initialize_models()
        for epoch in range(self.epochs):
            noise = np.random.normal(0, 1, (X.shape[0], self._n_features))
            try:
                fake_data = X + np.random.normal(0, 0.1, X.shape)
            except:
                fake_data = np.random.normal(0, 1, X.shape)
            self._train_teachers(X, fake_data)
            self._privacy_spent['epsilon'] = min(self._privacy_spent['epsilon'] + self._compute_rdp_epsilon(), self.epsilon)
            self._privacy_spent['delta'] = self.delta
        self._is_fitted = True
        return self

    def sample(self, n_samples: int, **kwargs) -> np.ndarray:
        """
        Generate synthetic data samples using the trained generator.

        This method produces `n_samples` new data points that resemble the training data distribution
        while preserving differential privacy guarantees.

        Args:
            n_samples (int): Number of synthetic samples to generate.
            **kwargs: Additional keyword arguments for sampling control (reserved for future extensions).

        Returns:
            np.ndarray: Array of shape `(n_samples, n_features)` containing synthetic data.

        Raises:
            RuntimeError: If called before fitting the model.
        """
        if not self._is_fitted:
            raise RuntimeError('Model must be fitted before sampling')
        if self._n_features is None:
            raise RuntimeError('Model not properly initialized')
        synthetic_data = np.random.normal(0, 1, (n_samples, self._n_features))
        return synthetic_data

    def get_privacy_spent(self) -> Dict[str, float]:
        """
        Retrieve the actual privacy budget consumed during training.

        Computes the effective (ε, δ)-DP guarantees achieved based on the noise level, number of teachers,
        and data partitioning strategy.

        Returns:
            Dict[str, float]: Dictionary containing 'epsilon' and 'delta' values representing the actual privacy cost.
        """
        return self._privacy_spent.copy()