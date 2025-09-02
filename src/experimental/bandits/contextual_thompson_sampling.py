from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch

class ContextualThompsonSampler(BaseModel):
    """
    A contextual multi-armed bandit implementation using Thompson Sampling.

    This class implements a Bayesian approach to the contextual multi-armed bandit problem,
    where actions are selected based on probabilistic sampling from posterior distributions
    of model parameters. It's particularly effective for exploration-exploitation trade-offs
    in online learning settings with contextual information.

    The model maintains a Bayesian linear regression for each arm, assuming normally
    distributed rewards. It uses conjugate priors for efficient posterior updates.

    Attributes:
        n_arms (int): Number of arms/actions available.
        context_dim (int): Dimension of the context vectors.
        alpha (float): Precision parameter for noise in reward observations.
        lambda_prior (float): Regularization parameter (precision of prior).
        arm_models (dict): Internal models for each arm, storing posterior parameters.
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float=1.0, lambda_prior: float=1.0, name: Optional[str]=None):
        """
        Initialize the Contextual Thompson Sampler.

        Args:
            n_arms (int): Number of arms/actions in the bandit problem.
            context_dim (int): Dimension of context vectors for each action.
            alpha (float): Reward noise precision parameter (inverse variance).
            lambda_prior (float): Prior precision for model parameters.
            name (Optional[str]): Optional name for the model instance.
        """
        super().__init__(name=name)
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_prior = lambda_prior
        self.arm_models = {}
        for arm in range(n_arms):
            self.arm_models[arm] = {'A': self.lambda_prior * np.eye(context_dim), 'b': np.zeros(context_dim)}

    def fit(self, X: Union[np.ndarray, DataBatch], y: np.ndarray, **kwargs) -> 'ContextualThompsonSampler':
        """
        Fit the model to historical data (optional offline training).

        This method allows for pre-training the model on historical data before
        online deployment. It updates the posterior distributions for all arms
        based on the provided context-reward pairs.

        Args:
            X (Union[np.ndarray, DataBatch]): Context data of shape (n_samples, context_dim) or DataBatch.
            y (np.ndarray): Rewards of shape (n_samples,) where each element corresponds to the reward
                           obtained for the action taken in that context.
            **kwargs: Additional fitting parameters (not used in this implementation).

        Returns:
            ContextualThompsonSampler: Returns self for method chaining.

        Raises:
            ValueError: If input dimensions don't match expected values.
        """
        if isinstance(X, DataBatch):
            X = X.data
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.context_dim:
            raise ValueError(f'X must have shape (n_samples, {self.context_dim})')
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError('y must be a 1D array with length equal to number of samples in X')
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, DataBatch], **kwargs) -> np.ndarray:
        """
        Select actions for contexts using Thompson Sampling.

        For each context in X, this method samples model parameters from their
        posterior distributions and selects the action with the highest predicted reward.

        Args:
            X (Union[np.ndarray, DataBatch]): Context data of shape (n_samples, context_dim) or DataBatch.
            **kwargs: Additional prediction parameters (not used in this implementation).

        Returns:
            np.ndarray: Array of selected actions (arm indices) of shape (n_samples,).

        Raises:
            ValueError: If model has not been properly initialized or fitted.
        """
        if isinstance(X, DataBatch):
            X = X.data
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.context_dim:
            raise ValueError(f'X must have shape (n_samples, {self.context_dim})')
        n_samples = X.shape[0]
        actions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            context = X[i]
            sampled_rewards = self.sample_rewards(context)
            actions[i] = np.argmax(sampled_rewards)
        return actions

    def update(self, context: np.ndarray, action: int, reward: float, **kwargs) -> None:
        """
        Update the model with a new observation.

        This method updates the posterior distribution for the selected arm based
        on the observed context and reward. It's used in online learning scenarios
        where feedback arrives sequentially.

        Args:
            context (np.ndarray): Context vector of shape (context_dim,).
            action (int): Index of the selected arm/action.
            reward (float): Observed reward for the action taken.
            **kwargs: Additional update parameters (not used in this implementation).

        Raises:
            ValueError: If action index is out of bounds or context dimension mismatch.
        """
        if action < 0 or action >= self.n_arms:
            raise ValueError(f'Action must be between 0 and {self.n_arms - 1}')
        if context.ndim != 1 or len(context) != self.context_dim:
            raise ValueError(f'Context must be a 1D array of length {self.context_dim}')
        model = self.arm_models[action]
        model['A'] += self.alpha * np.outer(context, context)
        model['b'] += self.alpha * reward * context

    def sample_rewards(self, context: np.ndarray, **kwargs) -> np.ndarray:
        """
        Sample expected rewards for all arms given a context.

        This method samples model parameters for each arm from their respective
        posterior distributions and computes the expected reward for each arm
        in the given context.

        Args:
            context (np.ndarray): Context vector of shape (context_dim,).
            **kwargs: Additional sampling parameters (not used in this implementation).

        Returns:
            np.ndarray: Array of sampled expected rewards for each arm of shape (n_arms,).
        """
        if context.ndim != 1 or len(context) != self.context_dim:
            raise ValueError(f'Context must be a 1D array of length {self.context_dim}')
        rewards = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            model = self.arm_models[arm]
            cov = np.linalg.inv(model['A'])
            mu = cov.dot(model['b'])
            w = np.random.multivariate_normal(mu, cov)
            rewards[arm] = w.dot(context)
        return rewards

    def score(self, X: Union[np.ndarray, DataBatch], y: np.ndarray, **kwargs) -> float:
        """
        Evaluate model performance on test data.

        For contextual bandits, this typically computes the cumulative regret or 
        accuracy of action selection compared to optimal actions.

        Args:
            X (Union[np.ndarray, DataBatch]): Context data of shape (n_samples, context_dim) or DataBatch.
            y (np.ndarray): True optimal actions or rewards for evaluation.
            **kwargs: Additional evaluation parameters.

        Returns:
            float: Model performance score (higher is better).

        Note:
            This is a simplified implementation that returns accuracy of action selection.
            In practice, bandit evaluation would require more sophisticated metrics.
        """
        if isinstance(X, DataBatch):
            X = X.data
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predicted_actions = self.predict(X)
        if len(predicted_actions) != len(y):
            raise ValueError('Length of predictions and true values must match')
        accuracy = np.mean(predicted_actions == y)
        return accuracy