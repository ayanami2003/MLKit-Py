from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch

class EpsilonGreedyBandit(BaseModel):
    """
    Epsilon-Greedy Multi-Armed Bandit strategy for contextual and non-contextual bandit problems.
    
    This class implements the epsilon-greedy exploration strategy, where actions are selected
    randomly with probability epsilon and greedily (based on estimated rewards) with probability
    1-epsilon. It supports both contextual and non-contextual bandit settings.
    
    Attributes:
        n_arms (int): Number of arms/actions available.
        epsilon (float): Probability of exploration (random action selection).
        context_dim (Optional[int]): Dimension of context vectors, if applicable.
        arm_rewards (np.ndarray): Accumulated rewards for each arm.
        arm_counts (np.ndarray): Number of times each arm has been pulled.
        name (str): Name identifier for the bandit model.
        is_fitted (bool): Whether the model has been fitted.
    """

    def __init__(self, n_arms: int, epsilon: float=0.1, context_dim: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the Epsilon-Greedy Bandit.
        
        Args:
            n_arms (int): Number of arms/actions in the bandit problem.
            epsilon (float): Exploration probability (0 <= epsilon <= 1). Defaults to 0.1.
            context_dim (Optional[int]): Dimension of context vectors. If provided, enables contextual bandit.
            name (Optional[str]): Name identifier for the model.
        """
        super().__init__(name=name)
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.context_dim = context_dim
        self.arm_rewards = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms)
        if context_dim is not None:
            self.arm_weights = np.zeros((n_arms, context_dim))
        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, DataBatch], y: np.ndarray, **kwargs) -> 'EpsilonGreedyBandit':
        """
        Fit the bandit model to historical data (optional for epsilon-greedy).
        
        In online learning settings, this might not be used, but it's included for compatibility
        with batch training scenarios.
        
        Args:
            X (Union[np.ndarray, DataBatch]): Context vectors (if contextual) or dummy input.
            y (np.ndarray): Rewards for each action taken.
            **kwargs: Additional fitting parameters.
            
        Returns:
            EpsilonGreedyBandit: The fitted model instance.
        """
        if isinstance(X, DataBatch):
            X = X.data
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, DataBatch], **kwargs) -> np.ndarray:
        """
        Select actions based on the epsilon-greedy strategy.
        
        With probability epsilon, selects a random action. With probability 1-epsilon,
        selects the action with the highest estimated reward.
        
        Args:
            X (Union[np.ndarray, DataBatch]): Context vectors for contextual bandits or dummy input.
            **kwargs: Additional prediction parameters.
            
        Returns:
            np.ndarray: Selected actions (arm indices).
        """
        if isinstance(X, DataBatch):
            X = X.data
        if X.ndim == 1:
            if self.context_dim is not None:
                if len(X) != self.context_dim:
                    raise ValueError(f'Context vector dimension mismatch. Expected {self.context_dim}, got {len(X)}')
                X = X.reshape(1, -1)
            else:
                X = X.reshape(1, -1)
        elif X.ndim == 2 and self.context_dim is not None:
            if X.shape[1] != self.context_dim:
                raise ValueError(f'Context matrix dimension mismatch. Expected {self.context_dim} columns, got {X.shape[1]}')
        elif X.ndim > 2:
            raise ValueError('X must be at most 2-dimensional')
        n_samples = X.shape[0]
        actions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if np.random.rand() < self.epsilon:
                actions[i] = np.random.randint(self.n_arms)
            else:
                if self.context_dim is not None:
                    context = X[i]
                    expected_rewards = self.arm_weights @ context
                else:
                    avg_rewards = np.divide(self.arm_rewards, self.arm_counts, out=np.full_like(self.arm_rewards, -np.inf), where=self.arm_counts != 0)
                    if np.all(avg_rewards == -np.inf):
                        actions[i] = np.random.randint(self.n_arms)
                        continue
                    expected_rewards = avg_rewards
                actions[i] = np.argmax(expected_rewards)
        return actions

    def update(self, context: Optional[np.ndarray], action: int, reward: float, **kwargs) -> None:
        """
        Update the bandit model with a new observation.
        
        Updates the reward estimates for the selected action based on the received reward.
        
        Args:
            context (Optional[np.ndarray]): Context vector for the current observation (if contextual).
            action (int): Index of the action/arm selected.
            reward (float): Reward received for the selected action.
            **kwargs: Additional update parameters.
        """
        if action < 0 or action >= self.n_arms:
            raise ValueError(f'Action must be between 0 and {self.n_arms - 1}')
        if self.context_dim is not None:
            if context is None:
                raise ValueError('Context must be provided for contextual bandit')
            if len(context) != self.context_dim:
                raise ValueError(f'Context dimension mismatch. Expected {self.context_dim}, got {len(context)}')
            predicted_reward = np.dot(self.arm_weights[action], context)
            error = reward - predicted_reward
            learning_rate = 0.1 / (1 + self.arm_counts[action])
            self.arm_weights[action] += learning_rate * error * context
        elif context is not None:
            pass
        self.arm_counts[action] += 1
        self.arm_rewards[action] += (reward - self.arm_rewards[action]) / self.arm_counts[action]

    def get_action_probabilities(self, context: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Get the probability distribution over actions for a given context.
        
        Args:
            context (Optional[np.ndarray]): Context vector for contextual bandits.
            
        Returns:
            np.ndarray: Probability of selecting each action.
        """
        if self.context_dim is not None:
            if context is None:
                raise ValueError('Context must be provided for contextual bandit')
            if len(context) != self.context_dim:
                raise ValueError(f'Context dimension mismatch. Expected {self.context_dim}, got {len(context)}')
            expected_rewards = self.arm_weights @ context
        else:
            avg_rewards = np.divide(self.arm_rewards, self.arm_counts, out=np.full_like(self.arm_rewards, 0), where=self.arm_counts != 0)
            expected_rewards = avg_rewards
        if self.context_dim is None and np.sum(self.arm_counts) == 0:
            return np.full(self.n_arms, 1.0 / self.n_arms)
        probabilities = np.full(self.n_arms, self.epsilon / self.n_arms)
        best_action = np.argmax(expected_rewards)
        probabilities[best_action] += 1 - self.epsilon
        return probabilities

    def score(self, X: Union[np.ndarray, DataBatch], y: np.ndarray, **kwargs) -> float:
        """
        Evaluate model performance on test data.
        
        For contextual bandits, this typically computes the accuracy of action selection
        compared to optimal actions.
        
        Args:
            X (Union[np.ndarray, DataBatch]): Context data of shape (n_samples, context_dim) or DataBatch.
            y (np.ndarray): True optimal actions or rewards for evaluation.
            **kwargs: Additional evaluation parameters.
            
        Returns:
            float: Model performance score (higher is better).
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