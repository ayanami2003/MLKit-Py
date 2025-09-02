import numpy as np
from typing import Optional

class CrossEntropyLoss:
    """
    A class to compute the cross-entropy loss for classification tasks.

    Cross-entropy loss measures the performance of a classification model whose output is a probability value
    between 0 and 1. It increases as the predicted probability diverges from the actual label.
    """

    def __init__(self, epsilon: float=1e-15):
        """
        Initialize the CrossEntropyLoss with a small epsilon to avoid log(0).

        Args:
            epsilon (float): Small value to clip probabilities and prevent undefined logarithm operations.
        """
        self.epsilon = epsilon

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross-entropy loss between true labels and predicted probabilities.

        Args:
            y_true (np.ndarray): Ground truth labels, shape (n_samples,) or (n_samples, n_classes).
            y_pred (np.ndarray): Predicted probabilities, shape (n_samples,) or (n_samples, n_classes).

        Returns:
            float: Computed cross-entropy loss value.

        Raises:
            ValueError: If input shapes are incompatible.
        """
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f'Number of samples in y_true ({y_true.shape[0]}) and y_pred ({y_pred.shape[0]}) must match.')
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)
            y_true = y_true.reshape(-1)
            if not np.all(np.isin(y_true, [0, 1])):
                raise ValueError('Binary classification labels must be 0 or 1.')
            y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            if len(y_true.shape) == 1:
                num_classes = y_pred.shape[1]
                y_true_one_hot = np.eye(num_classes)[y_true]
            else:
                y_true_one_hot = y_true
            if y_true_one_hot.shape != y_pred.shape:
                raise ValueError(f'Shape mismatch: y_true {y_true_one_hot.shape} and y_pred {y_pred.shape} must match for multi-class.')
            y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            loss = -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))
        return loss

class LogisticLoss:
    """
    A class to compute the logistic loss (log loss) for binary classification tasks.

    Logistic loss is used in logistic regression and other probabilistic classifiers to measure
    the performance of a model that outputs probabilities between 0 and 1.
    """

    def __init__(self, epsilon: float=1e-15):
        """
        Initialize the LogisticLoss with a small epsilon to avoid log(0).

        Args:
            epsilon (float): Small value to clip probabilities and prevent undefined logarithm operations.
        """
        self.epsilon = epsilon

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the logistic loss between true labels and predicted probabilities.

        Args:
            y_true (np.ndarray): Ground truth binary labels, shape (n_samples,).
            y_pred (np.ndarray): Predicted probabilities, shape (n_samples,).

        Returns:
            float: Computed logistic loss value.

        Raises:
            ValueError: If input shapes are incompatible or labels are not binary.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape.')
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError('y_true must contain only binary labels (0 or 1).')
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return float(loss)


# ...(code omitted)...


class LogarithmicLoss:
    """
    A class to compute the logarithmic loss for probabilistic classification tasks.

    Note: This is another naming convention for what is commonly known as log loss or cross-entropy loss.
    It measures the performance of a classification model whose output is a probability value between 0 and 1.
    """

    def __init__(self, epsilon: float=1e-15):
        """
        Initialize the LogarithmicLoss with a small epsilon to avoid log(0).

        Args:
            epsilon (float): Small value to clip probabilities and prevent undefined logarithm operations.
        """
        self.epsilon = epsilon

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the logarithmic loss between true labels and predicted probabilities.

        Args:
            y_true (np.ndarray): Ground truth labels, shape (n_samples,) or (n_samples, n_classes).
            y_pred (np.ndarray): Predicted probabilities, shape (n_samples,) or (n_samples, n_classes).

        Returns:
            float: Computed logarithmic loss value.

        Raises:
            ValueError: If input shapes are incompatible or probabilities are outside [0, 1].
        """
        if not np.all((y_pred >= 0) & (y_pred <= 1)):
            raise ValueError('Predicted probabilities must be between 0 and 1.')
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f'Number of samples in y_true ({y_true.shape[0]}) and y_pred ({y_pred.shape[0]}) must match.')
        if len(y_pred.shape) == 1 or (len(y_pred.shape) == 2 and y_pred.shape[1] == 1):
            y_pred = y_pred.reshape(-1)
            y_true = y_true.reshape(-1)
            if not np.all(np.isin(y_true, [0, 1])):
                raise ValueError('Binary classification labels must be 0 or 1.')
            y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            if len(y_true.shape) == 1:
                num_classes = y_pred.shape[1]
                y_true_one_hot = np.eye(num_classes)[y_true]
            else:
                y_true_one_hot = y_true
            if y_true_one_hot.shape != y_pred.shape:
                raise ValueError(f'Shape mismatch: y_true {y_true_one_hot.shape} and y_pred {y_pred.shape} must match for multi-class.')
            y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            loss = -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))
        return float(loss)


# ...(code omitted)...


class MeanAbsoluteErrorLoss:
    """
    A class to compute the Mean Absolute Error (MAE) loss for regression and classification tasks.

    MAE loss computes the average of absolute differences between predicted and true values.
    It is less sensitive to outliers compared to Mean Squared Error and can be used in robust regression.
    """

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean absolute error loss between true and predicted values.

        Args:
            y_true (np.ndarray): Ground truth values, shape (n_samples,) or (n_samples, n_outputs).
            y_pred (np.ndarray): Predicted values, shape (n_samples,) or (n_samples, n_outputs).

        Returns:
            float: Computed mean absolute error loss value.

        Raises:
            ValueError: If input shapes are incompatible.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got y_true shape {y_true.shape} and y_pred shape {y_pred.shape}')
        if y_true.size == 0:
            return 0.0
        mae = np.mean(np.abs(y_true - y_pred))
        return float(mae)

class DiceLoss:
    """
    A class to compute the Dice loss for binary or multiclass segmentation/classification tasks,
    particularly effective for imbalanced datasets.

    Dice loss is based on the Dice coefficient, which measures overlap between predicted and true labels.
    It is widely used in medical image segmentation and other domains with significant class imbalance.
    """

    def __init__(self, smooth: float=1e-06):
        """
        Initialize the DiceLoss with a smoothing parameter to avoid division by zero.
        
        Args:
            smooth (float): Smoothing parameter to prevent division by zero during computation.
        """
        self.smooth = smooth

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Dice loss between true labels and predicted probabilities.
        
        Args:
            y_true (np.ndarray): Ground truth labels, shape (n_samples,) or (n_samples, n_classes).
            y_pred (np.ndarray): Predicted probabilities, shape (n_samples,) or (n_samples, n_classes).
            
        Returns:
            float: Computed Dice loss value.
            
        Raises:
            ValueError: If input shapes are incompatible.
        """
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f'Number of samples in y_true ({y_true.shape[0]}) and y_pred ({y_pred.shape[0]}) must match.')
        if len(y_pred.shape) == 1 or (len(y_pred.shape) == 2 and y_pred.shape[1] == 1):
            y_pred = y_pred.reshape(-1)
            y_true = y_true.reshape(-1)
            if not np.all(np.isin(y_true, [0, 1])):
                y_true = (y_true > 0.5).astype(float)
            intersection = np.sum(y_true * y_pred)
            union = np.sum(y_true) + np.sum(y_pred)
            dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return float(1.0 - dice_coefficient)
        else:
            if len(y_true.shape) == 1:
                num_classes = y_pred.shape[1]
                y_true_one_hot = np.eye(num_classes)[y_true.astype(int)]
            else:
                y_true_one_hot = y_true
            if y_true_one_hot.shape != y_pred.shape:
                raise ValueError(f'Shape mismatch: y_true {y_true_one_hot.shape} and y_pred {y_pred.shape} must match for multi-class.')
            intersection = np.sum(y_true_one_hot * y_pred, axis=0)
            union = np.sum(y_true_one_hot, axis=0) + np.sum(y_pred, axis=0)
            dice_coefficients = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return float(1.0 - np.mean(dice_coefficients))


# ...(code omitted)...


class ContrastiveLoss:
    """
    A class to compute the contrastive loss for learning embeddings where similar samples should be close
    and dissimilar samples should be far apart in the embedding space.

    Contrastive loss is commonly used in siamese networks and other metric learning scenarios.
    """

    def __init__(self, margin: float=1.0):
        """
        Initialize the ContrastiveLoss with a margin parameter.
        
        Args:
            margin (float): Minimum distance margin between dissimilar pairs.
        """
        self.margin = margin

    def compute(self, embeddings1: np.ndarray, embeddings2: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the contrastive loss between pairs of embeddings.
        
        Args:
            embeddings1 (np.ndarray): First set of embeddings, shape (n_pairs, embedding_dim).
            embeddings2 (np.ndarray): Second set of embeddings, shape (n_pairs, embedding_dim).
            y_true (np.ndarray): Binary labels indicating similarity (1) or dissimilarity (0),
                               shape (n_pairs,).
                               
        Returns:
            float: Average contrastive loss over all pairs.
            
        Raises:
            ValueError: If input shapes are incompatible.
        """
        if embeddings1.shape != embeddings2.shape:
            raise ValueError(f'Embeddings shapes must match. Got {embeddings1.shape} and {embeddings2.shape}')
        if embeddings1.ndim != 2:
            raise ValueError(f'Embeddings must be 2D arrays. Got shape {embeddings1.shape}')
        if y_true.shape[0] != embeddings1.shape[0]:
            raise ValueError(f'Number of labels ({y_true.shape[0]}) must match number of pairs ({embeddings1.shape[0]})')
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError('Labels must be binary (0 or 1)')
        diff = embeddings1 - embeddings2
        distances_squared = np.sum(diff ** 2, axis=1)
        distances = np.sqrt(distances_squared)
        similar_loss = y_true * distances_squared
        hinge_loss = np.maximum(0, self.margin - distances)
        dissimilar_loss = (1 - y_true) * hinge_loss ** 2
        total_loss = 0.5 * (similar_loss + dissimilar_loss)
        return float(np.mean(total_loss))

class TripletLoss:
    """
    A class to compute the triplet loss for learning embeddings by comparing anchor, positive, and negative samples.

    Triplet loss encourages the distance between an anchor and a positive sample (same class) to be smaller
    than the distance between the anchor and a negative sample (different class) by at least a margin.
    """

    def __init__(self, margin: float=1.0):
        """
        Initialize the TripletLoss with a margin parameter.

        Args:
            margin (float): Margin to enforce separation between positive and negative samples.
        """
        self.margin = margin

    def compute(self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
        """
        Compute the triplet loss using anchor, positive, and negative embeddings.

        Args:
            anchor (np.ndarray): Anchor embeddings, shape (n_samples, embedding_dim).
            positive (np.ndarray): Positive embeddings (same class as anchor), shape (n_samples, embedding_dim).
            negative (np.ndarray): Negative embeddings (different class from anchor), shape (n_samples, embedding_dim).

        Returns:
            float: Computed triplet loss value.

        Raises:
            ValueError: If input shapes are incompatible.
        """
        if anchor.shape != positive.shape or anchor.shape != negative.shape:
            raise ValueError(f'Input arrays must have the same shape. Got anchor: {anchor.shape}, positive: {positive.shape}, negative: {negative.shape}')
        pos_dist = np.sum(np.square(anchor - positive), axis=1)
        neg_dist = np.sum(np.square(anchor - negative), axis=1)
        losses = np.maximum(0, pos_dist - neg_dist + self.margin)
        return float(np.mean(losses))


# ...(code omitted)...


def tversky_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float=0.3, beta: float=0.7, smooth: float=1e-07) -> float:
    """
    Compute the Tversky loss for binary or multiclass classification problems, particularly useful for imbalanced datasets.

    The Tversky loss is a generalization of the Dice loss and Focal Loss that allows for adjusting the penalty
    for false positives and false negatives independently via the alpha and beta parameters respectively.

    Args:
        y_true (np.ndarray): Ground truth labels, shape (n_samples,) or (n_samples, n_classes).
        y_pred (np.ndarray): Predicted probabilities, shape (n_samples,) or (n_samples, n_classes).
        alpha (float): Weight for false positives. Lower values reduce false positive penalty.
        beta (float): Weight for false negatives. Lower values reduce false negative penalty.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: Computed Tversky loss value.

    Raises:
        ValueError: If input shapes are incompatible or parameters are out of valid ranges.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f'Input shapes are incompatible: y_true {y_true.shape} and y_pred {y_pred.shape}')
    if y_true.size == 0:
        raise ValueError('Input arrays cannot be empty')
    if not 0 <= alpha <= 1:
        raise ValueError(f'Alpha must be between 0 and 1, got {alpha}')
    if not 0 <= beta <= 1:
        raise ValueError(f'Beta must be between 0 and 1, got {beta}')
    if smooth < 0:
        raise ValueError(f'Smooth must be non-negative, got {smooth}')
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    tp = np.sum(y_true_flat * y_pred_flat)
    fp = np.sum((1 - y_true_flat) * y_pred_flat)
    fn = np.sum(y_true_flat * (1 - y_pred_flat))
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky_index