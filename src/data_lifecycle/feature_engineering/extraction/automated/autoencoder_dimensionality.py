from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class AutoencoderDimensionalityReducer(BaseTransformer):

    def __init__(self, encoding_dim: int, hidden_layers: Optional[List[int]]=None, activation: str='relu', epochs: int=100, batch_size: int=32, learning_rate: float=0.001, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the AutoencoderDimensionalityReducer.
        
        Parameters
        ----------
        encoding_dim : int
            The dimension of the encoded (compressed) representation
        hidden_layers : Optional[List[int]], optional
            The sizes of hidden layers in the autoencoder. If None, a single hidden layer
            with size halfway between input and encoding dimensions is used
        activation : str, optional
            Activation function to use in the autoencoder layers (default: 'relu')
        epochs : int, optional
            Number of training epochs (default: 100)
        batch_size : int, optional
            Batch size for training (default: 32)
        learning_rate : float, optional
            Learning rate for the optimizer (default: 0.001)
        random_state : Optional[int], optional
            Random seed for reproducibility (default: None)
        name : Optional[str], optional
            Name of the transformer instance (default: None)
        """
        super().__init__(name=name)
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _initialize_weights(self, shape):
        """Initialize weights with Xavier initialization"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)

    def _activate(self, x):
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            x = np.clip(x, -500, 500)
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            return x

    def _activation_derivative(self, x):
        """Compute derivative of activation function"""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            return np.ones_like(x)

    def _build_network(self, input_dim):
        """Build the autoencoder network architecture"""
        if self.hidden_layers is None:
            hidden_size = max(self.encoding_dim, (input_dim + self.encoding_dim) // 2)
            self._hidden_layers = [hidden_size]
        else:
            self._hidden_layers = self.hidden_layers
        layer_sizes = [input_dim] + self._hidden_layers + [self.encoding_dim]
        self._encoder_weights = []
        self._encoder_biases = []
        for i in range(len(layer_sizes) - 1):
            w = self._initialize_weights((layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self._encoder_weights.append(w)
            self._encoder_biases.append(b)
        layer_sizes.reverse()
        self._decoder_weights = []
        self._decoder_biases = []
        for i in range(len(layer_sizes) - 1):
            w = self._initialize_weights((layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self._decoder_weights.append(w)
            self._decoder_biases.append(b)

    def _forward_pass(self, x):
        """Forward pass through the autoencoder"""
        encoder_activations = [x]
        current = x
        for i in range(len(self._encoder_weights)):
            z = np.dot(current, self._encoder_weights[i]) + self._encoder_biases[i]
            current = self._activate(z)
            encoder_activations.append(current)
        encoded = current
        decoder_activations = [encoded]
        current = encoded
        for i in range(len(self._decoder_weights)):
            z = np.dot(current, self._decoder_weights[i]) + self._decoder_biases[i]
            current = self._activate(z)
            decoder_activations.append(current)
        decoded = current
        return (encoded, decoded, encoder_activations, decoder_activations)

    def _backward_pass(self, x, encoder_activations, decoder_activations):
        """Backward pass to compute gradients"""
        output_error = decoder_activations[-1] - x
        decoder_dW = []
        decoder_db = []
        error = output_error
        for i in reversed(range(len(self._decoder_weights))):
            dW = np.dot(decoder_activations[i].T, error) / x.shape[0]
            db = np.mean(error, axis=0, keepdims=True)
            decoder_dW.insert(0, dW)
            decoder_db.insert(0, db)
            if i > 0:
                error = np.dot(error, self._decoder_weights[i].T) * self._activation_derivative(decoder_activations[i])
        encoder_dW = []
        encoder_db = []
        for i in range(len(self._encoder_weights)):
            layer_idx = len(self._encoder_weights) - 1 - i
            dW = np.dot(encoder_activations[layer_idx].T, error) / x.shape[0]
            db = np.mean(error, axis=0, keepdims=True)
            encoder_dW.append(dW)
            encoder_db.append(db)
            if layer_idx > 0:
                error = np.dot(error, self._encoder_weights[layer_idx].T) * self._activation_derivative(encoder_activations[layer_idx])
        encoder_dW.reverse()
        encoder_db.reverse()
        return (encoder_dW, encoder_db, decoder_dW, decoder_db)

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'AutoencoderDimensionalityReducer':
        """
        Train the autoencoder on the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to train the autoencoder on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for fitting (not currently used)
            
        Returns
        -------
        AutoencoderDimensionalityReducer
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        self._input_shape = X.shape
        input_dim = X.shape[1]
        self._build_network(input_dim)
        for epoch in range(self.epochs):
            if self.random_state is not None:
                np.random.seed(self.random_state + epoch)
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            for i in range(0, X.shape[0], self.batch_size):
                batch = X_shuffled[i:i + self.batch_size]
                (encoded, decoded, encoder_activations, decoder_activations) = self._forward_pass(batch)
                (encoder_dW, encoder_db, decoder_dW, decoder_db) = self._backward_pass(batch, encoder_activations, decoder_activations)
                for j in range(len(self._encoder_weights)):
                    self._encoder_weights[j] -= self.learning_rate * encoder_dW[j]
                    self._encoder_biases[j] -= self.learning_rate * encoder_db[j]
                for j in range(len(self._decoder_weights)):
                    self._decoder_weights[j] -= self.learning_rate * decoder_dW[j]
                    self._decoder_biases[j] -= self.learning_rate * decoder_db[j]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply dimensionality reduction using the trained encoder.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for transformation (not currently used)
            
        Returns
        -------
        FeatureSet
            Transformed data with reduced dimensionality
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        current = X
        for i in range(len(self._encoder_weights)):
            z = np.dot(current, self._encoder_weights[i]) + self._encoder_biases[i]
            current = self._activate(z)
        return FeatureSet(features=current, names=[f'encoded_feature_{i}' for i in range(current.shape[1])])

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reconstruct the original space from the encoded representation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Encoded data to reconstruct. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for inverse transformation (not currently used)
            
        Returns
        -------
        FeatureSet
            Reconstructed data in the original feature space
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        current = X
        for i in range(len(self._decoder_weights)):
            z = np.dot(current, self._decoder_weights[i]) + self._decoder_biases[i]
            current = self._activate(z)
        return FeatureSet(features=current, names=[f'reconstructed_feature_{i}' for i in range(current.shape[1])])