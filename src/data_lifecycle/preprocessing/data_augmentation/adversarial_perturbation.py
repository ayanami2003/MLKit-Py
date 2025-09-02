from typing import Optional
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class AdversarialPerturbationTransformer(BaseTransformer):
    """
    A transformer that applies adversarial perturbations to input features for data augmentation.
    
    This class generates small, intentional perturbations to input data that are designed to
    challenge model robustness. It can be used as part of a data augmentation pipeline to
    improve model generalization and resilience to input variations.
    
    The transformer supports various perturbation methods and allows control over the
    magnitude and nature of the applied perturbations.
    
    Parameters
    ----------
    epsilon : float, default=0.01
        Magnitude of the perturbation to apply. Controls how strong the adversarial
        perturbation will be.
    perturbation_method : str, default='fgsm'
        Method used to generate perturbations. Supported methods include:
        - 'fgsm': Fast Gradient Sign Method
        - 'rand': Random perturbation
    random_state : int, optional
        Random seed for reproducibility of random perturbations.
    preserve_original : bool, default=True
        Whether to include the original samples in the output in addition to the
        perturbed samples.
    name : str, optional
        Name of the transformer instance.
    
    Attributes
    ----------
    epsilon_ : float
        Actual epsilon value used for perturbations.
    perturbation_method_ : str
        Actual perturbation method used.
    """

    def __init__(self, epsilon: float=0.01, perturbation_method: str='fgsm', random_state: Optional[int]=None, preserve_original: bool=True, name: Optional[str]=None):
        super().__init__(name=name)
        self.epsilon = epsilon
        self.perturbation_method = perturbation_method
        self.random_state = random_state
        self.preserve_original = preserve_original
        if perturbation_method not in ['fgsm', 'rand']:
            raise ValueError(f"Unsupported perturbation method: {perturbation_method}. Supported methods are: 'fgsm', 'rand'")

    def fit(self, data: FeatureSet, **kwargs) -> 'AdversarialPerturbationTransformer':
        """
        Fit the transformer to the input data.
        
        For adversarial perturbation, fitting primarily validates the input data
        and initializes internal state. No actual model training occurs.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to fit the transformer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        AdversarialPerturbationTransformer
            Self instance for method chaining.
        """
        self.feature_set_ = data
        self.epsilon_ = self.epsilon
        self.perturbation_method_ = self.perturbation_method
        self.fitted_ = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply adversarial perturbations to the input data.
        
        Generates perturbed versions of the input samples according to the configured
        perturbation method and magnitude.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed features with adversarial perturbations applied.
        """
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' before 'transform'.")
        if self.random_state is not None:
            np.random.seed(self.random_state)
        original_features = data.features
        if self.perturbation_method_ == 'rand':
            perturbations = self._generate_random_perturbations(original_features.shape)
        elif self.perturbation_method_ == 'fgsm':
            perturbations = self._generate_fgsm_perturbations(original_features)
        else:
            raise ValueError(f'Unsupported perturbation method: {self.perturbation_method_}')
        perturbed_features = original_features + perturbations * self.epsilon_
        if self.preserve_original:
            combined_features = np.vstack([original_features, perturbed_features])
        else:
            combined_features = perturbed_features
        sample_ids = None
        if data.sample_ids is not None:
            if self.preserve_original:
                sample_ids = data.sample_ids + [f'{sid}_perturbed' for sid in data.sample_ids]
            else:
                sample_ids = [f'{sid}_perturbed' for sid in data.sample_ids]
        result = FeatureSet(features=combined_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=sample_ids, metadata=data.metadata.copy() if data.metadata else {}, quality_scores=data.quality_scores.copy() if data.quality_scores else {})
        return result

    def _generate_random_perturbations(self, shape):
        """Generate random perturbations using uniform distribution."""
        return np.random.uniform(-1, 1, shape)

    def _generate_fgsm_perturbations(self, features):
        """Generate FGSM-style perturbations (sign of random gradients)."""
        gradients = np.random.randn(*features.shape)
        return np.sign(gradients)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the transformation (not supported for adversarial perturbations).
        
        As adversarial perturbations are intentionally destructive transformations,
        exact inversion is not possible. This method raises a NotImplementedError.
        
        Parameters
        ----------
        data : FeatureSet
            Transformed features (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            This method always raises NotImplementedError.
            
        Raises
        ------
        NotImplementedError
            Adversarial perturbations cannot be inverted.
        """
        raise NotImplementedError('Adversarial perturbations cannot be inverted.')