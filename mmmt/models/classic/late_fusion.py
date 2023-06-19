import numpy as np
import logging


class LateFusion:
    """
    Basic late fusion
    """

    def __init__(self, n_mods, n_classes, weights=None):
        """
        Args:
            n_mods: number of modalities
            n_classes: number of classes/tasks
        """
        self.n_mods = n_mods
        self.n_classes = n_classes

        """weights: weights used for weighted average with size [n_mods, n_classes]. Initialized with equal weights."""
        if weights is None:
            self.weights = np.ones((n_mods, n_classes), float) / n_mods
        else:
            self.weights = weights

    def apply_fusion(self, predictions):
        """
        Calculate fused predictions using weighted average

        preditions: predictions made by all every model with size [n_mod, n_sample, n_class]
                    n_mod is the number of models/modalities
                    n_sample is the numebr of data samples
                    n_class is the number of classes

        return fused of size [n_sample, n_class].
        """
        fused = np.zeros((predictions.shape[1], self.n_classes))
        logging.info(fused.shape, self.weights.shape, predictions.shape)
        for modality in range(self.n_mods):
            for class_id in range(self.n_classes):
                fused[:, class_id] = (
                    fused[:, class_id]
                    + self.weights[modality, class_id]
                    * predictions[modality, :, class_id]
                )
        return fused
