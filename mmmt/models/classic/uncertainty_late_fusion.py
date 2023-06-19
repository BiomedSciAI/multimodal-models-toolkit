import numpy as np
from mmmt.models.classic.late_fusion import LateFusion


class UncertaintyLateFusion(LateFusion):
    """
    Implemention of the late fusion method described in the following paper.
    Wang, Hongzhi, Vaishnavi Subramanian, and Tanveer Syeda-Mahmood.
    Modeling uncertainty in multi-modal fusion for lung cancer survival analysis.
    In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), pp. 1169-1172. IEEE, 2021.
    """

    def __init__(self, n_mods, n_classes):
        """
        Args:
            n_mods: number of modalities
            n_classes: number of classes/tasks
        """
        super().__init__(n_mods, n_classes)
        self.k = 0
        self.alpha = 0.01

    def solve_weight(self, m):
        """
        Compute the fusion weights of a class_id from the covariance matrix.

        Args:
            m: covariance matrix of a class_id

        Returns:
            Fusion weights for the different models/modalities and one class_id
        """
        invM = np.linalg.inv(m)
        w = np.matmul(invM, np.ones((m.shape[0], 1)))
        w /= np.sum(w)
        return w

    def compute_fusion_weights(self, predictions, ground_truth):
        """
        Calculate fusion weights with model selection

        Args:
            preditions: predictions made by every model with size [n_mod, n_sample, n_class]
                        n_mod is the number of models/modalities
                        n_sample is the number of data samples
                        n_class is the number of classes
            ground_truth: one-hot representation of the ground truth with size  [n_sample, n_class]
            K: number of models to be selected for fusion. Only the top K models will be used for fusion, default K=n_mod.
            alpha: weight for adding identity matrix to make the covariance matrix well conditioned. Typical value can be 0.1 or 0.01

        Returns:
            fusion_weights of size [n_mod, n_class].
               fusion_weights[:, L] is the fusion weights for class L. Note that there are only K non zero values.
        """
        self.n_mods = predictions.shape[0]
        n_samples = predictions.shape[1]
        self.n_classes = predictions.shape[2]
        if self.k == -1 or self.k > self.n_mods:
            self.k = self.n_mods

        selected = np.zeros((self.n_classes, self.n_mods), int)
        errors = np.zeros((self.n_mods, n_samples, self.n_classes), float)
        sumerrors = np.zeros((self.n_classes, self.n_mods), float)
        weights = np.zeros((self.n_mods, self.n_classes))

        for modality in range(self.n_mods):
            errors[modality, :, :] = np.abs(ground_truth - predictions[modality, :, :])
            for class_id in range(self.n_classes):
                sumerrors[class_id, modality] += np.sum(errors[modality, :, class_id])

        for class_id in range(self.n_classes):
            inds = np.argsort(sumerrors[class_id, :])
            selected[class_id, :] = inds

        convariance_matrix = np.zeros((self.n_classes, self.k, self.k))

        for class_id in range(self.n_classes):
            for mod1_index in range(self.k):
                for mod2_index in range(mod1_index, self.k):
                    jerror = (
                        errors[selected[class_id, mod1_index], :, class_id]
                        * errors[selected[class_id, mod2_index], :, class_id]
                    )
                    common_error = np.sum(jerror)
                    convariance_matrix[class_id, mod1_index, mod2_index] = common_error
                    convariance_matrix[
                        class_id, mod2_index, mod1_index
                    ] = convariance_matrix[class_id, mod1_index, mod2_index]

        for class_id in range(self.n_classes):
            convariance_matrix[class_id, :, :] = convariance_matrix[class_id, :, :] / (
                np.max(convariance_matrix[class_id, :, :]) + 1e-10
            )
            m = convariance_matrix[class_id, :, :]
            for mod_index in range(self.k):
                m[mod_index, mod_index] += self.alpha
            w = self.solve_weight(m)
            weights[selected[class_id, 0 : self.k], class_id] = np.matrix.flatten(w)
        return weights
