import numpy as np


import logging
import math

logging.basicConfig(level=logging.INFO)


class GeneralFileLoader:
    """
    General file loader
    Load graph and labels, separate training, validation and testing sets.
    """

    def __init__(self, dataset_name, split_ratios, seed=0):
        """
        Args:
            dataset_name: name of the dataset to retrieve
            split_ratios: list of 3 floating number for training, validation, testing split ratios,
            the sum has to give 1
            seed: pseudo-random number generator seed
        """

        self.dataset_name = dataset_name
        self.data = None
        self.selected_classes = None
        self.num_label = None
        self.n_train_samples = None
        self.n_val_samples = None
        self.n_test_samples = None

        # check split ratio
        assert math.isclose(sum(split_ratios), 1), "split ratios do not sum 1"

    @staticmethod
    def set_random_seed(seed=0):
        """
        Set the pseudo-random number generator seed, this function is only containing the set seed for numpy,
        which is used for data splits. For network initialization additional set seed functions should be used.

        Args:
            seed: pseudo-random number generator seed
        """
        np.random.seed(seed)

    def build_graph(self):
        """
        Build and return graph, labels, data split, the number of classes of the dataset,
        and the number of relation types in the graph.

        Returns:
            graph, labels, data split, the number of classes of the dataset,
            and the number of relation types in the graph
        """
        g, labels, data_splits, n_classes, num_rels = None, None, None, None, None

        return g, labels, data_splits, n_classes, num_rels
