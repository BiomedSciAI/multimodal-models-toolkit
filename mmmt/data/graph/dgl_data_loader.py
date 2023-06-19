import numpy as np
import torch

from dgl.contrib.data import load_data

from .general_file_loader import GeneralFileLoader
from .data_to_graph import heterograph_creator

import logging

logging.basicConfig(level=logging.INFO)


class DGLFileLoader(GeneralFileLoader):
    """
    DGL file loader
    Load graph and labels, separate training, validation and testing sets.
    """

    def __init__(self, dataset_name, split_ratios, seed):
        """
        Args:
            dataset_name: name of the dataset to retrieve
            split_ratios: list of 3 floating number for training, validation, testing split ratios,
            the sum has to give 1
            seed: pseudo-random number generator seed
        """

        super().__init__(dataset_name, split_ratios, seed)

        if self.dataset_name == "AIFB":

            # load data
            self.data = load_data(dataset="aifb")

            # prepare split ratios
            self.set_random_seed(seed)
            self.num_label = len(self.data.train_idx) + len(self.data.test_idx)

            self.n_train_samples = int(np.ceil(split_ratios[0] * self.num_label))
            self.n_val_samples = int(np.ceil(split_ratios[1] * self.num_label))
            self.n_test_samples = (
                len(self.data.labels) - self.n_train_samples - self.n_val_samples
            )

        else:
            logging.error(
                "The dataset " + self.dataset_name + " has not yet been implemented"
            )
            raise ValueError("Unknown dataset!")

        logging.info(dataset_name + " has been loaded")

    def build_graph(self):
        """
        Build and return graph, labels, data split, the number of classes of the dataset,
        and the number of relation types in the graph.

        Returns:
            graph, labels, data split, the number of classes of the dataset,
            and the number of relation types in the graph
        """

        # split set
        def_train_idx = self.data.train_idx
        def_test_idx = self.data.test_idx

        labeled_indices = np.concatenate((def_train_idx, def_test_idx), axis=0)

        ptation = np.random.permutation(self.num_label)
        train_idx = labeled_indices[ptation[: self.n_train_samples]]
        val_idx = labeled_indices[
            ptation[self.n_train_samples : self.n_train_samples + self.n_val_samples]
        ]
        test_idx = labeled_indices[ptation[self.n_train_samples + self.n_val_samples :]]
        data_splits = [train_idx, val_idx, test_idx]

        # create graph
        g = heterograph_creator(self.data)

        # prepare label
        labels = self.data.labels
        labels = torch.from_numpy(labels).view(-1)

        # extract graph properties
        num_rels = len(g.etypes)
        n_classes = self.data.num_classes

        logging.info("Graph and metadata is ready")

        return g, labels, data_splits, n_classes, num_rels


if __name__ == "__main__":

    DGL_FL = DGLFileLoader("AIFB", [0.7, 0.2, 0.1], 0)
    g, labels, data_splits, n_classes, num_rels = DGL_FL.build_graph()
