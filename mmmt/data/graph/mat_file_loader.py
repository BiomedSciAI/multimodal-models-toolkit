import os
import dgl
import urllib

import numpy as np
import torch
import scipy

from .general_file_loader import GeneralFileLoader
from .data_to_graph import create_edge_list

import logging

logging.basicConfig(level=logging.INFO)


class MatFileLoader(GeneralFileLoader):
    """
    MAT file loader
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

        if self.dataset_name == "ACM":

            # load data
            data_url = "https://data.dgl.ai/dataset/ACM.mat"
            tmp_path = "./tmp"
            os.makedirs(tmp_path, exist_ok=True)

            data_file_path = os.path.join(tmp_path, "ACM.mat")

            urllib.request.urlretrieve(data_url, data_file_path)
            self.data = scipy.io.loadmat(data_file_path)

            # prepare split ratios
            # for ACM dataset the class 6 has no samples, therefore we are not including it
            self.selected_classes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]

            self.set_random_seed(seed)
            self.num_label = self.data["PvsC"].shape[0]

            self.n_train_samples = int(np.ceil(self.num_label * split_ratios[0]))  # 800
            self.n_val_samples = int(np.ceil(self.num_label * split_ratios[1]))  # 200
            self.n_test_samples = (
                self.num_label - self.n_train_samples - self.n_val_samples
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

        # processing of MAT file for ACM is very specific, therefore currently isolated in the relative if branch
        if self.dataset_name == "ACM":
            # split set
            pvc = self.data["PvsC"].tocsr()
            p_selected = pvc[:, self.selected_classes].tocoo()
            pid = p_selected.row

            shuffle = np.random.permutation(pid)
            train_idx = torch.tensor(shuffle[0 : self.n_train_samples]).long()
            val_idx = torch.tensor(
                shuffle[
                    self.n_train_samples : self.n_train_samples + self.n_val_samples
                ]
            ).long()
            test_idx = torch.tensor(
                shuffle[self.n_train_samples + self.n_val_samples :]
            ).long()
            data_splits = [train_idx, val_idx, test_idx]

            # create graph
            ppA = self.data["PvsA"].dot(self.data["PvsA"].transpose()) > 1
            ppL = self.data["PvsL"].dot(self.data["PvsL"].transpose()) >= 1
            ppP = self.data["PvsP"]

            g = dgl.heterograph(
                {
                    ("feat", "0", "feat"): create_edge_list(ppA),
                    ("feat", "1", "feat"): create_edge_list(ppP),
                    ("feat", "2", "feat"): create_edge_list(ppL),
                }
            )

            # prepare labels
            labels = pvc.indices
            for ind, lbl in enumerate(self.selected_classes):
                labels[labels == lbl] = ind

            labels = torch.tensor(labels).long()

            # extract graph properties
            num_rels = len(g.etypes)
            n_classes = len(self.selected_classes)

            logging.info("Graph and metadata is ready")

            return g, labels, data_splits, n_classes, num_rels


if __name__ == "__main__":
    M_FL = MatFileLoader("ACM", [0.064, 0.016, 0.92], 0)
    g, labels, data_splits, n_classes, num_rels = M_FL.build_graph()
