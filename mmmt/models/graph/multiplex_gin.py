import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv


class MultiplexGIN(nn.Module):
    """
    Multiplex GIN framework for message passing via multiplex walks.
    """

    def __init__(self, module_input):
        """
        Args:
            module_input: dictionary for module initialization

        """
        super().__init__()

        self.convAC1 = GINConv(aggregator_type="mean", activation=F.leaky_relu)
        self.convCA1 = GINConv(aggregator_type="mean", activation=F.leaky_relu)

        self.convAC2 = GINConv(aggregator_type="mean", activation=F.leaky_relu)
        self.convCA2 = GINConv(aggregator_type="mean", activation=F.leaky_relu)

        self.out_size = 4 * module_input["node_emb_dim"]

    def forward(self, derived_graph):
        """
        Forward pass of the model.

        Args:
            derived_graph: dictionary of graph topology (key 'graph') and node features (key 'node_features')

        Returns:
            Graph latent representation
        """
        g1, g2 = derived_graph["graph"]
        node_features = derived_graph["node_features"]

        h_AC1 = self.convAC1(g1, node_features)
        h_CA1 = self.convCA1(g2, node_features)

        h = torch.cat((h_AC1, h_CA1), dim=1)

        h_AC2 = self.convAC2(g1, h)
        h_CA2 = self.convCA2(g2, h)

        # #aggregate across features
        h = torch.cat((h_AC2, h_CA2), dim=1)
        return h
