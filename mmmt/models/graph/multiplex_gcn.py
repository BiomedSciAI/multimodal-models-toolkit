import torch
from torch import nn
from dgl.nn.pytorch.conv import SGConv


class MultiplexGCN(nn.Module):
    """
    Multiplex GCN for message passing according to sGCN Conv for sparse graphs.
    """

    def __init__(self, module_input):
        """
        Args:
            module_input: dictionary for module initialization containing the following keys:
                - in_size: input feature dimension
                - gl_hidden_size: list of sizes of the hidden layers
        """
        super().__init__()

        in_size = module_input["node_emb_dim"]
        gl_hidden_size = module_input["gl_hidden_size"]

        self.convAC1 = SGConv(in_size, gl_hidden_size[0], allow_zero_in_degree=True)
        self.convCA1 = SGConv(in_size, gl_hidden_size[0], allow_zero_in_degree=True)

        self.convAC2 = SGConv(
            2 * gl_hidden_size[0], gl_hidden_size[1], allow_zero_in_degree=True
        )
        self.convCA2 = SGConv(
            2 * gl_hidden_size[0], gl_hidden_size[1], allow_zero_in_degree=True
        )

        self.out_size = gl_hidden_size[-1] * 2

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

        ef_1 = g1.edata["w"].float()
        ef_2 = g2.edata["w"].float()

        h_AC1 = self.convAC1(g1, node_features, edge_weight=ef_1)
        h_CA1 = self.convCA1(g2, node_features, edge_weight=ef_2)

        h = torch.cat((h_AC1, h_CA1), dim=1)

        h_AC2 = self.convAC2(g1, h, edge_weight=ef_1)
        h_CA2 = self.convCA2(g2, h, edge_weight=ef_2)

        # #aggregate across features
        h = torch.cat((h_AC2, h_CA2), dim=1)

        return h
