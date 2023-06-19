import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv


class mGNN(nn.Module):
    """
    mGNN framework for message passing from https://arxiv.org/abs/2109.10119
    """

    def __init__(self, module_input):
        """
        Args:
            module_input: dictionary for module initialization containing the following keys:
                - in_size: input feature dimension
                - gl_hidden_size: size of the hidden layer
                - num_att_heads: number of attention heads
                - n_layers: number of layers of the multiplex graphs, input of the graph model
        """
        super().__init__()

        in_size = module_input["node_emb_dim"]
        gl_hidden_size = module_input["gl_hidden_size"]
        num_att_heads = module_input["num_att_heads"]
        n_layers = module_input["n_layers"]

        self.n_layers = n_layers

        self.convC1 = GATConv(
            in_size, in_size, num_heads=num_att_heads, allow_zero_in_degree=True
        )
        self.convA1 = GATConv(
            in_size, in_size, num_heads=num_att_heads, allow_zero_in_degree=True
        )

        self.convC2 = GATConv(
            2 * in_size * num_att_heads,
            gl_hidden_size[0],
            num_heads=num_att_heads,
            allow_zero_in_degree=True,
        )
        self.convA2 = GATConv(
            2 * in_size * num_att_heads,
            gl_hidden_size[0],
            num_heads=num_att_heads,
            allow_zero_in_degree=True,
        )

        self.out_size = gl_hidden_size[-1] * 2 * num_att_heads

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

        h_C1 = self.convC1(g1, node_features).reshape(node_features.size()[0], -1)
        h_A1 = self.convA1(g2, node_features).reshape(node_features.size()[0], -1)

        h = F.leaky_relu(torch.cat((h_C1, h_A1), dim=1))

        h_C2 = self.convC2(g1, h).reshape(node_features.size()[0], -1)
        h_A2 = self.convA2(g2, h).reshape(node_features.size()[0], -1)

        # graph based readout
        h = torch.cat((h_C2, h_A2), dim=1)

        return h
