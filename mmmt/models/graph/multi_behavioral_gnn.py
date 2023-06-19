import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv


class MultiBehavioralGNN(nn.Module):
    """
    Multibehavioral GNN framework for message passing from https://dl.acm.org/doi/pdf/10.1145/3340531.3412119
    """

    def __init__(self, module_input):
        """
        Args:
            module_input: dictionary for module initialization containing the following keys:
                - in_size: input feature dimension
                - gl_hidden_size: size of the hidden layer
                - n_layers: number of concepts
        """
        super().__init__()

        in_size = module_input["node_emb_dim"]
        gl_hidden_size = module_input["gl_hidden_size"]
        n_layers = module_input["n_layers"]

        self.n_layers = n_layers

        self.convQ1 = GraphConv(in_size, gl_hidden_size[0], allow_zero_in_degree=True)
        self.convM1 = GraphConv(
            gl_hidden_size[0], gl_hidden_size[-1], allow_zero_in_degree=True
        )

        self.out_size = gl_hidden_size[-1]

    def forward(self, derived_graph):
        """
        Forward pass of the model.

        Args:
            derived_graph: dictionary of graph topology (key 'graph') and node features (key 'node_features')

        Returns:
            Graph latent representation
        """
        g1, g2 = derived_graph["graph"]
        Q_node_features = derived_graph["node_features"]
        h_Q1 = self.convQ1(g1, Q_node_features)
        h_Q1 = F.leaky_relu(h_Q1)

        # node based feature stack for Multiplex graph
        M_node_features = torch.kron(torch.ones(self.n_layers, 1), h_Q1)
        h_M1 = self.convM1(g2, M_node_features)
        h_M1 = F.leaky_relu(h_M1)

        return h_M1
