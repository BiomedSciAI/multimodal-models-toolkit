from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv


class GCN(nn.Module):
    """
    Baseline GCN
    """

    def __init__(self, module_input):
        """
        Args:
            module_input: dictionary for module initialization containing the following keys:
                - in_size: input feature dimension
                - gl_hidden_size: size of the hidden layer
        """
        super().__init__()

        in_size = module_input["node_emb_dim"]
        gl_hidden_size = module_input["gl_hidden_size"]

        # create layers
        self.layer1 = GraphConv(in_size, gl_hidden_size[0], allow_zero_in_degree=True)
        self.layer2 = GraphConv(
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
        g = derived_graph["graph"][0]  # only one graph is present
        node_features = derived_graph["node_features"]

        h = self.layer1(g, node_features)
        h = F.leaky_relu(h)
        h = self.layer2(g, h)

        return h
