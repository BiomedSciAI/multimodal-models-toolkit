import dgl
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv


class Relational_GCN(nn.Module):
    """
    Relational GCN in https://arxiv.org/pdf/1703.06103.pdf
    (Uses in-built Relational Graph Conv)
    """

    def __init__(self, module_input):
        """
        Args:
            module_input: dictionary for module initialization containing the following keys:
                - in_size: input feature dimension
                - n_layers: number of concepts
                - num_bases: number of bases of the RelGraphConv
                - gl_hidden_size: size of the hidden layer
        """
        super().__init__()

        in_size = module_input["node_emb_dim"]
        n_layers = module_input["n_layers"]
        num_bases = module_input["num_bases"]
        gl_hidden_size = module_input["gl_hidden_size"]

        # create layers
        self.layer1 = RelGraphConv(
            in_size,
            gl_hidden_size[0],
            num_rels=n_layers,
            regularizer="basis",
            num_bases=num_bases,
        )
        self.layer2 = RelGraphConv(
            gl_hidden_size[0],
            gl_hidden_size[-1],
            num_rels=n_layers,
            regularizer="basis",
            num_bases=num_bases,
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

        etype = dgl.to_homogeneous(g).edata[dgl.ETYPE]

        h = self.layer1(dgl.to_homogeneous(g), node_features, etype)
        h = F.leaky_relu(h)

        h = self.layer2(dgl.to_homogeneous(g), h, etype)

        return h
