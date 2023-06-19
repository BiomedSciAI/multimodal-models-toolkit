import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MLP model for combination with the graph model.
    """

    def __init__(
        self,
        num_nodes,
        in_size,
        hidden_size: list,
        out_size,
        dropout,
        add_softmax=False,
    ):
        """
        Args:
            num_nodes: number of nodes in the graph
            in_size: input feature dimension
            hidden_size: list of hidden layer dimension of the classification head,
            for one hidden layer len(hidden_layer) == 1
            out_size: output dimension, which normally corresponds to the number of classes
            add_softmax: whether to add a softmax layer at the end
        """
        super().__init__()

        self.in_size = in_size
        self.num_nodes = num_nodes
        self.out_size = out_size

        if in_size > 1:
            self.agg = torch.nn.Linear(in_size, 1)
        else:
            self.agg = None

        self.dp_out = torch.nn.Dropout(dropout)

        self.dense_layers = []

        in_features = self.num_nodes
        for out_features in hidden_size:
            self.dense_layers.append(torch.nn.Linear(in_features, out_features))
            in_features = out_features

        self.dense_layers = nn.ModuleList(self.dense_layers)

        if add_softmax:
            self.final_layer = nn.Sequential(
                torch.nn.Linear(hidden_size[-1], out_size), torch.nn.Softmax(2)
            )
        else:
            self.final_layer = torch.nn.Linear(hidden_size[-1], out_size)

    def forward(self, h):
        """
        Forward pass of the classifier head model.

        Args:
            h: graph latent representation

        Returns:
            Logits, with defined dimension out_size
        """

        if self.agg:
            h = self.agg(h.float())

            h = self.dp_out(h)

            h = h.transpose(2, 0)

        else:
            h = h[None, :, :]

            h = self.dp_out(h)

        for layer in self.dense_layers:
            h = F.leaky_relu(layer(h))

        h = self.final_layer(h)

        return h.squeeze(dim=0)
