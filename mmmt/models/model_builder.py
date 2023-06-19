from torch import nn
import logging


class ModelBuilder(nn.Module):
    """
    Builder to unite graph model and head model.
    """

    def __init__(self, graph_model_object, head_object, batch_size):
        """
        Args:
            graph_model_object: instance of class of the graph model
            head_object: instance of class of the head model
        """
        super().__init__()

        self.graph_model_object = graph_model_object
        self.head_object = head_object
        self.batch_size = batch_size

        self.check_compatibility()

    def check_compatibility(self):
        """
        Compatibility checks between the defined graph neural network and the configured head module.
        If incompatibility is detected and assertion error is raised, if graph model metadata is missing, an attribute error is raised.
        """
        if hasattr(self.head_object, "in_size") and hasattr(
            self.graph_model_object, "out_size"
        ):
            assert self.head_object.in_size == self.graph_model_object.out_size
        else:
            if hasattr(self.head_object, "in_size"):
                raise AttributeError(
                    "out_size attribute in graph_object is not provided"
                )
            else:
                raise AttributeError("in_size attribute in head_object is not provided")

    def forward(self, derived_graph):
        """
        Forward pass of the model.

        Args:
            derived_graph: dictionary of graph topology (key 'graph') and node features (key 'node_features')

        Returns:
            Graph label logits, with defined dimension out_size
        """

        h = self.graph_model_object(derived_graph)

        n_samples = h.shape[0] / self.head_object.num_nodes
        if n_samples != int(n_samples):
            logging.error(
                "irregular number of nodes in the batch: "
                + str(h.shape[0])
                + " vs. "
                + str(self.head_object.num_nodes)
            )
        if n_samples != self.batch_size:
            logging.debug(
                "number of samples per batch is not as defined, this should happen maximum once per epoch."
            )

        h = (
            h.reshape(-1, self.head_object.num_nodes, h.shape[1])
            .transpose(1, 0)
            .reshape(-1, h.shape[1])
            .reshape((self.head_object.num_nodes, -1, h.shape[1]))
        )

        # Note:
        #   from dgl documentation: in a batched graph is obtained by concatenating the corresponding features
        #   from all graphs in order.
        #   https://docs.dgl.ai/en/0.8.x/guide/training-graph.html?highlight=batch

        h = self.head_object(h)

        return h
