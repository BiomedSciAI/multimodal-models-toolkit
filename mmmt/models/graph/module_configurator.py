import mmmt

import logging


class ModuleConfigurator:
    """
    Configures the graph neural network taking the inputs from the user configuration stored in graph_module
    """

    def __init__(self, graph_module):
        """
        Args:
            graph_module: user input that defines which flavour to use for the graph neural network
        """
        self.graph_module = graph_module
        self.graph_module_identifier = graph_module["module_identifier"]

    def get_module(self, graph_sample):
        """
        Instantiate the chosen graph neural network and provides metadata for the construction of the head module downstream

        Args:
            graph_sample: a graph sample resulting from the combination of the available modalities

        Returns:
            A tuple containing the instantiated graph neural network, the input size needed for the head module and the number of nodes arriving to the head module
        """
        modules = {
            "mplex": mmmt.models.graph.MultiplexGIN,
            "mplex-prop": mmmt.models.graph.MultiplexGCN,
            "mgnn": mmmt.models.graph.mGNN,
            "multibehav": mmmt.models.graph.MultiBehavioralGNN,
            "gcn": mmmt.models.graph.GCN,
            "rgcn": mmmt.models.graph.Relational_GCN,
        }

        if self.graph_module_identifier in modules:
            return (
                modules[self.graph_module_identifier](self.graph_module),
                self.get_head_in_size(),
                self.get_head_num_nodes(graph_sample),
            )
        else:
            logging.error(
                self.graph_module_identifier
                + " method not implemented. Choose within "
                + str(modules.keys())
            )
            raise ValueError("Unknown method! Choose within " + str(modules.keys()))

    def get_head_in_size(self):
        """
        Provides the input size needed for the head module

        Returns:
            input size needed for the head module
        """
        gl_hidden_size = self.graph_module.get("gl_hidden_size", [0])
        node_emb_dim = self.graph_module.get("node_emb_dim", 0)
        num_att_heads = self.graph_module.get("num_att_heads", 1)
        sizes = {
            "mplex": 4 * node_emb_dim,
            "mplex-prop": gl_hidden_size[-1] * 2,
            "mgnn": gl_hidden_size[-1] * 2 * num_att_heads,
        }
        return sizes.get(self.graph_module_identifier, gl_hidden_size[-1])
        # no need to capture exceptions as already done in self.get_module

    def get_head_num_nodes(self, graph_sample):
        """
        Provides the number of nodes arriving to the head module

        Args:
            graph_sample: a graph sample resulting from the combination of the available modalities

        Returns:
            number of nodes arriving to the head module
        """
        if self.graph_module_identifier == "multibehav":
            return graph_sample[1].num_nodes()  # * self.graph_batch_size
        else:
            return graph_sample[0].num_nodes()  # * self.graph_batch_size
