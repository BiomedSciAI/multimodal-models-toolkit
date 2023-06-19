from fuse.data.ops.op_base import OpBase
from typing import List, Union
from fuse.utils.ndict import NDict
import logging

import mmmt


class OpBuildBaseGraph(OpBase):
    """
    Operator for building the base graph using mmmt library
    """

    def __init__(self, model, thresh_q):
        """Constructor method

        :param model: concept encoding model
        :type model: torch.nn.Module
        :param thresh_q: saliency threshold
        :type thresh_q: float
        """
        super().__init__()
        self.model = model
        self.thresh_q = thresh_q
        self.model.eval()
        logging.debug(self.model)

    def __call__(
        self,
        sample_dict: NDict,
        key_in_concat="data.input.multimodal",
        key_in_concept="data.forward_pass.multimodal",
        key_out="data.graph",
        **kwargs,
    ) -> Union[None, dict, List[dict]]:
        """produces a graph for a given sample

        :param sample_dict: sample dictionary
        :type sample_dict: NDict
        :param key_in_concat: input key (node features) in sample dict, defaults to "data.input.multimodal"
        :type key_in_concat: str, optional
        :param key_in_concept: concept key in sample dict, defaults to "data.forward_pass.multimodal"
        :type key_in_concept: str, optional
        :param key_out: graph key to generate, defaults to "data.graph"
        :type key_out: str, optional
        :return: updated sample dict
        :rtype: Union[None, dict, List[dict]]
        """

        (
            multi_graph,
            node_features,
        ) = mmmt.data.graph.data_to_graph.multigraph_object_creator(
            self.model,
            sample_dict[key_in_concept],
            sample_dict[key_in_concat],
            self.thresh_q,
        )

        sample_dict[key_out] = {"graph": multi_graph, "node_features": node_features}

        return sample_dict


class OpBuildDerivedGraph(OpBase):
    """
    Operator for building the derived graph (e.g. mplex) using mmmt library
    """

    def __init__(self, graph_module):
        """Constructor method

        :param graph_module: type of graph module to be used downstream, available values: ['mplex', 'mplex-prop', 'multibehav', 'mGNN']
        :type graph_module: str
        """
        super().__init__()
        self.graph_module = graph_module

    def __call__(
        self,
        sample_dict: NDict,
        key_in="data.base_graph",
        key_out="data.derived_graph",
        **kwargs,
    ) -> Union[None, dict, List[dict]]:
        """transforms a graph for a sample, to adapt it to the donwstream graph module

        :param sample_dict: sample dictionary
        :type sample_dict: NDict
        :param key_in: dict key with base graph, defaults to "data.base_graph"
        :type key_in: str, optional
        :param key_out: dict key where the transformed graph will be stored, defaults to "data.derived_graph"
        :type key_out: str, optional
        :return: update sample dict
        :rtype: Union[None, dict, List[dict]]
        """

        graph = sample_dict[key_in]["graph"]
        node_features = sample_dict[key_in]["node_features"]

        GT = mmmt.data.graph.graph_to_graph.GraphTransform(
            graph, node_features, self.graph_module
        )
        derived_graph, derived_node_features = GT.transform()

        sample_dict[key_out] = {
            "graph": derived_graph,
            "node_features": derived_node_features,
        }

        return sample_dict
