import os

import dgl
import graphviz
import matplotlib.pyplot as plt
import mlflow
import networkx as nx
import numpy as np
import nxviz as nv
from fuse.utils.ndict import NDict
from nxviz import annotate
from itertools import chain


class GraphVisualization:
    """Visualization of the graph(s) induced by a dataset"""

    def __init__(
        self,
        args_dict,
    ):
        """
        Args:
            args_dict (dict): arguments of DatasetGraphVisualization as defined in configuration yaml
        """

        self.selected_samples = args_dict["step_args"]["selected_samples"]
        self.fused_dataset_key = args_dict["step_args"]["io"]["fused_dataset_key"]
        self.file_prefix = args_dict["step_args"]["io"]["file_prefix"]
        self.feature_group_sizes = args_dict["step_args"]["feature_group_sizes"]

        self.visualization_config = args_dict["step_args"]
        self.mmmt_pipeline = args_dict["pipeline"]
        self.pipeline = args_dict["pipeline"]["fuse_pipeline"]
        self.root_dir = args_dict["root_dir"]
        self.mmmt_pipeline = args_dict["pipeline"]

    def __call__(
        self,
    ):
        """
        Build graphs from samples by extending the FuseMedML pipeline with specific operators

        Returns:
            train, validation and test sets with samples structured as graphs
        """
        with mlflow.start_run(run_name=f"{self.__class__.__qualname__}", nested=True):
            mlflow.log_params(NDict(self.visualization_config).flatten())

            node_categories = list(
                chain.from_iterable(
                    [
                        [category] * size
                        for category, size in self.feature_group_sizes.items()
                    ]
                )
            )

            for split_name, split_samples in self.selected_samples.items():
                selected_dataset = self.mmmt_pipeline[self.fused_dataset_key][
                    split_name
                ]
                if split_samples == "all":
                    filename = os.path.join(
                        self.root_dir, f"{self.file_prefix}_{split_name}_all.png"
                    )
                    self.visualize_dataset(
                        selected_dataset,
                        samples=None,
                        save_file=filename,
                        node_categories=node_categories,
                    )
                    mlflow.log_artifact(filename)
                else:
                    samples_name = "_".join(map(str, split_samples))
                    filename = os.path.join(
                        self.root_dir,
                        f"{self.file_prefix}_{split_name}_{samples_name}.png",
                    )
                    self.visualize_dataset(
                        selected_dataset,
                        samples=split_samples,
                        save_file=filename,
                        node_categories=node_categories,
                    )
                    mlflow.log_artifact(filename)

    @staticmethod
    def visualize_dataset(
        dataset,
        graph_key="data.base_graph.graph",
        node_names_key="names.data.input.multimodal",
        node_categories=None,
        samples=None,
        max_edge_thickness=20,
        save_file=None,
    ):
        """Visualization of a Graph Based Multimodal Representation

        :param dataset: FuseMedML dataset to be visualized
        :type dataset: FuseMedML dataset
        :param graph_key: dictionary key that points to the graph, defaults to "data.base_graph.graph"
        :type graph_key: str, optional
        :param node_names_key: key that contains names of the nodes, defaults to "names.data.input.multimodal"
        :type node_names_key: str, optional
        :param node_categories: list of categories for the nodes, defaults to None
        :type node_categories: [str], optional
        :param samples: visulize only some samples, defaults to None
        :type samples: [int], optional
        :param max_edge_thickness: maximum edge thickness, defaults to 20
        :type max_edge_thickness: int, optional
        :return: NetworkX graph used for visualization
        :rtype: nx.Graph
        """

        if node_categories is None:
            node_categories = dataset[0][node_names_key]
        node_names = dataset[0].get(node_names_key, node_categories)

        adj = np.zeros_like(
            dgl.to_homogeneous(dataset[0][graph_key])
            .adjacency_matrix()
            .to_dense()
            .numpy()
        )
        for sample in dataset:
            if not samples or sample["data"]["sample_id"] in samples:
                adj = (
                    adj
                    + dgl.to_homogeneous(sample[graph_key])
                    .adjacency_matrix()
                    .to_dense()
                    .numpy()
                )

        adj = max_edge_thickness * adj / adj.max()

        attrs = {
            num: {
                "value": num,
                "name": node_name,
                "group": node_categories[num].split(".")[-1],
            }
            for (num, node_name) in enumerate(node_names)
        }

        G = nx.from_numpy_matrix(adj)
        nx.set_node_attributes(G, attrs)

        plt.figure(figsize=(12, 12))
        pos = nv.nodes.circos(G, group_by="group", color_by="group")
        nv.edges.circos(G, pos, lw_by="weight")
        annotate.circos_group(G, group_by="group")
        nv.plots.despine()
        nv.plots.aspect_equal()

        if save_file is not None:
            plt.savefig(save_file, format="PNG")

        return G

    def visualize_encoding_strategy(encoding_strategy):

        concept_encoder_name = "Concept Encoder"

        encoding_graph = nx.DiGraph()
        encoding_graph.add_node(concept_encoder_name)
        for modality_key, modality_encoder in encoding_strategy.get(
            "modality_encoders"
        ).items():
            modality_name = modality_key.split()[-1]
            encoding_graph.add_node(modality_name)
            encoding_graph.add_edge(modality_name, concept_encoder_name)

        encoding_graph.add_edge(concept_encoder_name, "Graph")
        plt.figure(figsize=(12, 12))
        nx.draw_networkx(encoding_graph)

        return encoding_graph

    def visualize_fuse_pipeline(pipeline, name=None, source_keys=None):
        """Visualize FuseMedML pipeline

        :param pipeline: FuseMedML pipeline
        :type pipeline: FuseMedML pipeline
        :param name: name of the pipeline, defaults to None
        :type name: str, optional
        :param source_keys: names of the keys that are used as source, defaults to None
        :type source_keys: [str], optional
        :return: GraphViz version of the pipeline
        :rtype: graphviz graph
        """

        if name is None:
            name = pipeline.get_name()
        graph = graphviz.Digraph(name=name, strict=True, format="png")

        for step_id, pipeline_step in enumerate(pipeline._ops_and_kwargs):
            op_name = pipeline_step[0]
            op_params = pipeline_step[1]
            op_short_name = op_name.__str__().split()[0].split(".")[-1]

            if step_id == 0 and source_keys is not None:
                for skey in source_keys:
                    graph.edge(op_short_name, skey)

            op_inputs = []
            op_outputs = []

            for op_key, data_key in op_params.items():
                if op_key.startswith("key"):

                    if op_key.startswith("key_in") or op_key.startswith("keys_in"):
                        if isinstance(data_key, list):
                            op_inputs.extend(data_key)
                        else:
                            op_inputs.append(data_key)
                    if op_key.startswith("key_out") or op_key.startswith("keys_out"):
                        if isinstance(data_key, list):
                            op_outputs.extend(data_key)
                        else:
                            op_outputs.append(data_key)

                    if isinstance(data_key, list):  # multiple keys
                        for item in data_key:
                            graph.node(item, shape="plaintext")

                    else:  # single key
                        graph.node(data_key, shape="plaintext")

                if op_inputs and op_outputs:
                    for input in op_inputs:
                        for output in op_outputs:

                            op_node_id = f"{op_short_name}_{step_id}"

                            graph.node(op_node_id, label=op_short_name, shape="box")
                            graph.edge(input, op_node_id)
                            graph.edge(op_node_id, output)

        return graph
