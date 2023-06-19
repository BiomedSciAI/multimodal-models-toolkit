import dgl
import scipy.sparse as spp
import numpy as np

import logging


class GraphTransform:
    """
    Class containing methods to transform between graph structures
    """

    def __init__(
        self, graph: dgl.DGLGraph, node_features, graph_module="mplex", alpha=0.5
    ):
        """
        Args:
            graph: a graph in dgl format
            node_features: features of the nodes
            graph_module: the method to use for transformation, available values: ['mplex', 'mplex-prop', 'multibehav', 'mGNN']
            alpha: paramater to control the intra and inter layer message passing, only applicable for mplex, mplex-prop
        """
        self.graph = graph
        self.node_features = node_features
        self.graph_module = graph_module
        self.alpha = alpha

    def transform(self):
        """
        Coordinate the graph transformation.

        Returns:
            transformed graph
            If method is not recognized returns None
        """

        transformations = {
            "mplex": self.create_multiplex_graph_object,
            "mplex-prop": self.create_multiplex_prop_graph_object,
            "multibehav": self.create_multibehav_graph_object,
            "mgnn": self.create_mGNN_graph_object,
            "rgcn": self.pass_multigraph,
            "gcn": self.create_homogeneous_graph,
        }

        if self.graph_module in transformations:
            return transformations[self.graph_module]()
        else:
            logging.error(
                self.graph_module
                + " method not implemented. Choose within "
                + str(transformations.keys())
            )
            raise ValueError(
                "Unknown method! Choose within " + str(transformations.keys())
            )

    def compute_mplex_walk_mat(self):
        """
        Computes the multiplex walk matrices

        Returns:
            AC: first intra-layer, then same layer
            CA: first same layer, then intra-layer
        """
        e_types = self.graph.etypes
        number_e_types = len(e_types)
        number_nodes = self.graph.num_nodes()
        mplex_node_features = None

        adj_dimension = number_nodes * number_e_types
        intra_L = np.zeros((adj_dimension, adj_dimension))

        for ind_et, e_type in enumerate(e_types):
            adj = self.graph[e_type].adjacency_matrix(transpose=True).to_dense().numpy()
            intra_L[
                ind_et * number_nodes : ind_et * number_nodes + number_nodes,
                ind_et * number_nodes : ind_et * number_nodes + number_nodes,
            ] = adj
            if ind_et == 0:
                mplex_node_features = self.node_features
            else:
                mplex_node_features = np.concatenate(
                    (mplex_node_features, self.node_features), axis=0
                )

        mat_A = spp.coo_matrix(
            self.alpha * np.ones((number_e_types, number_e_types))
        ) - spp.coo_matrix(self.alpha * np.eye(number_e_types))
        mat_B = spp.coo_matrix(np.eye(number_nodes))

        mat_C = spp.coo_matrix((1 - self.alpha) * np.eye(number_e_types))

        C = spp.kron(mat_A, mat_B) + spp.kron(mat_C, mat_B)

        intra_L = spp.coo_matrix(intra_L)

        AC = intra_L.dot(C)
        CA = C.dot(intra_L)

        return AC, CA, mplex_node_features

    def create_multiplex_graph_object(
        self,
    ):
        """
        Creates multiplex graph from heterograph objects

        Returns:
            message passing object type
            g1: type I supra-adjacency
            g2: type II supra-adjacency
        """

        AC, CA, mplex_node_features = self.compute_mplex_walk_mat()

        g1 = dgl.from_scipy(spp.coo_matrix(AC))
        g2 = dgl.from_scipy(spp.coo_matrix(CA))

        # # type II graph can be inferred from type I graph TODO: verify
        # g2 = dgl.from_scipy(g1.adj(scipy_fmt="coo").T)

        return [g1, g2], mplex_node_features

    def create_multiplex_prop_graph_object(self):
        """
        Creates multiplex propagation graphs from heterograph and node features

        Returns:
            g1: Type I graph objects
            g2: Type II graph objects
        """

        AC, CA, mplex_node_features = self.compute_mplex_walk_mat()

        g1 = dgl.from_scipy(spp.coo_matrix(AC), eweight_name="w")
        g2 = dgl.from_scipy(spp.coo_matrix(CA), eweight_name="w")

        return [g1, g2], mplex_node_features

    def create_multibehav_graph_object(self):
        """
        Creates quotient graph and multilayered graph according to https://dl.acm.org/doi/pdf/10.1145/3340531.3412119

        Returns:
            quotient_graph: quotient graph
            mplex_graph: multilayered graph
        """
        e_types = self.graph.etypes
        number_e_types = len(e_types)
        number_nodes = self.graph.num_nodes()

        adj_dimension = number_nodes * number_e_types
        quotient_adj = np.zeros((number_nodes, number_nodes))
        intra_L = np.zeros((adj_dimension, adj_dimension))

        for ind_et, e_type in enumerate(e_types):
            adj = self.graph[e_type].adjacency_matrix(transpose=True).to_dense().numpy()
            intra_L[
                ind_et * number_nodes : ind_et * number_nodes + number_nodes,
                ind_et * number_nodes : ind_et * number_nodes + number_nodes,
            ] = adj
            quotient_adj = quotient_adj + adj / number_e_types

        quotient_adj = spp.coo_matrix(quotient_adj)
        C = spp.coo_matrix(
            np.kron(np.ones((number_e_types, number_e_types)), np.eye(number_nodes))
            - np.eye(number_nodes * number_e_types)
        )

        intra_L = spp.coo_matrix(intra_L)
        mplex_adj = intra_L + C

        quotient_graph = dgl.from_scipy(quotient_adj)
        mplex_graph = dgl.from_scipy(mplex_adj)

        return [quotient_graph, mplex_graph], self.node_features

    def create_mGNN_graph_object(self):
        """
        Creates intra and inter multigraph objects according to https://arxiv.org/pdf/2109.10119.pdf

        Returns:
            message passing object types
            g_inter_layer: inter graph
            g_intra_layer: intra graph
        """
        e_types = self.graph.etypes
        number_e_types = len(e_types)
        number_nodes = self.graph.num_nodes()
        mplex_node_features = None

        adj_dimension = number_nodes * number_e_types
        intra_L = np.zeros((adj_dimension, adj_dimension))

        for ind_et, e_type in enumerate(e_types):
            adj = self.graph[e_type].adjacency_matrix(transpose=True).to_dense().numpy()
            intra_L[
                ind_et * number_nodes : ind_et * number_nodes + number_nodes,
                ind_et * number_nodes : ind_et * number_nodes + number_nodes,
            ] = adj
            if ind_et == 0:
                mplex_node_features = self.node_features
            else:
                mplex_node_features = np.concatenate(
                    (mplex_node_features, self.node_features), axis=0
                )

        C = spp.coo_matrix(
            np.kron(np.ones((number_e_types, number_e_types)), np.eye(number_nodes))
            - np.eye(number_nodes * number_e_types)
        )

        intra_L = spp.coo_matrix(intra_L)

        g_inter_layer = dgl.from_scipy(C)
        g_intra_layer = dgl.from_scipy(intra_L)

        return [g_inter_layer, g_intra_layer], mplex_node_features

    def pass_multigraph(self):
        return [self.graph], self.node_features

    def create_homogeneous_graph(self):
        return [dgl.to_homogeneous(self.graph)], self.node_features
