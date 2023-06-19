import copy

import torch
import numpy as np
import scipy.sparse as spp
import dgl


def create_edge_list(adj):
    """
    Create the list of edges from an adjacency matrix.

    Args:
        adj: adjacency matrix in CSC format

    Returns:
        list_edge_tuples: list of edges, each edge is a tuple
    """
    rows, cols = adj.nonzero()
    list_edge_tuples = []

    for ind, r in enumerate(rows):
        list_edge_tuples.append((r, cols[ind]))

    return list_edge_tuples


def heterograph_creator(dataset):
    """
    Create a heterograph according to a dataset coming from DGL data loader.

    Args:
        dataset: a dataset from DGL loader utility

    Returns:
        hg: heterograph object with node ("feat")
    """

    g = {}
    edge_types = np.unique(dataset.edge_type)  # find unique edge types

    for j in range(len(edge_types)):
        # extract edge relations of type i
        i = edge_types[j]
        edge_list = (
            dataset.edge_src[dataset.edge_type == i],
            dataset.edge_dst[dataset.edge_type == i],
        )

        g[("feat", str(j), "feat")] = edge_list

    hg = dgl.heterograph(g)

    return hg


def multigraph_object_creator(
    common_encoder, common_embedding, concatenated_modality_inputs, thresh_q
):
    """
    Creates a multigraph from common encoder and embeddings using a quantile-based threshold

    Args:
        common_encoder : common encoder
        common_embedding: latent representation of all samples from common encoder
        concatenated_modality_inputs: concatenated embeddings of all samples
        thresh_q: quantile for thresholding

    Returns:
        multi_graphs: multigraph dgl object
    """

    num_rel_type = common_embedding.shape[
        0
    ]  # size of encoding - number of relation types
    num_nodes = concatenated_modality_inputs.shape[0]  # size of graph - number of nodes

    common_embedding_diff = torch.zeros(
        [num_nodes, num_rel_type]
    )  # initialise impact matrix

    for n in range(num_nodes):
        # compute feature impact
        concatenated_modality_inputs_copied = copy.deepcopy(
            concatenated_modality_inputs
        )
        concatenated_modality_inputs_copied[n] = 0.0
        common_embedding_perturbed = common_encoder(
            torch.from_numpy(concatenated_modality_inputs_copied)
        )
        common_embedding_diff[n] = abs(
            common_embedding_perturbed.detach().squeeze() - common_embedding
        )

    thresh = torch.quantile(common_embedding_diff, thresh_q, dim=0).expand_as(
        common_embedding_diff
    )

    impacted_features = (common_embedding_diff > thresh).float()

    g = {}

    n_edges = {}

    for rel_type in range(num_rel_type):

        # create planar adjacency matrix
        adj_matrix = impacted_features[:, rel_type].reshape((-1, 1))

        # removes self-connections
        adj_matrix_no_self_connections = spp.coo_matrix(
            (adj_matrix.mm(adj_matrix.transpose(0, 1))).mul(
                torch.ones(adj_matrix.shape[0]) - torch.eye(adj_matrix.shape[0])
            )
        )

        rows, cols = adj_matrix_no_self_connections.nonzero()

        n_edges[rel_type] = adj_matrix_no_self_connections.sum()
        list_edge_tuples = []
        for ind, r in enumerate(rows):
            list_edge_tuples.append((r, cols[ind]))

        # create planar graph from adjacency matrix, then add to heterograph
        g[("feat", str(rel_type), "feat")] = list_edge_tuples

    # create multigraph object, second argument ensures number of nodes are consistent across planes
    multi_graph = dgl.heterograph(g, {"feat": num_nodes})

    node_features = np.array(
        [[x] for x in concatenated_modality_inputs.tolist()]
    ).astype(np.float32)

    return multi_graph, node_features
