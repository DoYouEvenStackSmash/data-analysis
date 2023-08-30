#!/usr/bin/python3
import argparse
from clustering_imports import *
from kmeans import *
from kmedoids import *
from ClusterTreeNode import ClusterTreeNode
from collections import deque
import time
from IPython.display import display
import json
import networkx as nx
from loguru import logger
from graph_support import *


def search_tree(node_list, data_list, T):
    """
    Searches for the closest_idx point in data_list to target T
    Returns closest_idx of the nearest point to T, and the distance between them
    """
    n_curr = 0
    # search_list = []
    # search representative nodes
    while node_list[n_curr].data_refs is None:
        min_dist = float("inf")
        nn = 0
        for i in node_list[n_curr].children:
            dist = np.linalg.norm(node_list[i].val - T)
            if dist < min_dist:
                nn = i
                min_dist = dist
        # search_list.append(nn)
        n_curr = nn

    # search leaves
    closest_idx = 0
    min_dist = float("inf")
    for idx in node_list[n_curr].data_refs:
        # print(idx)
        dist = np.linalg.norm(data_list[idx] - T)
        if dist < min_dist:
            closest_idx = idx
            min_dist = dist
    # search_list.append(closest_idx)

    return closest_idx, min_dist


def search_tree_associations(node_list, data_list, input_list):
    """
    Computes matches using tree-based nearest neighbor search.

    Args:
        node_list (list of ClusterTreeNode): List of nodes representing a search tree structure.
        data_list (list of numpy arrays): List of data points for comparison.
        input_list (list of numpy arrays): List of input points to find matches for.

    Returns:
        di_match (list of ints): List of indices representing nearest neighbors found in the search tree.
        ds_match (list of floats): List of corresponding distances to nearest neighbors.
    """
    di_match_indices = []
    ds_match_distances = []

    start_time = time.perf_counter()

    # For each input point in the input list
    for input_index, input_point in enumerate(input_list):
        # Search the tree to find nearest neighbor and distance
        nearest_index, nearest_distance = search_tree(node_list, data_list, input_point)

        di_match_indices.append(nearest_index)
        ds_match_distances.append(nearest_distance)

    end_time = time.perf_counter()
    time_to_locate = end_time - start_time
    logger.info("{}".format(time_to_locate))
    # print("Time taken for tree-based search:", time_to_locate)

    return di_match_indices, ds_match_distances


def all_pairs_associations(data_list, input_list):
    """
    Computes nearest neighbor matches using all pairs comparison.

    Args:
        data_list (list of numpy arrays): List of data points for comparison.
        input_list (list of numpy arrays): List of input points to find matches for.

    Returns:
        mi_match (list of ints): List of indices representing nearest neighbors.
        ms_match (list of floats): List of corresponding distances to nearest neighbors.
    """
    mi_match_indices = []
    ms_match_distances = []

    start_time = time.perf_counter()

    # For each input point in the input list
    for input_index, input_point in enumerate(input_list):
        min_distance = float("inf")
        nearest_neighbor_index = 0

        # Compare the input point with all data points
        for data_index, data_point in enumerate(data_list):
            distance = np.linalg.norm(input_point - data_point)

            # Update nearest neighbor if a closer one is found
            if distance < min_distance:
                min_distance = distance
                nearest_neighbor_index = data_index

        mi_match_indices.append(nearest_neighbor_index)
        ms_match_distances.append(min_distance)

    end_time = time.perf_counter()
    total_execution_time = end_time - start_time

    logger.info("{}".format(total_execution_time))

    return mi_match_indices, ms_match_distances


def construct_tree(M, k=3, R=30, C=-1):
    """
    Builds a hierarchical clustering on the input data M
    Returns a flat tree as a list of nodes, where node_list[0] is the root.
    """
    # if size of data is greater than cutoff, we assume that the centroid does
    # not need to be a member of the dataset.
    if C < 0:
        C = max(int(len(M) / k**3), 50)  # cutoff threshold

    node_list = []
    node_queue = deque()
    data_queue = deque()

    node_list.append(ClusterTreeNode(0))
    node_queue.append(0)
    data_queue.append(M)

    while len(node_queue):
        node = node_list[node_queue.popleft()]
        data = data_queue.popleft()

        node.children = []
        # perform k means clustering.
        if len(data) > C:
            clusters = None
            centroids = initial_centroids(data, k)
            for r in range(R):
                clusters = assign_kmeans_clusters(data, centroids, bool(r == 0))

                new_centroids = update_centroids(clusters)

                # TODO: Investigate Estop behavior
                if new_centroids.shape != centroids.shape:
                    centroids = new_centroids
                    break

                if np.linalg.norm(new_centroids - centroids) == 0.0:
                    break
                centroids = new_centroids

            # create new nodes for centroids, and add each centroid/data pair to queues
            for i, ctr in enumerate(centroids):
                idx = len(node_list)
                node.children.append(idx)
                node_list.append(ClusterTreeNode(ctr))
                node_queue.append(idx)
                data_queue.append(np.array(clusters[i]))
        # perform k medioids clustering to ensure that the center is within the input data
        else:
            dlen = data.shape[0]
            mlist, distances = preprocess(data, k)

            total_sum = float("inf")
            clusters = None

            for _ in range(R):
                clusters = assign_clusters(dlen, mlist, distances)
                mlist = update_medioids(clusters, mlist, distances)
                new_sum = calculate_sum(clusters, mlist, distances)
                if new_sum == total_sum:
                    break
                total_sum = new_sum
            clusters, medioids = postprocess(data, clusters, mlist)

            # create new nodes for medioids, initialize node data arrays to hold the clusters
            for i, med in enumerate(medioids):
                idx = len(node_list)
                node.children.append(idx)
                node_list.append(ClusterTreeNode(med))
                node_list[idx].data_refs = np.array(clusters[i])

    return node_list


def construct_data_list(M, node_list):
    """
    Wrapper function for constructing a new data array by replacing data references to their indices
    Returns an array of data points, and replaces data_refs in leaf nodes
    """
    data_list = [np.empty(M.shape[1:]) for _ in range(M.shape[0])]
    img_count = 0

    for i, node in enumerate(node_list[1:]):
        if node.children is None:
            if node.data_refs is not None:
                data_idx = []
                for j, img in enumerate(node.data_refs):
                    if np.linalg.norm(img - node.val) == 0.0:
                        node.val_idx = img_count
                    data_list[img_count] = np.array(img)
                    data_idx.append(img_count)
                    img_count += 1
                node_list[i + 1].data_refs = data_idx
    # print(img_count)
    return data_list


def hierarchify_wrapper(filename, k, R, C):
    """
    Wrapper function for building a hierarchical clustering
    """
    M = data_loading_wrapper(filename)
    nl, dl = hierarchify(M, k, R, C)
    return nl, dl


def hierarchify(M, k=3, R=4, C=-1):
    """
    Wrapper function for execution of clustering
    """

    tree_build = time.perf_counter()
    node_list = construct_tree(M, k, R, C)
    tree_build = time.perf_counter() - tree_build

    # print("building data reference list...")
    data_build = time.perf_counter()
    data_list = construct_data_list(M, node_list)
    data_build = time.perf_counter() - data_build

    # N D k R tree_build data_build
    # Log the metrics in CSV format
    logger.info(
        "{},{},{},{},{},{}".format(
            M.shape[0], np.product(M.shape[1:]), k, R, tree_build, data_build
        )
    )
    return node_list, data_list


def data_loading_wrapper(filename):
    """
    Wrapper for data loader
    returns a loaded data array
    """
    M = dataloader(filename)
    return M


def serialize_wrapper(args, node_list, data_list):
    """
    Wrapper function for serializing a constructed clustering with params
    """
    params = {
        "input": args.input,
        "output": args.output,
        "k": args.clusters,
        "R": args.iterations,
        "C": args.cutoff,
    }
    output_prefix = args.output
    tree_dict = {
        "parameters": params,
        "resources": {
            "node_vals_file": f"{output_prefix}_tree_node_vals.npy",
            "data_list_file": f"{output_prefix}_tree_data_list.npy",
        },
    }

    tree_node_list = []
    tree_node_vals = []

    # handle root node
    tree_node_list.append(
        {
            "node_id": 0,  # position of root node in node_array
            "node_val_idx": None,  # position of node.val in tree_node_vals array
            "children": [
                c for c in node_list[0].children
            ],  # array of references to elements of tree_node_list
            "data_refs": None,
        }
    )

    for i, node in enumerate(node_list[1:]):
        if node.data_refs is not None:
            tree_node_list.append(
                {
                    "node_id": i + 1,  # position of node in tree_node_list
                    "node_val_idx": len(
                        tree_node_vals
                    ),  # position of node.val in tree_node_vals array
                    "children": None,  # children of a leaf node is None
                    "data_refs": [
                        didx for didx in node.data_refs
                    ],  # array of references to elements of data_list
                }
            )
            # special handling of node's value
            # since node.val for a leaf node is a member of the input data, we add the same member from
            # data_list using node.val_idx
            tree_node_vals.append(np.array(data_list[node.val_idx]))
        else:
            tree_node_list.append(
                {
                    "node_id": i + 1,  # position of node in tree_node_list
                    "node_val_idx": len(
                        tree_node_vals
                    ),  # position of node.val in tree_node_vals array
                    "children": [
                        c for c in node.children
                    ],  # array of references to other elements of tree_node_list
                    "data_refs": None,  # internal node has no data_refs
                }
            )
            # node.val of an internal node is generated by kmeans, and only exists in the node
            tree_node_vals.append(np.array(node.val))

    tree_dict["node_list"] = tree_node_list

    f = open(f"{output_prefix}_tree_hierarchy.json", "w")
    f.write(json.dumps(tree_dict, indent=2))
    f.close()

    np.save(f"{output_prefix}_tree_node_vals.npy", np.array(tree_node_vals))
    np.save(f"{output_prefix}_tree_data_list.npy", np.array(data_list))


def build_wrapper(args):
    """
    Wrapper function for constructing a hierarchical clustering from input and serializing the output
    """
    node_list, data_list = hierarchify_wrapper(
        args.input, args.clusters, args.iterations, args.cutoff
    )

    print("Building hierarchical clustering")
    if args.output is not None:
        serialize_wrapper(args, node_list, data_list)
    return node_list, data_list


def tree_loader(filename):
    """
    Wrapper function for deserializing a hierarchical clustering
    Returns a node_list of ClusterTreeNodes and a list of data points
    """

    """
    "node_list": [
        {
            "node_id": 1,
            "node_val_idx": 0, # refers to an element of node_valls
            "children": [
                4, # refers to an element of node_list
                5, # refers to  element of node_list
                6  # refers to the 6th element of node_list
            ],
            "data_refs": null # 
        },
        ]

    """
    f = open(filename, "r")
    parsed_data = json.loads(f.read())
    f.close()

    # load values of internal tree nodes
    tree_node_vals = np.load(parsed_data["resources"]["node_vals_file"])

    # load data referred to by leaf nodes
    data_list = np.load(parsed_data["resources"]["data_list_file"])

    # load node_list
    node_list_data = parsed_data["node_list"]

    # process root node
    root_node_data = node_list_data[0]
    root_node = ClusterTreeNode(
        val=root_node_data["node_id"],
        val_idx=root_node_data["node_val_idx"],
        children=root_node_data["children"],
        data=root_node_data["data_refs"],
    )

    node_list = [root_node]
    for j, node_data in enumerate(node_list_data[1:]):
        # if node_data["data_refs"] is
        node_list.append(
            ClusterTreeNode(
                val=tree_node_vals[
                    node_data["node_val_idx"]
                ],  # get value of node using node_val_idx
                val_idx=node_data[
                    "node_val_idx"
                ],  # get actual value of node using node_val_idx
                children=node_data["children"],
                data=node_data["data_refs"],
            )
        )

    return node_list, data_list


def load_wrapper(args):
    """
    Wrapper function for loading a hierarchical clustering from some static representation
    Returns a node_list and data_list
    """
    print("Loading hierarchical clustering")
    node_list, data_list = tree_loader(args.tree)
    if args.G:
        build_tree_diagram(node_list, data_list)
    print(len(data_list))
    # if args.G:
    #     graph_serialize(node_list, data_list)
    return


def search_wrapper(args):
    """
    Wrapper function for searching a hierarchical cluster tree for some list of points
    returns the list of associations and distances
    """
    node_list, data_list = tree_loader(args.tree)
    print(len(data_list))
    N = data_loading_wrapper(args.input)
    print("Searching hierarchical clustering")
    np.random.shuffle(N)
    print(len(N))
    st_idxs, st_dss = search_tree_associations(node_list, data_list, N)
    print(st_dss)

    if args.G:
        ap_idxs, ap_dss = all_pairs_associations(data_list, N)
        search_graph_serialize(node_list, data_list, st_idxs, st_dss, ap_idxs, ap_dss)
    else:
        ds_mat = setup_coeff(st_dss)
        mlist = [i for i in range(len(N))]
        display_correlation_matrix(mlist, st_idxs, ds_mat)


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Clustering Program")

    subparsers = parser.add_subparsers(title="Subcommands", dest="command")

    # Build subparser
    build_parser = subparsers.add_parser("build", help="Build hierarchical clustering")
    build_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input file with data to build hierarchical clustering",
    )
    build_parser.add_argument(
        "-k", "--clusters", type=int, default=3, help="Number of clusters (default 3)"
    )
    build_parser.add_argument(
        "-R",
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations (default 30)",
    )
    build_parser.add_argument(
        "-C",
        "--cutoff",
        type=int,
        default=45,
        help="Minimal number of elements in medoid cluster (default 45)",
    )
    build_parser.add_argument(
        "-o", "--output", help="Output file for saving the hierarchical clustering"
    )
    build_parser.set_defaults(func=build_wrapper)

    # Load subparser
    load_parser = subparsers.add_parser(
        "load", help="Load existing hierarchical clustering"
    )
    load_parser.add_argument(
        "-t", "--tree", required=True, help="JSON file containing hierarchy"
    )
    load_parser.add_argument(
        "-G", action="store_true", help="generate graph from tree data"
    )
    load_parser.set_defaults(func=load_wrapper)

    # Search subparser
    search_parser = subparsers.add_parser(
        "search", help="Search hierarchical clustering"
    )
    search_parser.add_argument(
        "-t", "--tree", required=True, help="JSON file containing hierarchy"
    )
    search_parser.add_argument(
        "-M",
        "--input",
        required=True,
        help="Large input file with data for search or building",
    )
    search_parser.add_argument(
        "-G", action="store_true", help="generate graph from tree data"
    )
    search_parser.set_defaults(func=search_wrapper)

    # Parse command-line arguments and call the appropriate function

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
