#!/usr/bin/python3
import argparse
from clustering_imports import *

# from kmeans import *
# from kmedoids import *
from ClusterTreeNode import ClusterTreeNode
from collections import deque
import time
from IPython.display import display
import json
import networkx as nx
from loguru import logger as LOGGER
from graph_support import *
from likelihood_scratch import *
from tree_build import *

import jax
import jax.numpy as jnp


def max_matrix(arr):
    magnitudes = np.sqrt(np.real(arr) ** 2 + np.imag(arr) ** 2)
    max_magnitude = np.max(magnitudes, axis=0)
    return max_magnitude


def param_loader(filename, count=1):
    return [{"id": i, "weight": 1} for i in range(count)]


def hierarchify_wrapper(filename, k=3, R=30, C=1, param_file=None, ctf_file=None):
    """Wrapper function for building a hierarchical clustering

    Args:
        filename (string): a filename for loading the data over which a clustering will be computed
        k (int, optional): Number of sublevel clusters  Defaults to 3.
        R (int, optional): Number of iterations for kmeans and kmedioids. Defaults to 4.
        C (int, optional): Cutoff threshold, maximum number of elements per cluster in the leaves of the tree. Defaults to -1.
        param_file (string, optional): a filename for a numpy array of parameters. Unused

    Returns:
        node_list (list(ClusterTreeNode)): Flattened tree
        [DatumT]: Ordered list of DatumT associated with the clusters. (M in a different order)
        param_list: legacy and unused
    """

    M = data_loading_wrapper(filename)
    if ctf_file != None:
        ml = ctf_wrapper(ctf_file)[0]
        for i, m in enumerate(M):
            M[i].m2 = ml
    # print(type(M[0].m1))
    node_list, data_list, param_list = hierarchify(M, k, R, C)
    return node_list, data_list, param_list


def ctf_wrapper(ctf_file):
    ctf_list = np.load(ctf_file)
    ctf_list = [max_matrix(ctf_list)]
    return ctf_list


def param_wrapper(param_file):
    """Loads a list of parameters from a file. Unused

    Args:
        param_file (string): a filename for a numpy array of parameters

    Returns:
        (list(strings)): A list of parameters
    """
    param_list = np.load(param_file)
    return param_list


def hierarchify(M, k=3, R=30, C=1):
    """Computes a hierarchical clustering over the input set M

    Args:
        M (DatumT): A list of DatumT objects
        k (int, optional): Number of sublevel clusters  Defaults to 3.
        R (int, optional): Number of iterations for kmeans and kmedioids. Defaults to 4.
        C (int, optional): Cutoff threshold, maximum number of elements per cluster in the leaves of the tree. Defaults to -1.

    Returns:
        node_list (list(ClusterTreeNode)): Flattened tree
        [DatumT]: Ordered list of DatumT associated with the clusters. (M in a different order)
        param_list: legacy and unused
    """

    global LOGGER
    tree_build = time.perf_counter()
    # print(len(M))
    node_list = construct_tree(M, k, R, C)
    # print(node_list)
    tree_build = time.perf_counter() - tree_build

    # print("building data reference list...")
    data_build = time.perf_counter()
    data_list, param_list = construct_data_list(
        node_list, [len(M), M[0].m1.shape[0], M[0].m1.shape[1]]
    )
    data_build = time.perf_counter() - data_build

    # N D k R tree_build data_build
    # Log the metrics in CSV format
    LOGGER.info(
        "{},{},{},{},{},{},{}".format(
            len(M), np.product(M[0].m1.shape), k, R, C, tree_build, data_build
        )
    )
    return node_list, data_list, param_list


def data_loading_wrapper(filename):
    """Wrapper for data loader. returns a loaded data array

    Args:
        filename (string): A flatbuffers filename

    Returns:
        M (list(DatumT)): A list of DatumT objects
    """

    M = None
    if filename.split(".")[-1] == "fbs":
        exp_fb = dl.load_flatbuffer(filename)
        exp_buf = DataSet.GetRootAsDataSet(exp_fb, 0)
        param_buf = extract_params(exp_buf)
        datum_buf = extract_datum(exp_buf)
        datumT_list = jax_lift_datum_buf_to_datumT(datum_buf, param_buf)
        M = datumT_list
    else:
        M = datum_loader(dataloader(filename))

    return M


def serialize_wrapper(args, node_list, data_list, param_list, tree_params=None):
    """Wrapper function for serializing a constructed clustering with params

    Args:
        node_list (list(ClusterTreeNode)): Flattened tree
        data_list (list(DatumT)): Ordered list of DatumT associated with the clusters.
        param_list: legacy and unused
    """
    print("serializing")
    # for n in node_list:
    #     if n.cluster_radius != 0:
    #         print(n.cluster_radius)
    params = {}
    if tree_params != None:
        params = tree_params
    else:
        params = {
            "input": args.input,
            "output": args.output,
            "params_file": args.params,
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
            "param_list_file": f"{output_prefix}_tree_param_list.npy",
        },
    }

    tree_node_list = []
    tree_node_vals = []
    tree_param_vals = param_list

    # handle root node
    tree_node_list.append(
        {
            "node_id": 0,  # position of root node in node_array
            "node_val_idx": None,  # position of node.val in tree_node_vals array
            "children": [
                c for c in node_list[0].children
            ],  # array of references to elements of tree_node_list
            "data_refs": None,
            "param_refs": None,
            "cluster_radius": node_list[0].cluster_radius,
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
                    # "param_refs": [pidx for pidx in node.param_refs],
                    "cluster_radius": node.cluster_radius,
                }
            )
            # special handling of node's value
            # since node.val for a leaf node is a member of the input data, we add the same member from
            # data_list using node.val_idx
            # print(data_list[node.val_idx].shape)
            tree_node_vals.append(data_list[node.val_idx])
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
                    # "param_refs": None,
                    "cluster_radius": node.cluster_radius,
                }
            )
            # node.val of an internal node is generated by kmeans, and only exists in the node
            # print(node.val.m1.)
            tree_node_vals.append(node.val.m1)  # .numpy())

    tree_dict["node_list"] = tree_node_list
    # tree_dict["param_list"] = tree_param_vals
    f = open(f"{output_prefix}_tree_hierarchy.json", "w")
    print(tree_dict)
    f.write(json.dumps(tree_dict, indent=2))
    f.close()
    # print(tree_node_vals)
    np.save(
        f"{output_prefix}_tree_node_vals.npy",
        np.stack([np.array(i) for i in tree_node_vals]),
    )
    np.save(
        f"{output_prefix}_tree_data_list.npy",
        np.stack([data_list[i] for i in range(len(data_list))]),
    )


def build_wrapper(args):
    """Wrapper function for constructing a hierarchical clustering from input and serializing the output

    Args:
        args: argparse object

    Returns:
        node_list (list(ClusterTreeNode)): Flattened tree
        data_list (list(DatumT)): Ordered list of DatumT associated with the clusters.
    """
    node_list, data_list, param_list = hierarchify_wrapper(
        args.input, args.clusters, args.iterations, args.cutoff, args.params, args.ctfs
    )

    print("Building hierarchical clustering")
    if args.output is not None:
        serialize_wrapper(args, node_list, data_list, param_list)
    # print("Foo")
    return node_list, data_list


def datum_loader(np_array_1, np_array_2=None):
    """Wrapper function to load a numpy array into datumT objects

    Args:
        np_array_1: a numpy array for m1 of DatumT
        np_array_2 (_type_, optional): a numpy array for m2 of DatumT. Defaults to None.

    Returns:
        datumT_arr (list(DatumT)): Ordered list of DatumT
    """

    mat2_gen = lambda np_array_2, x: np_array_2[x]
    if type(np_array_2) == type(None):
        mat2_gen = lambda np_array_2, x: 1
    else:
        print(len(np_array_2))
    # print(type(np_array_2[0]))
    datumT_arr = []
    for idx, mat1 in enumerate(np_array_1):
        mat2 = mat2_gen(np_array_2, idx)
        datumT_val = DatumT()
        datumT_val.m1 = jnp.array(mat1)
        datumT_val.m2 = jnp.array(mat2)
        datumT_arr.append(datumT_val)

    return datumT_arr


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
    tree_node_vals = datum_loader(np.load(parsed_data["resources"]["node_vals_file"]))

    # load data referred to by leaf nodes
    data_list = datum_loader(np.load(parsed_data["resources"]["data_list_file"]))

    # load node_list
    node_list_data = parsed_data["node_list"]
    # param_list = parsed_data["param_list"]
    tree_params = parsed_data["parameters"]

    # process root node
    root_node_data = node_list_data[0]
    root_node = ClusterTreeNode(
        val=root_node_data["node_id"],
        val_idx=root_node_data["node_val_idx"],
        children=root_node_data["children"],
        data=root_node_data["data_refs"],
        cluster_radius=root_node_data["cluster_radius"]
        # params=root_node_data["param_refs"],
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
                cluster_radius=node_data["cluster_radius"]
                # params=node_data["param_refs"],
            )
        )

    return node_list, data_list, tree_params


def load_wrapper(args):
    """
    Wrapper function for loading a hierarchical clustering from some static representation
    Returns a node_list and data_list
    """
    print("Loading hierarchical clustering")
    node_list, data_list, tree_params = tree_loader(args.tree)
    if args.G:
        build_tree_diagram(node_list, data_list)
    if args.output:
        param_list = []
        serialize_wrapper(args, node_list, data_list, param_list, tree_params)


def search_wrapper(args):
    """
    Wrapper function for searching a hierarchical cluster tree for some list of points
    returns the list of associations and distances
    """
    node_list, data_list, tree_params = tree_loader(args.tree)
    print(len(data_list))
    N = data_loading_wrapper(args.input)
    print("Searching hierarchical clustering")
    np.random.shuffle(N)

    st_idxs, st_dss = search_tree_associations(node_list, data_list, N)

    if args.G:
        ap_idxs, ap_dss = all_pairs_associations(data_list, N)
        search_graph_serialize(node_list, data_list, st_idxs, st_dss, ap_idxs, ap_dss)
    else:
        ds_mat = setup_coeff(st_dss)
        mlist = [i for i in range(len(N))]
        display_correlation_matrix(mlist, st_idxs, ds_mat)


def likelihood_wrapper(args):
    """Wrapper function for computing log likelihood of a preexisting tree.
    Serializes the results in a csv.

    Args:
        args: Argparse object
    """
    node_list, data_list, tree_params = tree_loader(args.models)
    N = None
    if args.ctfs != None:
        img_arr, ctf_arr = dataloader(args.images), dataloader(args.ctfs)
        # print(type(ctf_arr))
        N = datum_loader(img_arr, ctf_arr)
    else:
        N = data_loading_wrapper(args.images)

    if args.test:
        approximate_likelihood, true_likelihood = testbench_likelihood(
            node_list, data_list, N
        )
        out_file = f"{args.output}_likelihoods.csv"
        write_csv(np.real(approximate_likelihood), np.real(true_likelihood), out_file)
        return
    (
        search_tree_nn_likelihood,
        search_tree_whole_cluster_likelihood,
    ) = search_tree_likelihoods(node_list, data_list, N)
    search_file = f"{args.output}_likelihoods.csv"
    write_csv(
        np.real(search_tree_nn_likelihood),
        np.real(search_tree_whole_cluster_likelihood),
        search_file,
    )
    if args.true_likelihood:
        # return
        all_pairs_nn_likelihood, all_pairs_global_likelihood = global_scope_likelihoods(
            data_list, N
        )
        ap_file = "all_pairs_likelihoods.csv"
        write_csv(
            np.real(all_pairs_nn_likelihood),
            np.real(all_pairs_global_likelihood),
            ap_file,
        )


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
        "-p",
        "--params",
        required=False,
        help="Input file with ctfs to build hierarchical clustering",
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
    build_parser.add_argument("--ctfs", help="npy array containing ctfs")

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
    load_parser.add_argument("-o", "--output", help="output file for serialization")
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

    # likelihood parser
    likelihood_parser = subparsers.add_parser(
        "likelihood", help="likelihood calculation"
    )
    likelihood_parser.add_argument(
        "--models", required=True, help="json file containing hierarchy"
    )
    likelihood_parser.add_argument(
        "--images", required=True, help="npy array containing images"
    )
    likelihood_parser.add_argument("--ctfs", help="npy array containing ctfs")

    likelihood_parser.add_argument(
        "-o", "--output", help="prefix of output file for saving the densities"
    )
    likelihood_parser.add_argument(
        "-T",
        "--true_likelihood",
        action="store_true",
        help="calculate true likelihoods",
    )
    likelihood_parser.add_argument(
        "-test", action="store_true", help="fast access to testbench function"
    )

    likelihood_parser.set_defaults(func=likelihood_wrapper)
    # Parse command-line arguments and call the appropriate function

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
