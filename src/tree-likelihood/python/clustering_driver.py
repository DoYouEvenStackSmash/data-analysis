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
from loguru import logger
from graph_support import *
from likelihood_scratch import *
from tree_build import *


def param_loader(filename, count=1):
    return [{"id": i, "weight": 1} for i in range(count)]


def hierarchify_wrapper(filename, k, R, C, param_file=None):
    """
    Wrapper function for building a hierarchical clustering
    """
    M = data_loading_wrapper(filename)
    # # M = np.random.randint(0, 10, (10, 2, 2))
    P = None
    # if param_file != None:
    #     P = param_loader(param_file, len(M))
    # else:
    #     P = param_loader(filename, len(M))

    # print(P)
    nl, dl, pl = hierarchify(M, k, R, C)
    return nl, dl, pl


def param_wrapper(param_file):
    param_list = np.load(param_file)
    # return [{"id":i,"weight":1} for i in range(len(param_list))]
    return param_list


def hierarchify(M, k=3, R=4, C=-1):
    """
    Wrapper function for execution of clustering
    """
    global logger
    tree_build = time.perf_counter()
    print(len(M))
    node_list = construct_tree(M, k, R, C)
    print(node_list)
    tree_build = time.perf_counter() - tree_build

    # print("building data reference list...")
    data_build = time.perf_counter()
    data_list, param_list = construct_data_list(node_list, [len(M),M[0].m1.shape[0],M[0].m1.shape[1]])
    data_build = time.perf_counter() - data_build

    # N D k R tree_build data_build
    # Log the metrics in CSV format
    logger.info(
        "{},{},{},{},{},{},{}".format(
            len(M), np.product(M[0].m1.shape), k, R, C, tree_build, data_build
        )
    )
    return node_list, data_list, param_list


def data_loading_wrapper(filename):
    """
    Wrapper for data loader
    returns a loaded data array
    """
    M = None
    if filename.split(".")[-1] == "fbs":
        exp_fb = dl.load_flatbuffer(filename)
        exp_buf = DataSet.GetRootAsDataSet(exp_fb,0)
        param_buf = extract_params(exp_buf)
        datum_buf = extract_datum(exp_buf)
        datumT_list = lift_datum_buf_to_datumT(datum_buf, param_buf)
        M = datumT_list
    else:
        M = dataloader(filename)
    return M


def serialize_wrapper(args, node_list, data_list, param_list, tree_params=None):
    """
    Wrapper function for serializing a constructed clustering with params
    """
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
                }
            )
            # special handling of node's value
            # since node.val for a leaf node is a member of the input data, we add the same member from
            # data_list using node.val_idx
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
                }
            )
            # node.val of an internal node is generated by kmeans, and only exists in the node
            tree_node_vals.append(node.val.m1.numpy())

    tree_dict["node_list"] = tree_node_list
    # tree_dict["param_list"] = tree_param_vals
    f = open(f"{output_prefix}_tree_hierarchy.json", "w")
    f.write(json.dumps(tree_dict, indent=2))
    f.close()

    np.save(f"{output_prefix}_tree_node_vals.npy", np.array(tree_node_vals))
    np.save(f"{output_prefix}_tree_data_list.npy", np.stack([data_list[i] for i in range(len(data_list))]))


def build_wrapper(args):
    """
    Wrapper function for constructing a hierarchical clustering from input and serializing the output
    """
    node_list, data_list, param_list = hierarchify_wrapper(
        args.input, args.clusters, args.iterations, args.cutoff, args.params
    )

    print("Building hierarchical clustering")
    if args.output is not None:
        serialize_wrapper(args, node_list, data_list, param_list)
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
    # param_list = parsed_data["param_list"]
    tree_params = parsed_data["parameters"]

    # process root node
    root_node_data = node_list_data[0]
    root_node = ClusterTreeNode(
        val=root_node_data["node_id"],
        val_idx=root_node_data["node_val_idx"],
        children=root_node_data["children"],
        data=root_node_data["data_refs"],
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
        serialize_wrapper(args, node_list, data_list, param_list, tree_params)
    print(len(data_list))
    # if args.G:
    #     graph_serialize(node_list, data_list)
    return


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
    node_list, data_list, tree_params = tree_loader(args.models)

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
        np.real(search_tree_nn_likelihood), np.real(search_tree_whole_cluster_likelihood), search_file
    )
    if args.true_likelihood:
        # return
        all_pairs_nn_likelihood, all_pairs_global_likelihood = global_scope_likelihoods(
            data_list, N
        )
        ap_file = "all_pairs_likelihoods.csv"
        write_csv(np.real(all_pairs_nn_likelihood), np.real(all_pairs_global_likelihood), ap_file)
    # if args.output:


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
