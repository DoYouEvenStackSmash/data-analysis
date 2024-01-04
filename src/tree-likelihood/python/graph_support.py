import networkx as nx
from clustering_imports import *


def search_graph_serialize(node_list, data_list, st_idxs, st_dss, ap_idxs, ap_dss):
    """
    Serializes the results from a search, head to head between tree search and all pairs comparison
    """

    # constructs a dictionary for serialization prep
    comp_dict = {}
    for m in range(len(data_list)):
        comp_dict[m] = {
            "Tree": {"Index": [], "Distance": []},
            "Naive": {"Index": [], "Distance": []},
        }
    for i, a in enumerate(st_idxs):
        comp_dict[a]["Tree"]["Index"].append(f"T{int(i)}")
        comp_dict[a]["Tree"]["Distance"].append(float(st_dss[i]))

    for i, a in enumerate(ap_idxs):
        comp_dict[a]["Naive"]["Index"].append(f"N{int(i)}")
        comp_dict[a]["Naive"]["Distance"].append(float(ap_dss[i]))

    comp_dict = {"Nodes": [{i: v} for i, v in comp_dict.items()]}

    # populates the data array with the elements
    data = []
    for N in comp_dict["Nodes"]:
        for element, details in N.items():
            key = element
            tree_neighbors = [
                (
                    "green",
                    f"Data{element}",
                    details["Tree"]["Index"][i],
                    details["Tree"]["Distance"][i],
                )
                for i in range(len(details["Tree"]["Index"]))
            ]
            for t in tree_neighbors:
                data.append(t)
            naive_neighbors = [
                (
                    "yellow",
                    f"Data{element}",
                    details["Naive"]["Index"][i],
                    details["Naive"]["Distance"][i],
                )
                for i in range(len(details["Naive"]["Index"]))
            ]
            for t in naive_neighbors:
                data.append(t)

    for i, node in enumerate(node_list[1:]):
        if node.data_refs is not None:
            for idx in node.data_refs:
                data.append(
                    (
                        "purple",
                        f"Node{i+1}",
                        f"Data{idx}",
                        float(custom_distance(node.val,data_list[idx]).numpy()),
                    )
                )
        else:
            for c in node.children:
                data.append(
                    (
                        "blue",
                        f"Node{i+1}",
                        f"Node{c}",
                        float(custom_distance(node.val,node_list[c].val).numpy()),
                    )
                )
    print([type(i) for i in data[0]])
    
    # Create a directed graph
    G = nx.DiGraph()
    node_colors = {}
    for color, A, B, _ in data:
        if color == "blue":
            node_colors[A] = color
        if color == "purple":
            node_colors[A] = color
            node_colors[B] = color
        else:
            node_colors[B] = color
    print(node_colors)
    # Add nodes and edges based on relationships
    for color, A, B, distance in data:
        G.add_edge(A, B, weight=distance, color=color)
    nx.set_node_attributes(G, node_colors, "color")
    # Save the graph in GraphML format
    nx.write_graphml(G, "search_results.graphml")


def build_tree_diagram(node_list, data_list):
    print(node_list)
    """
    Given the node and data lists, creates an adjacency list which represents the tree
    Serializes the output as tree_representation.grpahml
    """
    SINGLE_ROOT = True  # unify clusters by taking mean of root node
    data = []
    # node_list[0].val = np.mean(
    #     np.array([node_list[i].val for i in node_list[0].children]), axis=0
    # )
    for i, n in enumerate(node_list):
        if i == 0 and not SINGLE_ROOT:
            continue
        if n.children is None:
            if n.data_refs is not None:
                for c in node_list[i].data_refs:
                    data.append(
                        (
                            "purple",
                            f"Node{i}",
                            f"Data{c}",
                            np.linalg.norm(node_list[i].val.m1 - data_list[c].m1),
                        )
                    )
            continue
        for c in node_list[i].children:
            # res_elem_tensor = jax_apply_d1m2_to_d2m1(node_list, node_list[i])
            # dist = jnp.linalg.norm(node_list[c].val - res_elem_tensor)
            data.append(
                (
                    "pink",
                    f"Node{i}",
                    f"Node{c}",
                    # 0
                    0
                    # np.linalg.norm(node_list[i].val - node_list[c].val),
                )
            )
        # Create a directed graph

    G = nx.DiGraph()
    node_colors = {}
    for color, A, B, _ in data:
        # if color == "blue":
        #     node_colors[A] = color
        if color == "purple":
            node_colors[A] = color
            node_colors[B] = color
        else:
            node_colors[A] = color

    # Add nodes and edges based on relationships
    for color, A, B, distance in data:
        G.add_edge(A, B, weight=distance, color=color)
    nx.set_node_attributes(G, node_colors, "color")
    # Save the graph in GraphML format
    nx.write_graphml(G, "tree_representation.graphml")
    return data
