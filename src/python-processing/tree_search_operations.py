from clustering_imports import *


def find_cluster(node_list, T):
    """
    Searches for the cluster containing(ish) T
    returns an index to a node in node_list
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
    return n_curr

def find_ctf_cluster(node_list_with_params, T):
    n_curr = 0
    # search_list = []
    # search representative nodes
    while node_list[n_curr].data_refs is None:
        min_dist = float("inf")
        nn = 0
        for i in node_list[n_curr].children:
            dist = np.linalg.norm(node_list[i].val - node_list[i].params)
            if dist < min_dist:
                nn = i
                min_dist = dist
        # search_list.append(nn)
        n_curr = nn
    return n_curr

def search_tree(node_list, data_list, T):
    """
    Searches for the closest_idx point in data_list to target T
    Returns closest_idx of the nearest point to T, and the distance between them
    """
    n_curr = find_cluster(node_list, T)

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
