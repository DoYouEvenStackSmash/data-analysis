#!/usr/bin/python3
import argparse
from clustering_imports import *
from kmeans import *
from kmedoids import *
from ClusterTreeNode import ClusterTreeNode
from collections import deque
import time
from IPython.display import display


def search_tree(node_list, data_list, T):
    """
    Searches for the closest_idx point in data_list to target T
    Returns closest_idx of the nearest point to T, and the distance between them
    """
    n_curr = 0
    # search_list = []
    # search representative nodes
    while node_list[n_curr].data_refs is None:
        min_dist = float('inf')
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
    min_dist = float('inf')
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
    
    print("Time taken for tree-based search:", time_to_locate)
    
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
        min_distance = float('inf')
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
    
    print("Total time taken for pairwise comparison:", total_execution_time)
    
    return mi_match_indices, ms_match_distances

def construct_tree(M, k=3, R=30, C=-1):
    """
    Builds a hierarchical clustering on the input data M
    Returns a flat tree as a list of nodes, where node_list[0] is the root.
    """
    # if size of data is greater than cutoff, we assume that the centroid does
    # not need to be a member of the dataset. 
    if C < 0:
        C = max(int(len(M) / k**3), 50) # cutoff threshold
    # SUPER_CUTOFF = CUTOFF * k
    # CUTOFF = int(len(M) / k**2) # cutoff threshold
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
            for _ in range(R):
                clusters = assign_kmeans_clusters(data, centroids)

                new_centroids = update_centroids(clusters)
                # convergence early stop condition
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
    pl = []
    total_vals = 0
    clusters = []
    centroids = []
    for i,node in enumerate(node_list[1:]):
        if node.children is None:
            if node.data_refs is not None:
                data_idx = []
                for j,img in enumerate(node.data_refs):
                    if np.linalg.norm(img - node.val) == 0.0:
                        node.val_idx = img_count
                    data_list[img_count] = np.array(img)
                    data_idx.append(img_count)
                    img_count += 1
                node_list[i + 1].data_refs = data_idx
    return data_list

def hierarchify_wrapper(filename, k, R, C):
    """
    Wrapper function for building a hierarchical clustering
    """
    M = data_loading_wrapper(filename)
    nl,dl = hierarchify(M, k, R, C)
    return nl,dl

def hierarchify(M, k=3, R=4,C=-1):
    """
    Wrapper function for execution of clustering
    """
    print("starting search tree construction:\n\tN = {}\n\tD = {}\n\tk = {}\n\tR = {}".format(M.shape[0],np.product(M.shape[1:]), k, R))
    tree_build = time.perf_counter()
    node_list = construct_tree(M,k,R,C)
    tree_build = time.perf_counter() - tree_build
    print("{}:\t{}".format("Tree build time:",tree_build))

    print("building data reference list...")
    data_build = time.perf_counter()
    data_list = construct_data_list(M, node_list)
    data_build = time.perf_counter() - data_build
    print("{}:\t{}".format("list build time:",data_build))
    print("total:\t{}".format(tree_build + data_build))

    return node_list, data_list

def data_loading_wrapper(filename):
    """
    Wrapper for data loader
    returns a loaded data array
    """
    M = dataloader(filename)
    return M


# import argparse

def build_wrapper(input_file, k, R, C):
    """
    Wrapper function for building hierarchical clusters
    """
    print("Building hierarchical clustering...")
    nl,dl = hierarchify_wrapper(input_file, k, R, C)

def search_wrapper(input_file, M, k, R, C):
    """
    Wrapper function for building a hierarchical cluster tree, and searching through it
    """
    print(f"Input file: {input_file}")
    print(f"Large input file: {M}")
    node_list,data_list = hierarchify_wrapper(input_file, k, R, C)
    N = data_loading_wrapper(M)
    print("Performing search operation...")
    st_idxs, st_dss = search_tree_associations(node_list, data_list, N)
    print(st_dss)
    
    

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Clustering Program")
    parser.add_argument("-hc", "--hierarchify", help="Input file to build hierarchical clustering", required=True)
    parser.add_argument("-k", "--clusters", type=int, default=3, help="Number of clusters (default 3)")
    parser.add_argument("-R", "--iterations", type=int, default=30, help="Number of iterations (default 30)")
    parser.add_argument("-C", "--cutoff", type=int, default=45, help="Minimal number of elements in medoid cluster (default 45)")
    parser.add_argument("-search", action="store_true", help="Perform search operation")
    parser.add_argument("-M", "--input_file", help="Large input file with data for search operation")

    args = parser.parse_args()

    if args.search:
        if args.input_file is None:
            print("Error: -M/--input_file is required for search operation")
            return
        search_wrapper(args.hierarchify, args.input_file, args.clusters, args.iterations, args.cutoff)
    else:
        build_wrapper(args.hierarchify, args.clusters, args.iterations, args.cutoff)
    
if __name__ == "__main__":
    main()
