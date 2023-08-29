#!/usr/bin/python3
from clustering_imports import *
import time
from IPython.display import display

def dataloader(filename):
    """
    Loads data from a file
    """
    # Check if the file exists
    if not os.path.exists(filename):
        print("File does not exist.")
        sys.exit()
    else:
        # Load the array using np.load()
        loaded_array = np.load(filename, allow_pickle=True)
        # Display the loaded array
        print("Loaded Array:")
        print(loaded_array.shape)
        return loaded_array


def initial_centroids(d, k):
    start_center = np.random.randint(len(d))  # Choose a random starting center index
    centroids = [d[start_center]]  # Initialize centroids list with the starting center

    # Create a deque of indices for non-center points
    not_chosen = deque(i for i in range(len(d)) if i != start_center)
    chosen = {start_center}  # Set containing chosen center indices

    # Compute initial centroids using k-means++
    centroids = kmeanspp(d, k, centroids, not_chosen, chosen)
    # print(centroids)
    return centroids

def construct_tree(M, k=3, R=30):
    """
    Builds a hierarchical clustering on the input data M
    Returns a flat tree as a list of nodes, where node_list[0] is the root.
    """
    # if size of data is greater than cutoff, we assume that the centroid does
    # not need to be a member of the dataset. 
    CUTOFF = int(len(M) / k**2) # cutoff threshold
    # SUPER_CUTOFF = CUTOFF * k
    # CUTOFF = int(len(M) / k**2) # cutoff threshold
    node_list = []
    node_queue = deque()
    data_queue = deque()

    node_list.append(TreeNode(0))
    node_queue.append(0)
    data_queue.append(M)

    while len(node_queue):

        node = node_list[node_queue.popleft()]
        data = data_queue.popleft()
        
        node.children = []
        # perform k means clustering.
        if len(data) > CUTOFF:
            
            clusters = None
            centroids = initial_centroids(data, k)
            for _ in range(R):
                clusters = assign_kmeans_clusters(data, centroids)
                
                # # TODO: investigate emergency brake for empty clusters
                # if len(clusters) < k:
                #     centroids = update_centroids(clusters)
                #     break
                
                new_centroids = update_centroids(clusters)
                # convergence early stop condition
                if np.linalg.norm(new_centroids - centroids) == 0.0:
                    break
                centroids = new_centroids
            
            # create new nodes for centroids, and add each centroid/data pair to queues
            for i, ctr in enumerate(centroids):
                idx = len(node_list)
                node.children.append(idx)
                node_list.append(TreeNode(ctr))
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
                node_list.append(TreeNode(med))
                node_list[idx].data = np.array(clusters[i])

    return node_list


def search_tree(node_list, data_list, T):
    """
    Searches for the closest_idx point in data_list to target T
    Returns closest_idx of the nearest point to T, and the distance between them
    """
    n_curr = 0
    # search_list = []
    # search representative nodes
    while node_list[n_curr].data is None:
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
    for idx in node_list[n_curr].data:
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
        node_list (list of TreeNode): List of nodes representing a search tree structure.
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


def clustering_wrapper(filename, algorithm="kmeans", iterations=100, k=4):
    """
    Wrapper function for execution of clustering
    """
    k = 3
    M = dataloader(filename)
    
    tree_build = time.perf_counter()
    node_list = construct_tree(M,k)
    tree_build = time.perf_counter() - tree_build
    print("{}\t{}".format("tree build: ", tree_build))
    
    data_list = [np.empty(M.shape[1:]) for _ in range(M.shape[0])]
    img_count = 0
    pl = []
    total_vals = 0
    clusters = []
    centroids = []
    for i,node in enumerate(node_list[1:]):
        if node.children is None:
            if node.data is not None:
                data_idx = []
                for j,img in enumerate(node.data):
                    if np.linalg.norm(img - node.val) == 0.0:
                        node.val_idx = img_count
                    data_list[img_count] = np.array(img)
                    data_idx.append(img_count)
                    img_count += 1
                node_list[i + 1].data = data_idx
                
                continue
            continue
    return node_list, data_list
    

    # for j in i.children:
    #     pl.append((i.idx, j))

    # print(generate_dot_graph(pl))
    # print(total_vals)

    # clusters, centroids = nested_kmeans(M)
    # print(len(clusters))
    # eval_cluster_inertia(clusters, centroids)

    # display_dendrogram(clusters, np.array(centroids))


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Clustering Algorithm")

    # Add the command-line arguments
    parser.add_argument(
        "--algorithm",
        choices=["kmeans", "kmedoids"],
        default="kmeans",
        help="Choice of clustering algorithm: 'kmeans' or 'kmedoids'",
    )

    parser.add_argument(
        "--filename", required=True, help="Path to input data file (.npy format)"
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=100,
        help="Number of iterations to run clustering algorithm (default: 100)",
    )

    parser.add_argument(
        "--k", type=int, default=3, help="Static number of clusters (default: 3)"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Display the parsed arguments
    print("Chosen algorithm:", args.algorithm)
    print("Filename:", args.filename)
    print("Number of iterations:", args.iter)
    print("Number of clusters:", args.k)
    clustering_wrapper(args.filename)


if __name__ == "__main__":
    main()
