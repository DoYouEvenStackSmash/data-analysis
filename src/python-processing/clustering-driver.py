#!/usr/bin/python3
import numpy as np
import os
import sys
import argparse
from kmedoids import *
from kmeans import *
from clustering_imports import *


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


def D_vect(di, dj):
    """
    Euclidean distance between two elements
    """
    return np.sqrt(np.sum(np.square(di - dj), axis=0))


def display_dendrogram(clusters, centroids):
    """
    Displays the clusters with the corresponding centroids and enclosing convex hull using matplotlib
    """
    reshaped_centers = centroids.reshape(-1, 2)
    cx_coords = reshaped_centers[:, 0]
    cy_coords = reshaped_centers[:, 1]

    # Prepare data for linkage and dendrogram
    all_data = np.concatenate(clusters)
    Z = linkage(all_data, "ward")

    plt.figure(figsize=(10, 5))

    # Plot dendrogram
    dendrogram(Z)

    # Plot data points and centroids
    for i, m in enumerate(clusters):
        reshaped_data = m.reshape(-1, 2)
        plt.scatter(
            reshaped_data[:, 0],
            reshaped_data[:, 1],
            c=colors[i],
            label="Cluster " + str(i + 1),
            s=10,
        )
        plt.scatter(cx_coords[i], cy_coords[i], marker="x", color=colors[i], s=50)

    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.title("Dendrogram of Hierarchical Clustering")
    plt.legend()
    plt.show()


def display_clusters(clusters, centroids):
    """
    Displays the clusters with the corresponding centroids and enclosing convex hull using matplotlib
    """
    ci = 0
    reshaped_centers = centroids.reshape(-1, 2)
    cx_coords = reshaped_centers[:, 0]
    cy_coords = reshaped_centers[:, 1]
    for i, m in enumerate(clusters):
        reshaped_data = m.reshape(-1, 2)
        # Compute the convex hull using Graham Scan
        if len(m) > 2:
            convex_hull = graham_scan_convex_hull(reshaped_data)
            convex_hull.append(convex_hull[0])

            # Convert the convex hull points to a numpy array for visualization
            convex_hull_array = np.array(convex_hull)

            # Plot the original points and the convex hull
            plt.plot(
                convex_hull_array[:, 0],
                convex_hull_array[:, 1],
                c=colors[i + ci],
                marker="x",
                linewidth=len(m)
                / 10,  # adjust width of the line according to the hierarchy
                label="Convex Hull",
            )
        plt.scatter(
            reshaped_data[:, 0],
            reshaped_data[:, 1],
            c=colors[i + ci],
            label="Points",
            s=1,
        )
        x_coords = reshaped_data[:, 0]
        y_coords = reshaped_data[:, 1]
        plt.scatter(x_coords, y_coords, marker="o", color=colors[i + ci], s=1)
        plt.scatter(cx_coords[i], cy_coords[i], marker="x", color=colors[i + ci], s=10)
    plt.show()


def sift_data(clusters, centroids, r):
    """
    Experimental behavior for creating a ball around the centroids for excluding outliers
    Returns a numpy array of the excluded points
    """
    data = []
    for i, k in enumerate(centroids):
        for idx, pt in enumerate(clusters[i]):
            if D_vect(k, pt) > r:
                data.append(np.array(clusters[i][idx]))
    return np.array(data)


def eval_cluster_inertia(clusters, centroids):
    """
    Calculates the inertia of each cluster
    """
    min_rad = float("inf")
    max_rad = 0.0
    total_sum = 0.0
    msums = [0] * len(centroids)
    for i, ctr in enumerate(centroids):
        for pt in clusters[i]:
            msums[i] += D_vect(ctr, pt)
        total_sum += msums[i]
        min_rad = min(min_rad, msums[i])
        max_rad = max(max_rad, msums[i])
        print("{}\t{}".format(ctr, msums[i]))


def nested_kmeans(M):
    """
    Builds a hierarchical clustering on input space M

    """
    k = 3  # Number of clusters for k-means
    max_iter = 30  # Maximum number of iterations for k-means
    start_center = np.random.randint(
        M.shape[0]
    )  # Choose a random starting center index
    centroids = [M[start_center]]  # Initialize centroids list with the starting center

    # Create a deque of indices for non-center points
    not_chosen = deque(i for i in range(len(M)) if i != start_center)
    chosen = {start_center}  # Set containing chosen center indices

    # Compute initial centroids using k-means++
    centroids = kmeanspp(M, k, centroids, not_chosen, chosen)

    cdict = {}  # Dictionary to store clusters at each level
    cq = deque()  # Queue for centroids
    dq = deque()  # Queue for data

    cq.append(centroids)  # Enqueue initial centroids
    dq.append(M)  # Enqueue the entire data array
    centroid_arr = []  # List to store all centroids

    cutoff = 65  # Cutoff point for switching to k-medioids
    while bool(cq):
        centroids = cq.popleft()  # Dequeue centroids
        data = dq.popleft()  # Dequeue data

        # If data size is below the cutoff, use k-medioids clustering
        if len(data) < cutoff:
            clusters, centroids = kmedioids(data, k, max_iter=30)
            for i, ctr in enumerate(centroids):
                ci = np.array(clusters[i])

                # Store the cluster data in cdict and the centroid in centroid_arr
                cdict[len(centroid_arr)] = ci
                centroid_arr.append(ctr)
            continue  # Skip further processing for this level

        else:
            # Apply regular k-means clustering
            for _ in range(max_iter):
                clusters = assign_kmeans_clusters(data, centroids)
                new_centroids = update_centroids(clusters)
                if np.array_equal(centroids, new_centroids):
                    break
                centroids = new_centroids

        # Process clusters and enqueue data for the next level
        for i, ctr in enumerate(centroids):
            ci = np.array(clusters[i])

            # Store the cluster data in cdict and the centroid in centroid_arr
            cdict[len(centroid_arr)] = ci
            centroid_arr.append(ctr)

            # Enqueue the centroid and cluster data for the next level
            cq.append(ctr)
            dq.append(ci)

    clusters = []
    centroids = []
    for k, v in cdict.items():
        centroids.append(centroid_arr[k])
        clusters.append(v)

    return clusters, centroids  # Return the final clusters and centroids


class Node:
    def __init__(self, val, idx, children=None, data=None):
        self.val = val
        self.idx = idx
        self.children = children
        self.data = None


def initial_centroids(d, k):
    start_center = np.random.randint(len(d))  # Choose a random starting center index
    centroids = [d[start_center]]  # Initialize centroids list with the starting center

    # Create a deque of indices for non-center points
    not_chosen = deque(i for i in range(len(d)) if i != start_center)
    chosen = {start_center}  # Set containing chosen center indices

    # Compute initial centroids using k-means++
    centroids = kmeanspp(d, k, centroids, not_chosen, chosen)
    return centroids


def construct_tree(M, k, C=65, R=30):
    node_list = [Node(-1, 0)]
    root = 0
    node_q = deque()  # queue to hold node indices
    data_q = deque()  # queue to hold data arrays
    node_s = deque()  # stack to hold node indices
    data_s = deque()  # stack to hold data arrays

    # initialization of queues
    node_q.append(root)
    data_q.append(M)
    print(len(M))

    while (len(node_q) and len(data_q)) or (len(node_s) and len(data_s)):
        while len(node_q) and len(data_q):
            n = node_list[node_q.popleft()]
            d = data_q.popleft()

            # If data size is below the cutoff, use k-medioids clustering
            if len(d) < C:
                clusters, centroids = kmedioids(d, k, R)
                # n.children = []
                for i, ctr in enumerate(centroids):
                    idx = len(node_list)
                    # n.children.append(idx)
                    node_list.append(Node(ctr, idx))
                    node_list[idx].data = np.array(clusters[i])
            else:
                node_s.append(n.idx)
                data_s.append(d)

        while len(node_s) and len(data_s):
            n = node_list[node_s.pop()]
            d = data_s.pop()
            centroids = initial_centroids(d, k)
            clusters = []
            for _ in range(R):
                clusters = assign_kmeans_clusters(d, centroids)
                new_centroids = update_centroids(clusters)
                if np.array_equal(centroids, new_centroids):
                    break
                centroids = new_centroids
            n.children = []
            for i, ctr in enumerate(centroids):
                idx = len(node_list)
                n.children.append(idx)
                node_list.append(Node(ctr, idx))
                node_q.append(idx)
                data_q.append(clusters[i])
            break
    return node_list


def generate_dot_graph(xy_pairs):
    dot_graph = "digraph G {\n"

    # Add nodes
    nodes = set()
    for x, y in xy_pairs:
        nodes.add(x)
        nodes.add(y)

    for node in nodes:
        dot_graph += f"    {node};\n"

    # Add edges
    for x, y in xy_pairs:
        dot_graph += f"    {x} -> {y};\n"

    dot_graph += "}"
    return dot_graph


# Example list of (x, y) pairs
# xy_pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]

# Generate dot format graph representation
# dot_representation = generate_dot_graph(xy_pairs)
# print(dot_representation)


def clustering_wrapper(filename, algorithm="kmeans", iterations=100, k=4):
    """
    Wrapper function for execution of clustering
    """
    k = 3
    M = dataloader(filename)
    # print(preprocess(M, k))
    # sys.exit()
    # M = M.reshape(M.shape[0], M.shape[1] ** 2)
    nl = construct_tree(M, k)
    # l = [i for i in nl if i.children != None]
    pl = []
    total_vals = 0
    clusters = []
    centroids = []
    for i in nl[1:]:
        if i.children is None:
            if i.data is not None:
                #     continue
                clusters.append(i.data)
                centroids.append(i.val)
                continue
            print(i.idx)
            continue

            # total_vals += len(i.data)

        for j in i.children:
            pl.append((i.idx, j))

    print(generate_dot_graph(pl))
    # print(total_vals)

    # clusters, centroids = nested_kmeans(M)
    # print(len(clusters))
    # eval_cluster_inertia(clusters, centroids)
    display_clusters(clusters, np.array(centroids))
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
