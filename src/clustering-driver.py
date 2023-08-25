#!/usr/bin/python3
import numpy as np
import os
import sys
import argparse
from kmedoids import *
from kmeans import *
from clustering_imports import *


def dataloader(filename):
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


def clustering_wrapper(filename, algorithm="kmeans", iterations=100, k=3):
    """
    Wrapper function for execution of clustering
    """
    M = dataloader(filename)
    nested_kmeans(M)
    # # kmeans_clusters, kmeans_mlist = kmeans(M, k, max_iter=100)
    # # print("kmeans")
    # # eval_clustering(kmeans_clusters, kmeans_mlist)
    # kmedioids_clusters, kmedioids_mlist = kmedioids(M, k, max_iter=100)
    # print("kmedioids")
    # thres = eval_clustering(kmedioids_clusters, kmedioids_mlist)
    # print(thres)
    # second_layer_M = sift_data(kmedioids_clusters, kmedioids_mlist, thres)
    # print(second_layer_M.shape)
    # print(second_layer_M)
    # kmedioids_clusters_2, kmedioids_mlist_2 = kmedioids(second_layer_M, 6, max_iter=100)
    # display_clusters([kmedioids_clusters,kmedioids_clusters_2],[kmedioids_mlist,kmedioids_mlist_2])



def display_clusters(clusters, centroids):
    ci = 0
    reshaped_centers = centroids.reshape(-1, 2)
    cx_coords = reshaped_centers[:, 0]
    cy_coords = reshaped_centers[:, 1]
    for i,m in enumerate(clusters):
        # if len(m) > 3:
        # continue
        reshaped_data = m.reshape(-1, 2)
        # print(reshaped_data.shape)
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
                c=colors[i+ci],
                marker="x",
                linewidth=len(m)/40,
                label="Convex Hull",
            )
        plt.scatter(
            reshaped_data[:, 0], reshaped_data[:, 1], c=colors[i+ci], label="Points", s=1
        )
        x_coords = reshaped_data[:, 0]
        y_coords = reshaped_data[:, 1]
        plt.scatter(x_coords, y_coords, marker="o", color=colors[i+ci], s=1)
        plt.scatter(cx_coords[i], cy_coords[i], marker="x", color=colors[i+ci], s=10)
        # break
    plt.show()

def sift_data(clusters, centroids, r):
    data = []
    for i,k in enumerate(centroids):
        for idx,pt in enumerate(clusters[i]):
            if D_vect(k, pt) > r:
                data.append(np.array(clusters[i][idx]))
    return np.array(data)


def eval_clustering(clusters, centroids):
    """
    Calculates the inertia of each cluster
    """
    min_rad = float('inf')
    max_rad = 0.
    total_sum = 0.
    msums = [0] * len(centroids)
    for i, ctr in enumerate(centroids):
        for pt in clusters[i]:
            msums[i] += D_vect(ctr, pt)
        total_sum += msums[i]
        min_rad = min(min_rad, msums[i])
        max_rad = max(max_rad, msums[i])
        print("{}\t{}".format(ctr, msums[i]))
    return 15

def nested_kmeans(M):
    k = 3
    max_iter = 30
    start_center = np.random.randint(M.shape[0])
    centroids = [M[start_center]]

    not_chosen = deque(i for i in range(len(M)) if i != start_center)
    chosen = {start_center}

    # compute initial centroids
    centroids = kmeanspp(M, k, centroids, not_chosen, chosen)

    cdict = {}
    cq = deque()
    dq = deque()

    cq.append(centroids)
    dq.append(M)
    centroid_arr = []
    
    cutoff = 65
    while bool(cq):
        centroids = cq.popleft()
        data = dq.popleft()
        if len(data) < cutoff:
            clusters, centroids = kmedioids(data, k, max_iter=100)
            for i,ctr in enumerate(centroids):
                ci = np.array(clusters[i])
                
                cdict[len(centroid_arr)] = ci
                centroid_arr.append(ctr)
            continue

        else:
            for _ in range(max_iter):
                clusters = assign_kmeans_clusters(data, centroids)
                new_centroids = update_centroids(clusters)
                if np.array_equal(centroids, new_centroids):
                    break
                centroids = new_centroids
        
        for i,ctr in enumerate(centroids):
            ci = np.array(clusters[i])
            
            cdict[len(centroid_arr)] = ci
            centroid_arr.append(ctr)
            cq.append(ctr)
            dq.append(ci)
    clusters = []
    centroids = []
    for k,v in cdict.items():
        centroids.append(centroid_arr[k])
        clusters.append(v)
    display_clusters(clusters, np.array(centroids))

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
