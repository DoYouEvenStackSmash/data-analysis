#!/usr/bin/python3
import numpy as np
from collections import deque
from clustering_imports import *


def weighted_sample(weights):
    """
    Sample a weighted probability distribution
    returns an index
    """
    # normalize total_w to be in [0,1]
    total_w = weights / np.sum(weights)
    sample_val = np.random.uniform(0, 1)
    for idx, w in enumerate(total_w):
        sample_val -= w
        if sample_val <= 0:
            return idx
    return len(weights) - 1


def distance(k, m):
    return np.linalg.norm(k - m)
    # return np.sqrt(np.sum(np.power(k - m, 2)))


def assign_kmeans_clusters(data, centroids):
    """
    Create clusters by assigning data to centroids
    Returns a list of np array clusters
    """
    clusters = [[] for _ in centroids]
    min_dist = float("inf")
    nn = None
    pdist = None
    for img in data:
        min_dist = float("inf")
        nn = None
        for idx, ctr in enumerate(centroids):
            if np.array_equal(img, ctr):
                continue
            pdist = distance(img, ctr)
            if pdist < min_dist:
                min_dist = pdist
                nn = idx
        if nn != None:
            clusters[nn].append(img)

    # convert clusters lists to numpy arrays
    for i in range(len(clusters)):
        if not len(clusters[i]):
            continue
        clusters[i] = np.stack([j for j in clusters[i]])

    return clusters


# def assign_kmeans_clusters(data, centroids):
#     """
#     Create clusters by assigning data to centroids
#     Returns a list of np array clusters
#     """
#     clusters = [[] for _ in centroids]

#     stack = [(data, centroids, clusters)]

#     while stack:
#         data, centroids, clusters = stack.pop()

#         if len(data) == 0:
#             continue

#         if len(centroids) == 1:
#             clusters[0].extend(data)
#             continue

#         mid = len(data) // 2
#         left_data, right_data = data[:mid], data[mid:]
#         left_centroids, right_centroids = centroids, centroids
#         left_clusters, right_clusters = [[] for _ in centroids], [[] for _ in centroids]

#         stack.append((left_data, left_centroids, left_clusters))
#         stack.append((right_data, right_centroids, right_clusters))

#     merged_clusters = []
#     for i, cluster_data in enumerate(clusters):
#         if len(cluster_data) > 0:
#             cluster_centroid = min(centroids, key=lambda ctr: distance(cluster_data[0], ctr))
#             merged_clusters.append(cluster_data)

#     return merged_clusters


def update_centroids(clusters):
    """
    Evaluate new centroids based on the existing clusters
    Returns an array of np arrays
    """
    return np.array(
        [
            np.mean(cluster, axis=0)
            if len(cluster)
            else np.zeros((1, cluster.shape[1]))
            for cluster in clusters
        ]
    )


def kmeanspp(M, k, centroids, not_chosen, chosen):
    """
    Compute a probably-better-than-random set of k centroids using kmeans++
    Returns a np array of k centroids
    """

    for _ in range(k - 1):
        weights = np.zeros(len(not_chosen))

        # distance lambda function
        D = lambda ck, m: np.sqrt(
            np.sum(np.array([np.power(i, 2) for i in (ck - m).flatten()]))
        )
        for idx, mdx in enumerate(not_chosen):
            m = M[mdx]
            min_dist = float("inf")
            for ck in centroids:
                min_dist = min(min_dist, D(ck, m))
            weights[idx] = np.power(min_dist, 2)

        selected_point = weighted_sample(weights)
        centroids.append(M[not_chosen[selected_point]])

        chosen.add(not_chosen[selected_point])
        not_chosen.remove(not_chosen[selected_point])

    centroids = np.array(centroids)
    return centroids


def kmeans(M, k, max_iter=100):
    """
    K means clustering algorithm. Given an array M, constructs k clusters over max_iter iterations
    Returns a numpy array of centroids, and an array of np arrays of clusters
    """
    # initialize a random starting centroid
    start_center = np.random.randint(M.shape[0])
    centroids = [M[start_center]]

    not_chosen = deque(i for i in range(len(M)) if i != start_center)
    chosen = {start_center}

    # compute initial centroids
    centroids = kmeanspp(M, k, centroids, not_chosen, chosen)

    # run k means for max_iter iterations
    for _ in range(max_iter):
        clusters = assign_kmeans_clusters(M, centroids)
        new_centroids = update_centroids(clusters)
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids
