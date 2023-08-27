import sys

sys.path.append("..")

from clustering_imports import *


def assign_clusters(dlen, mlist, distances):
    """
    Implicitly assigns elements to clusters using the distances matrix
    Returns a list of clusters
    """
    clusters = [[] for _ in mlist]

    for idx in range(dlen):
        distances_to_medioids = distances[idx, mlist]
        nearest_medioid_index = np.argmin(distances_to_medioids)
        clusters[nearest_medioid_index].append(idx)

    return clusters


def calculate_sum(clusters, mlist, distances):
    """
    Calculates the total sum over all clusters of distances between elements and centroids
    """
    total_sum = np.sum(
        [distances[idx, midx] for midx in mlist for idx in clusters[mlist.index(midx)]]
    )
    return total_sum


def update_medioids(clusters, mlist, distances):
    """
    Evaluate new medioids based on the existing clusters
    Returns an array of np arrays
    """
    new_mlist = []

    for midx in mlist:
        cluster = clusters[mlist.index(midx)]
        cluster_distances = np.sum(distances[cluster][:, cluster], axis=1)
        new_mlist.append(cluster[np.argmin(cluster_distances)])

    return new_mlist


# import numpy as np


class Datum:
    def __init__(self, idx):
        self.idx = idx
        self.v = None


# def preprocess(M, k=3):
#     """
#     Preprocesses M according to k medioids algorithm
#     Computes an initial set of medioids, and a distances matrix between every pair of points
#     Returns a list of indices referring to M, and a distances matrix
#     """
#     n = M.shape[0]

#     # Flatten the 2x2 matrices to 1D arrays for pairwise calculations
#     M_flat = M
#     pairwise_distances = None
#     if len(M.shape) == 3:
#         M_flat = M.reshape(n, -1)
#         pairwise_distances = np.sqrt(np.sum(np.square(M_flat[:, np.newaxis] - M_flat), axis=2))
#     else:
#          pairwise_distances = np.sqrt(np.sum(np.square(M_flat[:, np.newaxis, :] - M_flat), axis=2))

#     # Step 1-2: Calculate denominators efficiently
#     denominators = np.sum(pairwise_distances, axis=1)
#     # Calculate v values using vectorized operations
#     v_values = pairwise_distances / denominators[:, np.newaxis]

#     np.fill_diagonal(v_values, 0)  # Set diagonal values to 0

#     v_sums = np.sum(v_values, axis=1)

#     # Initialize objects using list comprehension
#     data = [Datum(idx) for idx in range(n)]

#     # Assign calculated v values to data objects
#     for j in range(n):
#         data[j].v = v_sums[j]

#     # Sort the data objects by v values
#     sortkey = lambda d: d.v
#     sorted_data = sorted(data, key=sortkey)

#     # Get the indices of the k medioids
#     medioid_indices = [d.idx for d in sorted_data[:k]]

#     return medioid_indices, pairwise_distances


def preprocess(M, k=3):
    """
    Preprocesses M according to k medioids algorithm
    Computes an initial set of medioids, and a distances matrix between every pair of points
    Returns a list of indices referring to M, and a distances matrix
    """
    n = M.shape[0]

    distances = np.sqrt(np.sum(np.square(M[:, np.newaxis, :] - M), axis=2))

    # Step 1-2: Calculate denominators efficiently
    denominators = np.sum(distances, axis=1)
    # Calculate v values using vectorized operations
    v_values = distances / denominators[:, np.newaxis]
    np.fill_diagonal(v_values, 0)  # Set diagonal values to 0
    v_sums = np.sum(v_values, axis=1)
    # Initialize objects using list comprehension
    data = [Datum(idx) for idx in range(n)]

    # Assign calculated v values to data objects
    for j in range(n):
        data[j].v = v_sums[j]
        # Sort the medioid indices by v values

    # medioid_indices = medioid_indices[np.argsort(v_sums[medioid_indices])]
    sortkey = lambda d: d.v
    sorted_data = sorted(data, key=sortkey)
    medioid_indices = np.argpartition(np.array([d.v for d in sorted_data]), k)[:k]
    mlist = list(medioid_indices)
    return mlist, distances


def postprocess(M, clusters, mlist):
    """
    Postprocesses the clusters and mlist to convert from indices of M, to explicit elements of M
    Returns a list of numpy arrays clusters and a numpy array medioids
    """
    medioids = np.array([M[idx] for idx in mlist])
    for i in range(len(mlist)):
        clusters[i] = np.stack(np.array([M[j] for j in clusters[i]]))
    return clusters, medioids


def kmedioids(M, k=5, max_iter=100):
    """
    K medioids algorithm. Creates clusters such that the centroids are members of the data.
    Returns a list of np arrays of clusters, and a numpy array of medioids.
    """
    n = M.shape[0]
    mlist, distances = preprocess(M, k)

    total_sum = float("inf")
    for _ in range(max_iter):
        clusters = assign_clusters(n, mlist, distances)
        mlist = update_medioids(clusters, mlist, distances)
        new_sum = calculate_sum(clusters, mlist, distances)
        if new_sum == total_sum:
            break
        total_sum = new_sum

    clusters, medioids = postprocess(M, clusters, mlist)
    return clusters, medioids
