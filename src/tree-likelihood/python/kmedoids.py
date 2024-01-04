#!/usr/bin/python3
from clustering_imports import *

import jax
import jax.numpy as jnp
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
    total_sum = sum(
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
        cluster = jnp.array(clusters[mlist.index(midx)])
        cluster_distances = jnp.sum(distances[cluster][:, cluster], axis=1)
        new_mlist.append(cluster[jnp.argmin(cluster_distances)])

    return new_mlist


def compute_distance_matrix(M_flat):
    """
    Build an nxn matrix of all pairwise distances between elements
    """
    num_tensors, width, height = M_flat.shape

    # Reshape M_flat to (num_tensors, width*height)
    M_flat_reshaped = M_flat.reshape((num_tensors, -1))

    # Compute pairwise distances
    pairwise_distances = jnp.linalg.norm(M_flat_reshaped[:, None] - M_flat_reshaped, axis=2)

    return pairwise_distances


def preprocess(M, k=3):
    """
    Preprocesses M according to k medioids algorithm
    Computes an initial set of medioids, and a distances matrix between every pair of points
    Returns a list of indices referring to M, and a distances matrix
    """
    n = len(M)
    print(type(M[0].m1))
    # Flatten the 2x2 matrices to 1D arrays for pairwise calculations
    pairwise_distances = None
    
    # M_flat = M.reshape(n, -1)
    M_flat = jnp.stack([m.m1 for m in M])
    print(type(M_flat))#.shape)
    pairwise_distances = compute_distance_matrix(M_flat)
    # for p in pairwise_distances:
    #     print(p)
    # Step 1-2: Calculate denominators efficiently
    denominators = jnp.sum(pairwise_distances, axis=1)

    # Calculate v values using vectorized operations in PyTorch
    v_values = pairwise_distances / denominators.reshape(-1, 1)

    # Set diagonal values to 0
    v_values_jax = jnp.array(v_values)

    # Equivalent JAX code for v_values - torch.diag(v_values.diag())
    v_values_jax = v_values_jax - jnp.diag(jnp.diag(v_values_jax))

    # Sum along axis 1
    v_sums = jnp.sum(v_values_jax, axis=1)

    # Initialize objects using list comprehension
    data = [(idx, v_sums[idx].item()) for idx in range(n)]

    # Sort the data objects by v values
    sortkey = lambda d: d[1]
    sorted_data = sorted(data, key=sortkey)

    # Get the indices of the k medioids
    medioid_indices = [d[0] for d in sorted_data[:k]]

    # Convert pairwise_distances to NumPy array if needed
    pairwise_distances_np = pairwise_distances#.numpy()

    return medioid_indices, pairwise_distances_np



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
