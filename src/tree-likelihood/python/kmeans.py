#!/usr/bin/python3

import numpy as np

from collections import deque
from clustering_imports import *


def initial_centroids(d, k):
    start_center = torch.random.randint(len(d))  # Choose a random starting center index
    centroids = [d[start_center]]  # Initialize centroids list with the starting center

    # Create a deque of indices for non-center points
    not_chosen = deque(i for i in range(len(d)) if i != start_center)
    chosen = {start_center}  # Set containing chosen center indices

    # Compute initial centroids using k-means++
    centroids = kmeanspp(d, k, centroids, not_chosen, chosen)
    # print(centroids)
    return centroids


def weighted_sample(weights):
    """
    Sample a weighted probability distribution
    returns an index
    """
    # normalize total_w to be in [0,1]
    total_w = weights / torch.sum(weights)
    sample_val = np.random.uniform(0, 1)
    for idx, w in enumerate(total_w):
        sample_val -= w
        if sample_val <= 0:
            return idx
    return len(weights) - 1


def custom_distance(k, m):
    return torch.linalg.norm(k.m1 - m.m1)
    # return torch.sqrt(torch.sum(torch.power(k - m, 2)))


def assign_kmeans_clusters(data, centroids, FIRST_FLAG=False, k=3):
    """
    Create clusters by assigning data to centroids
    Returns a list of np array clusters
    """
    clusters = [[] for _ in centroids]
    min_dist = float("inf")
    nn = None
    pdist = None
    for i, img in enumerate(data):
        min_dist = float("inf")
        nn = None
        for idx, ctr in enumerate(centroids):
            if FIRST_FLAG:
                if torch.tensor_equal(data[i], ctr):
                    continue
            pdist = custom_distance(data[i], ctr)
            if pdist < min_dist:
                min_dist = pdist
                nn = idx
        if nn != None:
            clusters[nn].append(data[i])

    # convert clusters lists to numpy arrays
    # torch.random.randint(len(data))
    new_clusters = []
    for i in range(len(clusters)):
        if not len(clusters[i]):
            # TODO: Investigate Estop Behavior
            # val = torch.stack(torch.tensor([data[torch.random.randint(len(data))]]))
            # new_clusters.append(val)
            continue
        new_clusters.append(torch.stack(torch.tensor([j for j in clusters[i]])))

    return new_clusters


def update_centroids(clusters):
    """
    Evaluate new centroids based on the existing clusters
    Returns an array of np arrays
    """
    return torch.tensor([torch.mean(cluster, axis=0) for cluster in clusters])


def kmeanspp(M, k, centroids, not_chosen, chosen):
    """
    Compute a probably-better-than-random set of k centroids using kmeans++
    Returns a np array of k centroids
    """

    for _ in range(k - 1):
        weights = torch.zeros(len(not_chosen))

        # distance lambda function
        D = lambda ck, m: torch.linalg.norm(ck - m)
        for idx, mdx in enumerate(not_chosen):
            m = M[mdx]
            min_dist = float("inf")
            for ck in centroids:
                min_dist = min(min_dist, D(ck, m))
            weights[idx] = torch.power(min_dist, 2)

        selected_point = weighted_sample(weights)
        centroids.append(M[not_chosen[selected_point]])

        chosen.add(not_chosen[selected_point])
        not_chosen.remove(not_chosen[selected_point])

    centroids = torch.tensor(centroids)
    return centroids


def kmeans(M, k, max_iter=100):
    """
    K means clustering algorithm. Given an array M, constructs k clusters over max_iter iterations
    Returns a numpy array of centroids, and an array of np arrays of clusters
    """
    # initialize a random starting centroid
    start_center = torch.random.randint(M.shape[0])
    centroids = [M[start_center]]

    not_chosen = deque(i for i in range(len(M)) if i != start_center)
    chosen = {start_center}

    # compute initial centroids
    centroids = kmeanspp(M, k, centroids, not_chosen, chosen)

    # run k means for max_iter iterations
    for _ in range(max_iter):
        clusters = assign_kmeans_clusters(M, centroids)
        new_centroids = update_centroids(clusters)
        if torch.tensor_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


def kmeanspp_refs(data_store, data_ref_arr, k):
    dlen = len(data_ref_arr)
    start_center = np.random.randint(dlen)
    not_chosen = deque(i for i in range(dlen) if i != start_center)

    centroids = [data_store[data_ref_arr[start_center]]]

    # chosen = {start_center}
    weights = None  # torch.zeros(dlen)
    min_dist = float("inf")

    for _ in range(k - 1):
        weights = torch.zeros(len(not_chosen))
        for idx, mdx in enumerate(not_chosen):
            min_dist = float("inf")
            m = data_store[data_ref_arr[mdx]]
            for ctx, ctr in enumerate(centroids):
                min_dist = min(min_dist, custom_distance(m, ctr))

            weights[idx] = torch.square(min_dist)

        selected_point = weighted_sample(weights)
        # print(selected_point)
        # print(len(not_chosen))
        centroids.append(data_store[data_ref_arr[not_chosen[selected_point]]])

        # chosen.add(not_chosen[selected_point])
        not_chosen.remove(not_chosen[selected_point])
        if not len(not_chosen):
            break
    
    return centroids


def kmeans_refs(data_store, data_ref_arr, centroids, FIRST_FLAG=False):
    """
    Perform K-Means clustering iterations for a subset of data references.

    Args:
        data_store (list of numpy arrays): Data points to be clustered.
        data_ref_arr (list of ints): Indices referring to elements in data_store.
        centroids (list of numpy arrays): Initial cluster centroids.
        FIRST_FLAG (bool): Flag to indicate the first iteration (default False).

    Returns:
        new_data_ref_clusters (list of lists): Updated cluster assignments for data references.
        new_centroids (numpy array): Updated cluster centroids.
    """

    data_ref_clusters = [[] for _ in centroids]
    nn = None
    pdist = None
    min_dist = float("inf")
    for dref_idx, dref in enumerate(data_ref_arr):
        min_dist = float("inf")
        nn = None
        for ctx, ctr in enumerate(centroids):
            if FIRST_FLAG:
                if torch.equal(data_store[dref].m1, ctr.m1):
                    continue


            pdist = custom_distance(data_store[dref],ctr)
            if pdist < min_dist:
                min_dist = pdist
                nn = ctx
        if nn != None:
            data_ref_clusters[nn].append(data_ref_arr[dref_idx])
        else:
            print("no parent?")

    new_data_ref_clusters = []
    new_centroids = []
    for i in range(len(data_ref_clusters)):
        if not len(data_ref_clusters[i]):
            continue
        new_data_ref_clusters.append(data_ref_clusters[i])
        new_centroids.append(DatumT())
        new_centroids[-1].m1 = torch.mean(torch.stack([data_store[i].m1 for i in new_data_ref_clusters[-1]]),axis=0)
    
    return new_data_ref_clusters, new_centroids
