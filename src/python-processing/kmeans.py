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
    for i, img in enumerate(data):
        min_dist = float("inf")
        nn = None
        for idx, ctr in enumerate(centroids):
            # if np.array_equal(data[i], ctr):
            #     continue
            pdist = distance(data[i], ctr)
            if pdist < min_dist:
                min_dist = pdist
                nn = idx
        if nn != None:
            clusters[nn].append(data[i])

    # convert clusters lists to numpy arrays
    new_clusters = []
    for i in range(len(clusters)):
        if not len(clusters[i]):
            # clusters[i] = np.empty(shape)
            continue
        new_clusters.append(np.stack(np.array([j for j in clusters[i]])))

    return new_clusters


def update_centroids(clusters):
    """
    Evaluate new centroids based on the existing clusters
    Returns an array of np arrays
    """
    return np.array(
        [
            np.mean(cluster, axis=0)
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
        D = lambda ck, m: np.linalg.norm(ck - m)
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
